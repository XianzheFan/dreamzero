#!/bin/bash
#SBATCH --job-name=df_rf_eval
#SBATCH --account=nvr_lpr_agentic
#SBATCH --partition=batch_block1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=03:50:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/logs/df_rf_eval/%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/logs/df_rf_eval/%j.err
#
# Closed-loop RoboFactory eval for the bimanual DreamZero policy.
# Server (dreamzero env, GPU 0) + ManiSkill client (RoboFactory env, GPU 1)
# on the same batch node, talking over localhost ws.
#
# Submit from a login node:
#   ssh oci-nrt-cs-001-login-02
#   cd /lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/dreamzero
#   CKPT_DIR=/lustre/.../robofactory_bimanual_liftbarrier_v4_long_50k \
#   CKPT_SETTING=checkpoint-2000 \
#   sbatch scripts/eval/robofactory_eval_slurm.sh
#
# All defaults below point at the v4_long_50k run / checkpoint-2000.

CKPT_DIR=${CKPT_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/checkpoints/robofactory_bimanual_liftbarrier_v4_long_50k}
CKPT_SETTING=${CKPT_SETTING:-checkpoint-2000}
TASK=${TASK:-LiftBarrier-rf}
NUM_EPISODES=${NUM_EPISODES:-10}
SEED_START=${SEED_START:-1000}
MAX_STEPS=${MAX_STEPS:-300}
REPLAN_EVERY=${REPLAN_EVERY:-8}
PORT=${PORT:-5011}
PROMPT=${PROMPT:-"the two robot arms lift the steel barrier together off the table"}
# Set SAVE_VIDEO_PRED=1 to have the server VAE-decode its denoised video
# latents at each infer() call and write one mp4 per agent. Adds ~5-15s
# per inference (heavy VAE decode); use only for diagnostics.
SAVE_VIDEO_PRED=${SAVE_VIDEO_PRED:-0}

REPO_DIR=/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/dreamzero
CONDA_BASE=/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/miniconda3
DREAMZERO_PY=${CONDA_BASE}/envs/dreamzero/bin/python
ROBOFACTORY_ENV=${CONDA_BASE}/envs/RoboFactory
ROBOFACTORY_PY=${ROBOFACTORY_ENV}/bin/python

EVAL_TAG=$(basename "${CKPT_DIR}")_${CKPT_SETTING}
OUT_BASE=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/logs/df_rf_eval/${EVAL_TAG}_${SLURM_JOB_ID:-manual}
mkdir -p "${OUT_BASE}" "$(dirname "${OUT_BASE}")"
SERVER_LOG=${OUT_BASE}/server.log
CLIENT_LOG=${OUT_BASE}/client.log
RESULTS_JSON=${OUT_BASE}/results.json
VIDEO_DIR=${OUT_BASE}/videos

echo "[$(date)] eval start"
echo "  CKPT_DIR=${CKPT_DIR}"
echo "  CKPT_SETTING=${CKPT_SETTING}"
echo "  TASK=${TASK}  num_episodes=${NUM_EPISODES}  seed_start=${SEED_START}"
echo "  out_dir=${OUT_BASE}"

# Same CUDA setup as the training slurm — batch nodes ship the host NV
# user-space libs in /usr/lib/x86_64-linux-gnu, no container-side cuda toolkit.
export CUDA_HOME=${CONDA_BASE}/envs/dreamzero
export PATH=${CUDA_HOME}/bin:${PATH}

# Multi-agent sparse hub attention applies an attn_mask FlashAttention 2
# doesn't support; force torch (eager) backend (same as training/serve).
export ATTENTION_BACKEND=torch
export HYDRA_FULL_ERROR=1

cleanup() {
    if [ -n "${SERVER_PID:-}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[$(date)] stopping policy server (pid=${SERVER_PID})"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ----- 1) launch policy server on GPU 0 (dreamzero env) -----
SAVE_FLAG=""
if [ "${SAVE_VIDEO_PRED}" = "1" ]; then
    SAVE_FLAG="--save-video-pred --video-pred-dir ${OUT_BASE}/video_pred"
fi
echo "[$(date)] launching server -> ${SERVER_LOG}"
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-} \
"${DREAMZERO_PY}" -m eval_utils.bimanual_policy_server \
    --ckpt-dir "${CKPT_DIR}" \
    --ckpt-setting "${CKPT_SETTING}" \
    --host 127.0.0.1 --port "${PORT}" \
    --image-h 240 --image-w 320 \
    --num-frames 33 --action-horizon 24 \
    ${SAVE_FLAG} \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

# ----- 2) wait until server is listening (or it dies) -----
echo "[$(date)] waiting for server ready (server pid=${SERVER_PID})"
DEADLINE=$(( SECONDS + 900 ))   # 15 min wall for the 4-step load
while true; do
    if grep -q "server listening on" "${SERVER_LOG}" 2>/dev/null; then
        echo "[$(date)] server is up"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[$(date)] FATAL: server exited before listening. Last 40 lines:"
        tail -n 40 "${SERVER_LOG}"
        exit 1
    fi
    if [ "${SECONDS}" -ge "${DEADLINE}" ]; then
        echo "[$(date)] FATAL: server didn't come up within 15 min. Last 40 lines:"
        tail -n 40 "${SERVER_LOG}"
        exit 1
    fi
    sleep 10
done

# ----- 3) run RoboFactory eval client on GPU 1 -----
# Sapien needs an NVIDIA Vulkan ICD it can talk to. Point it at the
# sapien-bundled JSON; if the batch node's NV driver is properly mounted
# (unlike the jupyter container), this is enough.
SAPIEN_VKLIB=${ROBOFACTORY_ENV}/lib/python3.9/site-packages/sapien/vulkan_library
export VK_ICD_FILENAMES=${SAPIEN_VKLIB}/nvidia_icd.json
export __EGL_VENDOR_LIBRARY_FILENAMES=${SAPIEN_VKLIB}/10_nvidia.json
export LD_LIBRARY_PATH=${ROBOFACTORY_ENV}/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

echo "[$(date)] launching client -> ${CLIENT_LOG}"
CUDA_VISIBLE_DEVICES=1 \
"${ROBOFACTORY_PY}" "${REPO_DIR}/scripts/eval/eval_robofactory_ws.py" \
    --host 127.0.0.1 --port "${PORT}" \
    --task "${TASK}" \
    --seed-start "${SEED_START}" \
    --num-episodes "${NUM_EPISODES}" \
    --max-steps "${MAX_STEPS}" \
    --replan-every "${REPLAN_EVERY}" \
    --prompt "${PROMPT}" \
    --log "${RESULTS_JSON}" \
    --video-dir "${VIDEO_DIR}" \
    > "${CLIENT_LOG}" 2>&1
CLIENT_RC=$?

echo "[$(date)] client exited rc=${CLIENT_RC}"
echo "--- last 20 lines of client log ---"
tail -n 20 "${CLIENT_LOG}"
echo "--- results.json ---"
if [ -f "${RESULTS_JSON}" ]; then
    "${DREAMZERO_PY}" -c "import json; d=json.load(open('${RESULTS_JSON}')); print(f\"success_rate={d.get('success_rate')}  n={d.get('n_completed')}/{d.get('n_target')}\")"
fi

exit "${CLIENT_RC}"
