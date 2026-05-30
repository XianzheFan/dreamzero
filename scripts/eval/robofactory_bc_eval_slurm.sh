#!/bin/bash
#SBATCH --job-name=rf_bc_eval
#SBATCH --account=nvr_lpr_agentic
#SBATCH --partition=batch_block1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/logs/rf_bc_eval/%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/logs/rf_bc_eval/%j.err
#
# Closed-loop RoboFactory eval for the lightweight multi-view BC policy.
#
# Submit from a login node:
#   BC_CKPT=/lustre/.../robofactory_bc_multiview_liftbarrier_v1/best.pt \
#   NUM_EPISODES=10 DUMP_ACTIONS=1 sbatch scripts/eval/robofactory_bc_eval_slurm.sh

BC_CKPT=${BC_CKPT:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/checkpoints/robofactory_bc_multiview_liftbarrier_v1/best.pt}
TASK=${TASK:-LiftBarrier-rf}
NUM_EPISODES=${NUM_EPISODES:-10}
SEED_START=${SEED_START:-1000}
MAX_STEPS=${MAX_STEPS:-300}
REPLAN_EVERY=${REPLAN_EVERY:-8}
PROGRESS_STEPS=${PROGRESS_STEPS:-300}
PORT=${PORT:-5011}
PROMPT=${PROMPT:-"the two robot arms lift the steel barrier together off the table"}
DUMP_ACTIONS=${DUMP_ACTIONS:-0}
SAVE_VIDEO=${SAVE_VIDEO:-1}

REPO_DIR=/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/dreamzero
CONDA_BASE=/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/miniconda3
DREAMZERO_PY=${CONDA_BASE}/envs/dreamzero/bin/python
ROBOFACTORY_ENV=${CONDA_BASE}/envs/RoboFactory
ROBOFACTORY_PY=${ROBOFACTORY_ENV}/bin/python

EVAL_TAG=$(basename "$(dirname "${BC_CKPT}")")_$(basename "${BC_CKPT}" .pt)
OUT_BASE=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/logs/rf_bc_eval/${EVAL_TAG}_${SLURM_JOB_ID:-manual}
mkdir -p "${OUT_BASE}" "$(dirname "${OUT_BASE}")"
SERVER_LOG=${OUT_BASE}/server.log
CLIENT_LOG=${OUT_BASE}/client.log
RESULTS_JSON=${OUT_BASE}/results.json
VIDEO_DIR=${OUT_BASE}/videos
DUMP_DIR=${OUT_BASE}/action_dump

echo "[$(date)] eval start"
echo "  BC_CKPT=${BC_CKPT}"
echo "  TASK=${TASK}  num_episodes=${NUM_EPISODES}  seed_start=${SEED_START}"
echo "  max_steps=${MAX_STEPS} replan_every=${REPLAN_EVERY} progress_steps=${PROGRESS_STEPS}"
echo "  out_dir=${OUT_BASE}"

export CUDA_HOME=${CONDA_BASE}/envs/dreamzero
export PATH=${CUDA_HOME}/bin:${PATH}
export HYDRA_FULL_ERROR=1

cleanup() {
    if [ -n "${SERVER_PID:-}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[$(date)] stopping policy server (pid=${SERVER_PID})"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[$(date)] launching BC server -> ${SERVER_LOG}"
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-} \
"${DREAMZERO_PY}" -m eval_utils.robofactory_bc_policy_server \
    --ckpt "${BC_CKPT}" \
    --progress-steps "${PROGRESS_STEPS}" \
    --host 127.0.0.1 --port "${PORT}" \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "[$(date)] waiting for server ready (server pid=${SERVER_PID})"
DEADLINE=$(( SECONDS + 300 ))
while true; do
    if grep -q "BC policy server listening" "${SERVER_LOG}" 2>/dev/null; then
        echo "[$(date)] server is up"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[$(date)] FATAL: server exited before listening. Last 40 lines:"
        tail -n 40 "${SERVER_LOG}"
        exit 1
    fi
    if [ "${SECONDS}" -ge "${DEADLINE}" ]; then
        echo "[$(date)] FATAL: server didn't come up within 5 min. Last 40 lines:"
        tail -n 40 "${SERVER_LOG}"
        exit 1
    fi
    sleep 2
done

SAPIEN_VKLIB=${ROBOFACTORY_ENV}/lib/python3.9/site-packages/sapien/vulkan_library
export VK_ICD_FILENAMES=${SAPIEN_VKLIB}/nvidia_icd.json
export __EGL_VENDOR_LIBRARY_FILENAMES=${SAPIEN_VKLIB}/10_nvidia.json
export LD_LIBRARY_PATH=${ROBOFACTORY_ENV}/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

DUMP_FLAG=""
if [ "${DUMP_ACTIONS}" = "1" ]; then
    DUMP_FLAG="--dump-actions ${DUMP_DIR}"
fi
VIDEO_FLAG=""
if [ "${SAVE_VIDEO}" = "1" ]; then
    VIDEO_FLAG="--video-dir ${VIDEO_DIR}"
fi

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
    ${VIDEO_FLAG} \
    ${DUMP_FLAG} \
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
