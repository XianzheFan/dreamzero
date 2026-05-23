#!/bin/bash
#SBATCH --job-name=df_rf_bimanual
#SBATCH --account=nvr_lpr_agentic
#SBATCH --partition=batch_block1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=224
#SBATCH --time=03:50:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/logs/df_rf_bimanual/%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/logs/df_rf_bimanual/%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#
# Chained 48h training for DreamZero RoboFactory bimanual.
# - batch_block1 = 4h max; we ask 3h50m so SIGUSR1 fires with ~5min slack
#   and the next job can be queued before we hit the wall.
# - resume: experiment.py auto-picks up the latest checkpoint in OUTPUT_DIR,
#   so each chained job picks up where the previous left off.
# - chaining: at the end of this script we sbatch ourselves with
#   --dependency=afterany so the chain keeps going until checkpoint-${TARGET_STEPS}
#   exists, at which point we exit without resubmitting.
#
# Submit from a login node (NOT a compute node):
#   ssh oci-nrt-cs-001-login-02
#   cd /lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/dreamzero
#   sbatch scripts/train/robofactory_bimanual_slurm.sh
#
# To override target/step counts:
#   TARGET_STEPS=12000 MAX_STEPS=12000 sbatch --export=ALL ...

# NOTE: do not enable `set -u` — the cuda-nvcc conda activate hook references
# NVCC_PREPEND_FLAGS without a default, which trips nounset.

# ----- config (override via --export=ALL,FOO=bar) -----
TARGET_STEPS=${TARGET_STEPS:-12000}      # chain stops once this ckpt exists
MAX_STEPS=${MAX_STEPS:-$TARGET_STEPS}    # trainer's max_steps
BATCH_SIZE=${BATCH_SIZE:-1}
SAVE_STEPS=${SAVE_STEPS:-500}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
REPORT_TO=${REPORT_TO:-wandb}
WANDB_PROJECT=${WANDB_PROJECT:-dreamzero_robofactory}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-robofactory_bimanual_liftbarrier_v3_droid}
# Pin wandb's run id so chained jobs append to the SAME run instead of
# spawning a new one each segment. Stored next to ckpts so all chain links
# read it. (Generated on first launch if missing.)
WANDB_ID_FILE="${OUTPUT_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/checkpoints/robofactory_bimanual_liftbarrier_v3_droid}/wandb_run_id"
OUTPUT_DIR=${OUTPUT_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/checkpoints/robofactory_bimanual_liftbarrier_v3_droid}
ROBOFACTORY_DATA_ROOT=${ROBOFACTORY_DATA_ROOT:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/data/robofactory_lerobot_v2/LiftBarrier-rf}

REPO_DIR=/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/dreamzero
CONDA_BASE=/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/miniconda3
CONDA_ENV=dreamzero
LOG_DIR=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/logs/df_rf_bimanual
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# ----- already done? bail out before consuming the slot -----
if [ -d "$OUTPUT_DIR/checkpoint-${TARGET_STEPS}" ]; then
    echo "[$(date)] Target checkpoint-${TARGET_STEPS} already present. Done."
    exit 0
fi

# ----- USR1 handler: forward to training so trainer can flush a ckpt -----
trap 'echo "[$(date)] Received USR1, forwarding to training..."; kill -USR1 "${TRAIN_PID}" 2>/dev/null || true' USR1

# NCCL config (fine for single node, kept for parity with backfill template)
export NCCL_IB_QPS_PER_CONNECTION=4

# Wandb: persistent run id across chained jobs + lustre-backed state dir
mkdir -p "$(dirname "$WANDB_ID_FILE")"
if [ ! -s "$WANDB_ID_FILE" ]; then
    # 8-char hex slug, e.g. "1a3f9b27"
    head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n' | head -c 8 > "$WANDB_ID_FILE"
    echo "[$(date)] Generated new wandb run id: $(cat $WANDB_ID_FILE)"
fi
export WANDB_RUN_ID="$(cat $WANDB_ID_FILE)"
export WANDB_RESUME=allow
export WANDB_DIR="${OUTPUT_DIR}"
export WANDB_CACHE_DIR="${OUTPUT_DIR}/wandb_cache"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

# CUDA env — batch nodes do not have /usr/local/cuda (that's container-only).
# We installed nvcc/cuda-nvcc 12.8 into the dreamzero conda env on /lustre/,
# so point CUDA_HOME at the env root (conda installs cuda toolkit with the
# usual {bin,lib,include} layout under the env prefix).
export CUDA_HOME="${CONDA_BASE}/envs/${CONDA_ENV}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# ----- run the training (same script used for the nohup smoke) -----
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"

NUM_GPUS=8 \
MAX_STEPS=${MAX_STEPS} \
BATCH_SIZE=${BATCH_SIZE} \
SAVE_STEPS=${SAVE_STEPS} \
LEARNING_RATE=${LEARNING_RATE} \
REPORT_TO=${REPORT_TO} \
WANDB_PROJECT=${WANDB_PROJECT} \
WANDB_RUN_NAME=${WANDB_RUN_NAME} \
OUTPUT_DIR=${OUTPUT_DIR} \
ROBOFACTORY_DATA_ROOT=${ROBOFACTORY_DATA_ROOT} \
    bash scripts/train/robofactory_bimanual_training.sh &
TRAIN_PID=$!
START_TS=${SECONDS}
wait "${TRAIN_PID}"
TRAIN_RC=$?
DURATION=$(( SECONDS - START_TS ))
echo "[$(date)] Training exited with rc=${TRAIN_RC} after ${DURATION}s"

# ----- chain next job unless target reached -----
if [ -d "$OUTPUT_DIR/checkpoint-${TARGET_STEPS}" ]; then
    echo "[$(date)] Reached checkpoint-${TARGET_STEPS}. Chain complete."
    exit 0
fi

# Safety: don't chain on quick fatal errors (e.g. import / config bug).
# Preemption / wall-time exit fires USR1 first, well past the 600s mark.
if [ "${TRAIN_RC}" -ne 0 ] && [ "${DURATION}" -lt 600 ]; then
    echo "[$(date)] Fast failure (rc=${TRAIN_RC}, ${DURATION}s) — NOT chaining. Fix the bug then resubmit manually."
    exit "${TRAIN_RC}"
fi

# afterany so a preemption still triggers the next job (it will resume).
NEXT=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} \
    --export=ALL,TARGET_STEPS=${TARGET_STEPS},MAX_STEPS=${MAX_STEPS},BATCH_SIZE=${BATCH_SIZE},SAVE_STEPS=${SAVE_STEPS},LEARNING_RATE=${LEARNING_RATE},REPORT_TO=${REPORT_TO},WANDB_PROJECT=${WANDB_PROJECT},WANDB_RUN_NAME=${WANDB_RUN_NAME},OUTPUT_DIR=${OUTPUT_DIR},ROBOFACTORY_DATA_ROOT=${ROBOFACTORY_DATA_ROOT} \
    "${REPO_DIR}/scripts/train/robofactory_bimanual_slurm.sh")
echo "[$(date)] Queued follow-up job: ${NEXT}"
