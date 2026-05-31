#!/bin/bash
#SBATCH --job-name=dz_rt_bimanual
#SBATCH --account=nvr_lpr_agentic
#SBATCH --partition=batch_short
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=224
#SBATCH --time=01:50:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/logs/dz_rt_bimanual/%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/logs/dz_rt_bimanual/%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#
# Chained 8xH100 training for DreamZero RoboTwin Franka bimanual
# (beat_block_hammer-rt-dzrel), using the shared-global camera layout and
# DreamZero's canonical relative-action training path for joint targets.
#
# Submit from a login node:
#   cd /lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/dreamzero
#   sbatch scripts/train/robotwin_bimanual_slurm.sh
#
# Override target/step counts:
#   TARGET_STEPS=50000 MAX_STEPS=50000 sbatch --export=ALL ...

# NOTE: do not enable `set -u`.

# ----- config -----
TARGET_STEPS=${TARGET_STEPS:-50000}
MAX_STEPS=${MAX_STEPS:-$TARGET_STEPS}
BATCH_SIZE=${BATCH_SIZE:-1}
SAVE_STEPS=${SAVE_STEPS:-500}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
REPORT_TO=${REPORT_TO:-wandb}
WANDB_PROJECT=${WANDB_PROJECT:-dreamzero_robotwin}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-robotwin_franka_bimanual_beat_block_hammer_droid_ma_sg_dzrel_50k}
OUTPUT_DIR=${OUTPUT_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/checkpoints/robotwin_franka_bimanual_beat_block_hammer_droid_ma_sg_dzrel_50k}
ROBOTWIN_DATA_ROOT=${ROBOTWIN_DATA_ROOT:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/data/robotwin_lerobot_v2/beat_block_hammer-rt-dzrel}

WANDB_ID_FILE="${OUTPUT_DIR}/wandb_run_id"

REPO_DIR=/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/dreamzero
CONDA_BASE=/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/miniconda3
CONDA_ENV=dreamzero
LOG_DIR=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/logs/dz_rt_bimanual
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# ----- already done? -----
if [ -d "$OUTPUT_DIR/checkpoint-${TARGET_STEPS}" ]; then
    echo "[$(date)] Target checkpoint-${TARGET_STEPS} already present. Done."
    exit 0
fi

# ----- USR1 handler: forward to training -----
trap 'echo "[$(date)] Received USR1, forwarding to training..."; kill -USR1 "${TRAIN_PID}" 2>/dev/null || true' USR1
CANCELLED=0
trap 'echo "[$(date)] Received TERM/INT, cancelling training..."; CANCELLED=1; kill -TERM "${TRAIN_PID}" 2>/dev/null || true' TERM INT

export NCCL_IB_QPS_PER_CONNECTION=4

# Wandb persistent run id across chained jobs
mkdir -p "$(dirname "$WANDB_ID_FILE")"
if [ ! -s "$WANDB_ID_FILE" ]; then
    head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n' | head -c 8 > "$WANDB_ID_FILE"
    echo "[$(date)] Generated new wandb run id: $(cat $WANDB_ID_FILE)"
fi
export WANDB_RUN_ID="$(cat $WANDB_ID_FILE)"
export WANDB_RESUME=allow
export WANDB_DIR="${OUTPUT_DIR}"
export WANDB_CACHE_DIR="${OUTPUT_DIR}/wandb_cache"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

# CUDA env from conda
export CUDA_HOME="${CONDA_BASE}/envs/${CONDA_ENV}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"

echo "[$(date)] Starting training: data=$ROBOTWIN_DATA_ROOT, output=$OUTPUT_DIR, max_steps=$MAX_STEPS"

NUM_GPUS=8 \
MAX_STEPS=${MAX_STEPS} \
BATCH_SIZE=${BATCH_SIZE} \
SAVE_STEPS=${SAVE_STEPS} \
LEARNING_RATE=${LEARNING_RATE} \
REPORT_TO=${REPORT_TO} \
WANDB_PROJECT=${WANDB_PROJECT} \
WANDB_RUN_NAME=${WANDB_RUN_NAME} \
OUTPUT_DIR=${OUTPUT_DIR} \
ROBOTWIN_DATA_ROOT=${ROBOTWIN_DATA_ROOT} \
    bash scripts/train/robotwin_bimanual_training.sh &
TRAIN_PID=$!
START_TS=${SECONDS}
wait "${TRAIN_PID}"
TRAIN_RC=$?
DURATION=$(( SECONDS - START_TS ))
echo "[$(date)] Training exited with rc=${TRAIN_RC} after ${DURATION}s"

# ----- chain next job unless target reached -----
if [ "${CANCELLED}" = "1" ]; then
    echo "[$(date)] Cancelled by signal. NOT chaining."
    exit "${TRAIN_RC}"
fi

if [ "${TRAIN_RC}" -ne 0 ] && grep -Eq "OutOfMemoryError|CUDA out of memory" "${LOG_DIR}/${SLURM_JOB_ID}.err" 2>/dev/null; then
    echo "[$(date)] CUDA OOM detected. NOT chaining. Adjust memory settings and resubmit manually."
    exit "${TRAIN_RC}"
fi

if [ -d "$OUTPUT_DIR/checkpoint-${TARGET_STEPS}" ]; then
    echo "[$(date)] Reached checkpoint-${TARGET_STEPS}. Chain complete."
    exit 0
fi

# Safety: don't chain on quick fatal errors.
if [ "${TRAIN_RC}" -ne 0 ] && [ "${DURATION}" -lt 600 ]; then
    echo "[$(date)] Fast failure (rc=${TRAIN_RC}, ${DURATION}s) — NOT chaining. Fix the bug then resubmit manually."
    exit "${TRAIN_RC}"
fi

NEXT=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} \
    --partition="${SLURM_JOB_PARTITION:-batch_short}" \
    --export=ALL,TARGET_STEPS=${TARGET_STEPS},MAX_STEPS=${MAX_STEPS},BATCH_SIZE=${BATCH_SIZE},SAVE_STEPS=${SAVE_STEPS},LEARNING_RATE=${LEARNING_RATE},REPORT_TO=${REPORT_TO},WANDB_PROJECT=${WANDB_PROJECT},WANDB_RUN_NAME=${WANDB_RUN_NAME},OUTPUT_DIR=${OUTPUT_DIR},ROBOTWIN_DATA_ROOT=${ROBOTWIN_DATA_ROOT} \
    "${REPO_DIR}/scripts/train/robotwin_bimanual_slurm.sh")
echo "[$(date)] Queued follow-up job: ${NEXT} on partition=${SLURM_JOB_PARTITION:-batch_short}"
