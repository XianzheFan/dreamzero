#!/bin/bash
#SBATCH --job-name=rf_bc_train
#SBATCH --account=nvr_lpr_agentic
#SBATCH --partition=batch_block1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/logs/rf_bc_train/%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/logs/rf_bc_train/%j.err

DATA_ROOT=${DATA_ROOT:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/data/robofactory_lerobot_v2/LiftBarrier-rf}
OUTPUT_DIR=${OUTPUT_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/checkpoints/robofactory_bc_multiview_liftbarrier_v1}
IMAGE_SIZE=${IMAGE_SIZE:-96}
EPOCHS=${EPOCHS:-80}
BATCH_SIZE=${BATCH_SIZE:-128}
HIDDEN_DIM=${HIDDEN_DIM:-512}
LR=${LR:-3e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
VAL_FRACTION=${VAL_FRACTION:-0.15}
SEED=${SEED:-42}
GRIPPER_WEIGHT=${GRIPPER_WEIGHT:-6.0}
FIRST_ACTION_WEIGHT=${FIRST_ACTION_WEIGHT:-2.0}
GRIPPER_BCE_WEIGHT=${GRIPPER_BCE_WEIGHT:-0.5}

REPO_DIR=/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/dreamzero
CONDA_BASE=/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/miniconda3
DREAMZERO_PY=${CONDA_BASE}/envs/dreamzero/bin/python

echo "[$(date)] BC training start"
echo "  DATA_ROOT=${DATA_ROOT}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  IMAGE_SIZE=${IMAGE_SIZE} EPOCHS=${EPOCHS} BATCH_SIZE=${BATCH_SIZE}"

mkdir -p "${OUTPUT_DIR}"
export CUDA_HOME=${CONDA_BASE}/envs/dreamzero
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}

cd "${REPO_DIR}"
CUDA_VISIBLE_DEVICES=0 "${DREAMZERO_PY}" scripts/train/robofactory_bc_training.py \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --image-size "${IMAGE_SIZE}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --val-fraction "${VAL_FRACTION}" \
    --seed "${SEED}" \
    --gripper-weight "${GRIPPER_WEIGHT}" \
    --first-action-weight "${FIRST_ACTION_WEIGHT}" \
    --gripper-bce-weight "${GRIPPER_BCE_WEIGHT}"
