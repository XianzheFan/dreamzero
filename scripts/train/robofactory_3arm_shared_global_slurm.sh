#!/bin/bash
# Login-node launcher for shared-global 3-arm DreamZero RoboFactory training.
#
# Defaults to ThreeRobotsStackCube-rf. Override ROBOFACTORY_TASK or
# ROBOFACTORY_DATA_ROOT for another converted 3-arm task, for example
# CameraAlignment-rf.
#
# Usage:
#   ssh oci-nrt-cs-001-login-02
#   cd /lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/dreamzero
#   bash scripts/train/robofactory_3arm_shared_global_slurm.sh

set -e

ROBOFACTORY_TASK=${ROBOFACTORY_TASK:-ThreeRobotsStackCube-rf}
SHARED_GLOBAL=${SHARED_GLOBAL:-1}
TARGET_STEPS=${TARGET_STEPS:-50000}
MAX_STEPS=${MAX_STEPS:-50000}
BATCH_SIZE=${BATCH_SIZE:-1}
SAVE_STEPS=${SAVE_STEPS:-500}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
REPORT_TO=${REPORT_TO:-wandb}
WANDB_PROJECT=${WANDB_PROJECT:-dreamzero_robofactory}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-robofactory_3arm_${ROBOFACTORY_TASK}_shared_global_dzrel_50k}
ACTION_LOSS_WEIGHT=${ACTION_LOSS_WEIGHT:-5.0}
GRIPPER_ACTION_LOSS_WEIGHT=${GRIPPER_ACTION_LOSS_WEIGHT:-6.0}
ACTION_PREFIX_LOSS_WEIGHT=${ACTION_PREFIX_LOSS_WEIGHT:-2.0}
ACTION_PREFIX_LOSS_LEN=${ACTION_PREFIX_LOSS_LEN:-8}
OUTPUT_DIR=${OUTPUT_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/checkpoints/robofactory_3arm_${ROBOFACTORY_TASK}_shared_global_dzrel_50k}
ROBOFACTORY_DATA_ROOT=${ROBOFACTORY_DATA_ROOT:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/data/robofactory_lerobot_v2/${ROBOFACTORY_TASK}}

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PARTITION_ARG=""
if [ -n "${PARTITION:-}" ]; then
    PARTITION_ARG="--partition=${PARTITION}"
fi
TIME_ARG=""
if [ -n "${TIME_LIMIT:-}" ]; then
    TIME_ARG="--time=${TIME_LIMIT}"
fi

echo "[$(date)] launching shared-global 3-arm training:"
echo "  ROBOFACTORY_TASK=${ROBOFACTORY_TASK}"
echo "  SHARED_GLOBAL=${SHARED_GLOBAL}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  WANDB_RUN_NAME=${WANDB_RUN_NAME}"
echo "  TARGET_STEPS=${TARGET_STEPS}"
echo "  SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
echo "  ACTION_LOSS_WEIGHT=${ACTION_LOSS_WEIGHT}"
echo "  GRIPPER_ACTION_LOSS_WEIGHT=${GRIPPER_ACTION_LOSS_WEIGHT}"
echo "  ACTION_PREFIX_LOSS_WEIGHT=${ACTION_PREFIX_LOSS_WEIGHT}"
echo "  ACTION_PREFIX_LOSS_LEN=${ACTION_PREFIX_LOSS_LEN}"
echo "  PARTITION=${PARTITION:-(default batch_block1)}"
echo "  TIME_LIMIT=${TIME_LIMIT:-(script default)}"

mkdir -p "$OUTPUT_DIR"

exec sbatch ${PARTITION_ARG} ${TIME_ARG} --export=ALL,\
SHARED_GLOBAL=${SHARED_GLOBAL},\
TARGET_STEPS=${TARGET_STEPS},\
MAX_STEPS=${MAX_STEPS},\
BATCH_SIZE=${BATCH_SIZE},\
SAVE_STEPS=${SAVE_STEPS},\
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT},\
LEARNING_RATE=${LEARNING_RATE},\
REPORT_TO=${REPORT_TO},\
WANDB_PROJECT=${WANDB_PROJECT},\
WANDB_RUN_NAME=${WANDB_RUN_NAME},\
ACTION_LOSS_WEIGHT=${ACTION_LOSS_WEIGHT},\
GRIPPER_ACTION_LOSS_WEIGHT=${GRIPPER_ACTION_LOSS_WEIGHT},\
ACTION_PREFIX_LOSS_WEIGHT=${ACTION_PREFIX_LOSS_WEIGHT},\
ACTION_PREFIX_LOSS_LEN=${ACTION_PREFIX_LOSS_LEN},\
OUTPUT_DIR=${OUTPUT_DIR},\
ROBOFACTORY_DATA_ROOT=${ROBOFACTORY_DATA_ROOT},\
NUM_ARMS=3 \
    "${REPO_DIR}/scripts/train/robofactory_bimanual_slurm.sh"
