#!/bin/bash
# Launch the multi-agent (bimanual) DreamZero inference server on the
# dreamzero conda env (Python 3.11). Stays alive until killed; pair
# with the RoboTwin eval client at policy/DreamZero/eval.sh.
#
# Usage:
#   PORT=5001 CKPT_DIR=/path/to/training_output CKPT_SETTING=checkpoint-10 \
#     bash scripts/inference/serve_bimanual.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PORT=${PORT:-5001}
HOST=${HOST:-0.0.0.0}
CKPT_DIR=${CKPT_DIR:-"/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/checkpoints/robotwin_bimanual_smoke"}
CKPT_SETTING=${CKPT_SETTING:-checkpoint-10}
# Match the LeRobot v2 mp4 raw dimensions written by
# scripts/data/robofactory_to_lerobot_v2.py (the transform chain checks
# input resolution exactly; the in-chain Resize downsizes to model target).
IMAGE_H=${IMAGE_H:-240}
IMAGE_W=${IMAGE_W:-320}
NUM_FRAMES=${NUM_FRAMES:-33}
ACTION_HORIZON=${ACTION_HORIZON:-24}

# Multi-agent sparse hub attention applies an attn_mask FlashAttention 2
# doesn't support; force the torch (eager) backend (same as training).
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-torch}
export HYDRA_FULL_ERROR=1

if [ ! -d "$CKPT_DIR/$CKPT_SETTING" ]; then
    echo "ERROR: $CKPT_DIR/$CKPT_SETTING not found" >&2
    exit 1
fi

echo "Starting bimanual policy server on ${HOST}:${PORT}"
echo "  CKPT_DIR=$CKPT_DIR"
echo "  CKPT_SETTING=$CKPT_SETTING"

exec python -m eval_utils.bimanual_policy_server \
    --ckpt-dir "$CKPT_DIR" \
    --ckpt-setting "$CKPT_SETTING" \
    --host "$HOST" \
    --port "$PORT" \
    --image-h "$IMAGE_H" \
    --image-w "$IMAGE_W" \
    --num-frames "$NUM_FRAMES" \
    --action-horizon "$ACTION_HORIZON"
