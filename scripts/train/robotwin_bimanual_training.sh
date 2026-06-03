#!/bin/bash
# DreamZero RoboTwin (Franka/Panda) bimanual smoke training.
#
# RoboTwin franka-panda exposes the same 7+1 per-arm tensor layout as
# RoboFactory, but uses its own ``robotwin`` embodiment/config namespace so
# simulator-specific gripper semantics stay separate. RoboTwin gripper commands
# are [0, 1], open=1.0, close=0.0. The converter stores absolute next-qpos
# action targets; the RoboTwin data config enables DreamZero's relative-action
# path for joint keys only.
#
# Usage:
#   ROBOTWIN_DATA_ROOT=/path/to/lerobot_v2/stack_blocks_two-rt \
#   OUTPUT_DIR=$HOME/checkpoints/robotwin_bimanual_smoke \
#   PRETRAINED_DIR=/path/to/DreamZero-DROID \
#   bash scripts/train/robotwin_bimanual_training.sh

export HYDRA_FULL_ERROR=1
# Multi-agent sparse hub attention uses a custom token-routing topology.
# ``flex`` runs the masked path through torch.compile'd flex_attention and
# avoids the legacy torch math kernel's dense [B,H,N,N] score matrix.
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-flex}

# RoboTwin's 33-frame bimanual batches sit close to the 80 GB H100 limit.
# Keep recomputation on and default to the shared-global data layout so the
# scene camera is encoded once instead of being duplicated into both agents.
GRAD_CKPT=${GRAD_CKPT:-true}
# ZeRO-2 CPU offload gives a few extra GB of GPU headroom for long runs.
DEEPSPEED_CFG=${DEEPSPEED_CFG:-groot/vla/configs/deepspeed/zero2_offload.json}
DATA_CFG=${DATA_CFG:-dreamzero/robotwin_franka_bimanual_shared_global}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

ROBOTWIN_DATA_ROOT=${ROBOTWIN_DATA_ROOT:-"/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/data/robotwin_lerobot_v2/stack_blocks_two-rt"}
OUTPUT_DIR=${OUTPUT_DIR:-"/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/checkpoints/robotwin_franka_bimanual_shared_global_dzrel_smoke"}
NUM_GPUS=${NUM_GPUS:-1}
MAX_STEPS=${MAX_STEPS:-10}
BATCH_SIZE=${BATCH_SIZE:-1}
SAVE_STEPS=${SAVE_STEPS:-$MAX_STEPS}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
REPORT_TO=${REPORT_TO:-none}
WANDB_PROJECT=${WANDB_PROJECT:-dreamzero_robotwin_smoke}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-robotwin_franka_bimanual_shared_global_dzrel_smoke}
ACTION_LOSS_WEIGHT=${ACTION_LOSS_WEIGHT:-5.0}
GRIPPER_ACTION_LOSS_WEIGHT=${GRIPPER_ACTION_LOSS_WEIGHT:-6.0}
GRIPPER_CLOSE_ACTION_LOSS_WEIGHT=${GRIPPER_CLOSE_ACTION_LOSS_WEIGHT:-4.0}
GRIPPER_CLOSE_THRESHOLD=${GRIPPER_CLOSE_THRESHOLD:-0.0}
GRIPPER_ACTION_DIMS=${GRIPPER_ACTION_DIMS:-7}
GRIPPER_CLEAN_ACTION_LOSS_WEIGHT=${GRIPPER_CLEAN_ACTION_LOSS_WEIGHT:-2.0}
GRIPPER_CLEAN_CLOSE_ACTION_LOSS_WEIGHT=${GRIPPER_CLEAN_CLOSE_ACTION_LOSS_WEIGHT:-4.0}
GRIPPER_CLEAN_MAX_SIGMA=${GRIPPER_CLEAN_MAX_SIGMA:-0.75}
GRIPPER_BINARY_ACTION_LOSS_WEIGHT=${GRIPPER_BINARY_ACTION_LOSS_WEIGHT:-4.0}
GRIPPER_BINARY_CLOSE_ACTION_LOSS_WEIGHT=${GRIPPER_BINARY_CLOSE_ACTION_LOSS_WEIGHT:-6.0}
GRIPPER_BINARY_LOGIT_SCALE=${GRIPPER_BINARY_LOGIT_SCALE:-4.0}
GRIPPER_BINARY_MAX_SIGMA=${GRIPPER_BINARY_MAX_SIGMA:-0.75}
ACTION_PREFIX_LOSS_WEIGHT=${ACTION_PREFIX_LOSS_WEIGHT:-2.0}
ACTION_PREFIX_LOSS_LEN=${ACTION_PREFIX_LOSS_LEN:-8}

WAN_CKPT_DIR=${WAN_CKPT_DIR:-"/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/checkpoints/umt5-xxl"}
PRETRAINED_DIR=${PRETRAINED_DIR:-"/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/checkpoints/DreamZero-DROID"}

if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.1-I2V-14B-480P not found at $WAN_CKPT_DIR. Downloading from HuggingFace..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi
if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi
if [ ! -d "$ROBOTWIN_DATA_ROOT" ]; then
    echo "ERROR: RoboTwin LeRobot v2 dataset not found at $ROBOTWIN_DATA_ROOT"
    echo "Run scripts/data/robotwin_to_lerobot_v2.py first."
    exit 1
fi

echo "RoboTwin Franka bimanual: data=$DATA_CFG  gradient_checkpointing=$GRAD_CKPT  deepspeed=$DEEPSPEED_CFG"
echo "action_loss_weight=$ACTION_LOSS_WEIGHT  gripper_action_loss_weight=$GRIPPER_ACTION_LOSS_WEIGHT  gripper_close_action_loss_weight=$GRIPPER_CLOSE_ACTION_LOSS_WEIGHT  gripper_close_threshold=$GRIPPER_CLOSE_THRESHOLD  gripper_action_dims=[$GRIPPER_ACTION_DIMS]  gripper_clean_action_loss_weight=$GRIPPER_CLEAN_ACTION_LOSS_WEIGHT  gripper_clean_close_action_loss_weight=$GRIPPER_CLEAN_CLOSE_ACTION_LOSS_WEIGHT  gripper_clean_max_sigma=$GRIPPER_CLEAN_MAX_SIGMA  action_prefix_loss_weight=$ACTION_PREFIX_LOSS_WEIGHT  action_prefix_loss_len=$ACTION_PREFIX_LOSS_LEN"
echo "gripper_binary_action_loss_weight=$GRIPPER_BINARY_ACTION_LOSS_WEIGHT  gripper_binary_close_action_loss_weight=$GRIPPER_BINARY_CLOSE_ACTION_LOSS_WEIGHT  gripper_binary_logit_scale=$GRIPPER_BINARY_LOGIT_SCALE  gripper_binary_max_sigma=$GRIPPER_BINARY_MAX_SIGMA"

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=$REPORT_TO \
    wandb_project=$WANDB_PROJECT \
    +training_args.run_name=$WANDB_RUN_NAME \
    data=$DATA_CFG \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=bimanual_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=$LEARNING_RATE \
    training_args.deepspeed="$DEEPSPEED_CFG" \
    ++training_args.gradient_checkpointing=$GRAD_CKPT \
    save_steps=$SAVE_STEPS \
    training_args.warmup_ratio=0.0 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=$BATCH_SIZE \
    max_steps=$MAX_STEPS \
    weight_decay=1e-5 \
    save_total_limit=5 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=${DATALOADER_NUM_WORKERS:-4} \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    robotwin_data_root=$ROBOTWIN_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=$PRETRAINED_DIR \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true \
    ++action_head_cfg.config.action_loss_weight=$ACTION_LOSS_WEIGHT \
    ++action_head_cfg.config.gripper_action_loss_weight=$GRIPPER_ACTION_LOSS_WEIGHT \
    ++action_head_cfg.config.gripper_close_action_loss_weight=$GRIPPER_CLOSE_ACTION_LOSS_WEIGHT \
    ++action_head_cfg.config.gripper_close_threshold=$GRIPPER_CLOSE_THRESHOLD \
    ++action_head_cfg.config.gripper_action_dims=[$GRIPPER_ACTION_DIMS] \
    ++action_head_cfg.config.gripper_clean_action_loss_weight=$GRIPPER_CLEAN_ACTION_LOSS_WEIGHT \
    ++action_head_cfg.config.gripper_clean_close_action_loss_weight=$GRIPPER_CLEAN_CLOSE_ACTION_LOSS_WEIGHT \
    ++action_head_cfg.config.gripper_clean_max_sigma=$GRIPPER_CLEAN_MAX_SIGMA \
    ++action_head_cfg.config.gripper_binary_action_loss_weight=$GRIPPER_BINARY_ACTION_LOSS_WEIGHT \
    ++action_head_cfg.config.gripper_binary_close_action_loss_weight=$GRIPPER_BINARY_CLOSE_ACTION_LOSS_WEIGHT \
    ++action_head_cfg.config.gripper_binary_logit_scale=$GRIPPER_BINARY_LOGIT_SCALE \
    ++action_head_cfg.config.gripper_binary_max_sigma=$GRIPPER_BINARY_MAX_SIGMA \
    ++action_head_cfg.config.action_prefix_loss_weight=$ACTION_PREFIX_LOSS_WEIGHT \
    ++action_head_cfg.config.action_prefix_loss_len=$ACTION_PREFIX_LOSS_LEN \
    ++action_head_cfg.config.use_gradient_checkpointing=$GRAD_CKPT \
    ++action_head_cfg.config.max_state_dim=8 \
    ++action_head_cfg.config.action_dim=8 \
    ++action_head_cfg.config.diffusion_model_cfg.num_agents=2 \
    ++action_head_cfg.config.diffusion_model_cfg.max_state_dim=8 \
    ++action_head_cfg.config.diffusion_model_cfg.action_dim=8 \
    ++action_head_cfg.config.diffusion_model_cfg.concat_first_frame_latent=false \
    ++action_head_cfg.config.diffusion_model_cfg.in_dim=16
