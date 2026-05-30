#!/bin/bash
# DreamZero RoboTwin (Franka/Panda) bimanual smoke training.
#
# Mirrors robofactory_bimanual_training.sh exactly except for the data
# root and run name, because RoboTwin franka-panda exposes the same
# 7+1 dim per-arm layout that the ``robofactory`` embodiment tag
# already encodes. The RoboTwin data is produced by
# ``scripts/data/robotwin_to_lerobot_v2.py`` (which writes the
# embodiment_tag as ``robofactory`` so it slots in to the same configs).
#
# Usage:
#   ROBOTWIN_DATA_ROOT=/path/to/lerobot_v2/beat_block_hammer-rt \
#   OUTPUT_DIR=$HOME/checkpoints/robotwin_bimanual_smoke \
#   PRETRAINED_DIR=/path/to/DreamZero-DROID \
#   bash scripts/train/robotwin_bimanual_training.sh

export HYDRA_FULL_ERROR=1
# Multi-agent sparse hub attention uses a custom token-routing topology.
# ``flex`` runs the masked path through torch.compile'd flex_attention and
# avoids the legacy torch math kernel's dense [B,H,N,N] score matrix.
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-flex}

# RoboTwin's 33-frame, 3-view bimanual batches still sit close to the
# 80 GB H100 limit. Keep recomputation on by default; FlexAttention
# removes the dense score-matrix blowup while gradient checkpointing
# keeps per-layer activations from filling the card.
GRAD_CKPT=${GRAD_CKPT:-true}
DEEPSPEED_CFG=${DEEPSPEED_CFG:-groot/vla/configs/deepspeed/zero2.json}
DATA_CFG=${DATA_CFG:-dreamzero/robotwin_franka_bimanual_relative}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

ROBOTWIN_DATA_ROOT=${ROBOTWIN_DATA_ROOT:-"/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/data/robotwin_lerobot_v2/beat_block_hammer-rt"}
OUTPUT_DIR=${OUTPUT_DIR:-"/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/checkpoints/robotwin_franka_bimanual_smoke"}
NUM_GPUS=${NUM_GPUS:-1}
MAX_STEPS=${MAX_STEPS:-10}
BATCH_SIZE=${BATCH_SIZE:-1}
SAVE_STEPS=${SAVE_STEPS:-$MAX_STEPS}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
REPORT_TO=${REPORT_TO:-none}
WANDB_PROJECT=${WANDB_PROJECT:-dreamzero_robotwin_smoke}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-robotwin_franka_bimanual_smoke}

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
    ++action_head_cfg.config.use_gradient_checkpointing=$GRAD_CKPT \
    ++action_head_cfg.config.max_state_dim=8 \
    ++action_head_cfg.config.action_dim=8 \
    ++action_head_cfg.config.diffusion_model_cfg.num_agents=2 \
    ++action_head_cfg.config.diffusion_model_cfg.max_state_dim=8 \
    ++action_head_cfg.config.diffusion_model_cfg.action_dim=8 \
    ++action_head_cfg.config.diffusion_model_cfg.concat_first_frame_latent=false \
    ++action_head_cfg.config.diffusion_model_cfg.in_dim=16
