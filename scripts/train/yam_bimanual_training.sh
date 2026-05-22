#!/bin/bash
# DreamZero YAM **bimanual** smoke training (PR 9e+9f).
#
# Wires:
#   - data=dreamzero/yam_bimanual_relative  (P=2 axis on state/action)
#   - model/dreamzero/transform=bimanual_cotrain  (per-agent V-tiling)
#   - num_agents=2 on CausalWanModel, per-agent state/action dims 7/7
#   - partial load from DreamZero-AgiBot single-agent ckpt (shape-mismatch
#     filter in base.py drops the per-agent state_encoder /
#     action_encoder / action_decoder / patch_embedding tensors -- the
#     DiT body still loads from pretrained).
#
# Smoke defaults: 3 steps, single GPU, batch size 1. Bump
# MAX_STEPS / NUM_GPUS / per-device batch for real runs.
#
# Usage:
#   YAM_DATA_ROOT=/lustre/.../yam_v2 \
#   OUTPUT_DIR=$HOME/checkpoints/yam_bimanual_smoke \
#   bash scripts/train/yam_bimanual_training.sh

export HYDRA_FULL_ERROR=1

YAM_DATA_ROOT=${YAM_DATA_ROOT:-"/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/data/yam_v2"}
OUTPUT_DIR=${OUTPUT_DIR:-"/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/checkpoints/yam_bimanual_smoke"}
NUM_GPUS=${NUM_GPUS:-1}
MAX_STEPS=${MAX_STEPS:-3}
BATCH_SIZE=${BATCH_SIZE:-1}

WAN_CKPT_DIR=${WAN_CKPT_DIR:-"/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/checkpoints/umt5-xxl"}
PRETRAINED_DIR=${PRETRAINED_DIR:-"/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/checkpoints/DreamZero-AgiBot"}

# Auto-download Wan2.1 base weights if missing (same as single-agent script).
if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.1-I2V-14B-480P not found at $WAN_CKPT_DIR. Downloading from HuggingFace..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi
if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi
if [ ! -d "$YAM_DATA_ROOT" ]; then
    echo "ERROR: YAM bimanual dataset not found at $YAM_DATA_ROOT"
    exit 1
fi

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=none \
    wandb_project=dreamzero_bimanual_smoke \
    data=dreamzero/yam_bimanual_relative \
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
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=$MAX_STEPS \
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
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    yam_data_root=$YAM_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=$PRETRAINED_DIR \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true \
    ++action_head_cfg.config.max_state_dim=7 \
    ++action_head_cfg.config.action_dim=7 \
    ++action_head_cfg.config.diffusion_model_cfg.num_agents=2 \
    ++action_head_cfg.config.diffusion_model_cfg.max_state_dim=7 \
    ++action_head_cfg.config.diffusion_model_cfg.action_dim=7 \
    ++action_head_cfg.config.diffusion_model_cfg.concat_first_frame_latent=false \
    ++action_head_cfg.config.diffusion_model_cfg.in_dim=16
