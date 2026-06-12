#!/bin/bash
# DreamZero RoboFactory multi-arm training (2 / 3 / 4 arms).
#
# Set ``NUM_ARMS=3`` or ``NUM_ARMS=4`` to switch to the 3-arm
# (CameraAlignment-rf, ThreeRobotsStackCube-rf) or 4-arm (TakePhoto-rf)
# variants -- the script picks the matching data config and propagates
# ``num_agents`` into ``diffusion_model_cfg``. Set ``SHARED_GLOBAL=1``
# for the shared-global layout currently wired for 2-arm and 3-arm
# RoboFactory training. Default is 2-arm duplicated-global.
#
# Usage:
#   ROBOFACTORY_DATA_ROOT=/path/to/lerobot_v2/LiftBarrier-rf \
#   OUTPUT_DIR=$HOME/checkpoints/robofactory_bimanual_smoke \
#   bash scripts/train/robofactory_bimanual_training.sh
#
#   # 3-arm:
#   NUM_ARMS=3 \
#   ROBOFACTORY_DATA_ROOT=/path/to/lerobot_v2/CameraAlignment-rf \
#   OUTPUT_DIR=$HOME/checkpoints/robofactory_3arm_smoke \
#   bash scripts/train/robofactory_bimanual_training.sh

export HYDRA_FULL_ERROR=1
# Multi-agent sparse hub attention uses a custom token-routing topology
# that FlashAttention 2 cannot express via its built-in causal/window
# flags. Two backends support the masked path:
#   * ``torch`` -- F.scaled_dot_product_attention(attn_mask=[1,1,N,N])
#                  falls back to the O(N^2) math kernel; ~40 s/step at
#                  N ~ 15k tokens (the P=2 sparse-hub case).
#   * ``flex``  -- builds a ``BlockMask`` from the same sparse-hub rule
#                  and runs it through a torch.compile'd ``flex_attention``
#                  Triton kernel that skips empty blocks; ~5-10x faster on
#                  H100 at the same N.
# Default to flex; set ATTENTION_BACKEND=torch to pin the legacy path.
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-flex}

NUM_ARMS=${NUM_ARMS:-2}
# PR 23: SHARED_GLOBAL=1 swaps the data config to the variant that emits
# a separate ``video_global`` stream + per-agent wrist-only video (no
# duplicated global view).
SHARED_GLOBAL=${SHARED_GLOBAL:-0}
# Wan2.1/DROID I2V was pretrained with [latent; first-frame mask/latent]
# patch-embedding channels (16 + 20 = 36). Keep that structure by default
# so video denoising sees the same conditioning path as the base model.
# Set CONCAT_FIRST_FRAME_LATENT=false DIFFUSION_IN_DIM=16 only for
# latent-only ablations.
CONCAT_FIRST_FRAME_LATENT=${CONCAT_FIRST_FRAME_LATENT:-true}
DIFFUSION_IN_DIM=${DIFFUSION_IN_DIM:-36}
case "$NUM_ARMS" in
    2)
        if [ "$SHARED_GLOBAL" = "1" ]; then
            DATA_CFG="dreamzero/robofactory_bimanual_shared_global"
        else
            DATA_CFG="dreamzero/robofactory_bimanual_relative"
        fi
        ;;
    3)
        if [ "$SHARED_GLOBAL" = "1" ]; then
            DATA_CFG="dreamzero/robofactory_3arm_shared_global"
        else
            DATA_CFG="dreamzero/robofactory_3arm_relative"
        fi
        ;;
    4) DATA_CFG="dreamzero/robofactory_4arm_relative" ;;
    *) echo "Unsupported NUM_ARMS=$NUM_ARMS (expected 2, 3, or 4)" >&2; exit 1 ;;
esac
if [ "$SHARED_GLOBAL" = "1" ] && [ "$NUM_ARMS" -eq 4 ]; then
    echo "SHARED_GLOBAL=1 is currently only wired for NUM_ARMS=2 or NUM_ARMS=3" >&2
    exit 1
fi
# Gradient checkpointing trade-off, post FlexAttention switch (PR 20):
#  * P=2 under FlexAttention: the sparse-hub attention no longer
#    materializes the [B,H,N,N] score matrix (~7 GB/layer/forward under
#    the old math kernel), so the full 32-layer activation footprint
#    fits comfortably in 80 GB. Default GRAD_CKPT=false to recover the
#    ~30% backward-time tax of recomputation.
#  * P=3/4: each extra agent adds ~15 GB of activation memory; even with
#    flex_attention the working set is tight, so we keep grad ckpt on.
GRAD_CKPT=${GRAD_CKPT:-false}
if [ "$NUM_ARMS" -ge 3 ]; then GRAD_CKPT=true; fi

# DeepSpeed stage. ZeRO-2 keeps params replicated (46 GB/rank for the 23B
# model), which is fine for P=2 but leaves no headroom for the extra
# ~15 GB/agent activation in P=3/4 — OOM on cross-attn FFN.
#
# We tried ZeRO-3 first but it breaks Wan VAE's causal feat_cache: the VAE
# is called under torch.no_grad() with 9 sequential chunk forwards per
# encode, and ZeRO-3's per-forward param all-gather/release interleaves
# with the cache state, producing a shape mismatch on torch.cat
# (cache_x vs x). So we stay on ZeRO-2 but offload optimizer state to
# CPU: that frees ~34.5 GB/rank (sharded Adam state for 23B params),
# enough to fit P=3/4 activations without touching the VAE codepath.
# Cost: optim step is ~1.3-2x slower due to PCIe traffic.
if [ "$NUM_ARMS" -ge 3 ]; then
    DEEPSPEED_CFG=${DEEPSPEED_CFG:-groot/vla/configs/deepspeed/zero2_offload.json}
else
    DEEPSPEED_CFG=${DEEPSPEED_CFG:-groot/vla/configs/deepspeed/zero2.json}
fi

# Allocator fragmentation hint (PyTorch's own OOM message suggests this).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "NUM_ARMS=$NUM_ARMS  SHARED_GLOBAL=$SHARED_GLOBAL  data=$DATA_CFG  gradient_checkpointing=$GRAD_CKPT  deepspeed=$DEEPSPEED_CFG  concat_first_frame_latent=$CONCAT_FIRST_FRAME_LATENT  diffusion_in_dim=$DIFFUSION_IN_DIM"

ROBOFACTORY_DATA_ROOT=${ROBOFACTORY_DATA_ROOT:-"/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/data/robofactory_lerobot_v2/LiftBarrier-rf"}
OUTPUT_DIR=${OUTPUT_DIR:-"/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/checkpoints/robofactory_bimanual_smoke"}
NUM_GPUS=${NUM_GPUS:-1}
MAX_STEPS=${MAX_STEPS:-3}
BATCH_SIZE=${BATCH_SIZE:-1}
SAVE_STEPS=${SAVE_STEPS:-$MAX_STEPS}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
REPORT_TO=${REPORT_TO:-none}
WANDB_PROJECT=${WANDB_PROJECT:-dreamzero_robofactory_smoke}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-robofactory_bimanual_smoke}
DATASET_SHARD_SAMPLING_RATE=${DATASET_SHARD_SAMPLING_RATE:-1.0}
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
JOINT_PREFIX_LOSS_WEIGHT=${JOINT_PREFIX_LOSS_WEIGHT:-1.0}
JOINT_PREFIX_LOSS_LEN=${JOINT_PREFIX_LOSS_LEN:-0}
FIRST_CLOSE_JOINT_LOSS_WEIGHT=${FIRST_CLOSE_JOINT_LOSS_WEIGHT:-1.0}
FIRST_CLOSE_JOINT_LOSS_WINDOW_BEFORE=${FIRST_CLOSE_JOINT_LOSS_WINDOW_BEFORE:-0}
FIRST_CLOSE_JOINT_LOSS_WINDOW_AFTER=${FIRST_CLOSE_JOINT_LOSS_WINDOW_AFTER:-0}
PRE_CLOSE_JOINT_LOSS_WEIGHT=${PRE_CLOSE_JOINT_LOSS_WEIGHT:-1.0}
PRE_CLOSE_JOINT_LOSS_WINDOW_BEFORE=${PRE_CLOSE_JOINT_LOSS_WINDOW_BEFORE:-0}
OPEN_PHASE_JOINT_LOSS_WEIGHT=${OPEN_PHASE_JOINT_LOSS_WEIGHT:-1.0}
MODEL_MAX_STATE_DIM=${MODEL_MAX_STATE_DIM:-8}
MODEL_ACTION_DIM=${MODEL_ACTION_DIM:-8}
AGENT_STATE_PAD_DIM=${AGENT_STATE_PAD_DIM:-null}
AGENT_ACTION_PAD_DIM=${AGENT_ACTION_PAD_DIM:-null}

echo "save_steps=$SAVE_STEPS  save_total_limit=$SAVE_TOTAL_LIMIT"
echo "dataset_shard_sampling_rate=$DATASET_SHARD_SAMPLING_RATE"
echo "action_loss_weight=$ACTION_LOSS_WEIGHT  gripper_action_loss_weight=$GRIPPER_ACTION_LOSS_WEIGHT  gripper_close_action_loss_weight=$GRIPPER_CLOSE_ACTION_LOSS_WEIGHT  gripper_close_threshold=$GRIPPER_CLOSE_THRESHOLD  gripper_action_dims=[$GRIPPER_ACTION_DIMS]  gripper_clean_action_loss_weight=$GRIPPER_CLEAN_ACTION_LOSS_WEIGHT  gripper_clean_close_action_loss_weight=$GRIPPER_CLEAN_CLOSE_ACTION_LOSS_WEIGHT  gripper_clean_max_sigma=$GRIPPER_CLEAN_MAX_SIGMA  action_prefix_loss_weight=$ACTION_PREFIX_LOSS_WEIGHT  action_prefix_loss_len=$ACTION_PREFIX_LOSS_LEN"
echo "gripper_binary_action_loss_weight=$GRIPPER_BINARY_ACTION_LOSS_WEIGHT  gripper_binary_close_action_loss_weight=$GRIPPER_BINARY_CLOSE_ACTION_LOSS_WEIGHT  gripper_binary_logit_scale=$GRIPPER_BINARY_LOGIT_SCALE  gripper_binary_max_sigma=$GRIPPER_BINARY_MAX_SIGMA"
echo "first_close_joint_loss_weight=$FIRST_CLOSE_JOINT_LOSS_WEIGHT  first_close_joint_loss_window_before=$FIRST_CLOSE_JOINT_LOSS_WINDOW_BEFORE  first_close_joint_loss_window_after=$FIRST_CLOSE_JOINT_LOSS_WINDOW_AFTER"
echo "joint_prefix_loss_weight=$JOINT_PREFIX_LOSS_WEIGHT  joint_prefix_loss_len=$JOINT_PREFIX_LOSS_LEN  pre_close_joint_loss_weight=$PRE_CLOSE_JOINT_LOSS_WEIGHT  pre_close_joint_loss_window_before=$PRE_CLOSE_JOINT_LOSS_WINDOW_BEFORE  open_phase_joint_loss_weight=$OPEN_PHASE_JOINT_LOSS_WEIGHT"
echo "model_max_state_dim=$MODEL_MAX_STATE_DIM  model_action_dim=$MODEL_ACTION_DIM  agent_state_pad_dim=$AGENT_STATE_PAD_DIM  agent_action_pad_dim=$AGENT_ACTION_PAD_DIM"

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
if [ ! -d "$ROBOFACTORY_DATA_ROOT" ]; then
    echo "ERROR: RoboFactory LeRobot v2 dataset not found at $ROBOFACTORY_DATA_ROOT"
    echo "Run scripts/data/robofactory_to_lerobot_v2.py first."
    exit 1
fi

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
    save_total_limit=$SAVE_TOTAL_LIMIT \
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
    robofactory_data_root=$ROBOFACTORY_DATA_ROOT \
    dataset_shard_sampling_rate=$DATASET_SHARD_SAMPLING_RATE \
    ++agent_state_pad_dim=$AGENT_STATE_PAD_DIM \
    ++agent_action_pad_dim=$AGENT_ACTION_PAD_DIM \
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
    ++action_head_cfg.config.joint_prefix_loss_weight=$JOINT_PREFIX_LOSS_WEIGHT \
    ++action_head_cfg.config.joint_prefix_loss_len=$JOINT_PREFIX_LOSS_LEN \
    ++action_head_cfg.config.first_close_joint_loss_weight=$FIRST_CLOSE_JOINT_LOSS_WEIGHT \
    ++action_head_cfg.config.first_close_joint_loss_window_before=$FIRST_CLOSE_JOINT_LOSS_WINDOW_BEFORE \
    ++action_head_cfg.config.first_close_joint_loss_window_after=$FIRST_CLOSE_JOINT_LOSS_WINDOW_AFTER \
    ++action_head_cfg.config.pre_close_joint_loss_weight=$PRE_CLOSE_JOINT_LOSS_WEIGHT \
    ++action_head_cfg.config.pre_close_joint_loss_window_before=$PRE_CLOSE_JOINT_LOSS_WINDOW_BEFORE \
    ++action_head_cfg.config.open_phase_joint_loss_weight=$OPEN_PHASE_JOINT_LOSS_WEIGHT \
    ++action_head_cfg.config.max_state_dim=$MODEL_MAX_STATE_DIM \
    ++action_head_cfg.config.action_dim=$MODEL_ACTION_DIM \
    ++action_head_cfg.config.diffusion_model_cfg.num_agents=$NUM_ARMS \
    ++action_head_cfg.config.diffusion_model_cfg.max_state_dim=$MODEL_MAX_STATE_DIM \
    ++action_head_cfg.config.diffusion_model_cfg.action_dim=$MODEL_ACTION_DIM \
    ++action_head_cfg.config.diffusion_model_cfg.concat_first_frame_latent=$CONCAT_FIRST_FRAME_LATENT \
    ++action_head_cfg.config.diffusion_model_cfg.in_dim=$DIFFUSION_IN_DIM
