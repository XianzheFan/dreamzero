import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/train/robofactory_bimanual_training.sh"


def test_robofactory_training_script_uses_local_gripper_action_dim():
    script = SCRIPT_PATH.read_text()

    assert "GRIPPER_ACTION_DIMS=${GRIPPER_ACTION_DIMS:-7}" in script
    assert "gripper_action_dims=[$GRIPPER_ACTION_DIMS]" in script
    assert "++action_head_cfg.config.gripper_action_dims=[$GRIPPER_ACTION_DIMS]" in script


def test_robofactory_training_script_passes_binary_gripper_loss_knobs():
    script = SCRIPT_PATH.read_text()

    for marker in (
        "DYNAMICS_LOSS_WEIGHT=${DYNAMICS_LOSS_WEIGHT:-1.0}",
        "dynamics_loss_weight=$DYNAMICS_LOSS_WEIGHT",
        "++action_head_cfg.config.dynamics_loss_weight=$DYNAMICS_LOSS_WEIGHT",
        "GRIPPER_BINARY_ACTION_LOSS_WEIGHT=${GRIPPER_BINARY_ACTION_LOSS_WEIGHT:-4.0}",
        "GRIPPER_BINARY_CLOSE_ACTION_LOSS_WEIGHT=${GRIPPER_BINARY_CLOSE_ACTION_LOSS_WEIGHT:-6.0}",
        "GRIPPER_BINARY_LOGIT_SCALE=${GRIPPER_BINARY_LOGIT_SCALE:-4.0}",
        "GRIPPER_BINARY_MAX_SIGMA=${GRIPPER_BINARY_MAX_SIGMA:-0.75}",
        "gripper_binary_action_loss_weight=$GRIPPER_BINARY_ACTION_LOSS_WEIGHT",
        "++action_head_cfg.config.gripper_binary_action_loss_weight=$GRIPPER_BINARY_ACTION_LOSS_WEIGHT",
        "++action_head_cfg.config.gripper_binary_close_action_loss_weight=$GRIPPER_BINARY_CLOSE_ACTION_LOSS_WEIGHT",
        "++action_head_cfg.config.gripper_binary_logit_scale=$GRIPPER_BINARY_LOGIT_SCALE",
        "++action_head_cfg.config.gripper_binary_max_sigma=$GRIPPER_BINARY_MAX_SIGMA",
    ):
        assert marker in script


def test_robofactory_training_script_passes_action_delta_loss_knobs():
    script = SCRIPT_PATH.read_text()

    for marker in (
        "ACTION_DELTA_LOSS_WEIGHT=${ACTION_DELTA_LOSS_WEIGHT:-0.0}",
        "ACTION_JERK_LOSS_WEIGHT=${ACTION_JERK_LOSS_WEIGHT:-0.0}",
        "ACTION_DELTA_MAX_SIGMA=${ACTION_DELTA_MAX_SIGMA:-0.75}",
        "ACTION_DELTA_EXCLUDE_GRIPPER=${ACTION_DELTA_EXCLUDE_GRIPPER:-true}",
        "action_delta_loss_weight=$ACTION_DELTA_LOSS_WEIGHT",
        "action_jerk_loss_weight=$ACTION_JERK_LOSS_WEIGHT",
        "++action_head_cfg.config.action_delta_loss_weight=$ACTION_DELTA_LOSS_WEIGHT",
        "++action_head_cfg.config.action_jerk_loss_weight=$ACTION_JERK_LOSS_WEIGHT",
        "++action_head_cfg.config.action_delta_max_sigma=$ACTION_DELTA_MAX_SIGMA",
        "++action_head_cfg.config.action_delta_exclude_gripper=$ACTION_DELTA_EXCLUDE_GRIPPER",
    ):
        assert marker in script


def test_robofactory_training_script_passes_first_close_joint_loss_knobs():
    script = SCRIPT_PATH.read_text()

    for marker in (
        "FIRST_CLOSE_JOINT_LOSS_WEIGHT=${FIRST_CLOSE_JOINT_LOSS_WEIGHT:-1.0}",
        "FIRST_CLOSE_JOINT_LOSS_WINDOW_BEFORE=${FIRST_CLOSE_JOINT_LOSS_WINDOW_BEFORE:-0}",
        "FIRST_CLOSE_JOINT_LOSS_WINDOW_AFTER=${FIRST_CLOSE_JOINT_LOSS_WINDOW_AFTER:-0}",
        "first_close_joint_loss_weight=$FIRST_CLOSE_JOINT_LOSS_WEIGHT",
        "++action_head_cfg.config.first_close_joint_loss_weight=$FIRST_CLOSE_JOINT_LOSS_WEIGHT",
        "++action_head_cfg.config.first_close_joint_loss_window_before=$FIRST_CLOSE_JOINT_LOSS_WINDOW_BEFORE",
        "++action_head_cfg.config.first_close_joint_loss_window_after=$FIRST_CLOSE_JOINT_LOSS_WINDOW_AFTER",
    ):
        assert marker in script


def test_robofactory_training_script_passes_approach_joint_loss_knobs():
    script = SCRIPT_PATH.read_text()

    for marker in (
        "JOINT_PREFIX_LOSS_WEIGHT=${JOINT_PREFIX_LOSS_WEIGHT:-1.0}",
        "JOINT_PREFIX_LOSS_LEN=${JOINT_PREFIX_LOSS_LEN:-0}",
        "PRE_CLOSE_JOINT_LOSS_WEIGHT=${PRE_CLOSE_JOINT_LOSS_WEIGHT:-1.0}",
        "PRE_CLOSE_JOINT_LOSS_WINDOW_BEFORE=${PRE_CLOSE_JOINT_LOSS_WINDOW_BEFORE:-0}",
        "joint_prefix_loss_weight=$JOINT_PREFIX_LOSS_WEIGHT",
        "pre_close_joint_loss_weight=$PRE_CLOSE_JOINT_LOSS_WEIGHT",
        "++action_head_cfg.config.joint_prefix_loss_weight=$JOINT_PREFIX_LOSS_WEIGHT",
        "++action_head_cfg.config.joint_prefix_loss_len=$JOINT_PREFIX_LOSS_LEN",
        "++action_head_cfg.config.pre_close_joint_loss_weight=$PRE_CLOSE_JOINT_LOSS_WEIGHT",
        "++action_head_cfg.config.pre_close_joint_loss_window_before=$PRE_CLOSE_JOINT_LOSS_WINDOW_BEFORE",
    ):
        assert marker in script


def test_robofactory_training_script_uses_full_dataset_sampling_by_default():
    script = SCRIPT_PATH.read_text()

    assert "DATASET_SHARD_SAMPLING_RATE=${DATASET_SHARD_SAMPLING_RATE:-1.0}" in script
    assert "dataset_shard_sampling_rate=$DATASET_SHARD_SAMPLING_RATE" in script
    assert "dataset_shard_sampling_rate=$DATASET_SHARD_SAMPLING_RATE \\" in script


def test_robofactory_training_script_preserves_droid_i2v_patch_embedding_by_default():
    script = SCRIPT_PATH.read_text()

    assert "CONCAT_FIRST_FRAME_LATENT=${CONCAT_FIRST_FRAME_LATENT:-true}" in script
    assert "DIFFUSION_IN_DIM=${DIFFUSION_IN_DIM:-36}" in script
    assert (
        "++action_head_cfg.config.diffusion_model_cfg.concat_first_frame_latent="
        "$CONCAT_FIRST_FRAME_LATENT"
    ) in script
    assert "++action_head_cfg.config.diffusion_model_cfg.in_dim=$DIFFUSION_IN_DIM" in script
    assert "++action_head_cfg.config.diffusion_model_cfg.concat_first_frame_latent=false" not in script
    assert "++action_head_cfg.config.diffusion_model_cfg.in_dim=16" not in script


def test_robofactory_training_script_uses_droid_base_head_width_by_default():
    script = SCRIPT_PATH.read_text()

    for marker in (
        "MODEL_MAX_STATE_DIM=${MODEL_MAX_STATE_DIM:-64}",
        "MODEL_ACTION_DIM=${MODEL_ACTION_DIM:-32}",
        "AGENT_STATE_PAD_DIM=${AGENT_STATE_PAD_DIM:-64}",
        "AGENT_ACTION_PAD_DIM=${AGENT_ACTION_PAD_DIM:-32}",
        "model_max_state_dim=$MODEL_MAX_STATE_DIM",
        "agent_action_pad_dim=$AGENT_ACTION_PAD_DIM",
        "++agent_state_pad_dim=$AGENT_STATE_PAD_DIM",
        "++agent_action_pad_dim=$AGENT_ACTION_PAD_DIM",
        "++action_head_cfg.config.max_state_dim=$MODEL_MAX_STATE_DIM",
        "++action_head_cfg.config.action_dim=$MODEL_ACTION_DIM",
        "++action_head_cfg.config.diffusion_model_cfg.max_state_dim=$MODEL_MAX_STATE_DIM",
        "++action_head_cfg.config.diffusion_model_cfg.action_dim=$MODEL_ACTION_DIM",
    ):
        assert marker in script

    assert "MODEL_MAX_STATE_DIM=${MODEL_MAX_STATE_DIM:-8}" not in script
    assert "MODEL_ACTION_DIM=${MODEL_ACTION_DIM:-8}" not in script
    assert "AGENT_STATE_PAD_DIM=${AGENT_STATE_PAD_DIM:-null}" not in script
    assert "AGENT_ACTION_PAD_DIM=${AGENT_ACTION_PAD_DIM:-null}" not in script


def test_robofactory_training_script_passes_self_forcing_knobs_default_off():
    script = SCRIPT_PATH.read_text()

    for marker in (
        "SELF_FORCING_TRAIN=${SELF_FORCING_TRAIN:-false}",
        "SELF_FORCING_WARMUP_STEPS=${SELF_FORCING_WARMUP_STEPS:-0}",
        "SELF_FORCING_FAST_WRITEBACK=${SELF_FORCING_FAST_WRITEBACK:-false}",
        "self_forcing_train=$SELF_FORCING_TRAIN",
        "self_forcing_warmup_steps=$SELF_FORCING_WARMUP_STEPS",
        "self_forcing_fast_writeback=$SELF_FORCING_FAST_WRITEBACK",
        "++action_head_cfg.config.self_forcing_train=$SELF_FORCING_TRAIN",
        "++action_head_cfg.config.self_forcing_warmup_steps=$SELF_FORCING_WARMUP_STEPS",
        "++action_head_cfg.config.self_forcing_fast_writeback=$SELF_FORCING_FAST_WRITEBACK",
    ):
        assert marker in script


def test_robofactory_training_script_supports_separate_lora_warm_start():
    script = SCRIPT_PATH.read_text()

    assert "PRETRAINED_LORA_DIR=${PRETRAINED_LORA_DIR:-}" in script
    assert "pretrained_lora_path=${PRETRAINED_LORA_DIR:-null}" in script


def test_robofactory_training_script_is_valid_bash():
    subprocess.run(["bash", "-n", str(SCRIPT_PATH)], check=True)
