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


def test_robofactory_training_script_is_valid_bash():
    subprocess.run(["bash", "-n", str(SCRIPT_PATH)], check=True)
