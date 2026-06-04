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


def test_robofactory_training_script_is_valid_bash():
    subprocess.run(["bash", "-n", str(SCRIPT_PATH)], check=True)
