import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/train/robofactory_bimanual_training.sh"


def test_robofactory_training_script_uses_local_gripper_action_dim():
    script = SCRIPT_PATH.read_text()

    assert "GRIPPER_ACTION_DIMS=${GRIPPER_ACTION_DIMS:-7}" in script
    assert "gripper_action_dims=[$GRIPPER_ACTION_DIMS]" in script
    assert "++action_head_cfg.config.gripper_action_dims=[$GRIPPER_ACTION_DIMS]" in script


def test_robofactory_training_script_is_valid_bash():
    subprocess.run(["bash", "-n", str(SCRIPT_PATH)], check=True)
