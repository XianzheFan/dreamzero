from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / "osmo_workflows/robotwin/probe_dreamzero_gripperfix_code.yaml"
CODE_CACHE_URI = "swift://pdx.s8k.io/AUTH_team-gear/datasets/users/xianzhef/oci-migration/dreamzero_code_rawinspect_20260603"


def test_robotwin_code_probe_checks_s3_gripperfix_markers_and_patch_points():
    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    defaults = workflow["default-values"]
    assert defaults["workflow_name"] == "dreamzero-code-probe-gripperfix-xianzhef-20260603"
    assert defaults["code_s3_uri"] == CODE_CACHE_URI

    script = workflow["workflow"]["tasks"][0]["files"][0]["contents"]
    assert "swift://pdx.s8k.io/AUTH_team-gear/datasets/users/xianzhef/*" in script
    assert "Refusing non-xianzhef data URI" in script
    assert "DreamZero code cache commit" in script
    assert "OSMO_CODE_COMMIT" in script
    assert "dreamzero_gripperfix_upload_latest" in script
    assert "gripper_clean_action_loss_weight" in script
    assert "gripper_binary_action_loss_weight" in script
    assert "gripper_action_dims" in script
    assert "_reconstruct_clean_sample_from_flow_target" in script
    assert "global_condition_mode: current_repeat" in script
    assert "agent_action_dims: [[0, 8], [8, 16]]" in script
    assert "train_script_has_explicit_gripper_action_dims" in script
    assert "train_script_has_gripper_dim_patch_points" in script
