from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / "osmo_workflows/robotwin/prepare_stack_blocks_two_dataset.yaml"
README_PATH = REPO_ROOT / "osmo_workflows/robotwin/README.md"


def _python_heredocs(script: str) -> list[str]:
    lines = script.splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].endswith("<<'PY'") or lines[i].endswith("<<'PY'; then"):
            start = i + 1
            end = start
            while end < len(lines) and lines[end] != "PY":
                end += 1
            assert end < len(lines), f"missing PY heredoc terminator after line {i + 1}"
            blocks.append("\n".join(lines[start:end]))
            i = end
        i += 1
    return blocks


def test_stack_prepare_workflow_defaults_to_larger_post_training_dataset():
    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    defaults = workflow["default-values"]
    assert defaults["workflow_name"] == "robotwin-stack-blocks-two-dataset500-l40-xianzhef-20260603"
    assert defaults["target_episodes"] == "500"
    assert defaults["data_variant"] == "stack_blocks_two-rt-500"
    assert defaults["raw_data_s3_uri"].endswith(
        "/RoboTwin/data/stack_blocks_two/demo_full_franka"
    )
    assert defaults["converted_data_s3_uri"].endswith(
        "/data/robotwin_lerobot_v2/stack_blocks_two-rt-500"
    )

    script = workflow["workflow"]["tasks"][0]["files"][0]["contents"]
    assert 'ROBOTWIN_TARGET_EPISODES="${ROBOTWIN_TARGET_EPISODES:-{{target_episodes}}}"' in script
    assert "export ROBOTWIN_TARGET_EPISODES ROBOTWIN_NUM_EPISODES ROBOTWIN_EXPECTED_EPISODES" in script
    assert 'DATA_ROOT="/workspace/data/robotwin_lerobot_v2/${DATA_VARIANT}"' in script
    assert "RAW_BOOTSTRAP_DATA_S3_URI" in script
    assert 'ROBOTWIN_REPO_CACHE="${ROBOTWIN_REPO_CACHE:-${OSMO_CACHE_ROOT}/RoboTwin_${DATA_VARIANT}}"' in script
    assert 'ROBOTWIN_COLLECT_EXTRA_SEEDS="${ROBOTWIN_COLLECT_EXTRA_SEEDS:-50}"' in script
    assert "ROBOTWIN_COLLECT_EXTRA_SEEDS" in script
    assert 'ROBOTWIN_MAX_COLLECT_ATTEMPTS="${ROBOTWIN_MAX_COLLECT_ATTEMPTS:-80}"' in script
    assert "Bootstrapping raw demos from ${RAW_BOOTSTRAP_DATA_S3_URI}/" in script
    assert "normalize_robotwin_raw_layout" in script
    assert 'nested="${root}/demo_full_franka"' in script
    assert "Flattening nested RoboTwin raw data" in script
    assert script.count('normalize_robotwin_raw_layout "$RAW_DATA_ROOT"') == 2
    assert "has_expected_lerobot_episodes" in script
    assert "drop_last_partial_episode_if_needed" in script
    assert "collect_robotwin_until_expected" in script
    assert "parse_collect_error_episode" in script
    assert "repair_robotwin_failed_episode" in script
    assert "Cleaning RoboTwin failed/partial episode" in script
    assert "RoboTwin collect_data.py failed with status" in script
    assert "patch_robotwin_collect_data_skip_failed_replays" in script
    assert "RoboTwin OSMO skip failed replay episode" in script
    assert "Removed failed replay artifact" in script
    assert 'data["episode_num"] = collect_target' in script
    assert "collect_target = target + max(extra, 0)" in script
    assert 'script/collect_data.py stack_blocks_two "${TASK_CONFIG_NAME}"' in script
    assert '--num-episodes "${ROBOTWIN_NUM_EPISODES:--1}"' in script
    assert 'rm -rf "$DATA_ROOT"' in script
    assert "_validate_joint_action_vector" in script
    assert "action = state[1:].copy()" in script
    assert "relative_stats_dreamzero.json" in script
    assert '"embodiment_tag": "robotwin"' in script
    assert "Patched RoboTwin converter to write embodiment_tag=robotwin." in script
    assert "converter marker missing after patch" in script
    assert "RoboTwin gripper convention self-check passed" in script
    assert "range=[0,1], open=1.0, close=0.0" in script
    assert "np.clip(gripper_val, 0, 1)" in script
    assert "def close_gripper(self, arm_tag: ArmTag, pos: float = 0.0)" in script
    assert "def open_gripper(self, arm_tag: ArmTag, pos: float = 1.0)" in script


def test_stack_prepare_workflow_embedded_python_blocks_compile():
    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    script = workflow["workflow"]["tasks"][0]["files"][0]["contents"]
    heredocs = _python_heredocs(script)
    assert len(heredocs) >= 15
    for block in heredocs:
        compile(block, f"{WORKFLOW_PATH}:embedded-python", "exec")


def test_stack_prepare_readme_documents_isolated_1000_episode_submission():
    readme = README_PATH.read_text()

    assert "workflow-local RoboTwin cache prefixes" in readme
    assert "prepare_stack_blocks_two_dataset.yaml --pool groot-l40-04" in readme
    assert "workflow_name=robotwin-stack-blocks-two-dataset1000-l40-xianzhef-20260603" in readme
    assert "data_variant=stack_blocks_two-rt-1000" in readme
    assert "target_episodes=1000" in readme
    assert (
        "raw_data_s3_uri=s3://GearHome/users/xianzhef/oci-migration/"
        "RoboTwin/data/stack_blocks_two/demo_full_franka_1000"
    ) in readme
    assert (
        "raw_bootstrap_data_s3_uri=s3://GearHome/users/xianzhef/oci-migration/"
        "RoboTwin/data/stack_blocks_two/demo_full_franka"
    ) in readme
    assert (
        "converted_data_s3_uri=s3://GearHome/users/xianzhef/oci-migration/"
        "data/robotwin_lerobot_v2/stack_blocks_two-rt-1000"
    ) in readme
