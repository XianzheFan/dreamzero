from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / "osmo_workflows/robotwin/train_stack_blocks_two_shared_global.yaml"
TRAIN_SCRIPT_PATH = REPO_ROOT / "scripts/train/robotwin_bimanual_training.sh"
SLURM_SCRIPT_PATH = REPO_ROOT / "scripts/train/robotwin_bimanual_slurm.sh"
CODE_CACHE_URI = "swift://pdx.s8k.io/AUTH_team-gear/datasets/users/xianzhef/oci-migration/dreamzero_code_rawinspect_20260603"


def _task_by_name(workflow, name):
    return next(task for task in workflow["workflow"]["tasks"] if task["name"] == name)


def _python_heredocs(script: str) -> list[str]:
    lines = script.splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].endswith("<<'PY'"):
            start = i + 1
            end = start
            while end < len(lines) and lines[end] != "PY":
                end += 1
            assert end < len(lines), f"missing PY heredoc terminator after line {i + 1}"
            blocks.append("\n".join(lines[start:end]))
            i = end
        i += 1
    return blocks


def test_stack_train_workflow_uploads_eval_checkpoints_to_run_and_s3cache():
    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    defaults = workflow["default-values"]
    assert defaults["code_s3_uri"] == CODE_CACHE_URI
    assert defaults["data_variant"] == "stack_blocks_two-rt-500"
    assert defaults["expected_data_episodes"] == "500"
    assert defaults["converted_data_s3_uri"].endswith(
        "/data/robotwin_lerobot_v2/stack_blocks_two-rt-500"
    )
    assert defaults["restore_run_name"] == (
        "dreamzero-rt-stack-blocks-two-train-shared-global-gripperfix-xianzhef-20260601"
    )

    train_task = _task_by_name(workflow, "train")
    script = train_task["files"][0]["contents"]

    assert workflow["workflow"]["resources"]["default"]["gpu"] == 8
    assert "resource" not in train_task
    assert "inputs" not in train_task
    assert 'DATA_VARIANT="${DATA_VARIANT:-{{data_variant}}}"' in script
    assert 'CODE_S3_URI="${CODE_S3_URI:-{{code_s3_uri}}}"' in script
    assert "swift://pdx.s8k.io/AUTH_team-gear/datasets/users/xianzhef/*" in script
    assert "Refusing non-xianzhef data URI" in script
    assert "DreamZero code cache commit" in script
    assert "OSMO_CODE_COMMIT" in script
    assert 'EXPECTED_DATA_EPISODES="${EXPECTED_DATA_EPISODES:-{{expected_data_episodes}}}"' in script
    assert 'CONVERTED_DATA_S3_URI="${CONVERTED_DATA_S3_URI:-{{converted_data_s3_uri}}}"' in script
    assert 'DATA_ROOT="${DATA_PARENT}/${DATA_VARIANT}"' in script
    assert "install_workflow_fallbacks" in script
    assert "Installing workflow-local RoboTwin LeRobot inspector fallback." in script
    assert "Patched LossLoggerCallback to log gripper_binary_action_loss_avg" in script
    assert "Patched LeRobot relative_action key matching for action./state. prefixes" in script
    assert "Patched RoboTwin embodiment namespace separate from RoboFactory." in script
    assert 'ROBOTWIN = "robotwin"' in script
    assert "modality_config_robotwin" in script
    assert "robotwin: ${transform_robotwin}" in script
    assert "Patched RoboTwin training script to pass gripper_action_dims=[7]" in script
    assert "ensure_dataset_inspector_deps" in script
    assert '("pandas", "pandas"), ("pyarrow", "pyarrow"), ("numpy", "numpy")' in script
    assert 'export GRIPPER_ACTION_DIMS="${GRIPPER_ACTION_DIMS:-7}"' in script
    assert "gripper_action_dims" in script
    assert "GRIPPER_ACTION_DIMS=$GRIPPER_ACTION_DIMS" in script
    assert "python scripts/data/inspect_robotwin_lerobot.py" in script
    assert '--expected-episodes "$EXPECTED_DATA_EPISODES"' in script
    assert "--gripper-dims 7,15" in script
    assert "--close-threshold 0.5" in script
    assert "CACHE_OUTPUT_S3_URI=\"${CACHE_OUTPUT_S3_URI:-${BOOTSTRAP_RESTORE_S3_URI}}\"" in script
    assert "osmo data upload \"${OUTPUT_S3_URI}/\" \"$stage_dir\"" in script
    assert "osmo data upload \"${CACHE_OUTPUT_S3_URI}/\" \"$stage_dir\"" in script
    assert "is_complete_resume_checkpoint" in script
    assert '[ -f "${ckpt_dir}/trainer_state.json" ]' in script
    assert "Skipping incomplete restored checkpoint" in script
    assert "Skipping incomplete ${label} checkpoint" in script
    assert "No usable complete checkpoint-* directories found under ${src_uri}/" in script
    assert "No usable complete checkpoint-* directories found under ${src_dir}" in script


def test_stack_train_workflow_has_gripperfix_training_defaults_and_markers():
    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    workflow_script = _task_by_name(workflow, "train")["files"][0]["contents"]
    train_script = TRAIN_SCRIPT_PATH.read_text()

    for marker in (
        "ACTION_LOSS_WEIGHT=\"${ACTION_LOSS_WEIGHT:-5.0}\"",
        "GRIPPER_ACTION_LOSS_WEIGHT=\"${GRIPPER_ACTION_LOSS_WEIGHT:-6.0}\"",
        "GRIPPER_CLOSE_ACTION_LOSS_WEIGHT=\"${GRIPPER_CLOSE_ACTION_LOSS_WEIGHT:-4.0}\"",
        "GRIPPER_ACTION_DIMS=\"${GRIPPER_ACTION_DIMS:-7}\"",
        "GRIPPER_CLEAN_ACTION_LOSS_WEIGHT=\"${GRIPPER_CLEAN_ACTION_LOSS_WEIGHT:-2.0}\"",
        "GRIPPER_CLEAN_CLOSE_ACTION_LOSS_WEIGHT=\"${GRIPPER_CLEAN_CLOSE_ACTION_LOSS_WEIGHT:-4.0}\"",
        "GRIPPER_CLEAN_MAX_SIGMA=\"${GRIPPER_CLEAN_MAX_SIGMA:-0.75}\"",
        "GRIPPER_BINARY_ACTION_LOSS_WEIGHT=\"${GRIPPER_BINARY_ACTION_LOSS_WEIGHT:-4.0}\"",
        "GRIPPER_BINARY_CLOSE_ACTION_LOSS_WEIGHT=\"${GRIPPER_BINARY_CLOSE_ACTION_LOSS_WEIGHT:-6.0}\"",
        "GRIPPER_BINARY_LOGIT_SCALE=\"${GRIPPER_BINARY_LOGIT_SCALE:-4.0}\"",
        "GRIPPER_BINARY_MAX_SIGMA=\"${GRIPPER_BINARY_MAX_SIGMA:-0.75}\"",
        "ACTION_PREFIX_LOSS_LEN=\"${ACTION_PREFIX_LOSS_LEN:-8}\"",
        "bash scripts/train/robotwin_bimanual_training.sh",
    ):
        assert marker in workflow_script

    for marker in (
        "++action_head_cfg.config.gripper_clean_action_loss_weight=$GRIPPER_CLEAN_ACTION_LOSS_WEIGHT",
        "++action_head_cfg.config.gripper_binary_action_loss_weight=$GRIPPER_BINARY_ACTION_LOSS_WEIGHT",
        "++action_head_cfg.config.gripper_action_dims=[$GRIPPER_ACTION_DIMS]",
        "++action_head_cfg.config.action_prefix_loss_len=$ACTION_PREFIX_LOSS_LEN",
        "++action_head_cfg.config.diffusion_model_cfg.num_agents=2",
    ):
        assert marker in train_script

    for marker in (
        "GRIPPER_CLEAN_ACTION_LOSS_WEIGHT",
        "GRIPPER_BINARY_ACTION_LOSS_WEIGHT",
        "diffusion_model_cfg.num_agents=2",
        "gripper_clean_action_loss_weight",
        "gripper_binary_action_loss_weight",
        "gripper_binary_action_loss_avg",
        "close_fraction_raw_lt_threshold",
        "_normalize_relative_action_key",
        "_relative_action_key_matches",
        "robotwin:",
        "ROBOTWIN = \"robotwin\"",
        "modality_config_robotwin",
    ):
        assert marker in workflow_script


def test_stack_train_workflow_embedded_python_blocks_compile():
    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    script = _task_by_name(workflow, "train")["files"][0]["contents"]
    heredocs = _python_heredocs(script)
    assert len(heredocs) >= 8
    for block in heredocs:
        compile(block, f"{WORKFLOW_PATH}:embedded-python", "exec")


def test_robotwin_slurm_wrapper_preserves_gripper_training_knobs():
    script = SLURM_SCRIPT_PATH.read_text()

    for marker in (
        "GRIPPER_ACTION_DIMS=${GRIPPER_ACTION_DIMS:-7}",
        "GRIPPER_BINARY_ACTION_LOSS_WEIGHT=${GRIPPER_BINARY_ACTION_LOSS_WEIGHT:-4.0}",
        "GRIPPER_BINARY_CLOSE_ACTION_LOSS_WEIGHT=${GRIPPER_BINARY_CLOSE_ACTION_LOSS_WEIGHT:-6.0}",
        "GRIPPER_BINARY_LOGIT_SCALE=${GRIPPER_BINARY_LOGIT_SCALE:-4.0}",
        "GRIPPER_BINARY_MAX_SIGMA=${GRIPPER_BINARY_MAX_SIGMA:-0.75}",
        "dims=[$GRIPPER_ACTION_DIMS]",
    ):
        assert marker in script

    for marker in (
        "GRIPPER_ACTION_DIMS=${GRIPPER_ACTION_DIMS} \\",
        "GRIPPER_BINARY_ACTION_LOSS_WEIGHT=${GRIPPER_BINARY_ACTION_LOSS_WEIGHT} \\",
        "GRIPPER_BINARY_CLOSE_ACTION_LOSS_WEIGHT=${GRIPPER_BINARY_CLOSE_ACTION_LOSS_WEIGHT} \\",
        "GRIPPER_BINARY_LOGIT_SCALE=${GRIPPER_BINARY_LOGIT_SCALE} \\",
        "GRIPPER_BINARY_MAX_SIGMA=${GRIPPER_BINARY_MAX_SIGMA} \\",
    ):
        assert marker in script

    export_line = next(line for line in script.splitlines() if line.strip().startswith("--export=ALL"))
    for marker in (
        "GRIPPER_ACTION_DIMS=${GRIPPER_ACTION_DIMS}",
        "GRIPPER_BINARY_ACTION_LOSS_WEIGHT=${GRIPPER_BINARY_ACTION_LOSS_WEIGHT}",
        "GRIPPER_BINARY_CLOSE_ACTION_LOSS_WEIGHT=${GRIPPER_BINARY_CLOSE_ACTION_LOSS_WEIGHT}",
        "GRIPPER_BINARY_LOGIT_SCALE=${GRIPPER_BINARY_LOGIT_SCALE}",
        "GRIPPER_BINARY_MAX_SIGMA=${GRIPPER_BINARY_MAX_SIGMA}",
    ):
        assert marker in export_line
