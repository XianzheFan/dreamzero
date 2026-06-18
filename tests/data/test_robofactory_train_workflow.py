from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / "osmo_workflows/robofactory/train_liftbarrier_shared_global.yaml"
CODE_CACHE_URI = (
    "swift://pdx.s8k.io/AUTH_team-gear/datasets/users/xianzhef/oci-migration/"
    "dreamzero_code_liftbarrier_motionfix_950b11b_20260618"
)
EXPECTED_CODE_COMMIT = "950b11ba09dcfa8c02ee962d458872244f25b0ba"
RUN_NAME = "dz-rf2-lb500-motionw4-th02-50k-fresh-xz-20260618"
WORKFLOW_NAME = "dz-rf2-lb500-motionw4-th02-50k-fresh-storagefix-xz-20260618"


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


def test_liftbarrier_train_workflow_defaults_to_cached_code_and_500_episode_data():
    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    defaults = workflow["default-values"]
    assert defaults["workflow_name"] == WORKFLOW_NAME
    assert defaults["run_name"] == RUN_NAME
    assert defaults["restore_run_name"] == RUN_NAME
    assert defaults["code_s3_uri"] == CODE_CACHE_URI
    assert defaults["expected_code_commit"] == EXPECTED_CODE_COMMIT
    assert defaults["data_variant"] == "LiftBarrier-rf-500"
    assert defaults["expected_data_episodes"] == "500"
    assert defaults["data_s3_uri"] == (
        "s3://GearHome/users/xianzhef/oci-migration/data/robofactory_lerobot_v2/LiftBarrier-rf-500"
    )
    assert defaults["max_steps"] == "50000"
    assert defaults["joint_motion_action_loss_weight"] == "4.0"
    assert defaults["joint_motion_action_loss_threshold"] == "0.2"

    resources = workflow["workflow"]["resources"]["default"]
    assert resources["gpu"] == 8
    assert resources["platform"] == "dgx-h100"
    assert resources["memory"] == "1681Gi"
    assert resources["storage"] == "803Gi"


def test_liftbarrier_train_workflow_cleans_checkpoint_staging_to_avoid_eviction():
    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    script = _task_by_name(workflow, "train")["files"][0]["contents"]

    assert 'CHECKPOINT_UPLOAD_MAX_COUNT="${CHECKPOINT_UPLOAD_MAX_COUNT:-1}"' in script
    assert 'rm -rf "$stage_root"' in script
    assert 'tail -n "$CHECKPOINT_UPLOAD_MAX_COUNT"' in script
    assert 'rm -rf "$stage_dir"' in script
    assert 'rm -rf "$import_dir"' in script
    assert 'osmo data upload "${OUTPUT_S3_URI}/" "$stage_dir" || upload_status=$?' in script
    assert 'return "$upload_status"' in script


def test_liftbarrier_train_workflow_verifies_code_cache_and_robofactory_dataset():
    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    train_task = _task_by_name(workflow, "train")
    script = train_task["files"][0]["contents"]

    assert "resource" not in train_task
    assert "inputs" not in train_task
    assert 'CODE_S3_URI="${CODE_S3_URI:-{{code_s3_uri}}}"' in script
    assert 'EXPECTED_CODE_COMMIT="${EXPECTED_CODE_COMMIT:-{{expected_code_commit}}}"' in script
    assert "swift://pdx.s8k.io/AUTH_team-gear/datasets/users/xianzhef/*" in script
    assert "Refusing non-xianzhef data URI" in script
    assert "DreamZero code cache commit" in script
    assert "OSMO_CODE_COMMIT" in script
    assert "code cache is missing OSMO_CODE_COMMIT marker" in script

    for marker in (
        "DATASET_SHARD_SAMPLING_RATE",
        "gripper_binary_action_loss_weight",
        "_compute_gripper_binary_action_loss",
        "joint_motion_action_loss_weight",
        "agent_action_dims: [[0, 8], [8, 16]]",
    ):
        assert marker in script

    assert 'DATA_S3_URI="${DATA_S3_URI:-{{data_s3_uri}}}"' in script
    assert 'DATA_ROOT="${DATA_PARENT}/${DATA_VARIANT}"' in script
    assert "python scripts/data/inspect_robotwin_lerobot.py" in script
    assert '--expected-episodes "$EXPECTED_DATA_EPISODES"' in script
    assert "--expected-action-dim 16" in script
    assert "--expected-state-dim 16" in script
    assert "--gripper-dims 7,15" in script
    assert "--close-threshold 0.0" in script
    assert "--gripper-min -1.0" in script
    assert "--gripper-max 1.0" in script
    assert "--expected-embodiment-tag robofactory" in script
    assert '--legacy-embodiment-tags ""' in script


def test_liftbarrier_train_workflow_passes_shared_global_binary_gripper_training_knobs():
    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    script = _task_by_name(workflow, "train")["files"][0]["contents"]

    for marker in (
        "export NUM_GPUS=8",
        "export NUM_ARMS=2",
        "export SHARED_GLOBAL=1",
        'export DATASET_SHARD_SAMPLING_RATE="${DATASET_SHARD_SAMPLING_RATE:-1.0}"',
        'export ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-5.0}"',
        'export GRIPPER_ACTION_LOSS_WEIGHT="${GRIPPER_ACTION_LOSS_WEIGHT:-6.0}"',
        'export GRIPPER_CLOSE_ACTION_LOSS_WEIGHT="${GRIPPER_CLOSE_ACTION_LOSS_WEIGHT:-4.0}"',
        'export GRIPPER_CLOSE_THRESHOLD="${GRIPPER_CLOSE_THRESHOLD:-0.0}"',
        'export GRIPPER_ACTION_DIMS="${GRIPPER_ACTION_DIMS:-7}"',
        'export GRIPPER_CLEAN_ACTION_LOSS_WEIGHT="${GRIPPER_CLEAN_ACTION_LOSS_WEIGHT:-2.0}"',
        'export GRIPPER_CLEAN_CLOSE_ACTION_LOSS_WEIGHT="${GRIPPER_CLEAN_CLOSE_ACTION_LOSS_WEIGHT:-4.0}"',
        'export GRIPPER_CLEAN_MAX_SIGMA="${GRIPPER_CLEAN_MAX_SIGMA:-0.75}"',
        'export GRIPPER_BINARY_ACTION_LOSS_WEIGHT="${GRIPPER_BINARY_ACTION_LOSS_WEIGHT:-4.0}"',
        'export GRIPPER_BINARY_CLOSE_ACTION_LOSS_WEIGHT="${GRIPPER_BINARY_CLOSE_ACTION_LOSS_WEIGHT:-6.0}"',
        'export GRIPPER_BINARY_LOGIT_SCALE="${GRIPPER_BINARY_LOGIT_SCALE:-4.0}"',
        'export GRIPPER_BINARY_MAX_SIGMA="${GRIPPER_BINARY_MAX_SIGMA:-0.75}"',
        'export ACTION_PREFIX_LOSS_WEIGHT="${ACTION_PREFIX_LOSS_WEIGHT:-2.0}"',
        'export ACTION_PREFIX_LOSS_LEN="${ACTION_PREFIX_LOSS_LEN:-8}"',
        'export JOINT_MOTION_ACTION_LOSS_WEIGHT="${JOINT_MOTION_ACTION_LOSS_WEIGHT:-{{joint_motion_action_loss_weight}}}"',
        'export JOINT_MOTION_ACTION_LOSS_THRESHOLD="${JOINT_MOTION_ACTION_LOSS_THRESHOLD:-{{joint_motion_action_loss_threshold}}}"',
        'WAN_CKPT_DIR="${WAN_CKPT_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/checkpoints/Wan2.1-I2V-14B-480P}"',
        'TOKENIZER_DIR="${TOKENIZER_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/checkpoints/umt5-xxl}"',
        'PRETRAINED_DIR="${PRETRAINED_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/xianzhef/checkpoints/DreamZero-DROID}"',
        'Using existing Wan2.1 checkpoint at $WAN_CKPT_DIR',
        'Using existing umt5 tokenizer at $TOKENIZER_DIR',
        'Using existing DreamZero-DROID checkpoint at $PRETRAINED_DIR',
        'WAN_CKPT_DIR="/workspace/checkpoints/Wan2.1-I2V-14B-480P"',
        'TOKENIZER_DIR="/workspace/checkpoints/umt5-xxl-tokenizer"',
        'PRETRAINED_DIR="/workspace/checkpoints/DreamZero-DROID"',
        'cp -an "${resolved}/." "$dest/"',
        "bash scripts/train/robofactory_bimanual_training.sh",
    ):
        assert marker in script

    for marker in (
        "DATASET_SHARD_SAMPLING_RATE=$DATASET_SHARD_SAMPLING_RATE",
        "GRIPPER_BINARY_ACTION_LOSS_WEIGHT=$GRIPPER_BINARY_ACTION_LOSS_WEIGHT",
        "GRIPPER_BINARY_CLOSE_ACTION_LOSS_WEIGHT=$GRIPPER_BINARY_CLOSE_ACTION_LOSS_WEIGHT",
        "GRIPPER_BINARY_LOGIT_SCALE=$GRIPPER_BINARY_LOGIT_SCALE",
        "GRIPPER_BINARY_MAX_SIGMA=$GRIPPER_BINARY_MAX_SIGMA",
        "JOINT_MOTION_ACTION_LOSS_WEIGHT=$JOINT_MOTION_ACTION_LOSS_WEIGHT",
        "JOINT_MOTION_ACTION_LOSS_THRESHOLD=$JOINT_MOTION_ACTION_LOSS_THRESHOLD",
    ):
        assert marker in script


def test_liftbarrier_train_workflow_restores_and_uploads_run_and_s3cache_checkpoints():
    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    script = _task_by_name(workflow, "train")["files"][0]["contents"]

    assert 'OUTPUT_S3_URI="${RUN_S3_ROOT}/checkpoints"' in script
    assert 'CACHE_OUTPUT_S3_URI="${CACHE_OUTPUT_S3_URI:-s3://GearHome/users/xianzhef/oci-migration/dreamzero_s3cache/bootstrap_checkpoints/${RUN_NAME}}"' in script
    assert 'RESTORE_S3_URI="${RESTORE_S3_URI:-s3://GearHome/users/xianzhef/oci-migration/dreamzero_runs/${RESTORE_RUN_NAME}/checkpoints}"' in script
    assert 'RESTORE_CACHE_S3_URI="${RESTORE_CACHE_S3_URI:-s3://GearHome/users/xianzhef/oci-migration/dreamzero_s3cache/bootstrap_checkpoints/${RESTORE_RUN_NAME}}"' in script
    assert 'osmo data upload "${OUTPUT_S3_URI}/" "$stage_dir"' in script
    assert 'osmo data upload "${CACHE_OUTPUT_S3_URI}/" "$stage_dir"' in script
    assert "periodic_checkpoint_upload" in script
    assert "Periodic checkpoint upload tick" in script
    assert "restore_checkpoints_from_s3 \"$RESTORE_CACHE_S3_URI\" \"restore_cache\"" in script
    assert "restore_checkpoints_from_s3 \"$RESTORE_S3_URI\" \"restore_primary\"" in script
    assert "Skipping incomplete checkpoint" in script
    assert "Skipping incomplete restored checkpoint" in script
    assert "No usable complete checkpoint-* directories found under ${src_uri}/." in script
    assert "No previous checkpoints restored; training will start fresh." in script
    assert '[ -f "${ckpt_dir}/trainer_state.json" ]' in script
    assert '[ -f "${ckpt_dir}/latest" ]' in script
    assert "-name '*_optim_states.pt'" in script
    assert "-name '*_model_states.pt'" in script


def test_liftbarrier_train_workflow_embedded_python_blocks_compile():
    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    script = _task_by_name(workflow, "train")["files"][0]["contents"]
    heredocs = _python_heredocs(script)
    assert len(heredocs) == 3
    for block in heredocs:
        compile(block, f"{WORKFLOW_PATH}:embedded-python", "exec")
