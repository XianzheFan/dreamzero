from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / "osmo_workflows/robofactory/train_liftbarrier_shared_global.yaml"
STAGED_WORKFLOW_PATH = REPO_ROOT / "osmo_workflows/robofactory/train_liftbarrier_gamma_staged.yaml"
CODE_CACHE_URI = "swift://pdx.s8k.io/AUTH_team-gear/datasets/users/xianzhef/oci-migration/dreamzero_code_gamma_dense_teacher_20260621"
EXPECTED_CODE_COMMIT = "69508a3588c10be56d66c9ba7c2d308413001194"


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
    assert defaults["workflow_name"] == "dz-rf-sg-gammactx-lb500-50k-scratch-xianzhef-20260621"
    assert defaults["run_name"] == "dz-rf-sg-gammactx-lb500-50k-scratch-xianzhef-20260621"
    assert defaults["restore_run_name"] == ""
    assert defaults["code_s3_uri"] == CODE_CACHE_URI
    assert defaults["expected_code_commit"] == EXPECTED_CODE_COMMIT
    assert defaults["data_variant"] == "LiftBarrier-rf-500"
    assert defaults["expected_data_episodes"] == "500"
    assert defaults["data_s3_uri"] == (
        "s3://GearHome/users/xianzhef/oci-migration/data/robofactory_lerobot_v2/LiftBarrier-rf-500"
    )
    assert defaults["max_steps"] == "50000"

    resources = workflow["workflow"]["resources"]["default"]
    assert resources["gpu"] == 8
    assert resources["platform"] == "dgx-h100"
    assert resources["memory"] == "1681Gi"
    assert resources["storage"] == "803Gi"


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
    assert '[ -n "$uri" ] || continue' in script
    assert "DreamZero code cache commit" in script
    assert "OSMO_CODE_COMMIT" in script
    assert "code cache is missing OSMO_CODE_COMMIT marker" in script

    for marker in (
        "DATASET_SHARD_SAMPLING_RATE",
        "dynamics_loss_weight",
        "gripper_binary_action_loss_weight",
        "ROPE_AGENT_DIM",
        "GLOBAL_VIDEO_DROPOUT_PROB",
        "GLOBAL_VIDEO_TIMESTEP_MODE",
        "USE_SPARSE_HUB_ATTENTION",
        "multi_agent_shuffle_agents",
        "global_video_dropout_prob",
        "_compute_gripper_binary_action_loss",
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
        'export DYNAMICS_LOSS_WEIGHT="${DYNAMICS_LOSS_WEIGHT:-1.0}"',
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
        'export FIRST_CLOSE_JOINT_LOSS_WEIGHT="${FIRST_CLOSE_JOINT_LOSS_WEIGHT:-1.0}"',
        'export FIRST_CLOSE_JOINT_LOSS_WINDOW_BEFORE="${FIRST_CLOSE_JOINT_LOSS_WINDOW_BEFORE:-0}"',
        'export FIRST_CLOSE_JOINT_LOSS_WINDOW_AFTER="${FIRST_CLOSE_JOINT_LOSS_WINDOW_AFTER:-0}"',
        'export ROPE_AGENT_DIM="${ROPE_AGENT_DIM:-gamma}"',
        'export MULTI_AGENT_SHUFFLE_AGENTS="${MULTI_AGENT_SHUFFLE_AGENTS:-true}"',
        'export MULTI_AGENT_SAMPLE_AGENT_POOL="${MULTI_AGENT_SAMPLE_AGENT_POOL:-true}"',
        'export GLOBAL_VIDEO_DROPOUT_PROB="${GLOBAL_VIDEO_DROPOUT_PROB:-0.1}"',
        'export GLOBAL_VIDEO_TIMESTEP_MODE="${GLOBAL_VIDEO_TIMESTEP_MODE:-clean}"',
        'export GLOBAL_VIDEO_ATTENTION_MODE="${GLOBAL_VIDEO_ATTENTION_MODE:-read_only}"',
        'export USE_SPARSE_HUB_ATTENTION="${USE_SPARSE_HUB_ATTENTION:-true}"',
        "bash scripts/train/robofactory_bimanual_training.sh",
    ):
        assert marker in script

    for marker in (
        "DATASET_SHARD_SAMPLING_RATE=$DATASET_SHARD_SAMPLING_RATE",
        "DYNAMICS_LOSS_WEIGHT=$DYNAMICS_LOSS_WEIGHT",
        "GRIPPER_BINARY_ACTION_LOSS_WEIGHT=$GRIPPER_BINARY_ACTION_LOSS_WEIGHT",
        "GRIPPER_BINARY_CLOSE_ACTION_LOSS_WEIGHT=$GRIPPER_BINARY_CLOSE_ACTION_LOSS_WEIGHT",
        "GRIPPER_BINARY_LOGIT_SCALE=$GRIPPER_BINARY_LOGIT_SCALE",
        "GRIPPER_BINARY_MAX_SIGMA=$GRIPPER_BINARY_MAX_SIGMA",
        "FIRST_CLOSE_JOINT_LOSS_WEIGHT=$FIRST_CLOSE_JOINT_LOSS_WEIGHT",
        "FIRST_CLOSE_JOINT_LOSS_WINDOW_BEFORE=$FIRST_CLOSE_JOINT_LOSS_WINDOW_BEFORE",
        "FIRST_CLOSE_JOINT_LOSS_WINDOW_AFTER=$FIRST_CLOSE_JOINT_LOSS_WINDOW_AFTER",
        "ROPE_AGENT_DIM=$ROPE_AGENT_DIM",
        "MULTI_AGENT_SHUFFLE_AGENTS=$MULTI_AGENT_SHUFFLE_AGENTS",
        "MULTI_AGENT_SAMPLE_AGENT_POOL=$MULTI_AGENT_SAMPLE_AGENT_POOL",
        "GLOBAL_VIDEO_DROPOUT_PROB=$GLOBAL_VIDEO_DROPOUT_PROB",
        "GLOBAL_VIDEO_TIMESTEP_MODE=$GLOBAL_VIDEO_TIMESTEP_MODE",
        "GLOBAL_VIDEO_ATTENTION_MODE=$GLOBAL_VIDEO_ATTENTION_MODE",
        "USE_SPARSE_HUB_ATTENTION=$USE_SPARSE_HUB_ATTENTION",
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
    assert "No restore run configured; training will start fresh." in script
    assert '[ -f "${ckpt_dir}/trainer_state.json" ]' in script


def test_liftbarrier_train_workflow_embedded_python_blocks_compile():
    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    script = _task_by_name(workflow, "train")["files"][0]["contents"]
    heredocs = _python_heredocs(script)
    assert len(heredocs) == 3
    for block in heredocs:
        compile(block, f"{WORKFLOW_PATH}:embedded-python", "exec")


def test_liftbarrier_gamma_staged_workflow_runs_dense_teacher_then_sparse_student():
    with STAGED_WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    defaults = workflow["default-values"]
    assert defaults["workflow_name"] == "dz-rf-sg-gamma-staged-lb500-xianzhef-20260621"
    assert defaults["run_name"] == "dz-rf-sg-gamma-staged-lb500-xianzhef-20260621"
    assert defaults["restore_run_name"] == ""
    assert defaults["code_s3_uri"] == CODE_CACHE_URI
    assert defaults["expected_code_commit"] == EXPECTED_CODE_COMMIT
    assert defaults["stage1_max_steps"] == "10000"
    assert defaults["stage2_max_steps"] == "50000"

    script = _task_by_name(workflow, "train")["files"][0]["contents"]
    for marker in (
        'BASE_RUN_NAME="${RUN_NAME:-{{run_name}}}"',
        'STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-${BASE_RUN_NAME}-dense-teacher}"',
        'STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-${BASE_RUN_NAME}-sparse-student}"',
        'STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-{{stage1_max_steps}}}"',
        'STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-{{stage2_max_steps}}}"',
        'export STAGE1_DYNAMICS_LOSS_WEIGHT="${STAGE1_DYNAMICS_LOSS_WEIGHT:-2.0}"',
        'export STAGE1_ACTION_LOSS_WEIGHT="${STAGE1_ACTION_LOSS_WEIGHT:-1.0}"',
        'export STAGE1_GRIPPER_BINARY_ACTION_LOSS_WEIGHT="${STAGE1_GRIPPER_BINARY_ACTION_LOSS_WEIGHT:-1.0}"',
        'export STAGE2_ACTION_LOSS_WEIGHT="${STAGE2_ACTION_LOSS_WEIGHT:-$BASE_ACTION_LOSS_WEIGHT}"',
        'export STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${BASE_OUTPUT_DIR}/dense_teacher}"',
        'export STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${BASE_OUTPUT_DIR}/sparse_student}"',
        "latest_complete_checkpoint()",
        'sort -V',
        'export PRETRAINED_DIR="$stage_pretrained_dir"',
        "configure_stage_loss_weights()",
        'export DYNAMICS_LOSS_WEIGHT="$STAGE1_DYNAMICS_LOSS_WEIGHT"',
        'export DYNAMICS_LOSS_WEIGHT="$STAGE2_DYNAMICS_LOSS_WEIGHT"',
        'echo "DYNAMICS_LOSS_WEIGHT=$DYNAMICS_LOSS_WEIGHT"',
        'echo "ACTION_LOSS_WEIGHT=$ACTION_LOSS_WEIGHT"',
        '"dense-teacher-style"',
        '"false"',
        '"sparse-causal-student-style"',
        '"true"',
        'STAGE1_CKPT="$(latest_complete_checkpoint "$STAGE1_OUTPUT_DIR" || true)"',
        'Stage1 complete checkpoint selected for stage2 warm-start',
        'RESTORE_RUN_NAME is ignored by the staged workflow',
        'osmo data upload "${BASE_LOG_S3_URI}/" /tmp/train_liftbarrier_gamma_staged.log',
        "Staged Gamma training complete.",
    ):
        assert marker in script


def test_liftbarrier_gamma_staged_workflow_embedded_python_blocks_compile():
    with STAGED_WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    script = _task_by_name(workflow, "train")["files"][0]["contents"]
    heredocs = _python_heredocs(script)
    assert len(heredocs) == 3
    for block in heredocs:
        compile(block, f"{STAGED_WORKFLOW_PATH}:embedded-python", "exec")
