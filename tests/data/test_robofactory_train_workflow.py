from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / "osmo_workflows/robofactory/train_liftbarrier_shared_global.yaml"
STAGED_WORKFLOW_PATH = REPO_ROOT / "osmo_workflows/robofactory/train_liftbarrier_gamma_staged.yaml"
DROIDWIDTH_TEACHER_WORKFLOW_PATH = (
    REPO_ROOT / "osmo_workflows/robofactory/train_liftbarrier_gamma_droidwidth_teacher.yaml"
)
CODE_CACHE_URI = "swift://pdx.s8k.io/AUTH_team-gear/datasets/users/xianzhef/oci-migration/dreamzero_code_gamma_jerkloss_32e8028_20260621"
EXPECTED_CODE_COMMIT = "32e80283df4d1655045dacea9cc14ad49760b7d2"
REQUIRE_CURRENT_CODE_CACHE_MESSAGE = (
    "code_s3_uri and expected_code_commit must be set to a current uploaded DreamZero code cache"
)


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
    assert defaults["save_steps"] == "2000"

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
        "action_delta_loss_weight",
        "action_jerk_loss_weight",
        "gripper_binary_action_loss_weight",
        "ROPE_AGENT_DIM",
        "GLOBAL_VIDEO_DROPOUT_PROB",
        "GLOBAL_VIDEO_TIMESTEP_MODE",
        "ATTENTION_BACKEND",
        "USE_SPARSE_HUB_ATTENTION",
        "multi_agent_shuffle_agents",
        "global_video_dropout_prob",
        "_compute_action_delta_loss",
        "_compute_action_jerk_loss",
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
        'export ACTION_DELTA_LOSS_WEIGHT="${ACTION_DELTA_LOSS_WEIGHT:-0.2}"',
        'export ACTION_DELTA_MAX_SIGMA="${ACTION_DELTA_MAX_SIGMA:-0.75}"',
        'export ACTION_DELTA_EXCLUDE_GRIPPER="${ACTION_DELTA_EXCLUDE_GRIPPER:-true}"',
        'export FIRST_CLOSE_JOINT_LOSS_WEIGHT="${FIRST_CLOSE_JOINT_LOSS_WEIGHT:-1.0}"',
        'export FIRST_CLOSE_JOINT_LOSS_WINDOW_BEFORE="${FIRST_CLOSE_JOINT_LOSS_WINDOW_BEFORE:-0}"',
        'export FIRST_CLOSE_JOINT_LOSS_WINDOW_AFTER="${FIRST_CLOSE_JOINT_LOSS_WINDOW_AFTER:-0}"',
        'export ROPE_AGENT_DIM="${ROPE_AGENT_DIM:-gamma}"',
        'export MULTI_AGENT_SHUFFLE_AGENTS="${MULTI_AGENT_SHUFFLE_AGENTS:-true}"',
        'export MULTI_AGENT_SAMPLE_AGENT_POOL="${MULTI_AGENT_SAMPLE_AGENT_POOL:-true}"',
        'export GLOBAL_VIDEO_DROPOUT_PROB="${GLOBAL_VIDEO_DROPOUT_PROB:-0.1}"',
        'export GLOBAL_VIDEO_TIMESTEP_MODE="${GLOBAL_VIDEO_TIMESTEP_MODE:-clean}"',
        'export GLOBAL_VIDEO_ATTENTION_MODE="${GLOBAL_VIDEO_ATTENTION_MODE:-read_only}"',
        'export ATTENTION_BACKEND="${ATTENTION_BACKEND:-flex}"',
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
        "action_delta_loss_weight",
        "ACTION_JERK_LOSS_WEIGHT=$ACTION_JERK_LOSS_WEIGHT",
        "FIRST_CLOSE_JOINT_LOSS_WEIGHT=$FIRST_CLOSE_JOINT_LOSS_WEIGHT",
        "FIRST_CLOSE_JOINT_LOSS_WINDOW_BEFORE=$FIRST_CLOSE_JOINT_LOSS_WINDOW_BEFORE",
        "FIRST_CLOSE_JOINT_LOSS_WINDOW_AFTER=$FIRST_CLOSE_JOINT_LOSS_WINDOW_AFTER",
        "ROPE_AGENT_DIM=$ROPE_AGENT_DIM",
        "MULTI_AGENT_SHUFFLE_AGENTS=$MULTI_AGENT_SHUFFLE_AGENTS",
        "MULTI_AGENT_SAMPLE_AGENT_POOL=$MULTI_AGENT_SAMPLE_AGENT_POOL",
        "GLOBAL_VIDEO_DROPOUT_PROB=$GLOBAL_VIDEO_DROPOUT_PROB",
        "GLOBAL_VIDEO_TIMESTEP_MODE=$GLOBAL_VIDEO_TIMESTEP_MODE",
        "GLOBAL_VIDEO_ATTENTION_MODE=$GLOBAL_VIDEO_ATTENTION_MODE",
        "ATTENTION_BACKEND=$ATTENTION_BACKEND",
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

def test_liftbarrier_train_workflow_promotes_nested_model_cache_without_mv_failure():
    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    script = _task_by_name(workflow, "train")["files"][0]["contents"]

    assert 'cp -an "${resolved}/." "$dest/"' in script
    assert "mv -n -t" not in script


def test_liftbarrier_train_workflows_restore_minimal_wan_components():
    for path in (WORKFLOW_PATH, STAGED_WORKFLOW_PATH, DROIDWIDTH_TEACHER_WORKFLOW_PATH):
        with path.open() as f:
            workflow = yaml.safe_load(f)

        script = _task_by_name(workflow, "train")["files"][0]["contents"]

        assert "download_or_restore_wan_minimal()" in script
        assert 'download_or_restore_wan_minimal "$WAN_CKPT_DIR"' in script
        assert (
            'download_or_restore_model "Wan2.1-I2V-14B-480P" '
            '"Wan-AI/Wan2.1-I2V-14B-480P" "model" "$WAN_CKPT_DIR"'
            not in script
        )
        assert 'osmo data download --resume --regex "$wan_component_regex"' not in script
        assert "Downloading minimal Wan2.1 components from Hugging Face" in script
        assert "models_t5_umt5-xxl-enc-bf16.pth" in script
        assert "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" in script
        assert "Wan2.1_VAE.pth" in script
        assert "hf_hub_download" in script


def test_liftbarrier_train_workflows_download_droid_pretrained_from_huggingface():
    for path in (WORKFLOW_PATH, STAGED_WORKFLOW_PATH, DROIDWIDTH_TEACHER_WORKFLOW_PATH):
        with path.open() as f:
            workflow = yaml.safe_load(f)

        script = _task_by_name(workflow, "train")["files"][0]["contents"]

        assert "download_or_restore_droid_pretrained()" in script
        assert 'download_or_restore_droid_pretrained "$PRETRAINED_DIR"' in script
        assert (
            'download_or_restore_model "DreamZero-DROID" '
            '"GEAR-Dreams/DreamZero-DROID" "model" "$PRETRAINED_DIR"'
            not in script
        )
        assert "Downloading DreamZero-DROID from Hugging Face" in script
        assert 'hf_download "GEAR-Dreams/DreamZero-DROID" "model" "$dest"' in script


def test_liftbarrier_train_workflow_embedded_python_blocks_compile():
    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    script = _task_by_name(workflow, "train")["files"][0]["contents"]
    heredocs = _python_heredocs(script)
    assert len(heredocs) == 5
    for block in heredocs:
        compile(block, f"{WORKFLOW_PATH}:embedded-python", "exec")


def test_liftbarrier_gamma_staged_workflow_runs_dense_teacher_then_sparse_student():
    with STAGED_WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    defaults = workflow["default-values"]
    assert defaults["workflow_name"] == "dz-rf-sg-gamma-staged-lb500-xianzhef-20260621"
    assert defaults["run_name"] == "dz-rf-sg-gamma-staged-lb500-xianzhef-20260621"
    assert defaults["restore_run_name"] == ""
    assert defaults["stage1_external_run_name"] == ""
    assert defaults["code_s3_uri"] == ""
    assert defaults["expected_code_commit"] == ""
    assert defaults["stage1_max_steps"] == "10000"
    assert defaults["stage2_warmup_max_steps"] == "3000"
    assert defaults["stage2_max_steps"] == "37000"
    assert defaults["save_steps"] == "2000"

    script = _task_by_name(workflow, "train")["files"][0]["contents"]
    for marker in (
        'BASE_RUN_NAME="${RUN_NAME:-{{run_name}}}"',
        'STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-${BASE_RUN_NAME}-dense-teacher}"',
        'STAGE2_WARMUP_RUN_NAME="${STAGE2_WARMUP_RUN_NAME:-${BASE_RUN_NAME}-sparse-warmup}"',
        'STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-${BASE_RUN_NAME}-sparse-self-forcing}"',
        'STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-{{stage1_max_steps}}}"',
        'STAGE2_WARMUP_MAX_STEPS="${STAGE2_WARMUP_MAX_STEPS:-{{stage2_warmup_max_steps}}}"',
        'STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-{{stage2_max_steps}}}"',
        'STAGE1_EXTERNAL_RUN_NAME="${STAGE1_EXTERNAL_RUN_NAME:-{{stage1_external_run_name}}}"',
        'STAGE1_EXTERNAL_S3_URI="${STAGE1_EXTERNAL_S3_URI:-s3://GearHome/users/xianzhef/oci-migration/dreamzero_runs/${STAGE1_EXTERNAL_RUN_NAME}/checkpoints}"',
        'STAGE1_EXTERNAL_CACHE_S3_URI="${STAGE1_EXTERNAL_CACHE_S3_URI:-s3://GearHome/users/xianzhef/oci-migration/dreamzero_s3cache/bootstrap_checkpoints/${STAGE1_EXTERNAL_RUN_NAME}}"',
        'export STAGE1_DYNAMICS_LOSS_WEIGHT="${STAGE1_DYNAMICS_LOSS_WEIGHT:-2.0}"',
        'export STAGE1_ACTION_LOSS_WEIGHT="${STAGE1_ACTION_LOSS_WEIGHT:-1.0}"',
        'export STAGE1_GRIPPER_BINARY_ACTION_LOSS_WEIGHT="${STAGE1_GRIPPER_BINARY_ACTION_LOSS_WEIGHT:-1.0}"',
        'export STAGE1_ACTION_DELTA_LOSS_WEIGHT="${STAGE1_ACTION_DELTA_LOSS_WEIGHT:-$BASE_ACTION_DELTA_LOSS_WEIGHT}"',
        'export STAGE2_ACTION_LOSS_WEIGHT="${STAGE2_ACTION_LOSS_WEIGHT:-$BASE_ACTION_LOSS_WEIGHT}"',
        'export STAGE2_ACTION_DELTA_LOSS_WEIGHT="${STAGE2_ACTION_DELTA_LOSS_WEIGHT:-$BASE_ACTION_DELTA_LOSS_WEIGHT}"',
        'export STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${BASE_OUTPUT_DIR}/dense_teacher}"',
        'export STAGE2_WARMUP_OUTPUT_DIR="${STAGE2_WARMUP_OUTPUT_DIR:-${BASE_OUTPUT_DIR}/sparse_warmup}"',
        'export STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${BASE_OUTPUT_DIR}/sparse_student}"',
        'export ATTENTION_BACKEND="${ATTENTION_BACKEND:-flex}"',
        'export STAGE1_GLOBAL_VIDEO_ATTENTION_MODE="${STAGE1_GLOBAL_VIDEO_ATTENTION_MODE:-${GLOBAL_VIDEO_ATTENTION_MODE:-bidirectional}}"',
        'export STAGE2_WARMUP_GLOBAL_VIDEO_ATTENTION_MODE="${STAGE2_WARMUP_GLOBAL_VIDEO_ATTENTION_MODE:-${GLOBAL_VIDEO_ATTENTION_MODE:-read_only}}"',
        'export STAGE2_GLOBAL_VIDEO_ATTENTION_MODE="${STAGE2_GLOBAL_VIDEO_ATTENTION_MODE:-${GLOBAL_VIDEO_ATTENTION_MODE:-read_only}}"',
        'export GLOBAL_VIDEO_ATTENTION_MODE="$STAGE1_GLOBAL_VIDEO_ATTENTION_MODE"',
        'export STAGE1_SELF_FORCING_TRAIN="${STAGE1_SELF_FORCING_TRAIN:-false}"',
        'export STAGE1_SELF_FORCING_WARMUP_STEPS="${STAGE1_SELF_FORCING_WARMUP_STEPS:-0}"',
        'export STAGE1_SELF_FORCING_FAST_WRITEBACK="${STAGE1_SELF_FORCING_FAST_WRITEBACK:-false}"',
        'export STAGE2_WARMUP_SELF_FORCING_TRAIN="${STAGE2_WARMUP_SELF_FORCING_TRAIN:-false}"',
        'export STAGE2_WARMUP_SELF_FORCING_WARMUP_STEPS="${STAGE2_WARMUP_SELF_FORCING_WARMUP_STEPS:-0}"',
        'export STAGE2_WARMUP_SELF_FORCING_FAST_WRITEBACK="${STAGE2_WARMUP_SELF_FORCING_FAST_WRITEBACK:-false}"',
        'export STAGE2_SELF_FORCING_TRAIN="${STAGE2_SELF_FORCING_TRAIN:-true}"',
        'export STAGE2_SELF_FORCING_WARMUP_STEPS="${STAGE2_SELF_FORCING_WARMUP_STEPS:-0}"',
        'export STAGE2_SELF_FORCING_FAST_WRITEBACK="${STAGE2_SELF_FORCING_FAST_WRITEBACK:-false}"',
        "latest_complete_checkpoint()",
        "restore_stage1_external_checkpoint_from_s3()",
        "restore_stage1_external_checkpoint()",
        "Using external stage1 teacher checkpoint for stage2 warm-start",
        "External teacher LoRA will be loaded through PRETRAINED_LORA_DIR on top of DreamZero-DROID.",
        'restore_stage1_external_checkpoint_from_s3 "$STAGE1_EXTERNAL_CACHE_S3_URI" "external_stage1_cache"',
        'restore_stage1_external_checkpoint_from_s3 "$STAGE1_EXTERNAL_S3_URI" "external_stage1_primary"',
        "STAGE1_CKPT=\"$STAGE1_EXTERNAL_CKPT\"",
        'sort -V',
        'export PRETRAINED_DIR="$stage_pretrained_dir"',
        'export PRETRAINED_LORA_DIR="$stage_lora_dir"',
        "configure_stage_loss_weights()",
        'export DYNAMICS_LOSS_WEIGHT="$STAGE1_DYNAMICS_LOSS_WEIGHT"',
        'export DYNAMICS_LOSS_WEIGHT="$STAGE2_DYNAMICS_LOSS_WEIGHT"',
        'export SELF_FORCING_TRAIN="$STAGE1_SELF_FORCING_TRAIN"',
        'export SELF_FORCING_TRAIN="$STAGE2_WARMUP_SELF_FORCING_TRAIN"',
        'export SELF_FORCING_TRAIN="$STAGE2_SELF_FORCING_TRAIN"',
        'export SELF_FORCING_WARMUP_STEPS="$STAGE1_SELF_FORCING_WARMUP_STEPS"',
        'export SELF_FORCING_WARMUP_STEPS="$STAGE2_WARMUP_SELF_FORCING_WARMUP_STEPS"',
        'export SELF_FORCING_WARMUP_STEPS="$STAGE2_SELF_FORCING_WARMUP_STEPS"',
        'export SELF_FORCING_FAST_WRITEBACK="$STAGE1_SELF_FORCING_FAST_WRITEBACK"',
        'export SELF_FORCING_FAST_WRITEBACK="$STAGE2_WARMUP_SELF_FORCING_FAST_WRITEBACK"',
        'export SELF_FORCING_FAST_WRITEBACK="$STAGE2_SELF_FORCING_FAST_WRITEBACK"',
        'echo "DYNAMICS_LOSS_WEIGHT=$DYNAMICS_LOSS_WEIGHT"',
        'echo "ACTION_LOSS_WEIGHT=$ACTION_LOSS_WEIGHT"',
        'echo "ACTION_DELTA_LOSS_WEIGHT=$ACTION_DELTA_LOSS_WEIGHT"',
        'export STAGE_LABEL="$stage_label"',
        'export GLOBAL_VIDEO_ATTENTION_MODE="$stage_global_video_attention_mode"',
        'echo "PRETRAINED_LORA_DIR=$PRETRAINED_LORA_DIR"',
        'echo "GLOBAL_VIDEO_ATTENTION_MODE=$GLOBAL_VIDEO_ATTENTION_MODE"',
        'echo "ATTENTION_BACKEND=$ATTENTION_BACKEND"',
        'echo "SELF_FORCING_TRAIN=$SELF_FORCING_TRAIN"',
        'echo "SELF_FORCING_WARMUP_STEPS=$SELF_FORCING_WARMUP_STEPS"',
        'echo "SELF_FORCING_FAST_WRITEBACK=$SELF_FORCING_FAST_WRITEBACK"',
        "start_periodic_checkpoint_upload()",
        "stop_periodic_checkpoint_upload()",
        "start_checkpoint_slimmer()",
        "stop_checkpoint_slimmer()",
        "slim_checkpoint_once()",
        "copy_minimal_checkpoint()",
        "valid_action_dims",
        "per_agent_action_loss",
        'export CHECKPOINT_SLIM_INTERVAL_SECONDS="${CHECKPOINT_SLIM_INTERVAL_SECONDS:-30}"',
        'echo "Started checkpoint slimmer pid=${CHECKPOINT_SLIMMER_PID}, interval=${CHECKPOINT_SLIM_INTERVAL_SECONDS}, output_dir=${OUTPUT_DIR}"',
        'echo "Started periodic checkpoint uploader pid=${PERIODIC_UPLOADER_PID}, interval=${CHECKPOINT_UPLOAD_INTERVAL_SECONDS}, run=${RUN_NAME}, output_dir=${OUTPUT_DIR}"',
        "train_status=$?",
        '"dense-teacher-style"',
        '"false"',
        '"sparse-readonly-warmup-style"',
        '"sparse-readonly-self-forcing-style"',
        '"true"',
        'STAGE1_CKPT="$(latest_complete_checkpoint "$STAGE1_OUTPUT_DIR" || true)"',
        'Stage1 complete checkpoint selected for stage2 warm-start',
        'STAGE2_WARMUP_CKPT="$(latest_complete_checkpoint "$STAGE2_WARMUP_OUTPUT_DIR" || true)"',
        'Stage2 warmup checkpoint selected for self-forcing warm-start',
        'RESTORE_RUN_NAME is ignored by the staged workflow',
        'osmo data upload "${BASE_LOG_S3_URI}/" /tmp/train_liftbarrier_gamma_staged.log',
        "Staged Gamma training complete.",
        REQUIRE_CURRENT_CODE_CACHE_MESSAGE,
        "Pass --set-string code_s3_uri=... expected_code_commit=...",
    ):
        assert marker in script

    output_dir_idx = script.index('export OUTPUT_DIR="$stage_output_dir"')
    start_uploader_idx = script.index(
        "start_periodic_checkpoint_upload",
        output_dir_idx,
    )
    start_slimmer_idx = script.index("start_checkpoint_slimmer", output_dir_idx)
    assert output_dir_idx < start_slimmer_idx < start_uploader_idx
    assert output_dir_idx < start_uploader_idx
    assert 'interval=${CHECKPOINT_UPLOAD_INTERVAL_SECONDS}s' not in script
    assert "is_complete_minimal_checkpoint()" in script
    assert "experiment_cfg/conf.yaml" in script
    assert 'cp -a "${ckpt_dir}/." "${stage_dir}/${ckpt_name}/"' not in script
    assert 'cp -an "${resolved}/." "$dest/"' in script
    assert "mv -n -t" not in script

    assert '"sparse-causal-warmup-style"' not in script
    assert '"sparse-causal-self-forcing-style"' not in script

    stage2_warmup_idx = script.index('"sparse-readonly-warmup-style"')
    stage2_warmup_end = script.index('"stage2_warmup"', stage2_warmup_idx)
    stage2_warmup_block = script[stage2_warmup_idx:stage2_warmup_end]
    assert '"$DREAMZERO_DROID_PRETRAINED_DIR"' in stage2_warmup_block
    assert '"$STAGE1_CKPT"' in stage2_warmup_block
    assert '"$STAGE2_WARMUP_GLOBAL_VIDEO_ATTENTION_MODE"' in stage2_warmup_block
    assert stage2_warmup_block.index(
        '"$DREAMZERO_DROID_PRETRAINED_DIR"'
    ) < stage2_warmup_block.index('"$STAGE1_CKPT"')

    stage1_idx = script.index('"dense-teacher-style"')
    stage1_end = script.index('"stage1"', stage1_idx)
    stage1_block = script[stage1_idx:stage1_end]
    assert '"$STAGE1_GLOBAL_VIDEO_ATTENTION_MODE"' in stage1_block

    stage2_self_forcing_idx = script.index('"sparse-readonly-self-forcing-style"')
    stage2_self_forcing_end = script.index(
        '"stage2_self_forcing"',
        stage2_self_forcing_idx,
    )
    stage2_self_forcing_block = script[
        stage2_self_forcing_idx:stage2_self_forcing_end
    ]
    assert '"$DREAMZERO_DROID_PRETRAINED_DIR"' in stage2_self_forcing_block
    assert '"$STAGE2_WARMUP_CKPT"' in stage2_self_forcing_block
    assert '"$STAGE2_GLOBAL_VIDEO_ATTENTION_MODE"' in stage2_self_forcing_block
    assert stage2_self_forcing_block.index(
        '"$DREAMZERO_DROID_PRETRAINED_DIR"'
    ) < stage2_self_forcing_block.index('"$STAGE2_WARMUP_CKPT"')


def test_liftbarrier_gamma_staged_workflow_embedded_python_blocks_compile():
    with STAGED_WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    script = _task_by_name(workflow, "train")["files"][0]["contents"]
    heredocs = _python_heredocs(script)
    assert len(heredocs) == 5
    for block in heredocs:
        compile(block, f"{STAGED_WORKFLOW_PATH}:embedded-python", "exec")


def test_liftbarrier_gamma_droidwidth_teacher_workflow_preserves_droid_base_head_width():
    with DROIDWIDTH_TEACHER_WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    defaults = workflow["default-values"]
    assert defaults["workflow_name"] == "dz-rf-sg-gamma-dwteacher-bidir-nodrop-lb500-50k-xz-20260622"
    assert defaults["run_name"] == "dz-rf-sg-gamma-dwteacher-bidir-nodrop-lb500-50k-xz-20260622"
    assert defaults["code_s3_uri"] == ""
    assert defaults["expected_code_commit"] == ""
    assert defaults["stage1_max_steps"] == "50000"
    assert defaults["save_steps"] == "2000"
    assert defaults["action_jerk_loss_weight"] == "0.0"
    assert defaults["strict_resume_run_name"] == ""
    assert defaults["train_num_frames"] == "33"
    assert defaults["train_action_horizon"] == "24"
    assert defaults["train_num_frame_per_block"] == "2"
    assert defaults["train_num_action_per_block"] == "24"
    assert defaults["train_warmup_ratio"] == "0.0"
    assert defaults["train_weight_decay"] == "1e-5"
    assert defaults["train_max_chunk_size"] == "4"
    assert defaults["train_max_grad_norm"] == "0.1"

    train_task = _task_by_name(workflow, "train")
    assert train_task["args"] == ["/tmp/train_liftbarrier_gamma_droidwidth_teacher.sh"]
    script = train_task["files"][0]["contents"]
    assert "STAGE2_RUN_NAME" not in script
    assert "STAGE2_MAX_STEPS" not in script

    for marker in (
        'exec > >(tee /tmp/train_liftbarrier_gamma_droidwidth_teacher.log) 2>&1',
        'STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-${BASE_RUN_NAME}-teacher}"',
        'CHECKPOINT_UPLOAD_INTERVAL_SECONDS="${CHECKPOINT_UPLOAD_INTERVAL_SECONDS:-120}"',
        'STRICT_RESUME_RUN_NAME="${STRICT_RESUME_RUN_NAME:-{{strict_resume_run_name}}}"',
        'RESUME_OUTPUT_S3_URI="${RUN_S3_ROOT}/resume_checkpoints"',
        'CACHE_RESUME_OUTPUT_S3_URI="${CACHE_RESUME_OUTPUT_S3_URI:-s3://GearHome/users/xianzhef/oci-migration/dreamzero_s3cache/resume_checkpoints/${RUN_NAME}}"',
        'STRICT_RESUME_S3_URI="${STRICT_RESUME_S3_URI:-s3://GearHome/users/xianzhef/oci-migration/dreamzero_runs/${STRICT_RESUME_RUN_NAME}/resume_checkpoints}"',
        'STRICT_RESUME_CACHE_S3_URI="${STRICT_RESUME_CACHE_S3_URI:-s3://GearHome/users/xianzhef/oci-migration/dreamzero_s3cache/resume_checkpoints/${STRICT_RESUME_RUN_NAME}}"',
        'export MODEL_MAX_STATE_DIM="${MODEL_MAX_STATE_DIM:-64}"',
        'export MODEL_ACTION_DIM="${MODEL_ACTION_DIM:-32}"',
        'export AGENT_STATE_PAD_DIM="${AGENT_STATE_PAD_DIM:-64}"',
        'export AGENT_ACTION_PAD_DIM="${AGENT_ACTION_PAD_DIM:-32}"',
        'export TRAIN_NUM_FRAMES="${TRAIN_NUM_FRAMES:-{{train_num_frames}}}"',
        'export TRAIN_ACTION_HORIZON="${TRAIN_ACTION_HORIZON:-{{train_action_horizon}}}"',
        'export TRAIN_NUM_FRAME_PER_BLOCK="${TRAIN_NUM_FRAME_PER_BLOCK:-{{train_num_frame_per_block}}}"',
        'export TRAIN_NUM_ACTION_PER_BLOCK="${TRAIN_NUM_ACTION_PER_BLOCK:-{{train_num_action_per_block}}}"',
        'export TRAIN_WARMUP_RATIO="${TRAIN_WARMUP_RATIO:-{{train_warmup_ratio}}}"',
        'export TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-{{train_weight_decay}}}"',
        'export TRAIN_MAX_CHUNK_SIZE="${TRAIN_MAX_CHUNK_SIZE:-{{train_max_chunk_size}}}"',
        'export TRAIN_MAX_GRAD_NORM="${TRAIN_MAX_GRAD_NORM:-{{train_max_grad_norm}}}"',
        'export BASE_ACTION_JERK_LOSS_WEIGHT="${ACTION_JERK_LOSS_WEIGHT:-{{action_jerk_loss_weight}}}"',
        'export GLOBAL_VIDEO_DROPOUT_PROB="${GLOBAL_VIDEO_DROPOUT_PROB:-0.0}"',
        'export GLOBAL_VIDEO_ATTENTION_MODE="${GLOBAL_VIDEO_ATTENTION_MODE:-bidirectional}"',
        "droidwidth teacher requires GLOBAL_VIDEO_ATTENTION_MODE=bidirectional",
        "ALLOW_NONBIDIRECTIONAL_TEACHER=true",
        "Set ALLOW_NONBIDIRECTIONAL_TEACHER=true only for explicit ablations.",
        'export STAGE_LABEL="$stage_label"',
        'export PRESERVE_LOCAL_DEEPSPEED_CHECKPOINTS="${PRESERVE_LOCAL_DEEPSPEED_CHECKPOINTS:-true}"',
        'export UPLOAD_STRICT_RESUME_CHECKPOINTS="${UPLOAD_STRICT_RESUME_CHECKPOINTS:-true}"',
        'export STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${BASE_OUTPUT_DIR}/teacher}"',
        "Local checkpoint slimmer disabled; preserving DeepSpeed state for strict resume.",
        "is_complete_strict_resume_checkpoint()",
        "stage_resume_checkpoints_for_upload()",
        "upload_resume_checkpoints_once()",
        'osmo data upload "${RESUME_OUTPUT_S3_URI}/" "$resume_stage_dir"',
        'osmo data upload "${CACHE_RESUME_OUTPUT_S3_URI}/" "$resume_stage_dir"',
        "Skipping non-resumable checkpoint",
        'echo "MODEL_MAX_STATE_DIM=$MODEL_MAX_STATE_DIM"',
        'echo "MODEL_ACTION_DIM=$MODEL_ACTION_DIM"',
        'echo "AGENT_STATE_PAD_DIM=$AGENT_STATE_PAD_DIM"',
        'echo "AGENT_ACTION_PAD_DIM=$AGENT_ACTION_PAD_DIM"',
        'echo "TRAIN_NUM_FRAMES=$TRAIN_NUM_FRAMES"',
        'echo "TRAIN_ACTION_HORIZON=$TRAIN_ACTION_HORIZON"',
        'echo "TRAIN_NUM_FRAME_PER_BLOCK=$TRAIN_NUM_FRAME_PER_BLOCK"',
        'echo "TRAIN_NUM_ACTION_PER_BLOCK=$TRAIN_NUM_ACTION_PER_BLOCK"',
        'echo "TRAIN_WARMUP_RATIO=$TRAIN_WARMUP_RATIO"',
        'echo "TRAIN_WEIGHT_DECAY=$TRAIN_WEIGHT_DECAY"',
        'echo "TRAIN_MAX_CHUNK_SIZE=$TRAIN_MAX_CHUNK_SIZE"',
        'echo "TRAIN_MAX_GRAD_NORM=${TRAIN_MAX_GRAD_NORM:-unset}"',
        "DREAMZERO_DROID_PRETRAINED_DIR=\"$PRETRAINED_DIR\"",
        "restore_stage1_lora_from_s3()",
        "restore_stage1_lora_checkpoint()",
        "Attempting to restore droidwidth teacher LoRA warm-start from RESTORE_RUN_NAME",
        "Restored LoRA checkpoint will be loaded through PRETRAINED_LORA_DIR, not Trainer resume.",
        "restore_stage1_lora_from_s3 \"$RESTORE_CACHE_S3_URI\" \"restore_cache\"",
        "restore_stage1_lora_from_s3 \"$RESTORE_S3_URI\" \"restore_primary\"",
        "Selected LoRA warm-start checkpoint",
        "restore_strict_resume_from_s3()",
        "restore_strict_resume_checkpoint()",
        "Attempting strict DeepSpeed resume from STRICT_RESUME_RUN_NAME",
        "Strict resume requires full DeepSpeed checkpoint state; slim eval checkpoints are intentionally rejected.",
        "Skipping non-strict restored checkpoint",
        "ERROR: STRICT_RESUME_RUN_NAME was set, but no full DeepSpeed checkpoint could be restored.",
        "restore_strict_resume_checkpoint",
        'export PRETRAINED_LORA_DIR="${STAGE1_PRETRAINED_LORA_DIR:-}"',
        'echo "PRETRAINED_LORA_DIR=$PRETRAINED_LORA_DIR"',
        "No restore run configured; droidwidth teacher will start from DreamZero-DROID.",
        "No previous droidwidth teacher LoRA warm-start restored; training will start from DreamZero-DROID.",
        "restore_stage1_lora_checkpoint",
        '"droidwidth-teacher-style"',
        'TEACHER_CKPT="$(latest_complete_checkpoint "$STAGE1_OUTPUT_DIR" || true)"',
        "Teacher complete checkpoint selected",
        "Droidwidth Gamma teacher training complete.",
        'osmo data upload "${BASE_LOG_S3_URI}/" /tmp/train_liftbarrier_gamma_droidwidth_teacher.log',
        "bash scripts/train/robofactory_bimanual_training.sh",
        "valid_action_dims",
        "per_agent_action_loss",
        REQUIRE_CURRENT_CODE_CACHE_MESSAGE,
        "Pass --set-string code_s3_uri=... expected_code_commit=...",
    ):
        assert marker in script

    assert '"sparse-causal-student-style"' not in script
    assert 'GLOBAL_VIDEO_ATTENTION_MODE="${GLOBAL_VIDEO_ATTENTION_MODE:-read_only}"' not in script
    assert "Stage1 complete checkpoint selected for stage2 warm-start" not in script
    assert "RESTORE_RUN_NAME is ignored by the droidwidth teacher workflow" not in script
    assert "Restored checkpoints will be staged under STAGE1_OUTPUT_DIR" not in script
    stage_idx = script.index('"droidwidth-teacher-style"')
    stage_end = script.index('"$DREAMZERO_DROID_PRETRAINED_DIR"', stage_idx)
    stage_block = script[stage_idx:stage_end]
    assert '"false"' in stage_block


def test_liftbarrier_gamma_droidwidth_teacher_workflow_embedded_python_blocks_compile():
    with DROIDWIDTH_TEACHER_WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    script = _task_by_name(workflow, "train")["files"][0]["contents"]
    heredocs = _python_heredocs(script)
    assert len(heredocs) == 5
    for block in heredocs:
        compile(block, f"{DROIDWIDTH_TEACHER_WORKFLOW_PATH}:embedded-python", "exec")
