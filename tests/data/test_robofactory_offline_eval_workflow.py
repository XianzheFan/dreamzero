from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / "osmo_workflows/robofactory/offline_eval_liftbarrier_ckpt500.yaml"
CODE_CACHE_URI = (
    "swift://pdx.s8k.io/AUTH_team-gear/datasets/users/xianzhef/oci-migration/"
    "dreamzero_code_liftbarrier_motionfix_950b11b_20260618"
)
EXPECTED_CODE_COMMIT = "950b11ba09dcfa8c02ee962d458872244f25b0ba"
SOURCE_TRAIN_RUN_NAME = "dz-rf2-lb500-motionw4-th02-50k-scratch-xz-20260618"


def _task_by_name(workflow, name):
    return next(task for task in workflow["workflow"]["tasks"] if task["name"] == name)


def test_liftbarrier_offline_eval_workflow_defaults_to_ckpt500_and_cached_code():
    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    defaults = workflow["default-values"]
    assert defaults["workflow_name"] == "dz-rf2-lb500-motionw4-th02-offline-ckpt500-xz-20260618"
    assert defaults["source_train_run_name"] == SOURCE_TRAIN_RUN_NAME
    assert defaults["code_s3_uri"] == CODE_CACHE_URI
    assert defaults["expected_code_commit"] == EXPECTED_CODE_COMMIT
    assert defaults["data_variant"] == "LiftBarrier-rf-500"
    assert defaults["data_s3_uri"] == (
        "s3://GearHome/users/xianzhef/oci-migration/data/robofactory_lerobot_v2/LiftBarrier-rf-500"
    )
    assert defaults["eval_ckpt_s3_uri"] == (
        "s3://GearHome/users/xianzhef/oci-migration/dreamzero_s3cache/bootstrap_checkpoints/"
        f"{SOURCE_TRAIN_RUN_NAME}"
    )
    assert defaults["eval_ckpt_fallback_s3_uri"] == (
        "s3://GearHome/users/xianzhef/oci-migration/dreamzero_runs/"
        f"{SOURCE_TRAIN_RUN_NAME}/checkpoints"
    )
    assert defaults["ckpt_setting"] == "checkpoint-500"
    assert defaults["min_model_bytes"] == "100000000"
    assert defaults["num_batches"] == "4"
    assert defaults["gripper_class_threshold"] == "0.0"
    assert defaults["disable_torch_compile"] == "true"
    assert defaults["ckpt_wait_timeout_seconds"] == "7200"
    assert defaults["ckpt_wait_interval_seconds"] == "120"

    resources = workflow["workflow"]["resources"]["default"]
    assert resources["cpu"] == 15
    assert resources["gpu"] == 1
    assert resources["platform"] == "ovx-l40"
    assert resources["memory"] == "120Gi"
    assert resources["storage"] == "620Gi"


def test_liftbarrier_offline_eval_workflow_restores_complete_checkpoint_safely():
    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    script = _task_by_name(workflow, "offline-eval")["files"][0]["contents"]

    assert 'EVAL_CKPT_S3_URI="${EVAL_CKPT_S3_URI:-{{eval_ckpt_s3_uri}}}"' in script
    assert 'EVAL_CKPT_FALLBACK_S3_URI="${EVAL_CKPT_FALLBACK_S3_URI:-{{eval_ckpt_fallback_s3_uri}}}"' in script
    assert 'CKPT_WAIT_TIMEOUT_SECONDS="${CKPT_WAIT_TIMEOUT_SECONDS:-{{ckpt_wait_timeout_seconds}}}"' in script
    assert 'CKPT_WAIT_INTERVAL_SECONDS="${CKPT_WAIT_INTERVAL_SECONDS:-{{ckpt_wait_interval_seconds}}}"' in script
    assert 'DISABLE_DREAMZERO_TORCH_COMPILE="${DISABLE_DREAMZERO_TORCH_COMPILE:-{{disable_torch_compile}}}"' in script
    assert "export DISABLE_DREAMZERO_TORCH_COMPILE" in script
    assert 'restore_checkpoint_from_uri "$EVAL_CKPT_S3_URI" "s3cache"' in script
    assert 'restore_checkpoint_from_uri "$EVAL_CKPT_FALLBACK_S3_URI" "primary"' in script
    assert "restore_checkpoint_when_available()" in script
    assert 'rm -rf "$EVAL_CKPT_ROOT"' in script
    assert "checkpoint not available yet" in script
    assert "timed out waiting ${CKPT_WAIT_TIMEOUT_SECONDS}s" in script
    assert 'sleep "$CKPT_WAIT_INTERVAL_SECONDS"' in script
    assert "MIN_MODEL_BYTES" in script
    assert 'stat -c%s "${ckpt_dir}/model.safetensors"' in script
    assert 'Checkpoint model.safetensors is too small: ${size} < ${MIN_MODEL_BYTES}' in script
    assert '[ -f "${ckpt_dir}/trainer_state.json" ]' in script
    assert 'find "$EVAL_CKPT_ROOT" -type d -name "$CKPT_SETTING"' in script


def test_liftbarrier_offline_eval_workflow_checks_robofactory_data_and_runs_eval():
    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    script = _task_by_name(workflow, "offline-eval")["files"][0]["contents"]

    for marker in (
        "DreamZero code cache commit",
        "OSMO_CODE_COMMIT",
        "gripper_binary_action_loss_weight",
        "_sync_runtime_shape_from_config",
        "hf_download_tokenizer",
        "--include tokenizer_config.json",
        "resolve_tokenizer_root",
        'promote_model_root "$resolved_cache_root" "$dest"',
        "Keeping existing promoted cache entry",
        "Skipping self-referential cache entry",
        "python scripts/data/inspect_robotwin_lerobot.py",
        'CANONICAL_DATA_ROOT="/workspace/data/robofactory_lerobot_v2/${DATA_VARIANT}"',
        'ln -sfn "$DATA_ROOT" "$CANONICAL_DATA_ROOT"',
        'export ROBOFACTORY_DATA_ROOT_FOR_TRANSFORM="$CANONICAL_DATA_ROOT"',
        "Using RoboFactory LeRobot dataset root",
        "--expected-episodes 500",
        "--expected-action-dim 16",
        "--expected-state-dim 16",
        "--gripper-dims 7,15",
        "--close-threshold 0.0",
        "--gripper-min -1.0",
        "--gripper-max 1.0",
        "--expected-embodiment-tag robofactory",
        '--legacy-embodiment-tags ""',
        "python -m eval_utils.offline_eval_bimanual",
        "--ckpt-dir \"$EVAL_CKPT_ROOT\"",
        "--ckpt-setting \"$CKPT_SETTING\"",
        "--num-batches \"$NUM_BATCHES\"",
        "--data-root \"$CANONICAL_DATA_ROOT\"",
        "--gripper-class-threshold \"$GRIPPER_CLASS_THRESHOLD\"",
        "gripper_class_threshold=${GRIPPER_CLASS_THRESHOLD}",
        "disable_torch_compile=${DISABLE_DREAMZERO_TORCH_COMPILE}",
        "canonical_data_root=${CANONICAL_DATA_ROOT}",
        "open rate",
    ):
        assert marker in script

    assert 'osmo data upload "${OUT_S3_URI}/" /workspace/offline_eval_outputs' in script
    assert "offline_eval_stdout.txt" in script
    assert "summary.txt" in script


def test_liftbarrier_offline_eval_workflow_embedded_script_is_valid_bash(tmp_path):
    import subprocess

    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    script = _task_by_name(workflow, "offline-eval")["files"][0]["contents"]
    script_path = tmp_path / "offline_eval_liftbarrier.sh"
    script_path.write_text(script)
    subprocess.run(["bash", "-n", str(script_path)], check=True)
