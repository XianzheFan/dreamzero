from pathlib import Path
import subprocess

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = (
    REPO_ROOT
    / "osmo_workflows/robofactory/closedloop_liftbarrier_gamma_droidwidth_teacher_c2000_slim_eval_gb200_1seed_20260621.yaml"
)
H100_WORKFLOW_PATH = (
    REPO_ROOT
    / "osmo_workflows/robofactory/closedloop_liftbarrier_gamma_droidwidth_teacher_c2000_slim_eval_h100_1seed_20260621.yaml"
)


def _workflow_and_script(path=WORKFLOW_PATH):
    with path.open() as f:
        workflow = yaml.safe_load(f)
    task = workflow["workflow"]["groups"][0]["tasks"][0]
    return workflow, task, task["files"][0]["contents"]


def test_droidwidth_teacher_eval_points_to_teacher_checkpoint_prefix():
    workflow, task, script = _workflow_and_script()

    defaults = workflow["default-values"]
    assert workflow["workflow"]["name"] == "{{workflow_name}}"
    assert defaults["workflow_name"] == (
        "dz-rf-sg-gamma-dwteacher-c2000-slim-eval-gb200-1seed1000-xz-20260621"
    )
    assert defaults["run_name"] == (
        "dz-rf-sg-gamma-dwteacher-c2000-slim-eval-gb200-1seed1000-xz-20260621"
    )
    assert defaults["ckpt_setting"] == "checkpoint-2000"
    assert defaults["local_eval_ckpt_root"] == "gamma_droidwidth_teacher_c2000_slim_eval_1seed1000"
    assert defaults["eval_num_frames"] == "33"
    assert defaults["eval_action_horizon"] == "24"
    assert task["image"] == "nvcr.io/nvidian/gr00t_isaac:v1.5"
    assert 'RUN_NAME="{{run_name}}"' in script
    assert (
        "dreamzero_runs/dz-rf-sg-gamma-dwteacher-lb500-r4-actiondelta-xianzhef-20260621-teacher/"
        "checkpoints/dz-rf-sg-gamma-dwteacher-lb500-r4-actiondelta-xianzhef-20260621-teacher"
        in script
    )
    assert 'CKPT_SETTING="{{ckpt_setting}}"' in script
    assert 'LOCAL_EVAL_CKPT_ROOT="/workspace/eval_ckpts/{{local_eval_ckpt_root}}"' in script
    assert "dense-teacher" not in script


def test_droidwidth_teacher_eval_uses_current_frame_window_defaults():
    _, _, script = _workflow_and_script()

    assert 'VIDEO_PRED_WRIST_WINDOW_MODE="${VIDEO_PRED_WRIST_WINDOW_MODE:-action}"' in script
    assert (
        'SHARED_GLOBAL_WRIST_WINDOW_MODE="${SHARED_GLOBAL_WRIST_WINDOW_MODE:-history-current-first}"'
        in script
    )
    assert '--shared-global-wrist-window-mode "$SHARED_GLOBAL_WRIST_WINDOW_MODE"' in script
    assert '--video-pred-wrist-window-mode "$VIDEO_PRED_WRIST_WINDOW_MODE"' in script
    assert 'EVAL_NUM_FRAMES="{{eval_num_frames}}"' in script
    assert 'EVAL_ACTION_HORIZON="{{eval_action_horizon}}"' in script
    assert 'export CKPT_SETTING LOCAL_EVAL_CKPT_ROOT EVAL_NUM_FRAMES EVAL_ACTION_HORIZON' in script
    assert '--num-frames "$EVAL_NUM_FRAMES"' in script
    assert '--action-horizon "$EVAL_ACTION_HORIZON"' in script
    assert '"EVAL_NUM_FRAMES"' in script
    assert '"EVAL_ACTION_HORIZON"' in script
    assert "Validating eval window against checkpoint config" in script
    assert "EVAL_WINDOW_VALIDATION_OK" in script
    assert "eval_action_horizon mismatch" in script


def test_droidwidth_teacher_eval_uses_gb200_node_resources():
    workflow, _, _ = _workflow_and_script()

    resources = workflow["workflow"]["resources"]["default"]
    assert resources["platform"] == "gb200"
    assert resources["gpu"] == 4
    assert resources["cpu"] == 84
    assert resources["memory"] == "760Gi"


def test_droidwidth_teacher_gb200_model_only_uses_pyav_video_reader_fallback():
    _, _, script = _workflow_and_script()

    assert 'MODEL_ONLY_DIAG="${MODEL_ONLY_DIAG:-1}"' in script
    assert '.gb200_model_only_eval_deps_v2' in script
    assert '.gb200_model_only_eval_deps_v1' not in script
    assert "decord" not in script
    assert '"mani_skill==3.0.0b12" decord' not in script
    assert "offline_video_action_diagnostic" in script


def test_droidwidth_teacher_gb200_s3_model_cache_downloads_are_throttled():
    _, _, script = _workflow_and_script()

    assert "Wan2.1-I2V-14B-480P-minimal-eval" in script
    assert 'repo_id = "Wan-AI/Wan2.1-I2V-14B-480P"' in script
    assert "hf_hub_download" in script
    assert '"models_t5_umt5-xxl-enc-bf16.pth"' in script
    assert '"models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"' in script
    assert '"Wan2.1_VAE.pth"' in script
    assert '"${MODEL_CACHE_S3_URI}/Wan2.1-I2V-14B-480P/"' not in script
    assert '"${MODEL_CACHE_S3_URI}/DreamZero-DROID/"' not in script
    assert 'repo_id=os.environ["DROID_HF_REPO"]' in script
    assert "local_dir=str(target)" in script


def test_droidwidth_teacher_eval_publishes_artifacts_to_osmo_output():
    _, _, script = _workflow_and_script()

    assert 'OSMO_OUTPUT_DIR="{{output}}"' in script
    assert 'mkdir -p "${OSMO_OUTPUT_DIR}/eval_outputs"' in script
    assert 'cp -a /workspace/eval_outputs/. "${OSMO_OUTPUT_DIR}/eval_outputs/"' in script


def test_droidwidth_teacher_eval_writes_checkpoint_manifest():
    _, _, script = _workflow_and_script()

    assert "write_eval_manifest" in script
    assert "robofactory_droidwidth_eval_manifest_v1" in script
    assert "checkpoint_eval_manifest.json" in script
    assert "checkpoint_eval_manifest.txt" in script
    assert "DREAMZERO_GIT_COMMIT" in script
    assert "CKPT_SETTING" in script
    assert "runtime_provenance.json" in script
    assert "checkpoint_code_commit" in script
    assert "checkpoint_stage_label" in script
    assert "checkpoint_action_dim" in script
    assert "checkpoint_global_video_attention_mode" in script
    assert "EVAL_NUM_FRAMES" in script
    assert "EVAL_ACTION_HORIZON" in script
    assert "VIDEO_PRED_ROLLOUT_MODE" in script
    assert "REPLAN_EVERYS" in script
    assert "TEMPORAL_ACTION_ENSEMBLE_DECAY" in script
    assert 'write_eval_manifest "$status" || true' in script


def test_droidwidth_teacher_eval_embedded_script_is_valid_bash():
    _, _, script = _workflow_and_script()

    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_droidwidth_teacher_h100_eval_targets_h100_pool_resources():
    workflow, task, script = _workflow_and_script(H100_WORKFLOW_PATH)

    defaults = workflow["default-values"]
    assert workflow["workflow"]["name"] == "{{workflow_name}}"
    assert defaults["workflow_name"] == (
        "dz-rf-gamma-dw-af2-lb500-50k-c2000-eval-h100-s1000-xz-20260622"
    )
    assert defaults["run_name"] == (
        "dz-rf-gamma-dw-af2-lb500-50k-c2000-eval-h100-s1000-xz-20260622"
    )
    assert defaults["ckpt_run_name"] == (
        "dz-rf-sg-gamma-dwteacher-actionlossfix2-lb500-50k-xz-20260622-teacher"
    )
    assert defaults["ckpt_s3_base"] == (
        "s3://GearHome/users/xianzhef/oci-migration/dreamzero_runs/"
        "dz-rf-sg-gamma-dwteacher-actionlossfix2-lb500-50k-xz-20260622-teacher/checkpoints/"
        "dz-rf-sg-gamma-dwteacher-actionlossfix2-lb500-50k-xz-20260622-teacher"
    )
    assert defaults["ckpt_setting"] == "checkpoint-2000"
    assert (
        defaults["local_eval_ckpt_root"]
        == "gamma_droidwidth_teacher_actionlossfix2_lb500_50k_c2000_slim_eval_h100_1seed1000"
    )
    assert defaults["eval_num_frames"] == "33"
    assert defaults["eval_action_horizon"] == "24"
    assert defaults["video_pred_rollout_modes"] == "action noncausal"
    assert defaults["replan_everys"] == "24 12"
    assert defaults["joint_delta_scales"] == "1.0"
    assert defaults["joint_target_accel_limits"] == "0 0.08"
    assert defaults["replan_boundary_blend_steps"] == "4"
    assert defaults["temporal_action_ensemble_decays"] == "0.6"
    assert defaults["smoothing_profile_names"] == "raw smooth"
    assert defaults["smoothing_profile_blend_steps"] == "0 4"
    assert defaults["smoothing_profile_ensemble_decays"] == "0 0.6"
    assert task["image"].startswith("nvcr.io/nvidian/groot-ci-base-eval:")
    assert 'RUN_NAME="{{run_name}}"' in script
    assert 'CKPT_RUN_NAME="{{ckpt_run_name}}"' in script
    assert 'CKPT_S3_BASE="${CKPT_S3_BASE:-{{ckpt_s3_base}}}"' in script
    assert 'CKPT_AMLFS_BASE="${CKPT_AMLFS_BASE:-{{ckpt_amlfs_base}}}"' in script
    assert 'VIDEO_PRED_WRIST_WINDOW_MODE="${VIDEO_PRED_WRIST_WINDOW_MODE:-action}"' in script
    assert 'VIDEO_PRED_ROLLOUT_MODES="${VIDEO_PRED_ROLLOUT_MODES:-{{video_pred_rollout_modes}}}"' in script
    assert 'VIDEO_PRED_ROLLOUT_MODES="${VIDEO_PRED_ROLLOUT_MODE}"' in script
    assert "for VIDEO_PRED_ROLLOUT_MODE in ${VIDEO_PRED_ROLLOUT_MODES}; do" in script
    assert "Unknown VIDEO_PRED_ROLLOUT_MODE=${VIDEO_PRED_ROLLOUT_MODE}" in script
    assert (
        '--video-pred-dir "/workspace/eval_outputs/video_pred/'
        '${INFERENCE_MODE}_${RESET_TAG}_${VIDEO_PRED_ROLLOUT_MODE}"'
        in script
    )
    assert "_vpred_${rollout_tag}_rp${REPLAN_EVERY}" in script
    assert 'REPLAN_EVERYS="${REPLAN_EVERYS:-{{replan_everys}}}"' in script
    assert 'CKPT_SETTING="{{ckpt_setting}}"' in script
    assert 'LOCAL_EVAL_CKPT_ROOT="/workspace/eval_ckpts/{{local_eval_ckpt_root}}"' in script
    assert 'EVAL_NUM_FRAMES="{{eval_num_frames}}"' in script
    assert 'EVAL_ACTION_HORIZON="{{eval_action_horizon}}"' in script
    assert 'export CKPT_SETTING LOCAL_EVAL_CKPT_ROOT EVAL_NUM_FRAMES EVAL_ACTION_HORIZON' in script
    assert '--num-frames "$EVAL_NUM_FRAMES"' in script
    assert '--action-horizon "$EVAL_ACTION_HORIZON"' in script
    assert "Validating eval window against checkpoint config" in script
    assert "EVAL_WINDOW_VALIDATION_OK" in script
    assert "eval_action_horizon mismatch" in script
    assert 'ROBOFACTORY_RENDER_BACKEND="${ROBOFACTORY_RENDER_BACKEND:-sapien_cuda:0}"' in script
    assert 'ROBOFACTORY_ENABLE_SHADOW="${ROBOFACTORY_ENABLE_SHADOW:-0}"' in script
    assert 'ROBOFACTORY_SHADER_PACK="${ROBOFACTORY_SHADER_PACK:-default}"' in script
    assert 'ROBOFACTORY_RENDER_PREFLIGHT="${ROBOFACTORY_RENDER_PREFLIGHT:-1}"' in script
    assert 'DUMP_RGB_TRACE="${DUMP_RGB_TRACE:-1}"' in script
    assert 'DUMP_RGB_TRACE_FLAG="--dump-rgb-trace"' in script
    assert "--dump-rgb-trace" in script
    assert "--future-rgb-trace-dir /workspace/eval_outputs" in script
    assert 'SERVE_HTTP_ARTIFACTS="${SERVE_HTTP_ARTIFACTS:-0}"' in script
    assert 'if [ "${SERVE_HTTP_ARTIFACTS}" = "1" ]; then' in script
    assert "HTTP artifact serving disabled; uploaded artifacts and exiting to release H100 resources." in script
    assert 'osmo data upload "${RUN_S3_ROOT}/eval_outputs_http/" /workspace/eval_outputs_http || true' in script
    assert "action/noncausal pred-video eval complete" in script
    assert (
        'TEMPORAL_ACTION_ENSEMBLE_DECAYS="${TEMPORAL_ACTION_ENSEMBLE_DECAYS:-{{temporal_action_ensemble_decays}}}"'
        in script
    )
    assert 'SMOOTHING_PROFILE_NAMES="${SMOOTHING_PROFILE_NAMES:-{{smoothing_profile_names}}}"' in script
    assert 'SMOOTHING_PROFILE_BLEND_STEPS="${SMOOTHING_PROFILE_BLEND_STEPS:-{{smoothing_profile_blend_steps}}}"' in script
    assert 'SMOOTHING_PROFILE_ENSEMBLE_DECAYS="${SMOOTHING_PROFILE_ENSEMBLE_DECAYS:-{{smoothing_profile_ensemble_decays}}}"' in script
    assert 'SMOOTHING_PROFILE_NAMES="custom"' in script
    assert 'read -r -a SMOOTHING_PROFILE_NAME_ARRAY <<< "$SMOOTHING_PROFILE_NAMES"' in script
    assert "{#" not in script
    assert 'SMOOTHING_PROFILE_NAME_COUNT="$(wc -w <<< "$SMOOTHING_PROFILE_NAMES" | tr -d \' \')"' in script
    assert 'for SMOOTHING_PROFILE_INDEX in "${!SMOOTHING_PROFILE_NAME_ARRAY[@]}"; do' in script
    assert '--smoothing-profile "$SMOOTHING_PROFILE"' in script
    assert 'JOINT_DELTA_SCALES="${JOINT_DELTA_SCALES:-{{joint_delta_scales}}}"' in script
    assert 'JOINT_TARGET_ACCEL_LIMITS="${JOINT_TARGET_ACCEL_LIMITS:-{{joint_target_accel_limits}}}"' in script
    assert "for JOINT_TARGET_ACCEL_LIMIT in ${JOINT_TARGET_ACCEL_LIMITS}; do" in script
    assert (
        'REPLAN_BOUNDARY_BLEND_STEPS="${REPLAN_BOUNDARY_BLEND_STEPS:-{{replan_boundary_blend_steps}}}"'
        in script
    )
    assert '--joint-target-accel-limit "$JOINT_TARGET_ACCEL_LIMIT"' in script
    assert '"smoothing_profile": cfg.get("smoothing_profile")' in script
    assert '"target_accel_limit": cfg.get("joint_target_accel_limit")' in script
    assert "write_eval_manifest" in script
    assert "checkpoint_eval_manifest.json" in script
    assert "UPLOAD_ARTIFACTS_DONE=0" in script
    assert "Received termination signal; uploading partial eval artifacts" in script
    assert "trap 'on_signal 143' TERM" in script
    assert "trap 'on_signal 130' INT" in script
    assert (
        "One or more eval variants failed; artifacts were prepared and will be uploaded"
        in script
    )
    assert script.index('ARCHIVE="${SERVE_ROOT}/gamma_droidwidth_teacher_') < script.index(
        "One or more eval variants failed; artifacts were prepared"
    )
    assert "DREAMZERO_GIT_COMMIT" in script
    assert "CKPT_RUN_NAME" in script
    assert "runtime_provenance.json" in script
    assert "checkpoint_code_commit" in script
    assert "checkpoint_stage_label" in script
    assert "checkpoint_action_dim" in script
    assert "checkpoint_global_video_attention_mode" in script
    assert "TEMPORAL_ACTION_ENSEMBLE_DECAYS" in script
    assert "SMOOTHING_PROFILE_NAMES" in script
    assert 'write_eval_manifest "$status" || true' in script
    assert "Preflighting RoboFactory renderer before loading policy server" in script
    assert "ROBOFACTORY_RENDER_PREFLIGHT_OK" in script
    assert "render_backend=${ROBOFACTORY_RENDER_BACKEND}" in script
    assert "dense-teacher" not in script

    resources = workflow["workflow"]["resources"]["default"]
    assert resources["platform"] == "dgx-h100"
    assert resources["gpu"] == 8
    assert resources["cpu"] == 84
    assert resources["memory"] == "1681Gi"


def test_droidwidth_teacher_h100_eval_embedded_script_is_valid_bash():
    _, _, script = _workflow_and_script(H100_WORKFLOW_PATH)

    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
