from pathlib import Path
import subprocess

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = (
    REPO_ROOT
    / "osmo_workflows/robofactory/closedloop_liftbarrier_50kscratch_c50000_gamma_1seed_20260620.yaml"
)


def _workflow_and_script():
    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)
    task = workflow["workflow"]["groups"][0]["tasks"][0]
    return workflow, task["files"][0]["contents"]


def test_closedloop_gamma_workflow_targets_50k_checkpoint_and_gamma_ref():
    workflow, script = _workflow_and_script()

    assert workflow["workflow"]["name"] == (
        "dz-rf2-lb50k-c50000-gamma-1seed1000-xz-20260620"
    )
    assert 'DREAMZERO_GIT_REF="${DREAMZERO_GIT_REF:-gamma}"' in script
    assert "Resolved DreamZero ref ${DREAMZERO_GIT_REF} to ${DREAMZERO_GIT_COMMIT}" in script
    assert "dreamzero_code_commit.txt" in script
    assert "checkpoint-50000" in script
    assert (
        "robofactory_2arm_LiftBarrier-rf-500_droidwidth_droidwarm_"
        "openphasejoint_d344e8e_50k_scratch_h100x8"
    ) in script

    resources = workflow["workflow"]["resources"]["default"]
    assert resources["gpu"] == 8
    assert resources["platform"] == "dgx-h100"


def test_closedloop_gamma_workflow_runs_causal_cache_diagnostics():
    _, script = _workflow_and_script()

    for marker in (
        'INFERENCE_MODES="${INFERENCE_MODES:-causal_flowmatch}"',
        'RESET_CAUSAL_STATE_EACH_INFERS="${RESET_CAUSAL_STATE_EACH_INFERS:-0 1}"',
        'export MAI_WRITE_DENOISED_CONTEXT_CACHE="${MAI_WRITE_DENOISED_CONTEXT_CACHE:-1}"',
        'export MAI_ROLLING_NOISE="${MAI_ROLLING_NOISE:-1}"',
        '--shared-global-wrist-window-mode history-current-first',
        '--video-pred-rollout-mode "$VIDEO_PRED_ROLLOUT_MODE"',
        '"$RESET_ARG"',
        'RESET_TAG="persistcache"',
        'RESET_TAG="resetcache"',
        '--video-pred-dir "/workspace/eval_outputs/video_pred/${INFERENCE_MODE}_${RESET_TAG}"',
        "analyze_video_pred_quality.py",
        "/workspace/code/dreamzero/scripts/eval/analyze_video_pred_quality.py",
        'server_meta = data.get("server", {}).get("meta", {})',
        '"reset_causal_state_each_infer": server_meta.get("reset_causal_state_each_infer")',
        '"write_denoised_context_cache": server_meta.get("write_denoised_context_cache")',
        '"mai_rolling_noise": server_meta.get("mai_rolling_noise")',
        '"video_pred_rollout_mode": server_meta.get("video_pred_rollout_mode")',
        '"shared_global_wrist_window_mode": server_meta.get("shared_global_wrist_window_mode")',
        '--joint-target-slew-rate "$JOINT_TARGET_SLEW_RATE"',
        '--success-mode "$SUCCESS_MODE"',
        '--strict-success-min-grasp-count "$STRICT_SUCCESS_MIN_GRASP_COUNT"',
        '--left-gripper-close-after-step "$LEFT_GRIPPER_CLOSE_AFTER_STEP"',
        '--right-gripper-close-after-step "$RIGHT_GRIPPER_CLOSE_AFTER_STEP"',
    ):
        assert marker in script


def test_closedloop_gamma_workflow_video_quality_script_exists():
    assert (REPO_ROOT / "scripts/eval/analyze_video_pred_quality.py").is_file()


def test_closedloop_gamma_workflow_does_not_patch_dreamzero_at_runtime():
    _, script = _workflow_and_script()

    assert "git -C /workspace/code/dreamzero apply" not in script
    assert "<<'PATCH'" not in script


def test_closedloop_gamma_workflow_embedded_script_is_valid_bash():
    _, script = _workflow_and_script()

    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
