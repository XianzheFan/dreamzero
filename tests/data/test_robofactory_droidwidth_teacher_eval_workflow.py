from pathlib import Path
import subprocess

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = (
    REPO_ROOT
    / "osmo_workflows/robofactory/closedloop_liftbarrier_gamma_droidwidth_teacher_c500_slim_eval_gb200_1seed_20260621.yaml"
)
H100_WORKFLOW_PATH = (
    REPO_ROOT
    / "osmo_workflows/robofactory/closedloop_liftbarrier_gamma_droidwidth_teacher_c500_slim_eval_h100_1seed_20260621.yaml"
)


def _workflow_and_script(path=WORKFLOW_PATH):
    with path.open() as f:
        workflow = yaml.safe_load(f)
    task = workflow["workflow"]["groups"][0]["tasks"][0]
    return workflow, task, task["files"][0]["contents"]


def test_droidwidth_teacher_eval_points_to_teacher_checkpoint_prefix():
    workflow, task, script = _workflow_and_script()

    assert workflow["workflow"]["name"] == (
        "dz-rf-sg-gamma-dwteacher-c500-slim-eval-gb200-1seed1000-xz-20260621"
    )
    assert task["image"] == "nvcr.io/nvidian/gr00t_isaac:v1.5"
    assert (
        'RUN_NAME="dz-rf-sg-gamma-dwteacher-c500-slim-eval-gb200-1seed1000-xz-20260621"'
        in script
    )
    assert (
        "dreamzero_runs/dz-rf-sg-gamma-dwteacher-lb500-r4-actiondelta-xianzhef-20260621-teacher/"
        "checkpoints/dz-rf-sg-gamma-dwteacher-lb500-r4-actiondelta-xianzhef-20260621-teacher"
        in script
    )
    assert 'CKPT_SETTING="checkpoint-500"' in script
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
    assert (
        'osmo data download --resume --processes 1 --threads 4 '
        '"${MODEL_CACHE_S3_URI}/DreamZero-DROID/" "$DROID_IMPORT"'
        in script
    )


def test_droidwidth_teacher_eval_embedded_script_is_valid_bash():
    _, _, script = _workflow_and_script()

    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_droidwidth_teacher_h100_eval_targets_h100_pool_resources():
    workflow, task, script = _workflow_and_script(H100_WORKFLOW_PATH)

    assert workflow["workflow"]["name"] == (
        "dz-rf-sg-gamma-dwteacher-c500-slim-eval-h100-1seed1000-xz-20260621"
    )
    assert task["image"].startswith("nvcr.io/nvidian/groot-ci-base-eval:")
    assert (
        'RUN_NAME="dz-rf-sg-gamma-dwteacher-c500-slim-eval-h100-1seed1000-xz-20260621"'
        in script
    )
    assert (
        "dreamzero_runs/dz-rf-sg-gamma-dwteacher-lb500-r4-actiondelta-xianzhef-20260621-teacher/"
        "checkpoints/dz-rf-sg-gamma-dwteacher-lb500-r4-actiondelta-xianzhef-20260621-teacher"
        in script
    )
    assert 'VIDEO_PRED_WRIST_WINDOW_MODE="${VIDEO_PRED_WRIST_WINDOW_MODE:-action}"' in script
    assert 'ROBOFACTORY_RENDER_BACKEND="${ROBOFACTORY_RENDER_BACKEND:-sapien_cuda:0}"' in script
    assert 'ROBOFACTORY_ENABLE_SHADOW="${ROBOFACTORY_ENABLE_SHADOW:-0}"' in script
    assert 'ROBOFACTORY_SHADER_PACK="${ROBOFACTORY_SHADER_PACK:-default}"' in script
    assert 'ROBOFACTORY_RENDER_PREFLIGHT="${ROBOFACTORY_RENDER_PREFLIGHT:-1}"' in script
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
