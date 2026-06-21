from pathlib import Path
import subprocess

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = (
    REPO_ROOT
    / "osmo_workflows/robofactory/closedloop_liftbarrier_gamma_staged_r4_stage1_c500_slim_eval_gb200_1seed_20260621.yaml"
)


def _workflow_and_script():
    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)
    task = workflow["workflow"]["groups"][0]["tasks"][0]
    return workflow, task, task["files"][0]["contents"]


def test_gb200_gamma_eval_defaults_to_model_only_diag_without_maniskill():
    _, task, script = _workflow_and_script()

    assert task["image"] == "nvcr.io/nvidian/gr00t_isaac:v1.5"
    assert 'MODEL_ONLY_DIAG="${MODEL_ONLY_DIAG:-1}"' in script
    assert 'DREAMZERO_VENV="/mnt/amlfs-01/home/xianzhef/osmo_cache/dreamzero/venvs/gb200_rf_eval_py312"' in script
    assert "uv venv --python /usr/bin/python3.12 --system-site-packages" in script
    assert 'export PYTHONPATH="/workspace/code/dreamzero:${PYTHONPATH:-}"' in script
    assert 'if [ "$MODEL_ONLY_DIAG" != "1" ]; then' in script
    assert 'python -m pip install -q "mani_skill==3.0.0b12" decord' in script


def test_gb200_gamma_eval_runs_offline_video_action_diagnostic():
    _, _, script = _workflow_and_script()

    assert "eval_utils/offline_video_action_diagnostic.py" in script
    assert "python -m eval_utils.offline_video_action_diagnostic" in script
    assert "--inference-mode \"$MODEL_ONLY_INFERENCE_MODE\"" in script
    assert "model_only_video_pred_quality.json" in script
    assert "GB200 model-only diagnostics complete" in script


def test_gb200_gamma_eval_embedded_script_is_valid_bash():
    _, _, script = _workflow_and_script()

    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
