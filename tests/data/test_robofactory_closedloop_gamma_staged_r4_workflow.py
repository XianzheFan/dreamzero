from pathlib import Path
import subprocess

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = (
    REPO_ROOT
    / "osmo_workflows/robofactory/closedloop_liftbarrier_gamma_staged_r4_stage1_c500_slim_eval_1seed_20260621.yaml"
)


def _workflow_and_script():
    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)
    task = workflow["workflow"]["groups"][0]["tasks"][0]
    return workflow, task["files"][0]["contents"]


def test_gamma_staged_r4_eval_patches_nested_tokenizer_paths():
    _, script = _workflow_and_script()

    assert '"tokenizer_path": "/workspace/checkpoints/umt5-xxl"' in script
    assert (
        'text = text.replace("/workspace/checkpoints/umt5-xxl-tokenizer", '
        '"/workspace/checkpoints/umt5-xxl")'
    ) in script
    assert 'grep -n "tokenizer_path" "${candidate_dir}/experiment_cfg/conf.yaml"' in script


def test_gamma_staged_r4_eval_embedded_script_is_valid_bash():
    _, script = _workflow_and_script()

    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_gamma_staged_r4_eval_uses_full_h100_node_resources():
    workflow, _ = _workflow_and_script()

    resources = workflow["workflow"]["resources"]["default"]
    assert resources["platform"] == "dgx-h100"
    assert resources["gpu"] == 8
    assert resources["cpu"] == 84
    assert resources["memory"] == "1681Gi"
