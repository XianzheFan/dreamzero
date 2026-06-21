from pathlib import Path
import subprocess

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = (
    REPO_ROOT
    / "osmo_workflows/robofactory/closedloop_liftbarrier_gamma_staged_r4_stage1_c2000_slim_eval_1seed_20260621.yaml"
)
GB200_WORKFLOW_PATH = (
    REPO_ROOT
    / "osmo_workflows/robofactory/closedloop_liftbarrier_gamma_staged_r4_stage1_c2000_slim_eval_gb200_1seed_20260621.yaml"
)


def _workflow_and_script(path=WORKFLOW_PATH):
    with path.open() as f:
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
    assert 'FATAL: stale umt5-xxl-tokenizer path remains in checkpoint config' in script


def test_gamma_staged_r4_eval_embedded_script_is_valid_bash():
    _, script = _workflow_and_script()

    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_gamma_staged_r4_eval_uses_full_h100_node_resources():
    workflow, script = _workflow_and_script()

    defaults = workflow["default-values"]
    assert workflow["workflow"]["name"] == "{{workflow_name}}"
    assert defaults["workflow_name"] == (
        "dz-rf-sg-gamma-r4-stage1-c2000-slim-eval-1seed1000-xz-20260621"
    )
    assert defaults["run_name"] == (
        "dz-rf-sg-gamma-r4-stage1-c2000-slim-eval-1seed1000-xz-20260621"
    )
    assert defaults["ckpt_setting"] == "checkpoint-2000"
    assert defaults["local_eval_ckpt_root"] == (
        "gamma_staged_r4_stage1_c2000_slim_eval_1seed1000"
    )
    assert 'RUN_NAME="{{run_name}}"' in script
    assert 'CKPT_SETTING="{{ckpt_setting}}"' in script
    assert 'LOCAL_EVAL_CKPT_ROOT="/workspace/eval_ckpts/{{local_eval_ckpt_root}}"' in script

    resources = workflow["workflow"]["resources"]["default"]
    assert resources["platform"] == "dgx-h100"
    assert resources["gpu"] == 8
    assert resources["cpu"] == 84
    assert resources["memory"] == "1681Gi"


def test_gamma_staged_r4_gb200_eval_uses_gb200_node_resources():
    workflow, script = _workflow_and_script(GB200_WORKFLOW_PATH)

    defaults = workflow["default-values"]
    assert workflow["workflow"]["name"] == "{{workflow_name}}"
    assert defaults["workflow_name"] == (
        "dz-rf-sg-gamma-r4-stage1-c2000-slim-eval-gb200-1seed1000-xz-20260621"
    )
    assert defaults["run_name"] == (
        "dz-rf-sg-gamma-r4-stage1-c2000-slim-eval-gb200-1seed1000-xz-20260621"
    )
    assert defaults["ckpt_setting"] == "checkpoint-2000"
    assert defaults["local_eval_ckpt_root"] == (
        "gamma_staged_r4_stage1_c2000_slim_eval_1seed1000"
    )
    assert 'RUN_NAME="{{run_name}}"' in script
    assert 'CKPT_SETTING="{{ckpt_setting}}"' in script
    assert 'LOCAL_EVAL_CKPT_ROOT="/workspace/eval_ckpts/{{local_eval_ckpt_root}}"' in script
    assert 'SERVER_CUDA_VISIBLE_DEVICES="1"' in script
    assert 'CLIENT_CUDA_VISIBLE_DEVICES="0"' in script
    assert "FATAL: stale umt5-xxl-tokenizer path remains in checkpoint config" in script

    resources = workflow["workflow"]["resources"]["default"]
    assert resources["platform"] == "gb200"
    assert resources["gpu"] == 4
    assert resources["cpu"] == 84
    assert resources["memory"] == "760Gi"


def test_gamma_staged_r4_gb200_eval_embedded_script_is_valid_bash():
    _, script = _workflow_and_script(GB200_WORKFLOW_PATH)

    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
