from pathlib import Path
import importlib.util

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/eval/submit_robofactory_droidwidth_eval_grid.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("submit_robofactory_droidwidth_eval_grid", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_steps_default_grid_is_2k_spaced():
    module = _load_module()

    steps = module.checkpoint_steps(2000, 10000, 2000)

    assert steps == [2000, 4000, 6000, 8000, 10000]


def test_checkpoint_steps_rejects_bad_ranges():
    module = _load_module()

    with pytest.raises(ValueError):
        module.checkpoint_steps(2000, 10000, 0)
    with pytest.raises(ValueError):
        module.checkpoint_steps(0, 10000, 2000)
    with pytest.raises(ValueError):
        module.checkpoint_steps(12000, 10000, 2000)


def test_build_submit_command_uses_checkpoint_specific_names():
    module = _load_module()

    command = module.build_submit_command(
        workflow=Path("eval.yaml"),
        pool="groot-h100-02",
        priority="LOW",
        step=4000,
        tag="20260621",
        name_template=module.DEFAULT_NAME_TEMPLATE,
        local_root_template=module.DEFAULT_LOCAL_ROOT_TEMPLATE,
    )

    assert command[:4] == ["osmo", "workflow", "submit", "eval.yaml"]
    assert "--set-string" in command
    assert "ckpt_setting=checkpoint-4000" in command
    assert "ckpt_setting=checkpoint-2500" not in command
    assert (
        "workflow_name=dz-rf-sg-gamma-dwteacher-50kfrom0-c4000-slim-eval-h100-1seed1000-xz-20260621"
        in command
    )
    assert (
        "run_name=dz-rf-sg-gamma-dwteacher-50kfrom0-c4000-slim-eval-h100-1seed1000-xz-20260621"
        in command
    )
    assert "local_eval_ckpt_root=gamma_droidwidth_teacher_50kfrom0_c4000_slim_eval_h100_1seed1000" in command


def test_main_prints_dry_run_commands_without_submitting(capsys):
    module = _load_module()

    status = module.main(
        [
            "--workflow",
            "eval.yaml",
            "--tag",
            "20260621",
            "--start-step",
            "2000",
            "--max-step",
            "6000",
            "--interval",
            "2000",
        ]
    )

    assert status == 0
    out = capsys.readouterr().out
    assert out.count("osmo workflow submit eval.yaml") == 3
    assert "ckpt_setting=checkpoint-2000" in out
    assert "ckpt_setting=checkpoint-4000" in out
    assert "ckpt_setting=checkpoint-6000" in out
    assert "ckpt_setting=checkpoint-2500" not in out
