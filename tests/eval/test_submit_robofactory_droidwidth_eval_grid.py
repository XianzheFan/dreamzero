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
    assert module.DEFAULT_POOL == "groot-h100-01"
    assert "c2000" in module.DEFAULT_WORKFLOW
    assert "c500" not in module.DEFAULT_WORKFLOW


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
        pool="groot-h100-01",
        priority="LOW",
        step=4000,
        tag="20260621",
        name_template=module.DEFAULT_NAME_TEMPLATE,
        local_root_template=module.DEFAULT_LOCAL_ROOT_TEMPLATE,
        ckpt_run_name=module.DEFAULT_CKPT_RUN_NAME,
        ckpt_s3_base_value=module.checkpoint_s3_base(module.DEFAULT_CKPT_RUN_NAME),
        ckpt_amlfs_base_value=module.checkpoint_amlfs_base(module.DEFAULT_CKPT_RUN_NAME),
    )

    assert command[:4] == ["osmo", "workflow", "submit", "eval.yaml"]
    assert "--set-string" in command
    assert "ckpt_setting=checkpoint-4000" in command
    assert "ckpt_setting=checkpoint-2500" not in command
    assert (
        "workflow_name=dz-rf-sg-gamma-dwteacher-bidir-50k-c4000-slim-eval-h100-1seed1000-xz-20260621"
        in command
    )
    assert (
        "run_name=dz-rf-sg-gamma-dwteacher-bidir-50k-c4000-slim-eval-h100-1seed1000-xz-20260621"
        in command
    )
    assert "ckpt_run_name=dz-rf-sg-gamma-dwteacher-bidir-lb500-50k-xz-20260622-teacher" in command
    assert (
        "ckpt_s3_base=s3://GearHome/users/xianzhef/oci-migration/dreamzero_runs/"
        "dz-rf-sg-gamma-dwteacher-bidir-lb500-50k-xz-20260622-teacher/checkpoints/"
        "dz-rf-sg-gamma-dwteacher-bidir-lb500-50k-xz-20260622-teacher"
        in command
    )
    assert (
        "local_eval_ckpt_root=gamma_droidwidth_teacher_bidir_50k_c4000_slim_eval_h100_1seed1000"
        in command
    )


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
    assert "--pool groot-h100-01" in out
    assert "ckpt_run_name=dz-rf-sg-gamma-dwteacher-bidir-lb500-50k-xz-20260622-teacher" in out


def test_main_rejects_explicit_off_grid_steps_by_default():
    module = _load_module()

    with pytest.raises(SystemExit):
        module.main(
            [
                "--workflow",
                "eval.yaml",
                "--tag",
                "20260621",
                "--steps",
                "2000,3500",
            ]
        )


def test_main_allows_explicit_off_grid_steps_for_one_off_diagnostics(capsys):
    module = _load_module()

    status = module.main(
        [
            "--workflow",
            "eval.yaml",
            "--tag",
            "20260621",
            "--steps",
            "3500",
            "--allow-off-grid-steps",
        ]
    )

    assert status == 0
    out = capsys.readouterr().out
    assert "ckpt_setting=checkpoint-3500" in out


def test_listing_has_ready_checkpoint_accepts_complete_marker():
    module = _load_module()

    listing = """
    checkpoint-2000/.cache_complete
    checkpoint-2000/model.safetensors
    """

    assert module.listing_has_ready_checkpoint(listing)


def test_listing_has_ready_checkpoint_accepts_model_and_metadata():
    module = _load_module()

    listing = """
    model.safetensors
    trainer_state.json
    experiment_cfg/conf.yaml
    """

    assert module.listing_has_ready_checkpoint(listing)


def test_main_only_ready_filters_unavailable_steps(monkeypatch, capsys):
    module = _load_module()

    def fake_check_checkpoint_ready(*, osmo_binary, ckpt_s3_base_value, step):
        return module.ReadyCheck(
            step=step,
            uri=f"{ckpt_s3_base_value}/checkpoint-{step}/",
            ready=step == 4000,
            reason="ready" if step == 4000 else "missing model/trainer_state/experiment_cfg ready markers",
        )

    monkeypatch.setattr(module, "check_checkpoint_ready", fake_check_checkpoint_ready)

    status = module.main(
        [
            "--workflow",
            "eval.yaml",
            "--tag",
            "20260621",
            "--steps",
            "2000,4000",
            "--only-ready",
        ]
    )

    assert status == 0
    captured = capsys.readouterr()
    assert "ckpt_setting=checkpoint-4000" in captured.out
    assert "ckpt_setting=checkpoint-2000" not in captured.out
    assert "SKIP checkpoint-2000" in captured.err
    assert "READY checkpoint-4000" in captured.err


def test_main_only_ready_can_fail_when_none_ready(monkeypatch, capsys):
    module = _load_module()

    def fake_check_checkpoint_ready(*, osmo_binary, ckpt_s3_base_value, step):
        return module.ReadyCheck(
            step=step,
            uri=f"{ckpt_s3_base_value}/checkpoint-{step}/",
            ready=False,
            reason="missing model/trainer_state/experiment_cfg ready markers",
        )

    monkeypatch.setattr(module, "check_checkpoint_ready", fake_check_checkpoint_ready)

    status = module.main(
        [
            "--workflow",
            "eval.yaml",
            "--tag",
            "20260621",
            "--steps",
            "2000,4000",
            "--only-ready",
            "--fail-if-none-ready",
        ]
    )

    assert status == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No ready checkpoints matched the requested eval grid." in captured.err
