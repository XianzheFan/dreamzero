from pathlib import Path
import importlib.util
import subprocess

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
        dreamzero_git_ref="gamma",
        dreamzero_expected_git_commit="abc123",
    )

    assert command[:4] == ["osmo", "workflow", "submit", "eval.yaml"]
    assert "--set-string" in command
    assert "ckpt_setting=checkpoint-4000" in command
    assert "ckpt_setting=checkpoint-2500" not in command
    assert (
        "workflow_name=dz-rf-gamma-dw-af2-lb500-50k-c4000-eval-h100-s1000-xz-20260621"
        in command
    )
    assert (
        "run_name=dz-rf-gamma-dw-af2-lb500-50k-c4000-eval-h100-s1000-xz-20260621"
        in command
    )
    assert (
        "ckpt_run_name=dz-rf-sg-gamma-dwteacher-actionlossfix2-lb500-50k-xz-20260622-teacher"
        in command
    )
    assert (
        "ckpt_s3_base=s3://GearHome/users/xianzhef/oci-migration/dreamzero_runs/"
        "dz-rf-sg-gamma-dwteacher-actionlossfix2-lb500-50k-xz-20260622-teacher/checkpoints/"
        "dz-rf-sg-gamma-dwteacher-actionlossfix2-lb500-50k-xz-20260622-teacher"
        in command
    )
    assert (
        "local_eval_ckpt_root=gamma_droidwidth_teacher_actionlossfix2_lb500_50k_c4000_slim_eval_h100_1seed1000"
        in command
    )
    assert "dreamzero_git_ref=gamma" in command
    assert "dreamzero_expected_git_commit=abc123" in command
    assert "eval_num_frames=65" not in command
    assert "eval_action_horizon=24" not in command


def test_build_submit_command_can_override_eval_window():
    module = _load_module()

    command = module.build_submit_command(
        workflow=Path("eval.yaml"),
        pool="groot-h100-01",
        priority="LOW",
        step=2000,
        tag="longwin",
        name_template=module.DEFAULT_NAME_TEMPLATE,
        local_root_template=module.DEFAULT_LOCAL_ROOT_TEMPLATE,
        ckpt_run_name=module.DEFAULT_CKPT_RUN_NAME,
        ckpt_s3_base_value=module.checkpoint_s3_base(module.DEFAULT_CKPT_RUN_NAME),
        ckpt_amlfs_base_value=module.checkpoint_amlfs_base(module.DEFAULT_CKPT_RUN_NAME),
        eval_num_frames=65,
        eval_action_horizon=24,
    )

    assert "eval_num_frames=65" in command
    assert "eval_action_horizon=24" in command


def test_main_prints_dry_run_commands_without_submitting(monkeypatch, capsys):
    module = _load_module()
    monkeypatch.setattr(module, "current_git_head", lambda: "abc123")

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
    assert "ckpt_run_name=dz-rf-sg-gamma-dwteacher-actionlossfix2-lb500-50k-xz-20260622-teacher" in out
    assert "dreamzero_git_ref=gamma" in out
    assert "dreamzero_expected_git_commit=abc123" in out


def test_main_prints_eval_window_overrides(monkeypatch, capsys):
    module = _load_module()
    monkeypatch.setattr(module, "current_git_head", lambda: "abc123")

    status = module.main(
        [
            "--workflow",
            "eval.yaml",
            "--tag",
            "20260621",
            "--steps",
            "2000",
            "--eval-num-frames",
            "65",
            "--eval-action-horizon",
            "24",
        ]
    )

    assert status == 0
    out = capsys.readouterr().out
    assert "ckpt_setting=checkpoint-2000" in out
    assert "eval_num_frames=65" in out
    assert "eval_action_horizon=24" in out


def test_fullft_gate_preset_prints_three_10_seed_scale1_eval_commands(monkeypatch, capsys):
    module = _load_module()
    monkeypatch.setattr(module, "current_git_head", lambda: "abc123")

    status = module.main(
        [
            "--workflow",
            "eval.yaml",
            "--tag",
            "20260625",
            "--preset",
            "fullft-gate",
        ]
    )

    assert status == 0
    out = capsys.readouterr().out
    assert out.count("osmo workflow submit eval.yaml") == 3
    assert "ckpt_setting=checkpoint-10000" in out
    assert "ckpt_setting=checkpoint-20000" in out
    assert "ckpt_setting=checkpoint-30000" in out
    assert " ckpt_setting=checkpoint-2000 " not in out
    assert (
        "workflow_name=dz-rf-gamma-dwteacher-fullft-lb500-30k-c10000-"
        "eval-h100-s1000-s10-xz-20260625"
    ) in out
    assert (
        "ckpt_run_name=dz-rf-sg-gamma-dwteacher-fullft-lb500-30k-xz-20260625-teacher"
        in out
    )
    assert "num_episodes=10" in out
    assert "video_pred_rollout_modes=action" in out
    assert "'replan_everys=24 12'" in out
    assert "joint_delta_scales=1.0" in out
    assert "joint_target_accel_limits=0" in out
    assert "'smoothing_profile_names=raw smooth'" in out
    assert "'smoothing_profile_blend_steps=0 4'" in out
    assert "'smoothing_profile_ensemble_decays=0 0.6'" in out


def test_fullft_gate_preset_respects_explicit_steps_run_name_and_set_string(monkeypatch, capsys):
    module = _load_module()
    monkeypatch.setattr(module, "current_git_head", lambda: "abc123")

    status = module.main(
        [
            "--workflow",
            "eval.yaml",
            "--tag",
            "20260625",
            "--preset",
            "fullft-gate",
            "--steps",
            "12000",
            "--ckpt-run-name",
            "custom-fullft-teacher",
            "--name-template",
            "custom-c{step}-{tag}",
            "--set-string",
            "num_episodes=3",
        ]
    )

    assert status == 0
    out = capsys.readouterr().out
    assert "workflow_name=custom-c12000-20260625" in out
    assert "ckpt_setting=checkpoint-12000" in out
    assert "ckpt_setting=checkpoint-10000" not in out
    assert "ckpt_run_name=custom-fullft-teacher" in out
    assert "num_episodes=3" in out
    assert "num_episodes=10" not in out
    assert "joint_delta_scales=1.0" in out


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


def test_train_workflow_ready_check_uses_remote_checkpoint_markers(monkeypatch):
    module = _load_module()
    calls = []

    def fake_run(command):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ready: /outputs/checkpoint-2000\n", stderr="")

    monkeypatch.setattr(module, "run_command_with_pty", fake_run)

    result = module.check_checkpoint_ready_from_train_workflow(
        osmo_binary="osmo",
        workflow_name="train-wf",
        task_name="train",
        output_dir="/outputs",
        step=2000,
    )

    assert result == module.ReadyCheck(
        step=2000,
        uri="/outputs/checkpoint-2000",
        ready=True,
        reason="ready",
    )
    assert calls == [
        [
            "osmo",
            "workflow",
            "exec",
            "train-wf",
            "train",
            "--entry",
            calls[0][-1],
        ]
    ]
    assert "checkpoint-2000" in calls[0][-1]
    assert "trainer_state.json" in calls[0][-1]
    assert "experiment_cfg/conf.yaml" in calls[0][-1]


def test_train_workflow_ready_check_reports_incomplete_checkpoint(monkeypatch):
    module = _load_module()

    def fake_run(command):
        return subprocess.CompletedProcess(
            command,
            4,
            stdout="incomplete checkpoint: /outputs/checkpoint-4000\nmodel.safetensors\n",
            stderr="",
        )

    monkeypatch.setattr(module, "run_command_with_pty", fake_run)

    result = module.check_checkpoint_ready_from_train_workflow(
        osmo_binary="osmo",
        workflow_name="train-wf",
        task_name="train",
        output_dir="/outputs",
        step=4000,
    )

    assert result.step == 4000
    assert result.uri == "/outputs/checkpoint-4000"
    assert not result.ready
    assert "incomplete checkpoint" in result.reason
    assert "model.safetensors" in result.reason


def test_train_workflow_ready_check_requires_ready_marker_even_when_cli_status_is_zero(monkeypatch):
    module = _load_module()

    def fake_run(command):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="missing checkpoint directory: /outputs/checkpoint-2000\n",
            stderr="",
        )

    monkeypatch.setattr(module, "run_command_with_pty", fake_run)

    result = module.check_checkpoint_ready_from_train_workflow(
        osmo_binary="osmo",
        workflow_name="train-wf",
        task_name="train",
        output_dir="/outputs",
        step=2000,
    )

    assert result.step == 2000
    assert result.uri == "/outputs/checkpoint-2000"
    assert not result.ready
    assert "without ready marker" in result.reason
    assert "missing checkpoint directory" in result.reason


def test_main_train_workflow_ready_check_requires_workflow_name():
    module = _load_module()

    with pytest.raises(SystemExit):
        module.main(
            [
                "--workflow",
                "eval.yaml",
                "--tag",
                "20260621",
                "--steps",
                "2000",
                "--only-ready",
                "--ready-check-source",
                "train-workflow",
            ]
        )


def test_main_only_ready_can_use_train_workflow_source(monkeypatch, capsys):
    module = _load_module()

    def fake_check_checkpoint_ready_from_train_workflow(
        *,
        osmo_binary,
        workflow_name,
        task_name,
        output_dir,
        step,
    ):
        assert workflow_name == "active-train"
        assert task_name == "train"
        assert output_dir == "/outputs"
        return module.ReadyCheck(
            step=step,
            uri=f"{output_dir}/checkpoint-{step}",
            ready=step == 4000,
            reason="ready" if step == 4000 else "missing checkpoint directory",
        )

    monkeypatch.setattr(
        module,
        "check_checkpoint_ready_from_train_workflow",
        fake_check_checkpoint_ready_from_train_workflow,
    )

    status = module.main(
        [
            "--workflow",
            "eval.yaml",
            "--tag",
            "20260621",
            "--steps",
            "2000,4000",
            "--only-ready",
            "--ready-check-source",
            "train-workflow",
            "--ready-train-workflow",
            "active-train",
            "--ready-train-output-dir",
            "/outputs",
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


def test_main_skip_existing_filters_existing_workflow(monkeypatch, capsys):
    module = _load_module()

    def fake_check_workflow_exists(*, osmo_binary, workflow_name, suffix_scan_limit):
        assert suffix_scan_limit == 20
        return module.ExistingWorkflowCheck(
            name=workflow_name,
            exists="c2000" in workflow_name,
            reason="exists" if "c2000" in workflow_name else "not found",
        )

    monkeypatch.setattr(module, "check_workflow_exists", fake_check_workflow_exists)

    status = module.main(
        [
            "--workflow",
            "eval.yaml",
            "--tag",
            "20260621",
            "--steps",
            "2000,4000",
            "--skip-existing",
        ]
    )

    assert status == 0
    captured = capsys.readouterr()
    assert "ckpt_setting=checkpoint-4000" in captured.out
    assert "ckpt_setting=checkpoint-2000" not in captured.out
    assert "SKIP checkpoint-2000: workflow already exists" in captured.err
    assert "NEW checkpoint-4000" in captured.err


def test_main_skip_existing_reports_suffixed_workflow(monkeypatch, capsys):
    module = _load_module()

    def fake_check_workflow_exists(*, osmo_binary, workflow_name, suffix_scan_limit):
        return module.ExistingWorkflowCheck(
            name=f"{workflow_name}-2",
            exists=True,
            reason=f"exists as {workflow_name}-2",
        )

    monkeypatch.setattr(module, "check_workflow_exists", fake_check_workflow_exists)

    status = module.main(
        [
            "--workflow",
            "eval.yaml",
            "--tag",
            "20260621",
            "--steps",
            "2000",
            "--skip-existing",
        ]
    )

    assert status == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        "SKIP checkpoint-2000: workflow already exists: "
        "dz-rf-gamma-dw-af2-lb500-50k-c2000-eval-h100-s1000-xz-20260621-2"
    ) in captured.err


def test_main_skip_existing_noops_when_all_exist(monkeypatch, capsys):
    module = _load_module()

    def fake_check_workflow_exists(*, osmo_binary, workflow_name, suffix_scan_limit):
        return module.ExistingWorkflowCheck(name=workflow_name, exists=True, reason="exists")

    monkeypatch.setattr(module, "check_workflow_exists", fake_check_workflow_exists)

    status = module.main(
        [
            "--workflow",
            "eval.yaml",
            "--tag",
            "20260621",
            "--steps",
            "2000,4000",
            "--skip-existing",
        ]
    )

    assert status == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No new checkpoint eval workflows matched the requested grid." in captured.err


def test_check_workflow_exists_accepts_exact_match(monkeypatch):
    module = _load_module()
    queries = []

    def fake_run(command, check, capture_output, text):
        queries.append(command[3])
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.check_workflow_exists(
        osmo_binary="osmo",
        workflow_name="eval-wf",
        suffix_scan_limit=3,
    )

    assert result == module.ExistingWorkflowCheck(name="eval-wf", exists=True, reason="exists")
    assert queries == ["eval-wf"]


def test_check_workflow_exists_scans_osmo_auto_suffixes(monkeypatch):
    module = _load_module()
    queries = []

    def fake_run(command, check, capture_output, text):
        workflow_name = command[3]
        queries.append(workflow_name)
        return subprocess.CompletedProcess(
            command,
            0 if workflow_name == "eval-wf-2" else 1,
            stdout="{}" if workflow_name == "eval-wf-2" else "",
            stderr="" if workflow_name == "eval-wf-2" else "not found",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.check_workflow_exists(
        osmo_binary="osmo",
        workflow_name="eval-wf",
        suffix_scan_limit=3,
    )

    assert result == module.ExistingWorkflowCheck(
        name="eval-wf-2",
        exists=True,
        reason="exists as eval-wf-2",
    )
    assert queries == ["eval-wf", "eval-wf-1", "eval-wf-2"]
