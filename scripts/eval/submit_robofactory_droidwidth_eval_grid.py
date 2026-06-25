#!/usr/bin/env python3
"""Generate or submit 2k-spaced RoboFactory droidwidth teacher eval jobs."""

from __future__ import annotations

import argparse
import datetime as dt
import errno
import os
import pty
import select
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple


DEFAULT_WORKFLOW = (
    "osmo_workflows/robofactory/"
    "closedloop_liftbarrier_gamma_droidwidth_teacher_c2000_slim_eval_h100_1seed_20260621.yaml"
)
DEFAULT_POOL = "groot-h100-01"
DEFAULT_NAME_TEMPLATE = (
    "dz-rf-gamma-dw-af2-lb500-50k-c{step}-eval-h100-s1000-xz-{tag}"
)
DEFAULT_LOCAL_ROOT_TEMPLATE = (
    "gamma_droidwidth_teacher_actionlossfix2_lb500_50k_c{step}_slim_eval_h100_1seed1000"
)
DEFAULT_CKPT_RUN_NAME = "dz-rf-sg-gamma-dwteacher-actionlossfix2-lb500-50k-xz-20260622-teacher"
DEFAULT_CKPT_S3_RUNS_PREFIX = "s3://GearHome/users/xianzhef/oci-migration/dreamzero_runs"
DEFAULT_CKPT_AMLFS_RUNS_PREFIX = "/mnt/amlfs-01/home/xianzhef/osmo_cache/dreamzero/checkpoints"
DEFAULT_DREAMZERO_GIT_REF = "gamma"
DEFAULT_READY_TRAIN_TASK = "train"
DEFAULT_READY_TRAIN_OUTPUT_DIR = "/workspace/outputs/robofactory_liftbarrier_gamma_droidwidth_teacher/teacher"
FULLFT_GATE_STEPS = [10000, 20000, 30000]
FULLFT_2K_GATE_STEPS = list(range(2000, 30001, 2000))
FULLFT_GATE_NAME_TEMPLATE = (
    "dz-rf-gamma-dwteacher-fullft-offload-lb500-30k-c{step}-eval-h100-s1000-s10-xz-{tag}"
)
FULLFT_GATE_LOCAL_ROOT_TEMPLATE = (
    "gamma_droidwidth_teacher_fullft_offload_lb500_30k_c{step}_slim_eval_h100_seed1000_s10"
)
FULLFT_GATE_CKPT_RUN_NAME_TEMPLATE = (
    "dz-rf-sg-gamma-dwteacher-fullft-offload-lb500-30k-xz-{tag}-teacher"
)
FULLFT_GATE_SET_STRING_DEFAULTS = (
    ("num_episodes", "10"),
    ("video_pred_rollout_modes", "action"),
    ("replan_everys", "24 12"),
    ("joint_delta_scales", "1.0"),
    ("joint_target_accel_limits", "0"),
    ("smoothing_profile_names", "raw smooth"),
    ("smoothing_profile_blend_steps", "0 4"),
    ("smoothing_profile_ensemble_decays", "0 0.6"),
)
CLOSE_TIMING_DIAGNOSTIC_SET_STRING_DEFAULTS = (
    ("gripper_close_pairs", "33:33 52:52"),
)
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_MARKERS = ("model.safetensors", "model.safetensors.index.json")


class ReadyCheck(NamedTuple):
    step: int
    uri: str
    ready: bool
    reason: str


class ExistingWorkflowCheck(NamedTuple):
    name: str
    exists: bool
    reason: str


def run_command_with_pty(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run a command behind a PTY for CLIs that require terminal sizing."""
    master_fd, slave_fd = pty.openpty()
    chunks: list[bytes] = []
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            list(command),
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
        )
        os.close(slave_fd)
        slave_fd = -1
        while True:
            if process.poll() is not None:
                break
            readable, _, _ = select.select([master_fd], [], [], 0.2)
            if master_fd not in readable:
                continue
            try:
                data = os.read(master_fd, 8192)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise
            if not data:
                break
            chunks.append(data)
        while True:
            try:
                data = os.read(master_fd, 8192)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise
            if not data:
                break
            chunks.append(data)
        returncode = process.wait()
        stdout = b"".join(chunks).decode(errors="replace")
        return subprocess.CompletedProcess(list(command), returncode, stdout=stdout, stderr="")
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
        if slave_fd >= 0:
            os.close(slave_fd)
        os.close(master_fd)


def current_git_head(repo_root: Path = REPO_ROOT) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    head = result.stdout.strip()
    if not head:
        raise RuntimeError("git rev-parse HEAD returned an empty commit")
    return head


def checkpoint_s3_base(ckpt_run_name: str, *, runs_prefix: str = DEFAULT_CKPT_S3_RUNS_PREFIX) -> str:
    prefix = runs_prefix.rstrip("/")
    return f"{prefix}/{ckpt_run_name}/checkpoints/{ckpt_run_name}"


def checkpoint_amlfs_base(
    ckpt_run_name: str, *, runs_prefix: str = DEFAULT_CKPT_AMLFS_RUNS_PREFIX
) -> str:
    prefix = runs_prefix.rstrip("/")
    return f"{prefix}/{ckpt_run_name}/{ckpt_run_name}"


def checkpoint_s3_uri(ckpt_s3_base_value: str, step: int) -> str:
    return f"{ckpt_s3_base_value.rstrip('/')}/checkpoint-{step}/"


def checkpoint_steps(start_step: int, max_step: int, interval: int) -> list[int]:
    if interval <= 0:
        raise ValueError("interval must be positive")
    if start_step <= 0:
        raise ValueError("start_step must be positive")
    if max_step < start_step:
        raise ValueError("max_step must be greater than or equal to start_step")
    return list(range(start_step, max_step + 1, interval))


def parse_steps(value: str) -> list[int]:
    parts = [part.strip() for part in value.replace(",", " ").split()]
    if not parts:
        raise argparse.ArgumentTypeError("--steps must contain at least one checkpoint step")
    steps: list[int] = []
    for part in parts:
        try:
            step = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid checkpoint step: {part!r}") from exc
        if step <= 0:
            raise argparse.ArgumentTypeError("checkpoint steps must be positive")
        steps.append(step)
    return steps


def off_grid_steps(steps: Sequence[int], *, start_step: int, max_step: int, interval: int) -> list[int]:
    if interval <= 0:
        raise ValueError("interval must be positive")
    return [
        step
        for step in steps
        if step < start_step or step > max_step or (step - start_step) % interval != 0
    ]


def listing_has_ready_checkpoint(listing: str) -> bool:
    has_model = any(marker in listing for marker in MODEL_MARKERS)
    if ".cache_complete" in listing and has_model:
        return True
    has_trainer_state = "trainer_state.json" in listing
    has_experiment_cfg = "experiment_cfg/conf.yaml" in listing or "experiment_cfg" in listing
    return has_model and has_trainer_state and has_experiment_cfg


def check_checkpoint_ready(*, osmo_binary: str, ckpt_s3_base_value: str, step: int) -> ReadyCheck:
    uri = checkpoint_s3_uri(ckpt_s3_base_value, step)
    result = subprocess.run(
        [osmo_binary, "data", "list", "--no-pager", uri],
        check=False,
        capture_output=True,
        text=True,
    )
    listing = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0:
        reason = " ".join(line.strip() for line in listing.splitlines() if line.strip())
        if len(reason) > 240:
            reason = reason[:237] + "..."
        return ReadyCheck(
            step=step,
            uri=uri,
            ready=False,
            reason=f"osmo data list failed with status {result.returncode}: {reason}",
        )
    if listing_has_ready_checkpoint(listing):
        return ReadyCheck(step=step, uri=uri, ready=True, reason="ready")
    return ReadyCheck(
        step=step,
        uri=uri,
        ready=False,
        reason="missing model/trainer_state/experiment_cfg ready markers",
    )


def check_checkpoint_ready_from_train_workflow(
    *,
    osmo_binary: str,
    workflow_name: str,
    task_name: str,
    output_dir: str,
    step: int,
) -> ReadyCheck:
    ckpt_dir = f"{output_dir.rstrip('/')}/checkpoint-{step}"
    script = f"""
set -euo pipefail
ckpt_dir={shlex.quote(ckpt_dir)}
if [ ! -d "$ckpt_dir" ]; then
  echo "missing checkpoint directory: $ckpt_dir"
  exit 3
fi
has_model=0
if [ -f "$ckpt_dir/model.safetensors" ] || [ -f "$ckpt_dir/model.safetensors.index.json" ]; then
  has_model=1
fi
if [ "$has_model" -eq 1 ] && [ -f "$ckpt_dir/trainer_state.json" ] && [ -f "$ckpt_dir/experiment_cfg/conf.yaml" ]; then
  echo "ready: $ckpt_dir"
  exit 0
fi
echo "incomplete checkpoint: $ckpt_dir"
find "$ckpt_dir" -maxdepth 2 -type f | sed "s#^$ckpt_dir/##" | sort | head -80
exit 4
""".strip()
    result = run_command_with_pty(
        [
            osmo_binary,
            "workflow",
            "exec",
            workflow_name,
            task_name,
            "--entry",
            f"/bin/bash -lc {shlex.quote(script)}",
        ]
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    reason = " ".join(lines)
    if len(reason) > 240:
        reason = reason[:237] + "..."
    ready_marker = f"ready: {ckpt_dir}"
    if any(line == ready_marker for line in lines):
        return ReadyCheck(step=step, uri=ckpt_dir, ready=True, reason="ready")
    if result.returncode == 0:
        reason = (
            f"osmo workflow exec returned status 0 without ready marker: {reason}"
            if reason
            else "osmo workflow exec returned status 0 without ready marker"
        )
    return ReadyCheck(
        step=step,
        uri=ckpt_dir,
        ready=False,
        reason=reason or f"osmo workflow exec failed with status {result.returncode}",
    )


def workflow_name_for_step(*, name_template: str, step: int, tag: str) -> str:
    return name_template.format(step=step, tag=tag)


def _query_workflow_exists(*, osmo_binary: str, workflow_name: str) -> ExistingWorkflowCheck:
    result = subprocess.run(
        [osmo_binary, "workflow", "query", workflow_name, "--format-type", "json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return ExistingWorkflowCheck(name=workflow_name, exists=True, reason="exists")
    reason = result.stderr.strip() or result.stdout.strip() or f"query status {result.returncode}"
    return ExistingWorkflowCheck(name=workflow_name, exists=False, reason=reason)


def check_workflow_exists(
    *,
    osmo_binary: str,
    workflow_name: str,
    suffix_scan_limit: int = 20,
) -> ExistingWorkflowCheck:
    if suffix_scan_limit < 0:
        raise ValueError("suffix_scan_limit must be non-negative")
    exact = _query_workflow_exists(osmo_binary=osmo_binary, workflow_name=workflow_name)
    if exact.exists:
        return exact
    for suffix in range(1, suffix_scan_limit + 1):
        candidate = f"{workflow_name}-{suffix}"
        check = _query_workflow_exists(osmo_binary=osmo_binary, workflow_name=candidate)
        if check.exists:
            return ExistingWorkflowCheck(
                name=check.name,
                exists=True,
                reason=f"exists as {check.name}",
            )
    return exact


def build_submit_command(
    *,
    workflow: Path,
    pool: str,
    priority: str,
    step: int,
    tag: str,
    name_template: str,
    local_root_template: str,
    ckpt_run_name: str,
    ckpt_s3_base_value: str,
    ckpt_amlfs_base_value: str,
    dreamzero_git_ref: str = DEFAULT_DREAMZERO_GIT_REF,
    dreamzero_expected_git_commit: str = "",
    eval_num_frames: int | None = None,
    eval_action_horizon: int | None = None,
    extra_set_string: Sequence[str] = (),
    osmo_binary: str = "osmo",
) -> list[str]:
    workflow_name = name_template.format(step=step, tag=tag)
    local_root = local_root_template.format(step=step, tag=tag)
    set_string = [
        f"workflow_name={workflow_name}",
        f"run_name={workflow_name}",
        f"ckpt_run_name={ckpt_run_name}",
        f"ckpt_s3_base={ckpt_s3_base_value}",
        f"ckpt_amlfs_base={ckpt_amlfs_base_value}",
        f"ckpt_setting=checkpoint-{step}",
        f"local_eval_ckpt_root={local_root}",
        f"dreamzero_git_ref={dreamzero_git_ref}",
        f"dreamzero_expected_git_commit={dreamzero_expected_git_commit}",
    ]
    if eval_num_frames is not None:
        set_string.append(f"eval_num_frames={eval_num_frames}")
    if eval_action_horizon is not None:
        set_string.append(f"eval_action_horizon={eval_action_horizon}")
    set_string.extend(extra_set_string)
    return [
        osmo_binary,
        "workflow",
        "submit",
        str(workflow),
        "--pool",
        pool,
        "--priority",
        priority,
        "--set-string",
        *set_string,
    ]


def shell_quote(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def option_was_provided(argv: Sequence[str], option: str) -> bool:
    return any(part == option or part.startswith(option + "=") for part in argv)


def set_string_key(value: str) -> str:
    return value.split("=", 1)[0]


def add_set_string_defaults(set_strings: list[str], defaults: Sequence[tuple[str, str]]) -> None:
    existing_keys = {set_string_key(value) for value in set_strings}
    for key, value in defaults:
        if key not in existing_keys:
            set_strings.append(f"{key}={value}")
            existing_keys.add(key)


def apply_fullft_gate_preset(
    args: argparse.Namespace,
    raw_argv: Sequence[str],
    *,
    default_steps: Sequence[int] = FULLFT_GATE_STEPS,
) -> None:
    if not option_was_provided(raw_argv, "--steps"):
        args.steps = list(default_steps)
    if not option_was_provided(raw_argv, "--name-template"):
        args.name_template = FULLFT_GATE_NAME_TEMPLATE
    if not option_was_provided(raw_argv, "--local-root-template"):
        args.local_root_template = FULLFT_GATE_LOCAL_ROOT_TEMPLATE
    if not option_was_provided(raw_argv, "--ckpt-run-name"):
        args.ckpt_run_name = FULLFT_GATE_CKPT_RUN_NAME_TEMPLATE.format(tag=args.tag)
    add_set_string_defaults(args.set_string, FULLFT_GATE_SET_STRING_DEFAULTS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one OSMO eval submission per checkpoint on a fixed grid. "
            "The default grid is checkpoint-2000, checkpoint-4000, ..., checkpoint-50000."
        )
    )
    parser.add_argument("--workflow", type=Path, default=Path(DEFAULT_WORKFLOW))
    parser.add_argument("--pool", default=DEFAULT_POOL)
    parser.add_argument("--priority", default="LOW", choices=("HIGH", "NORMAL", "LOW"))
    parser.add_argument(
        "--preset",
        choices=("fullft-gate", "fullft-2k-gate"),
        help=(
            "Apply a named eval recipe. fullft-gate evaluates checkpoints "
            "10000/20000/30000 with 10 episodes, scale=1.0, action rollout only, "
            "and raw/smooth replan=24/12 settings. fullft-2k-gate applies the "
            "same metric to every 2k checkpoint from 2000 through 30000."
        ),
    )
    parser.add_argument("--start-step", type=int, default=2000)
    parser.add_argument("--max-step", type=int, default=50000)
    parser.add_argument("--interval", type=int, default=2000)
    parser.add_argument(
        "--steps",
        type=parse_steps,
        help=(
            "Explicit checkpoint steps, e.g. '2000,4000,6000'. Overrides the grid, "
            "but still must land on the 2k cadence unless --allow-off-grid-steps is set."
        ),
    )
    parser.add_argument(
        "--allow-off-grid-steps",
        action="store_true",
        help="Permit explicit --steps outside the configured start/max/interval grid for one-off diagnostics.",
    )
    parser.add_argument(
        "--tag",
        default=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d"),
        help="Suffix used in workflow/run names.",
    )
    parser.add_argument("--name-template", default=DEFAULT_NAME_TEMPLATE)
    parser.add_argument("--local-root-template", default=DEFAULT_LOCAL_ROOT_TEMPLATE)
    parser.add_argument(
        "--ckpt-run-name",
        default=DEFAULT_CKPT_RUN_NAME,
        help="Training stage run name that owns the checkpoint-* directories.",
    )
    parser.add_argument(
        "--ckpt-s3-base",
        help=(
            "Full S3 checkpoint base. Defaults to "
            f"{DEFAULT_CKPT_S3_RUNS_PREFIX}/<ckpt-run-name>/checkpoints/<ckpt-run-name>."
        ),
    )
    parser.add_argument(
        "--ckpt-amlfs-base",
        help=(
            "Full AMLFS checkpoint cache base. Defaults to "
            f"{DEFAULT_CKPT_AMLFS_RUNS_PREFIX}/<ckpt-run-name>/<ckpt-run-name>."
        ),
    )
    parser.add_argument(
        "--dreamzero-git-ref",
        default=DEFAULT_DREAMZERO_GIT_REF,
        help="DreamZero git ref cloned by the eval workflow. Defaults to gamma.",
    )
    parser.add_argument(
        "--dreamzero-expected-git-commit",
        help=(
            "Expected DreamZero commit after resolving --dreamzero-git-ref. "
            "Defaults to the current local HEAD; pass an empty string to disable the guard."
        ),
    )
    parser.add_argument(
        "--eval-num-frames",
        type=int,
        help="Override the eval server --num-frames template value for long-window ablations.",
    )
    parser.add_argument(
        "--eval-action-horizon",
        type=int,
        help="Override the eval server --action-horizon template value for action-window ablations.",
    )
    parser.add_argument(
        "--close-timing-diagnostic",
        action="store_true",
        help=(
            "Add a gripper-close timing sweep. This appends gripper_close_pairs='33:33 52:52' "
            "unless gripper_close_pairs is already provided through --set-string."
        ),
    )
    parser.add_argument(
        "--set-string",
        action="append",
        default=[],
        help="Additional OSMO --set-string key=value override. May be repeated.",
    )
    parser.add_argument("--osmo-binary", default="osmo")
    parser.add_argument(
        "--only-ready",
        action="store_true",
        help="Only emit/submit checkpoints whose S3 directory already contains complete checkpoint markers.",
    )
    parser.add_argument(
        "--ready-check-source",
        default="s3",
        choices=("s3", "train-workflow"),
        help=(
            "Where --only-ready checks checkpoint completeness. Use train-workflow for an active "
            "training run when local S3 credentials are unavailable."
        ),
    )
    parser.add_argument(
        "--ready-train-workflow",
        help="Training workflow name used when --ready-check-source=train-workflow.",
    )
    parser.add_argument(
        "--ready-train-task",
        default=DEFAULT_READY_TRAIN_TASK,
        help="Training task name used when --ready-check-source=train-workflow.",
    )
    parser.add_argument(
        "--ready-train-output-dir",
        default=DEFAULT_READY_TRAIN_OUTPUT_DIR,
        help="Checkpoint output directory inside the active training task.",
    )
    parser.add_argument(
        "--fail-if-none-ready",
        action="store_true",
        help="With --only-ready, return exit code 2 when no requested checkpoint is ready.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip steps whose generated OSMO workflow name already exists.",
    )
    parser.add_argument(
        "--existing-suffix-scan-limit",
        type=int,
        default=20,
        help=(
            "With --skip-existing, also query OSMO auto-suffixed names "
            "<workflow>-1..<workflow>-N. Defaults to 20."
        ),
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit the generated workflows. Without this flag, commands are printed only.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    if args.preset == "fullft-gate":
        apply_fullft_gate_preset(args, raw_argv)
    elif args.preset == "fullft-2k-gate":
        apply_fullft_gate_preset(args, raw_argv, default_steps=FULLFT_2K_GATE_STEPS)
    if args.close_timing_diagnostic:
        add_set_string_defaults(args.set_string, CLOSE_TIMING_DIAGNOSTIC_SET_STRING_DEFAULTS)
    steps = args.steps or checkpoint_steps(args.start_step, args.max_step, args.interval)
    ckpt_s3_base_value = args.ckpt_s3_base or checkpoint_s3_base(args.ckpt_run_name)
    ckpt_amlfs_base_value = args.ckpt_amlfs_base or checkpoint_amlfs_base(args.ckpt_run_name)
    dreamzero_expected_git_commit = (
        current_git_head()
        if args.dreamzero_expected_git_commit is None
        else args.dreamzero_expected_git_commit
    )
    if args.steps and not args.allow_off_grid_steps:
        bad_steps = off_grid_steps(
            steps,
            start_step=args.start_step,
            max_step=args.max_step,
            interval=args.interval,
        )
        if bad_steps:
            parser.error(
                "explicit --steps must stay on the configured eval cadence "
                f"{args.start_step}..{args.max_step} every {args.interval}; "
                f"off-grid steps: {','.join(str(step) for step in bad_steps)}. "
                "Use --allow-off-grid-steps only for one-off diagnostics."
            )
    if args.only_ready:
        if args.ready_check_source == "train-workflow" and not args.ready_train_workflow:
            parser.error("--ready-train-workflow is required with --ready-check-source=train-workflow")
        ready_steps: list[int] = []
        for step in steps:
            if args.ready_check_source == "train-workflow":
                check = check_checkpoint_ready_from_train_workflow(
                    osmo_binary=args.osmo_binary,
                    workflow_name=args.ready_train_workflow,
                    task_name=args.ready_train_task,
                    output_dir=args.ready_train_output_dir,
                    step=step,
                )
            else:
                check = check_checkpoint_ready(
                    osmo_binary=args.osmo_binary,
                    ckpt_s3_base_value=ckpt_s3_base_value,
                    step=step,
                )
            if check.ready:
                print(f"READY checkpoint-{step}: {check.uri}", file=sys.stderr, flush=True)
                ready_steps.append(step)
            else:
                print(
                    f"SKIP checkpoint-{step}: {check.reason} at {check.uri}",
                    file=sys.stderr,
                    flush=True,
                )
        steps = ready_steps
        if not steps:
            print("No ready checkpoints matched the requested eval grid.", file=sys.stderr, flush=True)
            if args.fail_if_none_ready:
                return 2
    if args.skip_existing and steps:
        new_steps: list[int] = []
        for step in steps:
            workflow_name = workflow_name_for_step(
                name_template=args.name_template,
                step=step,
                tag=args.tag,
            )
            check = check_workflow_exists(
                osmo_binary=args.osmo_binary,
                workflow_name=workflow_name,
                suffix_scan_limit=args.existing_suffix_scan_limit,
            )
            if check.exists:
                print(
                    f"SKIP checkpoint-{step}: workflow already exists: {check.name}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"NEW checkpoint-{step}: no existing workflow named {workflow_name}",
                    file=sys.stderr,
                    flush=True,
                )
                new_steps.append(step)
        steps = new_steps
        if not steps:
            print("No new checkpoint eval workflows matched the requested grid.", file=sys.stderr, flush=True)
    commands = [
        build_submit_command(
            workflow=args.workflow,
            pool=args.pool,
            priority=args.priority,
            step=step,
            tag=args.tag,
            name_template=args.name_template,
            local_root_template=args.local_root_template,
            ckpt_run_name=args.ckpt_run_name,
            ckpt_s3_base_value=ckpt_s3_base_value,
            ckpt_amlfs_base_value=ckpt_amlfs_base_value,
            dreamzero_git_ref=args.dreamzero_git_ref,
            dreamzero_expected_git_commit=dreamzero_expected_git_commit,
            eval_num_frames=args.eval_num_frames,
            eval_action_horizon=args.eval_action_horizon,
            extra_set_string=args.set_string,
            osmo_binary=args.osmo_binary,
        )
        for step in steps
    ]

    for command in commands:
        print(shell_quote(command), flush=True)
        if args.submit:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
