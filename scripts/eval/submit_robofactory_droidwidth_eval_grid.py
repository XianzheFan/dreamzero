#!/usr/bin/env python3
"""Generate or submit 2k-spaced RoboFactory droidwidth teacher eval jobs."""

from __future__ import annotations

import argparse
import datetime as dt
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
    "dz-rf-sg-gamma-dwteacher-bidir-nodrop-50k-c{step}-slim-eval-h100-1seed1000-xz-{tag}"
)
DEFAULT_LOCAL_ROOT_TEMPLATE = (
    "gamma_droidwidth_teacher_bidir_nodrop_50k_c{step}_slim_eval_h100_1seed1000"
)
DEFAULT_CKPT_RUN_NAME = "dz-rf-sg-gamma-dwteacher-bidir-nodrop-lb500-50k-xz-20260622-teacher"
DEFAULT_CKPT_S3_RUNS_PREFIX = "s3://GearHome/users/xianzhef/oci-migration/dreamzero_runs"
DEFAULT_CKPT_AMLFS_RUNS_PREFIX = "/mnt/amlfs-01/home/xianzhef/osmo_cache/dreamzero/checkpoints"
DEFAULT_DREAMZERO_GIT_REF = "gamma"
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
        return ReadyCheck(
            step=step,
            uri=uri,
            ready=False,
            reason=f"osmo data list failed with status {result.returncode}",
        )
    if listing_has_ready_checkpoint(listing):
        return ReadyCheck(step=step, uri=uri, ready=True, reason="ready")
    return ReadyCheck(
        step=step,
        uri=uri,
        ready=False,
        reason="missing model/trainer_state/experiment_cfg ready markers",
    )


def workflow_name_for_step(*, name_template: str, step: int, tag: str) -> str:
    return name_template.format(step=step, tag=tag)


def check_workflow_exists(*, osmo_binary: str, workflow_name: str) -> ExistingWorkflowCheck:
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
        *extra_set_string,
    ]
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
        "--submit",
        action="store_true",
        help="Actually submit the generated workflows. Without this flag, commands are printed only.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
        ready_steps: list[int] = []
        for step in steps:
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
            check = check_workflow_exists(osmo_binary=args.osmo_binary, workflow_name=workflow_name)
            if check.exists:
                print(
                    f"SKIP checkpoint-{step}: workflow already exists: {workflow_name}",
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
