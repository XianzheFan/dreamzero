#!/usr/bin/env python3
"""Generate or submit 2k-spaced RoboFactory droidwidth teacher eval jobs."""

from __future__ import annotations

import argparse
import datetime as dt
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path


DEFAULT_WORKFLOW = (
    "osmo_workflows/robofactory/"
    "closedloop_liftbarrier_gamma_droidwidth_teacher_c2000_slim_eval_h100_1seed_20260621.yaml"
)
DEFAULT_POOL = "groot-h100-01"
DEFAULT_NAME_TEMPLATE = (
    "dz-rf-sg-gamma-dwteacher-bidir-50k-c{step}-slim-eval-h100-1seed1000-xz-{tag}"
)
DEFAULT_LOCAL_ROOT_TEMPLATE = (
    "gamma_droidwidth_teacher_bidir_50k_c{step}_slim_eval_h100_1seed1000"
)
DEFAULT_CKPT_RUN_NAME = "dz-rf-sg-gamma-dwteacher-bidir-lb500-50k-xz-20260622-teacher"
DEFAULT_CKPT_S3_RUNS_PREFIX = "s3://GearHome/users/xianzhef/oci-migration/dreamzero_runs"
DEFAULT_CKPT_AMLFS_RUNS_PREFIX = "/mnt/amlfs-01/home/xianzhef/osmo_cache/dreamzero/checkpoints"


def checkpoint_s3_base(ckpt_run_name: str, *, runs_prefix: str = DEFAULT_CKPT_S3_RUNS_PREFIX) -> str:
    prefix = runs_prefix.rstrip("/")
    return f"{prefix}/{ckpt_run_name}/checkpoints/{ckpt_run_name}"


def checkpoint_amlfs_base(
    ckpt_run_name: str, *, runs_prefix: str = DEFAULT_CKPT_AMLFS_RUNS_PREFIX
) -> str:
    prefix = runs_prefix.rstrip("/")
    return f"{prefix}/{ckpt_run_name}/{ckpt_run_name}"


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
        "--set-string",
        action="append",
        default=[],
        help="Additional OSMO --set-string key=value override. May be repeated.",
    )
    parser.add_argument("--osmo-binary", default="osmo")
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
