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
DEFAULT_POOL = "groot-h100-02"
DEFAULT_NAME_TEMPLATE = (
    "dz-rf-sg-gamma-dwteacher-50kfrom0-c{step}-slim-eval-h100-1seed1000-xz-{tag}"
)
DEFAULT_LOCAL_ROOT_TEMPLATE = (
    "gamma_droidwidth_teacher_50kfrom0_c{step}_slim_eval_h100_1seed1000"
)


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


def build_submit_command(
    *,
    workflow: Path,
    pool: str,
    priority: str,
    step: int,
    tag: str,
    name_template: str,
    local_root_template: str,
    extra_set_string: Sequence[str] = (),
    osmo_binary: str = "osmo",
) -> list[str]:
    workflow_name = name_template.format(step=step, tag=tag)
    local_root = local_root_template.format(step=step, tag=tag)
    set_string = [
        f"workflow_name={workflow_name}",
        f"run_name={workflow_name}",
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
        help="Explicit checkpoint steps, e.g. '2000,4000,6000'. Overrides the grid.",
    )
    parser.add_argument(
        "--tag",
        default=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d"),
        help="Suffix used in workflow/run names.",
    )
    parser.add_argument("--name-template", default=DEFAULT_NAME_TEMPLATE)
    parser.add_argument("--local-root-template", default=DEFAULT_LOCAL_ROOT_TEMPLATE)
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
    args = build_parser().parse_args(argv)
    steps = args.steps or checkpoint_steps(args.start_step, args.max_step, args.interval)
    commands = [
        build_submit_command(
            workflow=args.workflow,
            pool=args.pool,
            priority=args.priority,
            step=step,
            tag=args.tag,
            name_template=args.name_template,
            local_root_template=args.local_root_template,
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
