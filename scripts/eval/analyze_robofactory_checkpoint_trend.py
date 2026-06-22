#!/usr/bin/env python3
"""Analyze whether RoboFactory checkpoint eval metrics improve with step.

Input can be one or more eval output roots, parent directories containing eval
outputs, or ``checkpoint_grid.json`` files written by
``summarize_robofactory_checkpoint_grid.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple

from scripts.eval.summarize_robofactory_checkpoint_grid import summarize_roots


class MetricSpec(NamedTuple):
    key: str
    direction: str
    label: str


DEFAULT_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("success_rate", "higher", "success_rate"),
    MetricSpec("success_count", "higher", "success_count"),
    MetricSpec("barrier_margin_best", "higher", "barrier_margin"),
    MetricSpec("target_min_mean_left", "lower", "left_target_dist"),
    MetricSpec("target_min_mean_right", "lower", "right_target_dist"),
    MetricSpec("mean_joint_step_jerk", "lower", "exec_jerk"),
    MetricSpec("mean_joint_accel_to_delta_ratio", "lower", "exec_accel_ratio"),
    MetricSpec("joint_delta_sign_flip_frac", "lower", "exec_flip_frac"),
    MetricSpec("mean_pred_chunk_joint_step_jerk", "lower", "pred_chunk_jerk"),
    MetricSpec(
        "mean_pred_chunk_joint_accel_to_delta_ratio",
        "lower",
        "pred_chunk_accel_ratio",
    ),
    MetricSpec("pred_chunk_joint_delta_sign_flip_frac", "lower", "pred_chunk_flip"),
    MetricSpec("max_model_replan_boundary_joint_jump", "lower", "model_boundary"),
    MetricSpec("raw_joint_saturation_frac", "lower", "raw_joint_sat"),
    MetricSpec("pred_conditioning_frame_mae_rgb_mean", "lower", "cond_t0_mae"),
    MetricSpec("action_pred_vs_future_mae_rgb_mean", "lower", "action_future_mae"),
    MetricSpec(
        "action_pred_vs_future_best_alignment_mae_rgb_mean",
        "lower",
        "action_best_mae",
    ),
    MetricSpec("noncausal_pred_vs_future_mae_rgb_mean", "lower", "noncausal_future_mae"),
    MetricSpec(
        "noncausal_pred_vs_future_best_alignment_mae_rgb_mean",
        "lower",
        "noncausal_best_mae",
    ),
    MetricSpec(
        "pred_vs_future_best_alignment_offset_abs_mean",
        "lower",
        "best_offset_abs",
    ),
    MetricSpec("vwin_action_pred_vs_future_mae_rgb_mean", "lower", "vwin_action_mae"),
    MetricSpec(
        "vwin_history_current_first_pred_vs_future_mae_rgb_mean",
        "lower",
        "vwin_hcf_mae",
    ),
)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _load_grid_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        rows = payload.get("rows")
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON object with rows or a rows list")
    return [row for row in rows if isinstance(row, dict)]


def load_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    json_rows: list[dict[str, Any]] = []
    root_paths: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if path.is_file() and path.suffix.lower() == ".json":
            json_rows.extend(_load_grid_json(path))
        else:
            root_paths.append(path)
    rows = json_rows
    if root_paths:
        rows.extend(summarize_roots(root_paths))
    rows.sort(
        key=lambda row: (
            row.get("checkpoint_step") is None,
            row.get("checkpoint_step") if row.get("checkpoint_step") is not None else 0,
            str(row.get("root") or row.get("ckpt_setting") or ""),
        )
    )
    return rows


def _linear_slope_per_2k(points: list[tuple[int, float]]) -> float | None:
    if len(points) < 2:
        return None
    xs = [step / 2000.0 for step, _ in points]
    ys = [value for _, value in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0.0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom


def analyze_rows(
    rows: list[dict[str, Any]],
    *,
    metrics: Iterable[MetricSpec] = DEFAULT_METRICS,
    min_points: int = 3,
) -> dict[str, Any]:
    trend: dict[str, Any] = {
        "checkpoint_count": len(rows),
        "checkpoint_steps": [row.get("checkpoint_step") for row in rows],
        "min_points": min_points,
        "metrics": {},
    }
    for spec in metrics:
        points: list[tuple[int, float]] = []
        for row in rows:
            step = row.get("checkpoint_step")
            value = _float_or_none(row.get(spec.key))
            if isinstance(step, int) and value is not None:
                points.append((step, value))
        if not points:
            continue
        first_step, first = points[0]
        last_step, last = points[-1]
        raw_delta = last - first
        improvement_delta = raw_delta if spec.direction == "higher" else -raw_delta
        raw_slope = _linear_slope_per_2k(points)
        improvement_slope = (
            None
            if raw_slope is None
            else raw_slope
            if spec.direction == "higher"
            else -raw_slope
        )
        enough = len(points) >= min_points
        trend["metrics"][spec.key] = {
            "label": spec.label,
            "direction": spec.direction,
            "point_count": len(points),
            "first_step": first_step,
            "first_value": first,
            "last_step": last_step,
            "last_value": last,
            "raw_delta": raw_delta,
            "improvement_delta": improvement_delta,
            "raw_slope_per_2k": raw_slope,
            "improvement_slope_per_2k": improvement_slope,
            "status": (
                "insufficient"
                if not enough
                else "improving"
                if improvement_slope is not None and improvement_slope > 0.0
                else "worsening"
                if improvement_slope is not None and improvement_slope < 0.0
                else "flat"
            ),
        }
    return trend


def _fmt(value: Any) -> str:
    if value is None:
        return "nan"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def print_trend(trend: dict[str, Any]) -> None:
    print(
        "metric\tdir\tn\tfirst_step\tfirst\tlast_step\tlast\t"
        "delta\timprove_slope_per_2k\tstatus"
    )
    metrics = trend.get("metrics", {})
    for key in metrics:
        item = metrics[key]
        print(
            "\t".join(
                [
                    item["label"],
                    item["direction"],
                    str(item["point_count"]),
                    str(item["first_step"]),
                    _fmt(item["first_value"]),
                    str(item["last_step"]),
                    _fmt(item["last_value"]),
                    _fmt(item["raw_delta"]),
                    _fmt(item["improvement_slope_per_2k"]),
                    item["status"],
                ]
            )
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="checkpoint_grid.json files or eval output roots/parents.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=3,
        help="Minimum metric points needed before calling a trend improving/worsening.",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    rows = load_rows(args.inputs)
    if not rows:
        raise SystemExit("no checkpoint rows found")
    trend = analyze_rows(rows, min_points=args.min_points)
    print_trend(trend)
    if args.json_out:
        args.json_out.write_text(
            json.dumps(trend, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote trend JSON: {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
