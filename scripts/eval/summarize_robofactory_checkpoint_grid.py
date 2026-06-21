"""Summarize RoboFactory eval artifacts across checkpoints.

Each closed-loop eval writes one ``eval_outputs`` tree containing:

* ``checkpoint_eval_manifest.json`` for provenance and checkpoint identity;
* ``scale_sweep_summary.json`` for physical/action diagnostics;
* ``video_pred_quality.json`` for predicted-video diagnostics.

This helper folds many eval output roots into one table so a 2k checkpoint
cadence can be compared without opening every artifact directory by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


CHECKPOINT_RE = re.compile(r"(?:checkpoint-|(?:^|[/_-])c)(\d+)\b")
SIGNAL_FILES = (
    "checkpoint_eval_manifest.json",
    "scale_sweep_summary.json",
    "video_pred_quality.json",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _checkpoint_step_from_text(text: str) -> int | None:
    match = CHECKPOINT_RE.search(text)
    if not match:
        return None
    return int(match.group(1))


def checkpoint_step(manifest: dict[str, Any], root: Path) -> int | None:
    ckpt_setting = str(manifest.get("ckpt_setting") or "")
    return (
        _checkpoint_step_from_text(ckpt_setting)
        or _checkpoint_step_from_text(str(root))
    )


def find_eval_roots(paths: Iterable[Path]) -> list[Path]:
    roots: set[Path] = set()
    for path in paths:
        path = path.expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            if path.name in SIGNAL_FILES:
                roots.add(path.parent.resolve())
            continue
        if any((path / name).is_file() for name in SIGNAL_FILES):
            roots.add(path.resolve())
        for signal in SIGNAL_FILES:
            for found in path.rglob(signal):
                roots.add(found.parent.resolve())
    return sorted(roots)


def _setting_score(row: dict[str, Any]) -> tuple[Any, ...]:
    success_count = _int_or_none(row.get("success_count")) or 0
    success_rate = _float_or_none(row.get("success_rate")) or 0.0
    margin = _float_or_none(row.get("barrier_margin_best"))
    target_left = _float_or_none(row.get("target_min_mean_left"))
    target_right = _float_or_none(row.get("target_min_mean_right"))
    target_mean = (
        (target_left + target_right) / 2.0
        if target_left is not None and target_right is not None
        else math.inf
    )
    raw_sat = _float_or_none(row.get("raw_joint_saturation_frac"))
    boundary_jump = _float_or_none(row.get("max_replan_boundary_joint_jump"))
    return (
        success_count,
        success_rate,
        margin if margin is not None else -math.inf,
        -target_mean,
        -(raw_sat if raw_sat is not None else math.inf),
        -(boundary_jump if boundary_jump is not None else math.inf),
    )


def best_sweep_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=_setting_score)


def _video_summary(root: Path) -> dict[str, Any]:
    path = root / "video_pred_quality.json"
    if not path.is_file():
        return {}
    payload = _load_json(path)
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return {}
    return summary


def _sweep_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "scale_sweep_summary.json"
    if not path.is_file():
        return []
    payload = _load_json(path)
    rows = payload.get("rows")
    return rows if isinstance(rows, list) else []


def summarize_eval_root(root: Path) -> dict[str, Any]:
    manifest_path = root / "checkpoint_eval_manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
    rows = _sweep_rows(root)
    best = best_sweep_row([row for row in rows if isinstance(row, dict)]) or {}
    video = _video_summary(root)
    step = checkpoint_step(manifest, root)
    return {
        "root": str(root),
        "checkpoint_step": step,
        "ckpt_setting": manifest.get("ckpt_setting")
        or (f"checkpoint-{step}" if step else ""),
        "eval_status": manifest.get("eval_status", ""),
        "run_name": manifest.get("run_name", ""),
        "dreamzero_git_commit": manifest.get("dreamzero_git_commit", ""),
        "checkpoint_code_commit": manifest.get("checkpoint_code_commit", ""),
        "checkpoint_expected_code_commit": manifest.get(
            "checkpoint_expected_code_commit", ""
        ),
        "checkpoint_stage_label": manifest.get("checkpoint_stage_label", ""),
        "checkpoint_action_dim": manifest.get("checkpoint_action_dim", ""),
        "checkpoint_diffusion_action_dim": manifest.get(
            "checkpoint_diffusion_action_dim", ""
        ),
        "checkpoint_num_agents": manifest.get("checkpoint_num_agents", ""),
        "checkpoint_agent_dim": manifest.get("checkpoint_agent_dim", ""),
        "checkpoint_global_video_attention_mode": manifest.get(
            "checkpoint_global_video_attention_mode", ""
        ),
        "checkpoint_global_video_timestep_mode": manifest.get(
            "checkpoint_global_video_timestep_mode", ""
        ),
        "checkpoint_use_sparse_hub_attention": manifest.get(
            "checkpoint_use_sparse_hub_attention", ""
        ),
        "checkpoint_train_architecture": manifest.get(
            "checkpoint_train_architecture", ""
        ),
        "video_pred_rollout_mode_counts": video.get("video_pred_rollout_mode_counts"),
        "video_pred_wrist_window_mode_counts": video.get("video_pred_wrist_window_mode_counts"),
        "best_setting_dir": best.get("setting_dir"),
        "best_replan_every": best.get("replan_every"),
        "best_action_representation": best.get("action_representation"),
        "best_scale": best.get("scale"),
        "best_slew": best.get("target_slew_rate"),
        "best_accel_limit": best.get("target_accel_limit"),
        "best_blend": best.get("replan_boundary_blend_steps"),
        "best_ensemble": best.get("temporal_action_ensemble_decay"),
        "success_count": best.get("success_count"),
        "success_rate": best.get("success_rate"),
        "target_min_mean_left": best.get("target_min_mean_left"),
        "target_min_mean_right": best.get("target_min_mean_right"),
        "barrier_margin_best": best.get("barrier_margin_best"),
        "left_grasp_episodes": best.get("left_grasp_episodes"),
        "right_grasp_episodes": best.get("right_grasp_episodes"),
        "mean_joint_step_delta": best.get("mean_joint_step_delta"),
        "max_joint_step_delta": best.get("max_joint_step_delta"),
        "mean_joint_step_accel": best.get("mean_joint_step_accel"),
        "mean_joint_accel_to_delta_ratio": best.get(
            "mean_joint_accel_to_delta_ratio"
        ),
        "max_joint_accel_to_delta_ratio": best.get(
            "max_joint_accel_to_delta_ratio"
        ),
        "joint_delta_sign_flip_frac": best.get("joint_delta_sign_flip_frac"),
        "mean_joint_delta_sign_flip_frac": best.get(
            "mean_joint_delta_sign_flip_frac"
        ),
        "mean_pred_chunk_joint_step_delta": best.get(
            "mean_pred_chunk_joint_step_delta"
        ),
        "mean_pred_chunk_joint_step_accel": best.get(
            "mean_pred_chunk_joint_step_accel"
        ),
        "mean_pred_chunk_joint_accel_to_delta_ratio": best.get(
            "mean_pred_chunk_joint_accel_to_delta_ratio"
        ),
        "pred_chunk_joint_delta_sign_flip_frac": best.get(
            "pred_chunk_joint_delta_sign_flip_frac"
        ),
        "max_model_replan_boundary_joint_jump": best.get(
            "max_model_replan_boundary_joint_jump"
        ),
        "mean_accel_limiter_correction_joint": best.get(
            "mean_accel_limiter_correction_joint"
        ),
        "max_replan_boundary_joint_jump": best.get("max_replan_boundary_joint_jump"),
        "raw_joint_saturation_frac": best.get("raw_joint_saturation_frac"),
        "raw_joint_saturation_frac_left": best.get("raw_joint_saturation_frac_left"),
        "raw_joint_saturation_frac_right": best.get("raw_joint_saturation_frac_right"),
        "raw_joint_saturation_frac_first_step": best.get("raw_joint_saturation_frac_first_step"),
        "raw_joint_saturation_frac_late_steps": best.get("raw_joint_saturation_frac_late_steps"),
        "raw_gripper_saturation_frac": best.get("raw_gripper_saturation_frac"),
        "joint_clamp_delta_mean": best.get("joint_clamp_delta_mean"),
        "joint_clamp_delta_max": best.get("joint_clamp_delta_max"),
        "pred_vs_future_mae_rgb_mean": video.get("pred_vs_future_mae_rgb_mean"),
        "pred_conditioning_frame_mae_rgb_mean": video.get(
            "pred_conditioning_frame_mae_rgb_mean"
        ),
        "pred_conditioning_frame_available_count": video.get(
            "pred_conditioning_frame_available_count"
        ),
        "pred_vs_future_first_frame_mae_rgb_mean": video.get(
            "pred_vs_future_first_frame_mae_rgb_mean"
        ),
        "pred_vs_future_last_frame_mae_rgb_mean": video.get(
            "pred_vs_future_last_frame_mae_rgb_mean"
        ),
        "pred_vs_future_mae_rgb_first_to_last_delta_mean": video.get(
            "pred_vs_future_mae_rgb_first_to_last_delta_mean"
        ),
        "pred_vs_future_best_alignment_offset_counts": video.get(
            "pred_vs_future_best_alignment_offset_counts"
        ),
        "pred_vs_future_best_alignment_offset_abs_mean": video.get(
            "pred_vs_future_best_alignment_offset_abs_mean"
        ),
        "pred_vs_future_best_alignment_mae_rgb_mean": video.get(
            "pred_vs_future_best_alignment_mae_rgb_mean"
        ),
        "pred_vs_future_best_alignment_improvement_rgb_mean": video.get(
            "pred_vs_future_best_alignment_improvement_rgb_mean"
        ),
        "temporal_absdiff_mean": video.get("temporal_absdiff_mean"),
        "temporal_freeze_frac_mean": video.get("temporal_freeze_frac_mean"),
        "laplacian_var_mean": video.get("laplacian_var_mean"),
        "sweep_rows": len(rows),
    }


def summarize_roots(roots: Iterable[Path]) -> list[dict[str, Any]]:
    rows = [summarize_eval_root(root) for root in find_eval_roots(roots)]
    rows.sort(
        key=lambda row: (
            row["checkpoint_step"] is None,
            (
                row["checkpoint_step"]
                if row["checkpoint_step"] is not None
                else str(row["root"])
            ),
            str(row["root"]),
        )
    )
    return rows


TABLE_COLUMNS = [
    ("ckpt", "ckpt_setting"),
    ("status", "eval_status"),
    ("ckpt_commit", "checkpoint_code_commit"),
    ("stage", "checkpoint_stage_label"),
    ("attn", "checkpoint_global_video_attention_mode"),
    ("actdim", "checkpoint_action_dim"),
    ("succ", "success_count"),
    ("rate", "success_rate"),
    ("setting", "best_setting_dir"),
    ("replan", "best_replan_every"),
    ("actrep", "best_action_representation"),
    ("scale", "best_scale"),
    ("blend", "best_blend"),
    ("ens", "best_ensemble"),
    ("accel_lim", "best_accel_limit"),
    ("L_target", "target_min_mean_left"),
    ("R_target", "target_min_mean_right"),
    ("margin", "barrier_margin_best"),
    ("L_grasp", "left_grasp_episodes"),
    ("R_grasp", "right_grasp_episodes"),
    ("joint_mean", "mean_joint_step_delta"),
    ("joint_max", "max_joint_step_delta"),
    ("accel_ratio", "mean_joint_accel_to_delta_ratio"),
    ("pred_accel", "mean_pred_chunk_joint_accel_to_delta_ratio"),
    ("flip_frac", "joint_delta_sign_flip_frac"),
    ("pred_flip", "pred_chunk_joint_delta_sign_flip_frac"),
    ("accel_corr", "mean_accel_limiter_correction_joint"),
    ("boundary", "max_replan_boundary_joint_jump"),
    ("model_boundary", "max_model_replan_boundary_joint_jump"),
    ("raw_sat", "raw_joint_saturation_frac"),
    ("raw_sat_L", "raw_joint_saturation_frac_left"),
    ("raw_sat_R", "raw_joint_saturation_frac_right"),
    ("clamp", "joint_clamp_delta_mean"),
    ("cond_t0", "pred_conditioning_frame_mae_rgb_mean"),
    ("future_mae", "pred_vs_future_mae_rgb_mean"),
    ("future_t0", "pred_vs_future_first_frame_mae_rgb_mean"),
    ("future_tlast", "pred_vs_future_last_frame_mae_rgb_mean"),
    ("future_drift", "pred_vs_future_mae_rgb_first_to_last_delta_mean"),
    ("future_best_off", "pred_vs_future_best_alignment_offset_counts"),
    ("future_best_mae", "pred_vs_future_best_alignment_mae_rgb_mean"),
    ("future_best_gain", "pred_vs_future_best_alignment_improvement_rgb_mean"),
    ("tempdiff", "temporal_absdiff_mean"),
]


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "nan"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def print_table(rows: list[dict[str, Any]]) -> None:
    print("\t".join(label for label, _ in TABLE_COLUMNS))
    for row in rows:
        print("\t".join(_fmt(row.get(key)) for _, key in TABLE_COLUMNS))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else [key for _, key in TABLE_COLUMNS]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="Eval output roots or parent directories containing eval outputs.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args(argv)

    rows = summarize_roots(args.roots)
    if not rows:
        raise SystemExit("no eval outputs found")
    print_table(rows)
    if args.json_out:
        args.json_out.write_text(
            json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote JSON summary: {args.json_out}", file=sys.stderr)
    if args.csv_out:
        write_csv(rows, args.csv_out)
        print(f"wrote CSV summary: {args.csv_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
