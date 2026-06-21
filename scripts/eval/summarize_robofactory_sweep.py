"""Summarize RoboFactory closed-loop sweep directories.

The clipped joint-scale diagnostic writes one directory per setting, e.g.
``jscale_1p25/results.json`` plus the output of
``analyze_gripper_dump.py --json-out action_dump_summary.json``. This helper
turns those per-setting files into one compact table focused on the physical
failure signals: TCP distance to grasp targets, grasp counts, lift margin,
joint-step size, and smoothing diagnostics.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Any


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _nested(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _mean_finite(values: list[Any]) -> float | None:
    floats = [
        value
        for value in (_float_or_none(v) for v in values)
        if value is not None
    ]
    if not floats:
        return None
    return sum(floats) / len(floats)


def _max_finite(values: list[Any]) -> float | None:
    floats = [
        value
        for value in (_float_or_none(v) for v in values)
        if value is not None
    ]
    if not floats:
        return None
    return max(floats)


def _setting_from_results(results: dict[str, Any], setting_dir: str) -> dict[str, Any]:
    cfg = results.get("eval_config", {})
    if not isinstance(cfg, dict):
        cfg = {}
    scale = cfg.get("joint_target_scale", cfg.get("joint_delta_scale"))
    reference = cfg.get(
        "joint_target_scale_reference",
        cfg.get("joint_delta_scale_reference"),
    )
    clip = cfg.get("joint_target_scale_clip", cfg.get("joint_delta_scale_clip"))
    boundary_blend_steps = cfg.get("replan_boundary_blend_steps")
    temporal_ensemble_decay = cfg.get("temporal_action_ensemble_decay")
    return {
        "setting_dir": os.path.basename(setting_dir),
        "replan_every": _float_or_none(cfg.get("replan_every")),
        "scale": _float_or_none(scale),
        "scale_reference": reference,
        "scale_clip": _float_or_none(clip),
        "target_slew_rate": _float_or_none(cfg.get("joint_target_slew_rate")),
        "target_accel_limit": _float_or_none(cfg.get("joint_target_accel_limit")),
        "replan_boundary_blend_steps": _float_or_none(boundary_blend_steps),
        "temporal_action_ensemble_decay": _float_or_none(temporal_ensemble_decay),
    }


def _joint_debug_stat(
    episodes: list[dict[str, Any]],
    debug_key: str,
    stat_key: str,
    reducer: str,
) -> float | None:
    values = [
        _nested(episode, "joint_debug", debug_key, stat_key)
        for episode in episodes
        if isinstance(episode, dict)
    ]
    if reducer == "mean":
        return _mean_finite(values)
    if reducer == "max":
        return _max_finite(values)
    raise ValueError(f"unsupported reducer: {reducer}")


def _norm_debug_nested_stat(
    norm_debugs: list[Any],
    debug_key: str,
    nested_key: str,
    reducer: str,
) -> float | None:
    values = []
    for debug in norm_debugs:
        if not isinstance(debug, dict):
            continue
        nested = debug.get(debug_key)
        if isinstance(nested, dict):
            values.append(nested.get(nested_key))
    if reducer == "mean":
        return _mean_finite(values)
    if reducer == "max":
        return _max_finite(values)
    raise ValueError(f"unsupported reducer: {reducer}")


def summarize_setting(setting_dir: str) -> dict[str, Any]:
    results_path = os.path.join(setting_dir, "results.json")
    dump_summary_path = os.path.join(setting_dir, "action_dump_summary.json")
    if not os.path.isfile(results_path):
        raise FileNotFoundError(results_path)

    results = _load_json(results_path)
    row = _setting_from_results(results, setting_dir)
    episodes = results.get("results", [])
    if not isinstance(episodes, list):
        episodes = []
    success_count = sum(1 for episode in episodes if isinstance(episode, dict) and episode.get("success"))
    row.update(
        {
            "episodes": len(episodes),
            "success_count": success_count,
            "success_rate": _float_or_none(results.get("success_rate")),
        }
    )

    if os.path.isfile(dump_summary_path):
        dump = _load_json(dump_summary_path)
        summary = dump.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        dump_episodes = dump.get("episodes", [])
        if not isinstance(dump_episodes, list):
            dump_episodes = []
        trace = summary.get("env_trace", {})
        if not isinstance(trace, dict):
            trace = {}
        norm_debugs = [
            episode.get("norm_debug", {})
            for episode in dump_episodes
            if isinstance(episode, dict)
        ]
        row.update(
            {
                "first_cmd_delta_mean": _mean_finite(
                    [
                        episode.get("first_cmd_delta_mean")
                        for episode in dump_episodes
                        if isinstance(episode, dict)
                    ]
                ),
                "first_cmd_delta_max": _max_finite(
                    [
                        episode.get("first_cmd_delta_max")
                        for episode in dump_episodes
                        if isinstance(episode, dict)
                    ]
                ),
                "target_min_mean_left": _float_or_none(
                    trace.get("left_tcp_to_grasp_target_min_mean")
                ),
                "target_min_mean_right": _float_or_none(
                    trace.get("right_tcp_to_grasp_target_min_mean")
                ),
                "barrier_margin_best": _float_or_none(
                    trace.get("success_margin_max_best")
                ),
                "barrier_margin_mean": _float_or_none(
                    trace.get("success_margin_max_mean")
                ),
                "left_grasp_episodes": trace.get("left_grasp_episodes"),
                "right_grasp_episodes": trace.get("right_grasp_episodes"),
                "all_grippers_decisive_close": summary.get(
                    "all_grippers_decisive_close"
                ),
                "any_gripper_never_closes": summary.get("any_gripper_never_closes"),
                "mean_joint_step_delta": _float_or_none(
                    summary.get("mean_joint_step_delta")
                ),
                "max_joint_step_delta": _float_or_none(
                    summary.get("max_joint_step_delta")
                ),
                "mean_joint_step_accel": _float_or_none(
                    summary.get("mean_joint_step_accel")
                ),
                "max_joint_step_accel": _float_or_none(
                    summary.get("max_joint_step_accel")
                ),
                "mean_joint_accel_to_delta_ratio": _float_or_none(
                    summary.get("mean_joint_accel_to_delta_ratio")
                ),
                "max_joint_accel_to_delta_ratio": _float_or_none(
                    summary.get("max_joint_accel_to_delta_ratio")
                ),
                "joint_delta_sign_flip_frac": _float_or_none(
                    summary.get("joint_delta_sign_flip_frac")
                ),
                "mean_joint_delta_sign_flip_frac": _float_or_none(
                    summary.get("mean_joint_delta_sign_flip_frac")
                ),
                "joint_delta_sign_flip_count": summary.get(
                    "joint_delta_sign_flip_count"
                ),
                "joint_delta_active_pair_count": summary.get(
                    "joint_delta_active_pair_count"
                ),
                "mean_pred_chunk_joint_step_delta": _float_or_none(
                    summary.get("mean_pred_chunk_joint_step_delta")
                ),
                "max_pred_chunk_joint_step_delta": _float_or_none(
                    summary.get("max_pred_chunk_joint_step_delta")
                ),
                "mean_pred_chunk_joint_step_accel": _float_or_none(
                    summary.get("mean_pred_chunk_joint_step_accel")
                ),
                "max_pred_chunk_joint_step_accel": _float_or_none(
                    summary.get("max_pred_chunk_joint_step_accel")
                ),
                "mean_pred_chunk_joint_accel_to_delta_ratio": _float_or_none(
                    summary.get("mean_pred_chunk_joint_accel_to_delta_ratio")
                ),
                "max_pred_chunk_joint_accel_to_delta_ratio": _float_or_none(
                    summary.get("max_pred_chunk_joint_accel_to_delta_ratio")
                ),
                "pred_chunk_joint_delta_sign_flip_frac": _float_or_none(
                    summary.get("pred_chunk_joint_delta_sign_flip_frac")
                ),
                "mean_model_replan_boundary_joint_jump": _float_or_none(
                    summary.get("mean_model_replan_boundary_joint_jump")
                ),
                "max_model_replan_boundary_joint_jump": _float_or_none(
                    summary.get("max_model_replan_boundary_joint_jump")
                ),
                "mean_replan_boundary_joint_jump": _float_or_none(
                    summary.get("mean_replan_boundary_joint_jump")
                ),
                "max_replan_boundary_joint_jump": _float_or_none(
                    summary.get("max_replan_boundary_joint_jump")
                ),
                "raw_joint_saturation_frac": _mean_finite(
                    [
                        debug.get("raw_joint_saturation_frac")
                        for debug in norm_debugs
                        if isinstance(debug, dict)
                    ]
                ),
                "raw_gripper_saturation_frac": _mean_finite(
                    [
                        debug.get("raw_gripper_saturation_frac")
                        for debug in norm_debugs
                        if isinstance(debug, dict)
                    ]
                ),
                "joint_clamp_delta_mean": _mean_finite(
                    [
                        debug.get("joint_clamp_delta_mean")
                        for debug in norm_debugs
                        if isinstance(debug, dict)
                    ]
                ),
                "joint_clamp_delta_max": _max_finite(
                    [
                        debug.get("joint_clamp_delta_max")
                        for debug in norm_debugs
                        if isinstance(debug, dict)
                    ]
                ),
                "raw_joint_saturation_frac_first_step": _mean_finite(
                    [
                        debug.get("raw_joint_saturation_frac_first_step")
                        for debug in norm_debugs
                        if isinstance(debug, dict)
                    ]
                ),
                "raw_joint_saturation_frac_late_steps": _mean_finite(
                    [
                        debug.get("raw_joint_saturation_frac_late_steps")
                        for debug in norm_debugs
                        if isinstance(debug, dict)
                    ]
                ),
                "raw_joint_saturation_frac_left": _norm_debug_nested_stat(
                    norm_debugs,
                    "raw_joint_saturation_frac_by_arm",
                    "left",
                    "mean",
                ),
                "raw_joint_saturation_frac_right": _norm_debug_nested_stat(
                    norm_debugs,
                    "raw_joint_saturation_frac_by_arm",
                    "right",
                    "mean",
                ),
                "joint_clamp_delta_mean_left": _norm_debug_nested_stat(
                    norm_debugs,
                    "joint_clamp_delta_mean_by_arm",
                    "left",
                    "mean",
                ),
                "joint_clamp_delta_mean_right": _norm_debug_nested_stat(
                    norm_debugs,
                    "joint_clamp_delta_mean_by_arm",
                    "right",
                    "mean",
                ),
                "mean_pre_blend_replan_boundary_joint_jump": _joint_debug_stat(
                    dump_episodes,
                    "pre_blend_replan_boundary_joint_jump",
                    "mean_abs",
                    "mean",
                ),
                "max_pre_blend_replan_boundary_joint_jump": _joint_debug_stat(
                    dump_episodes,
                    "pre_blend_replan_boundary_joint_jump",
                    "max_abs",
                    "max",
                ),
                "mean_pre_ensemble_replan_boundary_joint_jump": _joint_debug_stat(
                    dump_episodes,
                    "pre_ensemble_replan_boundary_joint_jump",
                    "mean_abs",
                    "mean",
                ),
                "max_pre_ensemble_replan_boundary_joint_jump": _joint_debug_stat(
                    dump_episodes,
                    "pre_ensemble_replan_boundary_joint_jump",
                    "max_abs",
                    "max",
                ),
                "mean_temporal_ensemble_correction_joint": _joint_debug_stat(
                    dump_episodes,
                    "temporal_ensemble_correction_joint",
                    "mean_abs",
                    "mean",
                ),
                "max_temporal_ensemble_correction_joint": _joint_debug_stat(
                    dump_episodes,
                    "temporal_ensemble_correction_joint",
                    "max_abs",
                    "max",
                ),
                "mean_accel_limiter_correction_joint": _joint_debug_stat(
                    dump_episodes,
                    "accel_limiter_correction_joint",
                    "mean_abs",
                    "mean",
                ),
                "max_accel_limiter_correction_joint": _joint_debug_stat(
                    dump_episodes,
                    "accel_limiter_correction_joint",
                    "max_abs",
                    "max",
                ),
                "mean_pre_slew_joint_step_delta": _joint_debug_stat(
                    dump_episodes,
                    "pre_slew_joint_step_delta",
                    "mean_abs",
                    "mean",
                ),
                "max_pre_slew_joint_step_delta": _joint_debug_stat(
                    dump_episodes,
                    "pre_slew_joint_step_delta",
                    "max_abs",
                    "max",
                ),
                "mean_slew_correction_joint": _joint_debug_stat(
                    dump_episodes,
                    "slew_correction_joint",
                    "mean_abs",
                    "mean",
                ),
                "max_slew_correction_joint": _joint_debug_stat(
                    dump_episodes,
                    "slew_correction_joint",
                    "max_abs",
                    "max",
                ),
            }
        )
    return row


def summarize_sweep(root: str, pattern: str = "jscale_*") -> list[dict[str, Any]]:
    setting_dirs = sorted(
        path for path in glob.glob(os.path.join(root, pattern)) if os.path.isdir(path)
    )
    rows = [summarize_setting(path) for path in setting_dirs]
    rows.sort(
        key=lambda row: (
            row["scale"] is None,
            row["scale"] if row["scale"] is not None else row["setting_dir"],
            row["replan_every"] is None,
            row["replan_every"] if row["replan_every"] is not None else row["setting_dir"],
            row["replan_boundary_blend_steps"] is None,
            (
                row["replan_boundary_blend_steps"]
                if row["replan_boundary_blend_steps"] is not None
                else row["setting_dir"]
            ),
            row["temporal_action_ensemble_decay"] is None,
            (
                row["temporal_action_ensemble_decay"]
                if row["temporal_action_ensemble_decay"] is not None
                else row["setting_dir"]
            ),
        )
    )
    return rows


def _fmt(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "nan"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        ("dir", "setting_dir"),
        ("replan", "replan_every"),
        ("scale", "scale"),
        ("clip", "scale_clip"),
        ("slew", "target_slew_rate"),
        ("blend", "replan_boundary_blend_steps"),
        ("ens", "temporal_action_ensemble_decay"),
        ("succ", "success_count"),
        ("eps", "episodes"),
        ("first_delta", "first_cmd_delta_mean"),
        ("first_max", "first_cmd_delta_max"),
        ("L_target", "target_min_mean_left"),
        ("R_target", "target_min_mean_right"),
        ("margin_best", "barrier_margin_best"),
        ("L_grasp", "left_grasp_episodes"),
        ("R_grasp", "right_grasp_episodes"),
        ("joint_mean", "mean_joint_step_delta"),
        ("joint_max", "max_joint_step_delta"),
        ("pred_joint", "mean_pred_chunk_joint_step_delta"),
        ("accel_mean", "mean_joint_step_accel"),
        ("accel_ratio", "mean_joint_accel_to_delta_ratio"),
        ("pred_accel_ratio", "mean_pred_chunk_joint_accel_to_delta_ratio"),
        ("flip_frac", "joint_delta_sign_flip_frac"),
        ("pred_flip", "pred_chunk_joint_delta_sign_flip_frac"),
        ("boundary_max", "max_replan_boundary_joint_jump"),
        ("model_boundary", "max_model_replan_boundary_joint_jump"),
        ("accel_lim", "target_accel_limit"),
        ("accel_corr", "mean_accel_limiter_correction_joint"),
        ("preblend_max", "max_pre_blend_replan_boundary_joint_jump"),
        ("ens_corr", "mean_temporal_ensemble_correction_joint"),
        ("slew_corr", "mean_slew_correction_joint"),
        ("raw_sat", "raw_joint_saturation_frac"),
        ("raw_sat_L", "raw_joint_saturation_frac_left"),
        ("raw_sat_R", "raw_joint_saturation_frac_right"),
        ("raw_sat_t0", "raw_joint_saturation_frac_first_step"),
        ("raw_sat_late", "raw_joint_saturation_frac_late_steps"),
        ("raw_grip_sat", "raw_gripper_saturation_frac"),
        ("clamp_mean", "joint_clamp_delta_mean"),
        ("clamp_L", "joint_clamp_delta_mean_left"),
        ("clamp_R", "joint_clamp_delta_mean_right"),
        ("clamp_max", "joint_clamp_delta_max"),
        ("never_close", "any_gripper_never_closes"),
    ]
    print("\t".join(label for label, _ in columns))
    for row in rows:
        print("\t".join(_fmt(row.get(key)) for _, key in columns))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Sweep output root containing jscale_* directories.")
    ap.add_argument("--pattern", default="jscale_*")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    rows = summarize_sweep(args.root, args.pattern)
    if not rows:
        raise SystemExit(f"no setting directories matching {args.pattern!r} under {args.root}")
    print_table(rows)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"rows": rows}, f, indent=2, sort_keys=True)
        print(f"wrote JSON summary: {args.json_out}")


if __name__ == "__main__":
    main()
