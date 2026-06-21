"""Analyze RoboFactory multi-arm action dumps.

``eval_robofactory_ws.py --dump-actions DIR`` writes one
``episode_<seed>.npz`` per rollout with:

* ``pred_chunk``: full predicted chunks, shape ``[n_infer, chunk_len, D]``.
  These are denormalized commands in controller units.
* ``action_norm_raw`` / ``action_norm_clipped`` (optional): normalized
  policy samples before/after inference-time clipping to ``[-1, 1]``.
* ``exec_action``: commands actually executed in the env, shape
  ``[n_steps, D]``.
* ``exec_action_pre_ensemble`` / ``exec_action_pre_blend`` /
  ``exec_action_pre_slew`` (optional): intermediate diagnostic commands
  before temporal action ensemble, replan-boundary blending, and slew
  limiting.
* ``obs_qpos``: qpos observed at each policy call, shape ``[n_infer, D]``.

By default, the script assumes each arm occupies an 8-D block:
7 joint dimensions followed by one gripper dimension. For 2 arms this
means gripper dims 7 and 15; for 3 arms, 7, 15, and 23. Override with
``--gripper-dims`` if a rollout uses a different layout.

* ``cmd > 0``: open.
* ``cmd < 0``: close.

The failure mode seen in early DreamZero checkpoints is easy to spot:
the arm joints move, but one or both grippers never issue a decisive
close command, or the gripper stays nearly flat around a dataset mean.
This script prints per-episode profiles and aggregate metrics so a
checkpoint can be triaged without watching every video.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

import numpy as np

DEFAULT_ARM_DIM = 8
DEFAULT_GRIPPER_OFFSET = 7
DATASET_ACTION_MEAN = -0.252
GT_DECISIVE_CLOSE_STEP = 43


def _profile(g: np.ndarray, n: int = 12) -> str:
    if g.size == 0:
        return ""
    step = max(1, len(g) // n)
    return " ".join(f"{i}:{g[i]:+.2f}" for i in range(0, len(g), step))


def _first_index(mask: np.ndarray) -> int | None:
    idx = np.where(mask)[0]
    return int(idx[0]) if idx.size else None


def _max_consecutive(mask: np.ndarray) -> int:
    best = cur = 0
    for value in mask.astype(bool):
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    return float(x)


def _parse_int_list(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise ValueError("empty integer list")
    return values


def _parse_str_list(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values:
        raise ValueError("empty label list")
    return values


def _infer_gripper_dims(
    action_dim: int,
    *,
    num_arms: int | None,
    arm_dim: int,
    gripper_offset: int,
    explicit_gripper_dims: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if explicit_gripper_dims is not None:
        gripper_dims = explicit_gripper_dims
    else:
        if arm_dim <= 0:
            raise ValueError(f"arm_dim must be positive, got {arm_dim}")
        if gripper_offset < 0 or gripper_offset >= arm_dim:
            raise ValueError(
                f"gripper_offset must be in [0, arm_dim), got {gripper_offset} for arm_dim={arm_dim}"
            )
        if num_arms is None:
            if action_dim % arm_dim != 0:
                raise ValueError(
                    f"cannot infer num_arms: action_dim={action_dim} is not divisible by arm_dim={arm_dim}; "
                    "pass --num-arms or --gripper-dims"
                )
            num_arms = action_dim // arm_dim
        if num_arms <= 0:
            raise ValueError(f"num_arms must be positive, got {num_arms}")
        gripper_dims = tuple(arm_dim * i + gripper_offset for i in range(num_arms))

    if len(set(gripper_dims)) != len(gripper_dims):
        raise ValueError(f"duplicate gripper dims: {gripper_dims}")
    bad = [dim for dim in gripper_dims if dim < 0 or dim >= action_dim]
    if bad:
        raise ValueError(f"gripper dims {bad} outside action dim {action_dim}")
    return gripper_dims


def _arm_labels(num_arms: int, custom_labels: tuple[str, ...] | None) -> tuple[str, ...]:
    if custom_labels is not None:
        if len(custom_labels) != num_arms:
            raise ValueError(
                f"--arm-labels has {len(custom_labels)} label(s), but inferred {num_arms} arm(s)"
            )
        return custom_labels
    if num_arms == 2:
        return ("left", "right")
    return tuple(f"arm{i}" for i in range(num_arms))


def _joint_dims(action_dim: int, gripper_dims: tuple[int, ...]) -> tuple[int, ...]:
    gripper_set = set(gripper_dims)
    return tuple(dim for dim in range(action_dim) if dim not in gripper_set)


def _gripper_metrics(
    values: np.ndarray,
    close_threshold: float,
    decisive_threshold: float,
) -> dict[str, Any]:
    if values.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "frac_close": 0.0,
            "first_close": None,
            "first_decisive_close": None,
            "max_consecutive_close": 0,
            "crossed_zero": False,
            "flat_near_dataset_mean": False,
            "never_closes": True,
        }
    close_mask = values < close_threshold
    decisive_mask = values < decisive_threshold
    first_decisive = _first_index(decisive_mask)
    first_close = _first_index(close_mask)
    has_open = bool(np.any(values > 0.0))
    has_close = bool(np.any(close_mask))
    crossed_zero = has_open and has_close
    flat = bool(values.std() < 0.15)
    near_mean = bool(abs(values.mean() - DATASET_ACTION_MEAN) < 0.1)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "frac_close": float(close_mask.mean()),
        "first_close": first_close,
        "first_decisive_close": first_decisive,
        "max_consecutive_close": _max_consecutive(close_mask),
        "crossed_zero": crossed_zero,
        "flat_near_dataset_mean": flat and near_mean,
        "never_closes": not has_close,
    }


def _chunk_close_offsets(
    pred_chunk: np.ndarray,
    dim: int,
    decisive_threshold: float,
) -> list[int | None]:
    offsets: list[int | None] = []
    for chunk in pred_chunk:
        offsets.append(_first_index(chunk[:, dim] < decisive_threshold))
    return offsets


def _range_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
        }
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def _abs_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "mean_abs": 0.0,
            "max_abs": 0.0,
            "p95_abs": 0.0,
        }
    abs_values = np.abs(values)
    return {
        "mean_abs": float(abs_values.mean()),
        "max_abs": float(abs_values.max()),
        "p95_abs": float(np.quantile(abs_values, 0.95)),
    }


def _joint_boundary_jumps(
    exec_action: np.ndarray,
    joint_dims: tuple[int, ...],
    infer_step: np.ndarray,
) -> np.ndarray:
    if not joint_dims or exec_action.shape[0] <= 1 or infer_step.size == 0:
        return np.zeros((0, len(joint_dims)), dtype=np.float32)
    jumps = []
    for raw_step in infer_step.astype(np.int64):
        step = int(raw_step)
        if step <= 0 or step >= exec_action.shape[0]:
            continue
        jumps.append(exec_action[step, joint_dims] - exec_action[step - 1, joint_dims])
    if not jumps:
        return np.zeros((0, len(joint_dims)), dtype=np.float32)
    return np.asarray(jumps, dtype=np.float32)


def _fmt_range(stats: dict[str, float]) -> str:
    return (
        f"min={stats['min']:+.3f} max={stats['max']:+.3f} "
        f"mean={stats['mean']:+.3f} std={stats['std']:.3f}"
    )


def _fmt_abs(stats: dict[str, float]) -> str:
    return (
        f"mean_abs={stats['mean_abs']:.3f} "
        f"p95_abs={stats['p95_abs']:.3f} "
        f"max_abs={stats['max_abs']:.3f}"
    )


def _round_list(values: np.ndarray, digits: int = 4) -> list[float]:
    if values.size == 0:
        return []
    return np.round(values.astype(np.float64), digits).tolist()


def _trace_column(
    trace: np.ndarray,
    columns: tuple[str, ...],
    name: str,
) -> np.ndarray | None:
    if trace.size == 0 or name not in columns:
        return None
    return trace[:, columns.index(name)]


def _finite(values: np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    return values[np.isfinite(values)]


def _first_true_step(
    steps: np.ndarray | None,
    values: np.ndarray | None,
    threshold: float = 0.5,
) -> int | None:
    if steps is None or values is None:
        return None
    mask = np.isfinite(values) & (values > threshold)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    return int(steps[idx[0]])


def _env_trace_debug(trace: np.ndarray, columns: tuple[str, ...]) -> dict[str, Any] | None:
    if trace.size == 0 or not columns:
        return None

    steps = _trace_column(trace, columns, "step")
    barrier_z = _finite(_trace_column(trace, columns, "barrier_z"))
    margin = _finite(_trace_column(trace, columns, "success_margin"))
    left_dist = _finite(_trace_column(trace, columns, "left_tcp_to_barrier"))
    right_dist = _finite(_trace_column(trace, columns, "right_tcp_to_barrier"))
    left_target_dist = _finite(_trace_column(trace, columns, "left_tcp_to_grasp_target"))
    right_target_dist = _finite(_trace_column(trace, columns, "right_tcp_to_grasp_target"))
    left_grasp = _trace_column(trace, columns, "left_grasping")
    right_grasp = _trace_column(trace, columns, "right_grasping")

    def _start_final_max(values: np.ndarray) -> dict[str, float | None]:
        if values.size == 0:
            return {"start": None, "final": None, "max": None, "min": None}
        return {
            "start": float(values[0]),
            "final": float(values[-1]),
            "max": float(values.max()),
            "min": float(values.min()),
        }

    def _min_final(values: np.ndarray) -> dict[str, float | None]:
        if values.size == 0:
            return {"min": None, "final": None}
        return {"min": float(values.min()), "final": float(values[-1])}

    return {
        "barrier_z": _start_final_max(barrier_z),
        "success_margin": _start_final_max(margin),
        "left_tcp_to_barrier": _min_final(left_dist),
        "right_tcp_to_barrier": _min_final(right_dist),
        "left_tcp_to_grasp_target": _min_final(left_target_dist),
        "right_tcp_to_grasp_target": _min_final(right_target_dist),
        "left_grasp_count": int(np.sum(np.isfinite(left_grasp) & (left_grasp > 0.5)))
        if left_grasp is not None
        else None,
        "right_grasp_count": int(np.sum(np.isfinite(right_grasp) & (right_grasp > 0.5)))
        if right_grasp is not None
        else None,
        "left_first_grasp_step": _first_true_step(steps, left_grasp),
        "right_first_grasp_step": _first_true_step(steps, right_grasp),
    }


def _fmt_start_final_max(stats: dict[str, float | None]) -> str:
    def fmt(value: float | None) -> str:
        return "nan" if value is None else f"{value:+.3f}"

    return (
        f"start={fmt(stats['start'])} final={fmt(stats['final'])} "
        f"min={fmt(stats['min'])} max={fmt(stats['max'])}"
    )


def _fmt_min_final(stats: dict[str, float | None]) -> str:
    def fmt(value: float | None) -> str:
        return "nan" if value is None else f"{value:.3f}"

    return f"min={fmt(stats['min'])} final={fmt(stats['final'])}"


def analyze_episode(
    path: str,
    close_threshold: float,
    decisive_threshold: float,
    print_profiles: bool,
    num_arms: int | None,
    arm_dim: int,
    gripper_offset: int,
    explicit_gripper_dims: tuple[int, ...] | None,
    custom_arm_labels: tuple[str, ...] | None,
) -> dict[str, Any]:
    d = np.load(path)
    seed = int(d["seed"])
    success = bool(d["success"])
    exec_action = np.asarray(d["exec_action"], dtype=np.float32)
    exec_action_pre_blend = (
        np.asarray(d["exec_action_pre_blend"], dtype=np.float32)
        if "exec_action_pre_blend" in d.files
        else None
    )
    exec_action_pre_ensemble = (
        np.asarray(d["exec_action_pre_ensemble"], dtype=np.float32)
        if "exec_action_pre_ensemble" in d.files
        else None
    )
    exec_action_pre_slew = (
        np.asarray(d["exec_action_pre_slew"], dtype=np.float32)
        if "exec_action_pre_slew" in d.files
        else None
    )
    pred_chunk = np.asarray(d["pred_chunk"], dtype=np.float32)
    action_norm_raw = (
        np.asarray(d["action_norm_raw"], dtype=np.float32)
        if "action_norm_raw" in d.files
        else None
    )
    action_norm_clipped = (
        np.asarray(d["action_norm_clipped"], dtype=np.float32)
        if "action_norm_clipped" in d.files
        else None
    )
    infer_step = np.asarray(d["infer_step"] if "infer_step" in d.files else [], dtype=np.int64)
    obs_qpos = np.asarray(d["obs_qpos"] if "obs_qpos" in d.files else [], dtype=np.float32)
    env_trace = np.asarray(d["env_trace"] if "env_trace" in d.files else [], dtype=np.float32)
    env_trace_columns = tuple(
        str(x) for x in np.asarray(d["env_trace_columns"] if "env_trace_columns" in d.files else [])
    )
    steps = int(exec_action.shape[0])

    if exec_action.ndim != 2:
        raise ValueError(f"{path}: exec_action must be [T, D], got {exec_action.shape}")
    if pred_chunk.ndim != 3:
        raise ValueError(f"{path}: pred_chunk must be [N, H, D], got {pred_chunk.shape}")
    action_dim = int(exec_action.shape[1])
    if int(pred_chunk.shape[2]) != action_dim:
        raise ValueError(
            f"{path}: pred_chunk action dim {pred_chunk.shape[2]} does not match "
            f"exec_action dim {action_dim}"
        )
    gripper_dims = _infer_gripper_dims(
        action_dim,
        num_arms=num_arms,
        arm_dim=arm_dim,
        gripper_offset=gripper_offset,
        explicit_gripper_dims=explicit_gripper_dims,
    )
    arm_labels = _arm_labels(len(gripper_dims), custom_arm_labels)
    joint_dims = _joint_dims(action_dim, gripper_dims)

    grip = {
        label: _gripper_metrics(exec_action[:, dim], close_threshold, decisive_threshold)
        for label, dim in zip(arm_labels, gripper_dims)
    }

    joint_step_delta = (
        np.diff(exec_action[:, joint_dims], axis=0)
        if steps > 1 and joint_dims
        else np.zeros((0, len(joint_dims)))
    )
    joint_step_accel = (
        np.diff(joint_step_delta, axis=0)
        if joint_step_delta.shape[0] > 1
        else np.zeros((0, len(joint_dims)))
    )
    replan_boundary_jump = _joint_boundary_jumps(
        exec_action,
        joint_dims,
        infer_step,
    )
    max_joint_step_delta = float(np.max(np.abs(joint_step_delta))) if joint_step_delta.size else 0.0
    mean_joint_step_delta = float(np.mean(np.abs(joint_step_delta))) if joint_step_delta.size else 0.0
    max_joint_step_accel = float(np.max(np.abs(joint_step_accel))) if joint_step_accel.size else 0.0
    mean_joint_step_accel = float(np.mean(np.abs(joint_step_accel))) if joint_step_accel.size else 0.0
    max_replan_boundary_joint_jump = (
        float(np.max(np.abs(replan_boundary_jump)))
        if replan_boundary_jump.size
        else 0.0
    )
    mean_replan_boundary_joint_jump = (
        float(np.mean(np.abs(replan_boundary_jump)))
        if replan_boundary_jump.size
        else 0.0
    )

    first_cmd_delta_mean = None
    first_cmd_delta_max = None
    joint_debug: dict[str, Any] = {
        "exec_joint": _range_stats(exec_action[:, joint_dims]) if joint_dims else None,
        "exec_joint_step_delta": _abs_stats(joint_step_delta),
        "exec_joint_step_accel": _abs_stats(joint_step_accel),
        "replan_boundary_joint_jump": _abs_stats(replan_boundary_jump),
        "pred_chunk_joint": _range_stats(pred_chunk[..., joint_dims]) if joint_dims else None,
    }
    if (
        joint_dims
        and exec_action_pre_ensemble is not None
        and exec_action_pre_ensemble.shape == exec_action.shape
    ):
        pre_ensemble_boundary_jump = _joint_boundary_jumps(
            exec_action_pre_ensemble,
            joint_dims,
            infer_step,
        )
        temporal_ensemble_target = (
            exec_action_pre_blend
            if exec_action_pre_blend is not None
            and exec_action_pre_blend.shape == exec_action.shape
            else exec_action
        )
        temporal_ensemble_correction = (
            exec_action_pre_ensemble[:, joint_dims]
            - temporal_ensemble_target[:, joint_dims]
        )
        joint_debug.update(
            {
                "pre_ensemble_replan_boundary_joint_jump": _abs_stats(
                    pre_ensemble_boundary_jump
                ),
                "temporal_ensemble_correction_joint": _abs_stats(
                    temporal_ensemble_correction
                ),
            }
        )
    if (
        joint_dims
        and exec_action_pre_blend is not None
        and exec_action_pre_blend.shape == exec_action.shape
    ):
        pre_blend_boundary_jump = _joint_boundary_jumps(
            exec_action_pre_blend,
            joint_dims,
            infer_step,
        )
        boundary_blend_target = (
            exec_action_pre_slew
            if exec_action_pre_slew is not None
            and exec_action_pre_slew.shape == exec_action.shape
            else exec_action
        )
        boundary_blend_correction = (
            exec_action_pre_blend[:, joint_dims] - boundary_blend_target[:, joint_dims]
        )
        joint_debug.update(
            {
                "pre_blend_replan_boundary_joint_jump": _abs_stats(
                    pre_blend_boundary_jump
                ),
                "boundary_blend_correction_joint": _abs_stats(boundary_blend_correction),
            }
        )
    if (
        joint_dims
        and exec_action_pre_slew is not None
        and exec_action_pre_slew.shape == exec_action.shape
    ):
        pre_slew_delta = (
            np.diff(exec_action_pre_slew[:, joint_dims], axis=0)
            if steps > 1
            else np.zeros((0, len(joint_dims)))
        )
        slew_correction = exec_action_pre_slew[:, joint_dims] - exec_action[:, joint_dims]
        joint_debug.update(
            {
                "pre_slew_joint_step_delta": _abs_stats(pre_slew_delta),
                "slew_correction_joint": _abs_stats(slew_correction),
            }
        )
    if (
        joint_dims
        and obs_qpos.ndim == 2
        and obs_qpos.shape[0] == pred_chunk.shape[0]
        and obs_qpos.shape[1] >= action_dim
    ):
        first_cmd_delta_signed = pred_chunk[:, 0, joint_dims] - obs_qpos[:, joint_dims]
        first_cmd_delta = np.abs(first_cmd_delta_signed)
        first_cmd_delta_mean = float(first_cmd_delta.mean())
        first_cmd_delta_max = float(first_cmd_delta.max())
        joint_debug.update(
            {
                "obs_qpos_joint": _range_stats(obs_qpos[:, joint_dims]),
                "first_cmd_delta_signed": _range_stats(first_cmd_delta_signed),
                "first_cmd_delta_abs": _abs_stats(first_cmd_delta_signed),
                "first_cmd_delta_abs_per_dim_mean": _round_list(
                    first_cmd_delta.mean(axis=0)
                ),
                "first_cmd_delta_abs_per_dim_max": _round_list(
                    first_cmd_delta.max(axis=0)
                ),
            }
        )

    pred_offsets = {
        label: _chunk_close_offsets(pred_chunk, dim, decisive_threshold)
        for label, dim in zip(arm_labels, gripper_dims)
    }

    norm_debug = None
    if action_norm_raw is not None and action_norm_clipped is not None:
        raw_grip = action_norm_raw[..., gripper_dims]
        clipped_grip = action_norm_clipped[..., gripper_dims]
        clamp_delta = np.abs(action_norm_raw - action_norm_clipped)
        raw_joint = action_norm_raw[..., joint_dims] if joint_dims else np.asarray([])
        clipped_joint = action_norm_clipped[..., joint_dims] if joint_dims else np.asarray([])
        joint_clamp_delta = (
            np.abs(raw_joint - clipped_joint)
            if raw_joint.size and clipped_joint.size
            else np.asarray([])
        )
        norm_debug = {
            "raw_gripper": _range_stats(raw_grip),
            "clipped_gripper": _range_stats(clipped_grip),
            "raw_gripper_by_arm": {
                label: _range_stats(action_norm_raw[..., dim])
                for label, dim in zip(arm_labels, gripper_dims)
            },
            "clipped_gripper_by_arm": {
                label: _range_stats(action_norm_clipped[..., dim])
                for label, dim in zip(arm_labels, gripper_dims)
            },
            "raw_saturation_frac": float((np.abs(action_norm_raw) >= 0.999).mean()),
            "raw_gripper_saturation_frac": float((np.abs(raw_grip) >= 0.999).mean()),
            "clamp_delta_mean": float(clamp_delta.mean()),
            "clamp_delta_max": float(clamp_delta.max()),
        }
        if raw_joint.size:
            norm_debug.update(
                {
                    "raw_joint": _range_stats(raw_joint),
                    "clipped_joint": _range_stats(clipped_joint),
                    "raw_joint_saturation_frac": float((np.abs(raw_joint) >= 0.999).mean()),
                    "joint_clamp_delta_mean": float(joint_clamp_delta.mean()),
                    "joint_clamp_delta_max": float(joint_clamp_delta.max()),
                }
            )

    trace_debug = _env_trace_debug(env_trace, env_trace_columns)

    episode = {
        "file": os.path.basename(path),
        "seed": seed,
        "success": success,
        "steps": steps,
        "num_infer": int(pred_chunk.shape[0]),
        "chunk_len": int(pred_chunk.shape[1]),
        "action_dim": action_dim,
        "num_arms": len(gripper_dims),
        "gripper_dims": list(gripper_dims),
        "arm_labels": list(arm_labels),
        "gripper": grip,
        "pred_decisive_close_offsets": pred_offsets,
        "max_joint_step_delta": max_joint_step_delta,
        "mean_joint_step_delta": mean_joint_step_delta,
        "max_joint_step_accel": max_joint_step_accel,
        "mean_joint_step_accel": mean_joint_step_accel,
        "max_replan_boundary_joint_jump": max_replan_boundary_joint_jump,
        "mean_replan_boundary_joint_jump": mean_replan_boundary_joint_jump,
        "first_cmd_delta_mean": first_cmd_delta_mean,
        "first_cmd_delta_max": first_cmd_delta_max,
        "joint_debug": joint_debug,
        "norm_debug": norm_debug,
        "env_trace_debug": trace_debug,
    }

    print(
        f"\n=== {episode['file']} seed={seed} success={success} "
        f"steps={steps} n_infer={pred_chunk.shape[0]} chunk_len={pred_chunk.shape[1]} ==="
    )
    print(
        f"  action_dim={action_dim} num_arms={len(gripper_dims)} "
        f"gripper_dims={list(gripper_dims)}"
    )
    for label, dim in zip(arm_labels, gripper_dims):
        m = grip[label]
        verdict = []
        if m["first_decisive_close"] is None:
            verdict.append(f"no cmd<{decisive_threshold:+.2f}")
        else:
            verdict.append(
                f"decisive close @{m['first_decisive_close']} "
                f"(GT~{GT_DECISIVE_CLOSE_STEP})"
            )
        if m["flat_near_dataset_mean"]:
            verdict.append(f"flat near dataset mean ({m['mean']:+.2f})")
        elif not m["crossed_zero"]:
            verdict.append("no open/close zero crossing")

        print(
            f"  {label} gripper dim {dim}: "
            f"min {m['min']:+.2f} max {m['max']:+.2f} "
            f"mean {m['mean']:+.2f} std {m['std']:.2f} "
            f"frac_close {m['frac_close']:.2f} "
            f"max_run {m['max_consecutive_close']}"
        )
        if print_profiles:
            print(f"     profile: {_profile(exec_action[:, dim])}")
        print(f"     -> {'; '.join(verdict)}")

    delta_msg = (
        f"  joint step delta: mean_abs={mean_joint_step_delta:.3f} "
        f"max_abs={max_joint_step_delta:.3f}"
    )
    if first_cmd_delta_mean is not None:
        delta_msg += (
            f" | first_cmd_vs_obs: mean_abs={first_cmd_delta_mean:.3f} "
            f"max_abs={first_cmd_delta_max:.3f}"
        )
    print(delta_msg)
    if joint_debug["exec_joint"] is not None:
        print(f"  exec joint range: {_fmt_range(joint_debug['exec_joint'])}")
        print(f"  pred chunk joint range: {_fmt_range(joint_debug['pred_chunk_joint'])}")
        print(f"  joint acceleration: {_fmt_abs(joint_debug['exec_joint_step_accel'])}")
        print(
            "  replan boundary joint jump: "
            f"{_fmt_abs(joint_debug['replan_boundary_joint_jump'])}"
        )
        if "pre_blend_replan_boundary_joint_jump" in joint_debug:
            print(
                "  pre-blend replan boundary joint jump: "
                f"{_fmt_abs(joint_debug['pre_blend_replan_boundary_joint_jump'])}"
            )
        if "pre_ensemble_replan_boundary_joint_jump" in joint_debug:
            print(
                "  pre-ensemble replan boundary joint jump: "
                f"{_fmt_abs(joint_debug['pre_ensemble_replan_boundary_joint_jump'])}"
            )
            print(
                "  temporal ensemble correction joint: "
                f"{_fmt_abs(joint_debug['temporal_ensemble_correction_joint'])}"
            )
        if "pre_slew_joint_step_delta" in joint_debug:
            print(
                "  pre-slew joint step delta: "
                f"{_fmt_abs(joint_debug['pre_slew_joint_step_delta'])}"
            )
            print(
                "  slew correction joint: "
                f"{_fmt_abs(joint_debug['slew_correction_joint'])}"
            )
        if "obs_qpos_joint" in joint_debug:
            print(f"  obs qpos joint range: {_fmt_range(joint_debug['obs_qpos_joint'])}")
            print(
                "  first_cmd_vs_obs signed: "
                f"{_fmt_range(joint_debug['first_cmd_delta_signed'])} | "
                f"{_fmt_abs(joint_debug['first_cmd_delta_abs'])}"
            )
            print(
                "  first_cmd_vs_obs per-dim mean_abs: "
                f"{joint_debug['first_cmd_delta_abs_per_dim_mean']}"
            )

    if infer_step.size:
        print(f"  infer steps: first={int(infer_step[0])} last={int(infer_step[-1])} count={infer_step.size}")
    if norm_debug is not None:
        rg = norm_debug["raw_gripper"]
        print(
            "  raw normalized gripper: "
            f"min={rg['min']:+.2f} max={rg['max']:+.2f} "
            f"sat_frac={norm_debug['raw_gripper_saturation_frac']:.2f} "
            f"clamp_delta_mean={norm_debug['clamp_delta_mean']:.3f} "
            f"clamp_delta_max={norm_debug['clamp_delta_max']:.3f}"
        )
        if "raw_joint" in norm_debug:
            rj = norm_debug["raw_joint"]
            print(
                "  raw normalized joints: "
                f"min={rj['min']:+.2f} max={rj['max']:+.2f} "
                f"sat_frac={norm_debug['raw_joint_saturation_frac']:.2f} "
                f"joint_clamp_delta_mean={norm_debug['joint_clamp_delta_mean']:.3f} "
                f"joint_clamp_delta_max={norm_debug['joint_clamp_delta_max']:.3f}"
            )

    if trace_debug is not None:
        print(
            "  barrier z: "
            f"{_fmt_start_final_max(trace_debug['barrier_z'])}"
        )
        print(
            "  success margin: "
            f"{_fmt_start_final_max(trace_debug['success_margin'])}"
        )
        print(
            "  tcp->barrier: "
            f"{arm_labels[0] if arm_labels else 'arm0'} "
            f"{_fmt_min_final(trace_debug['left_tcp_to_barrier'])} | "
            f"{arm_labels[1] if len(arm_labels) > 1 else 'arm1'} "
            f"{_fmt_min_final(trace_debug['right_tcp_to_barrier'])}"
        )
        print(
            "  grasping: "
            f"left_count={trace_debug['left_grasp_count']} "
            f"left_first={trace_debug['left_first_grasp_step']} "
            f"right_count={trace_debug['right_grasp_count']} "
            f"right_first={trace_debug['right_first_grasp_step']}"
        )
        print(
            "  tcp->grasp target: "
            f"{arm_labels[0] if arm_labels else 'arm0'} "
            f"{_fmt_min_final(trace_debug['left_tcp_to_grasp_target'])} | "
            f"{arm_labels[1] if len(arm_labels) > 1 else 'arm1'} "
            f"{_fmt_min_final(trace_debug['right_tcp_to_grasp_target'])}"
        )

    return episode


def _aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(episodes)
    success_count = sum(int(e["success"]) for e in episodes)
    labels = list(episodes[0]["arm_labels"]) if episodes else []

    def _count_grip(label: str, key: str) -> int:
        return sum(int(bool(e["gripper"][label][key])) for e in episodes)

    first_close_values = {}
    for label in labels:
        vals = [
            e["gripper"][label]["first_decisive_close"]
            for e in episodes
            if e["gripper"][label]["first_decisive_close"] is not None
        ]
        first_close_values[label] = {
            "count": len(vals),
            "mean": _safe_float(np.mean(vals)) if vals else None,
            "min": int(np.min(vals)) if vals else None,
            "max": int(np.max(vals)) if vals else None,
        }

    all_decisive = sum(
        int(all(e["gripper"][label]["first_decisive_close"] is not None for label in labels))
        for e in episodes
    )
    any_never_closes = sum(
        int(any(e["gripper"][label]["never_closes"] for label in labels))
        for e in episodes
    )
    any_flat_mean = sum(
        int(any(e["gripper"][label]["flat_near_dataset_mean"] for label in labels))
        for e in episodes
    )
    per_gripper_never_closes = {
        label: _count_grip(label, "never_closes")
        for label in labels
    }

    summary = {
        "episodes": n,
        "success_count": success_count,
        "success_rate": success_count / n if n else 0.0,
        "num_arms": len(labels),
        "arm_labels": labels,
        "all_grippers_decisive_close": all_decisive,
        "any_gripper_never_closes": any_never_closes,
        "any_gripper_flat_near_dataset_mean": any_flat_mean,
        "per_gripper_never_closes": per_gripper_never_closes,
        "first_decisive_close": first_close_values,
        "max_joint_step_delta": max(e["max_joint_step_delta"] for e in episodes) if episodes else 0.0,
        "mean_joint_step_delta": float(np.mean([e["mean_joint_step_delta"] for e in episodes])) if episodes else 0.0,
        "max_joint_step_accel": max(e["max_joint_step_accel"] for e in episodes) if episodes else 0.0,
        "mean_joint_step_accel": float(np.mean([e["mean_joint_step_accel"] for e in episodes])) if episodes else 0.0,
        "max_replan_boundary_joint_jump": max(
            e["max_replan_boundary_joint_jump"] for e in episodes
        ) if episodes else 0.0,
        "mean_replan_boundary_joint_jump": float(
            np.mean([e["mean_replan_boundary_joint_jump"] for e in episodes])
        ) if episodes else 0.0,
    }
    trace_episodes = [e["env_trace_debug"] for e in episodes if e.get("env_trace_debug") is not None]
    if trace_episodes:
        margin_max = [
            t["success_margin"]["max"]
            for t in trace_episodes
            if t["success_margin"]["max"] is not None
        ]
        left_min = [
            t["left_tcp_to_barrier"]["min"]
            for t in trace_episodes
            if t["left_tcp_to_barrier"]["min"] is not None
        ]
        right_min = [
            t["right_tcp_to_barrier"]["min"]
            for t in trace_episodes
            if t["right_tcp_to_barrier"]["min"] is not None
        ]
        left_target_min = [
            t["left_tcp_to_grasp_target"]["min"]
            for t in trace_episodes
            if t["left_tcp_to_grasp_target"]["min"] is not None
        ]
        right_target_min = [
            t["right_tcp_to_grasp_target"]["min"]
            for t in trace_episodes
            if t["right_tcp_to_grasp_target"]["min"] is not None
        ]
        summary["env_trace"] = {
            "episodes": len(trace_episodes),
            "success_margin_max_mean": _safe_float(np.mean(margin_max)) if margin_max else None,
            "success_margin_max_best": _safe_float(np.max(margin_max)) if margin_max else None,
            "left_tcp_to_barrier_min_mean": _safe_float(np.mean(left_min)) if left_min else None,
            "right_tcp_to_barrier_min_mean": _safe_float(np.mean(right_min)) if right_min else None,
            "left_tcp_to_grasp_target_min_mean": _safe_float(np.mean(left_target_min)) if left_target_min else None,
            "right_tcp_to_grasp_target_min_mean": _safe_float(np.mean(right_target_min)) if right_target_min else None,
            "left_grasp_episodes": sum(
                int((t["left_grasp_count"] or 0) > 0) for t in trace_episodes
            ),
            "right_grasp_episodes": sum(
                int((t["right_grasp_count"] or 0) > 0) for t in trace_episodes
            ),
        }
    if labels == ["left", "right"]:
        summary.update(
            {
                "both_grippers_decisive_close": all_decisive,
                "either_gripper_never_closes": any_never_closes,
                "either_gripper_flat_near_dataset_mean": any_flat_mean,
                "left_never_closes": per_gripper_never_closes["left"],
                "right_never_closes": per_gripper_never_closes["right"],
            }
        )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", help="Directory containing episode_<seed>.npz files.")
    ap.add_argument("--close-threshold", type=float, default=0.0)
    ap.add_argument("--decisive-threshold", type=float, default=-0.5)
    ap.add_argument(
        "--num-arms",
        type=int,
        default=None,
        help="Number of arms. Defaults to action_dim / arm_dim.",
    )
    ap.add_argument(
        "--arm-dim",
        type=int,
        default=DEFAULT_ARM_DIM,
        help="Action dimensions per arm when inferring gripper dims. Default: 8.",
    )
    ap.add_argument(
        "--gripper-offset",
        type=int,
        default=DEFAULT_GRIPPER_OFFSET,
        help="Per-arm gripper offset when inferring gripper dims. Default: 7.",
    )
    ap.add_argument(
        "--gripper-dims",
        default=None,
        help="Comma-separated explicit gripper dims, e.g. 7,15,23. Overrides --num-arms.",
    )
    ap.add_argument(
        "--arm-labels",
        default=None,
        help="Comma-separated labels for arms/grippers, e.g. panda0,panda1,panda2.",
    )
    ap.add_argument("--json-out", default=None, help="Optional path for machine-readable metrics.")
    ap.add_argument("--no-profiles", action="store_true", help="Hide per-step gripper profiles.")
    args = ap.parse_args()
    explicit_gripper_dims = _parse_int_list(args.gripper_dims)
    custom_arm_labels = _parse_str_list(args.arm_labels)

    files = sorted(glob.glob(os.path.join(args.dump_dir, "episode_*.npz")))
    if not files:
        raise SystemExit(f"no episode_*.npz under {args.dump_dir}")

    episodes = [
        analyze_episode(
            f,
            close_threshold=args.close_threshold,
            decisive_threshold=args.decisive_threshold,
            print_profiles=not args.no_profiles,
            num_arms=args.num_arms,
            arm_dim=args.arm_dim,
            gripper_offset=args.gripper_offset,
            explicit_gripper_dims=explicit_gripper_dims,
            custom_arm_labels=custom_arm_labels,
        )
        for f in files
    ]
    summary = _aggregate(episodes)

    print(f"\n==== summary over {summary['episodes']} episodes ====")
    print(f"  success: {summary['success_count']}/{summary['episodes']} ({summary['success_rate']:.1%})")
    print(
        f"  arms: {summary['num_arms']} ({', '.join(summary['arm_labels'])})"
    )
    print(
        "  all grippers decisive close: "
        f"{summary['all_grippers_decisive_close']}/{summary['episodes']}"
    )
    print(
        "  any gripper never closes: "
        f"{summary['any_gripper_never_closes']}/{summary['episodes']}"
    )
    print(
        "  any gripper flat near dataset mean: "
        f"{summary['any_gripper_flat_near_dataset_mean']}/{summary['episodes']}"
    )
    print(f"  per-gripper never closes: {summary['per_gripper_never_closes']}")
    print(f"  first decisive close: {summary['first_decisive_close']}")
    print(
        "  joint step delta: "
        f"mean_abs={summary['mean_joint_step_delta']:.3f} "
        f"max_abs={summary['max_joint_step_delta']:.3f}"
    )
    print(
        "  joint acceleration: "
        f"mean_abs={summary['mean_joint_step_accel']:.3f} "
        f"max_abs={summary['max_joint_step_accel']:.3f}"
    )
    print(
        "  replan boundary joint jump: "
        f"mean_abs={summary['mean_replan_boundary_joint_jump']:.3f} "
        f"max_abs={summary['max_replan_boundary_joint_jump']:.3f}"
    )
    if "env_trace" in summary:
        trace = summary["env_trace"]
        print(
            "  env trace: "
            f"episodes={trace['episodes']} "
            f"success_margin_max_mean={trace['success_margin_max_mean']} "
            f"success_margin_max_best={trace['success_margin_max_best']} "
            f"left_tcp_min_mean={trace['left_tcp_to_barrier_min_mean']} "
            f"right_tcp_min_mean={trace['right_tcp_to_barrier_min_mean']} "
            f"left_target_min_mean={trace['left_tcp_to_grasp_target_min_mean']} "
            f"right_target_min_mean={trace['right_tcp_to_grasp_target_min_mean']} "
            f"left_grasp_eps={trace['left_grasp_episodes']} "
            f"right_grasp_eps={trace['right_grasp_episodes']}"
        )

    if args.json_out:
        payload = {"summary": summary, "episodes": episodes}
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"  wrote JSON metrics: {args.json_out}")


if __name__ == "__main__":
    main()
