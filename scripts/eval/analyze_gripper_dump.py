"""Analyze RoboFactory multi-arm action dumps.

``eval_robofactory_ws.py --dump-actions DIR`` writes one
``episode_<seed>.npz`` per rollout with:

* ``pred_chunk``: full predicted chunks, shape ``[n_infer, chunk_len, D]``.
  These are denormalized commands in controller units.
* ``action_norm_raw`` / ``action_norm_clipped`` (optional): normalized
  policy samples before/after inference-time clipping to ``[-1, 1]``.
* ``exec_action``: commands actually executed in the env, shape
  ``[n_steps, D]``.
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
    max_joint_step_delta = float(np.max(np.abs(joint_step_delta))) if joint_step_delta.size else 0.0
    mean_joint_step_delta = float(np.mean(np.abs(joint_step_delta))) if joint_step_delta.size else 0.0

    first_cmd_delta_mean = None
    first_cmd_delta_max = None
    if (
        joint_dims
        and obs_qpos.ndim == 2
        and obs_qpos.shape[0] == pred_chunk.shape[0]
        and obs_qpos.shape[1] >= action_dim
    ):
        first_cmd_delta = np.abs(pred_chunk[:, 0, joint_dims] - obs_qpos[:, joint_dims])
        first_cmd_delta_mean = float(first_cmd_delta.mean())
        first_cmd_delta_max = float(first_cmd_delta.max())

    pred_offsets = {
        label: _chunk_close_offsets(pred_chunk, dim, decisive_threshold)
        for label, dim in zip(arm_labels, gripper_dims)
    }

    norm_debug = None
    if action_norm_raw is not None and action_norm_clipped is not None:
        raw_grip = action_norm_raw[..., gripper_dims]
        clipped_grip = action_norm_clipped[..., gripper_dims]
        clamp_delta = np.abs(action_norm_raw - action_norm_clipped)
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
        "first_cmd_delta_mean": first_cmd_delta_mean,
        "first_cmd_delta_max": first_cmd_delta_max,
        "norm_debug": norm_debug,
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

    if args.json_out:
        payload = {"summary": summary, "episodes": episodes}
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"  wrote JSON metrics: {args.json_out}")


if __name__ == "__main__":
    main()
