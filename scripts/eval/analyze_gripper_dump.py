"""Analyze RoboFactory bimanual action dumps.

``eval_robofactory_ws.py --dump-actions DIR`` writes one
``episode_<seed>.npz`` per rollout with:

* ``pred_chunk``: full predicted chunks, shape ``[n_infer, chunk_len, 16]``.
  These are denormalized commands in controller units.
* ``action_norm_raw`` / ``action_norm_clipped`` (optional): normalized
  policy samples before/after inference-time clipping to ``[-1, 1]``.
* ``exec_action``: commands actually executed in the env, shape
  ``[n_steps, 16]``.
* ``exec_chunk_index`` / ``exec_gripper_chunk_index`` (optional): the
  joint chunk index and gripper chunk index used for each env step.
* ``exec_gripper_source_infer_step`` (optional): env step at which the
  gripper command's source chunk was predicted; useful for queue-mode
  gripper execution diagnostics.
* ``obs_qpos``: qpos observed at each policy call, shape ``[n_infer, 16]``.

LiftBarrier gripper command dims are 7 (left) and 15 (right):

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

GRIP_DIMS = (7, 15)
JOINT_DIMS = tuple(range(0, 7)) + tuple(range(8, 15))
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


def _gripper_metrics(
    values: np.ndarray,
    close_threshold: float,
    decisive_threshold: float,
) -> dict[str, Any]:
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
    exec_chunk_index = np.asarray(
        d["exec_chunk_index"] if "exec_chunk_index" in d.files else [],
        dtype=np.int64,
    )
    exec_gripper_chunk_index = np.asarray(
        d["exec_gripper_chunk_index"] if "exec_gripper_chunk_index" in d.files else [],
        dtype=np.int64,
    )
    exec_gripper_source_infer_step = np.asarray(
        (
            d["exec_gripper_source_infer_step"]
            if "exec_gripper_source_infer_step" in d.files
            else []
        ),
        dtype=np.int64,
    )
    steps = int(exec_action.shape[0])

    if exec_action.ndim != 2 or exec_action.shape[1] < 16:
        raise ValueError(f"{path}: exec_action must be [T, >=16], got {exec_action.shape}")
    if pred_chunk.ndim != 3 or pred_chunk.shape[2] < 16:
        raise ValueError(f"{path}: pred_chunk must be [N, H, >=16], got {pred_chunk.shape}")

    grip = {
        "left": _gripper_metrics(exec_action[:, GRIP_DIMS[0]], close_threshold, decisive_threshold),
        "right": _gripper_metrics(exec_action[:, GRIP_DIMS[1]], close_threshold, decisive_threshold),
    }

    joint_step_delta = np.diff(exec_action[:, JOINT_DIMS], axis=0) if steps > 1 else np.zeros((0, len(JOINT_DIMS)))
    max_joint_step_delta = float(np.max(np.abs(joint_step_delta))) if joint_step_delta.size else 0.0
    mean_joint_step_delta = float(np.mean(np.abs(joint_step_delta))) if joint_step_delta.size else 0.0

    first_cmd_delta_mean = None
    first_cmd_delta_max = None
    if obs_qpos.ndim == 2 and obs_qpos.shape[0] == pred_chunk.shape[0] and obs_qpos.shape[1] >= 16:
        first_cmd_delta = np.abs(pred_chunk[:, 0, JOINT_DIMS] - obs_qpos[:, JOINT_DIMS])
        first_cmd_delta_mean = float(first_cmd_delta.mean())
        first_cmd_delta_max = float(first_cmd_delta.max())

    pred_offsets = {
        "left": _chunk_close_offsets(pred_chunk, GRIP_DIMS[0], decisive_threshold),
        "right": _chunk_close_offsets(pred_chunk, GRIP_DIMS[1], decisive_threshold),
    }

    gripper_source_offset = None
    gripper_queue_age = None
    if exec_chunk_index.size and exec_gripper_chunk_index.size:
        if exec_chunk_index.shape != exec_gripper_chunk_index.shape:
            raise ValueError(
                f"{path}: exec_chunk_index shape {exec_chunk_index.shape} "
                f"does not match exec_gripper_chunk_index shape "
                f"{exec_gripper_chunk_index.shape}"
            )
        offset = exec_gripper_chunk_index - exec_chunk_index
        gripper_source_offset = {
            "min": int(offset.min()),
            "max": int(offset.max()),
            "mean": float(offset.mean()),
        }
    if exec_gripper_source_infer_step.size:
        if exec_gripper_source_infer_step.shape[0] != steps:
            raise ValueError(
                f"{path}: exec_gripper_source_infer_step length "
                f"{exec_gripper_source_infer_step.shape[0]} does not match "
                f"steps {steps}"
            )
        age = np.arange(steps, dtype=np.int64) - exec_gripper_source_infer_step
        gripper_queue_age = {
            "min": int(age.min()),
            "max": int(age.max()),
            "mean": float(age.mean()),
        }

    norm_debug = None
    if action_norm_raw is not None and action_norm_clipped is not None:
        raw_grip = action_norm_raw[..., GRIP_DIMS]
        clipped_grip = action_norm_clipped[..., GRIP_DIMS]
        clamp_delta = np.abs(action_norm_raw - action_norm_clipped)
        norm_debug = {
            "raw_gripper": _range_stats(raw_grip),
            "clipped_gripper": _range_stats(clipped_grip),
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
        "gripper": grip,
        "pred_decisive_close_offsets": pred_offsets,
        "max_joint_step_delta": max_joint_step_delta,
        "mean_joint_step_delta": mean_joint_step_delta,
        "first_cmd_delta_mean": first_cmd_delta_mean,
        "first_cmd_delta_max": first_cmd_delta_max,
        "gripper_source_offset": gripper_source_offset,
        "gripper_queue_age": gripper_queue_age,
        "norm_debug": norm_debug,
    }

    print(
        f"\n=== {episode['file']} seed={seed} success={success} "
        f"steps={steps} n_infer={pred_chunk.shape[0]} chunk_len={pred_chunk.shape[1]} ==="
    )
    for label, dim in (("left", GRIP_DIMS[0]), ("right", GRIP_DIMS[1])):
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
            f"  {label[0].upper()} gripper dim {dim}: "
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
    if gripper_source_offset is not None:
        print(
            "  gripper chunk source offset: "
            f"min={gripper_source_offset['min']} "
            f"max={gripper_source_offset['max']} "
            f"mean={gripper_source_offset['mean']:.1f}"
        )
    if gripper_queue_age is not None:
        print(
            "  gripper source age: "
            f"min={gripper_queue_age['min']} "
            f"max={gripper_queue_age['max']} "
            f"mean={gripper_queue_age['mean']:.1f}"
        )
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

    def _count_grip(side: str, key: str) -> int:
        return sum(int(bool(e["gripper"][side][key])) for e in episodes)

    first_close_values = {}
    for side in ("left", "right"):
        vals = [
            e["gripper"][side]["first_decisive_close"]
            for e in episodes
            if e["gripper"][side]["first_decisive_close"] is not None
        ]
        first_close_values[side] = {
            "count": len(vals),
            "mean": _safe_float(np.mean(vals)) if vals else None,
            "min": int(np.min(vals)) if vals else None,
            "max": int(np.max(vals)) if vals else None,
        }

    both_decisive = sum(
        int(
            e["gripper"]["left"]["first_decisive_close"] is not None
            and e["gripper"]["right"]["first_decisive_close"] is not None
        )
        for e in episodes
    )
    either_never_closes = sum(
        int(e["gripper"]["left"]["never_closes"] or e["gripper"]["right"]["never_closes"])
        for e in episodes
    )
    either_flat_mean = sum(
        int(
            e["gripper"]["left"]["flat_near_dataset_mean"]
            or e["gripper"]["right"]["flat_near_dataset_mean"]
        )
        for e in episodes
    )

    return {
        "episodes": n,
        "success_count": success_count,
        "success_rate": success_count / n if n else 0.0,
        "both_grippers_decisive_close": both_decisive,
        "either_gripper_never_closes": either_never_closes,
        "either_gripper_flat_near_dataset_mean": either_flat_mean,
        "left_never_closes": _count_grip("left", "never_closes"),
        "right_never_closes": _count_grip("right", "never_closes"),
        "first_decisive_close": first_close_values,
        "max_joint_step_delta": max(e["max_joint_step_delta"] for e in episodes) if episodes else 0.0,
        "mean_joint_step_delta": float(np.mean([e["mean_joint_step_delta"] for e in episodes])) if episodes else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", help="Directory containing episode_<seed>.npz files.")
    ap.add_argument("--close-threshold", type=float, default=0.0)
    ap.add_argument("--decisive-threshold", type=float, default=-0.5)
    ap.add_argument("--json-out", default=None, help="Optional path for machine-readable metrics.")
    ap.add_argument("--no-profiles", action="store_true", help="Hide per-step gripper profiles.")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dump_dir, "episode_*.npz")))
    if not files:
        raise SystemExit(f"no episode_*.npz under {args.dump_dir}")

    episodes = [
        analyze_episode(
            f,
            close_threshold=args.close_threshold,
            decisive_threshold=args.decisive_threshold,
            print_profiles=not args.no_profiles,
        )
        for f in files
    ]
    summary = _aggregate(episodes)

    print(f"\n==== summary over {summary['episodes']} episodes ====")
    print(f"  success: {summary['success_count']}/{summary['episodes']} ({summary['success_rate']:.1%})")
    print(
        "  both grippers decisive close: "
        f"{summary['both_grippers_decisive_close']}/{summary['episodes']}"
    )
    print(
        "  either gripper never closes: "
        f"{summary['either_gripper_never_closes']}/{summary['episodes']}"
    )
    print(
        "  either gripper flat near dataset mean: "
        f"{summary['either_gripper_flat_near_dataset_mean']}/{summary['episodes']}"
    )
    print(
        "  first decisive close left/right: "
        f"{summary['first_decisive_close']['left']} / "
        f"{summary['first_decisive_close']['right']}"
    )
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
