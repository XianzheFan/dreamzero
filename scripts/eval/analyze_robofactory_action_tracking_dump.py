"""Analyze RoboFactory close-loop action dumps with qpos tracking fields.

The companion patch to ``eval_robofactory_ws.py`` writes, for every executed
step:

* ``exec_action``: absolute qpos target sent to ``pd_joint_pos``
* ``exec_qpos_before``: actual env qpos before the command
* ``exec_qpos_after``: actual env qpos after one env step

This script summarizes command size, actual movement, controller tracking, and
gripper timing. It also keeps compatibility with older dumps that only contain
``exec_action`` and ``obs_qpos`` by reporting the missing tracking fields.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


JOINT_DIMS = np.asarray([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14], dtype=np.int64)
GRIPPER_DIMS = np.asarray([7, 15], dtype=np.int64)


def _safe_stats(values: np.ndarray | list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def _first_index(mask: np.ndarray) -> int | None:
    idx = np.where(mask)[0]
    return int(idx[0]) if idx.size else None


def _as_bool(value: Any) -> bool:
    arr = np.asarray(value)
    if arr.shape == ():
        return bool(arr.item())
    return bool(arr.any())


def analyze_file(path: str, close_threshold: float) -> dict[str, Any]:
    d = np.load(path)
    exec_action = np.asarray(d["exec_action"], dtype=np.float32)
    pred_chunk = np.asarray(d["pred_chunk"], dtype=np.float32)
    obs_qpos = np.asarray(d["obs_qpos"], dtype=np.float32) if "obs_qpos" in d.files else None
    pre = np.asarray(d["exec_qpos_before"], dtype=np.float32) if "exec_qpos_before" in d.files else None
    post = np.asarray(d["exec_qpos_after"], dtype=np.float32) if "exec_qpos_after" in d.files else None
    infer_step = np.asarray(d["infer_step"], dtype=np.int64) if "infer_step" in d.files else None

    if exec_action.ndim != 2 or exec_action.shape[1] != 16:
        raise ValueError(f"{path}: exec_action must be [T, 16], got {exec_action.shape}")
    if pred_chunk.ndim != 3 or pred_chunk.shape[2] != 16:
        raise ValueError(f"{path}: pred_chunk must be [N, H, 16], got {pred_chunk.shape}")
    if pre is not None and pre.shape != exec_action.shape:
        raise ValueError(f"{path}: exec_qpos_before shape {pre.shape} != {exec_action.shape}")
    if post is not None and post.shape != exec_action.shape:
        raise ValueError(f"{path}: exec_qpos_after shape {post.shape} != {exec_action.shape}")

    grip = exec_action[:, GRIPPER_DIMS]
    close_mask = grip < close_threshold
    episode: dict[str, Any] = {
        "file": os.path.basename(path),
        "seed": int(np.asarray(d["seed"]).item()) if "seed" in d.files else None,
        "success": _as_bool(d["success"]) if "success" in d.files else False,
        "steps": int(exec_action.shape[0]),
        "num_infer": int(pred_chunk.shape[0]),
        "chunk_len": int(pred_chunk.shape[1]),
        "left_first_close_step": _first_index(close_mask[:, 0]) if close_mask.size else None,
        "right_first_close_step": _first_index(close_mask[:, 1]) if close_mask.size else None,
        "left_frac_close": float(close_mask[:, 0].mean()) if close_mask.size else 0.0,
        "right_frac_close": float(close_mask[:, 1].mean()) if close_mask.size else 0.0,
        "has_step_qpos_tracking": bool(pre is not None and post is not None),
    }

    if obs_qpos is not None and obs_qpos.ndim == 2 and obs_qpos.shape[1] == 16:
        first_cmd_delta = np.abs(pred_chunk[:, 0, JOINT_DIMS] - obs_qpos[:, JOINT_DIMS])
        episode["planned_first_cmd_joint_delta_l1"] = _safe_stats(first_cmd_delta.mean(axis=1))
        episode["planned_first_cmd_joint_delta_linf"] = _safe_stats(first_cmd_delta.max(axis=1))

    if pre is not None and post is not None:
        cmd = exec_action[:, JOINT_DIMS] - pre[:, JOINT_DIMS]
        actual = post[:, JOINT_DIMS] - pre[:, JOINT_DIMS]
        tracking = post[:, JOINT_DIMS] - exec_action[:, JOINT_DIMS]
        cmd_norm = np.linalg.norm(cmd, axis=1)
        actual_norm = np.linalg.norm(actual, axis=1)
        ratio = actual_norm / np.maximum(cmd_norm, 1e-8)
        episode.update(
            {
                "cmd_joint_delta_l1": _safe_stats(np.abs(cmd).mean(axis=1)),
                "cmd_joint_delta_linf": _safe_stats(np.abs(cmd).max(axis=1)),
                "actual_joint_delta_l1": _safe_stats(np.abs(actual).mean(axis=1)),
                "actual_joint_delta_linf": _safe_stats(np.abs(actual).max(axis=1)),
                "actual_to_cmd_joint_delta_ratio": _safe_stats(ratio),
                "post_to_target_joint_l1": _safe_stats(np.abs(tracking).mean(axis=1)),
                "post_to_target_joint_linf": _safe_stats(np.abs(tracking).max(axis=1)),
                "final_qpos": post[-1].tolist() if post.shape[0] else [],
            }
        )

        if obs_qpos is not None and infer_step is not None:
            boundary_err = []
            for idx, step in enumerate(infer_step.tolist()):
                if idx < obs_qpos.shape[0] and 0 <= step < pre.shape[0]:
                    boundary_err.append(float(np.abs(obs_qpos[idx] - pre[step]).mean()))
            episode["infer_obs_vs_step_qpos_l1"] = _safe_stats(boundary_err)

    if "action_norm_raw" in d.files and "action_norm_clipped" in d.files:
        raw = np.asarray(d["action_norm_raw"], dtype=np.float32)
        clipped = np.asarray(d["action_norm_clipped"], dtype=np.float32)
        episode["norm_raw_saturation_frac"] = float((np.abs(raw) >= 0.999).mean())
        episode["norm_gripper_raw_saturation_frac"] = float(
            (np.abs(raw[..., GRIPPER_DIMS]) >= 0.999).mean()
        )
        episode["norm_clip_delta_l1_mean"] = float(np.abs(raw - clipped).mean())
        episode["norm_clip_delta_linf"] = float(np.abs(raw - clipped).max())

    return episode


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        return {"num_episodes": 0, "success_rate": 0.0}
    ok = sum(1 for ep in episodes if ep.get("success"))

    def collect(path: tuple[str, ...]) -> list[float]:
        values = []
        for ep in episodes:
            cur: Any = ep
            missing = False
            for key in path:
                if not isinstance(cur, dict) or key not in cur:
                    missing = True
                    break
                cur = cur[key]
            if not missing:
                values.append(float(cur))
        return values

    return {
        "num_episodes": len(episodes),
        "success_count": ok,
        "success_rate": ok / len(episodes),
        "has_step_qpos_tracking_count": sum(
            1 for ep in episodes if ep.get("has_step_qpos_tracking")
        ),
        "cmd_joint_delta_l1_mean": _safe_stats(collect(("cmd_joint_delta_l1", "mean"))),
        "actual_joint_delta_l1_mean": _safe_stats(
            collect(("actual_joint_delta_l1", "mean"))
        ),
        "actual_to_cmd_ratio_mean": _safe_stats(
            collect(("actual_to_cmd_joint_delta_ratio", "mean"))
        ),
        "post_to_target_joint_l1_mean": _safe_stats(
            collect(("post_to_target_joint_l1", "mean"))
        ),
        "planned_first_cmd_joint_delta_l1_mean": _safe_stats(
            collect(("planned_first_cmd_joint_delta_l1", "mean"))
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--close-threshold", type=float, default=0.0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dump_dir, "episode_*.npz")))
    if not files:
        raise FileNotFoundError(f"no episode_*.npz under {args.dump_dir}")
    episodes = [analyze_file(path, args.close_threshold) for path in files]
    result = {
        "config": {
            "dump_dir": args.dump_dir,
            "close_threshold": args.close_threshold,
            "joint_dims": JOINT_DIMS.tolist(),
            "gripper_dims": GRIPPER_DIMS.tolist(),
        },
        "aggregate": aggregate(episodes),
        "episodes": episodes,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["aggregate"], indent=2), flush=True)
    for ep in episodes:
        target_l1 = ep.get("post_to_target_joint_l1", {}).get("mean", None)
        ratio = ep.get("actual_to_cmd_joint_delta_ratio", {}).get("mean", None)
        print(
            f"{ep['file']} seed={ep['seed']} success={ep['success']} steps={ep['steps']} "
            f"target_l1={target_l1} ratio={ratio} "
            f"first_close=({ep['left_first_close_step']},{ep['right_first_close_step']})",
            flush=True,
        )


if __name__ == "__main__":
    main()
