"""Trace a RoboFactory motion-planning expert rollout.

This is a companion to ``eval_robofactory_ws.py --dump-actions``. It runs the
task's built-in RoboFactory motion-planning solution while wrapping
``env.reset`` and ``env.step`` so the same barrier/TCP/grasp trace can be
compared against model closed-loop rollouts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.eval_robofactory_ws import ENV_TRACE_COLUMNS, collect_env_trace


def _flatten_bimanual_action(action: Any) -> np.ndarray | None:
    if action is None:
        return None
    if isinstance(action, dict):
        if "panda-0" not in action or "panda-1" not in action:
            return None
        left = np.asarray(action["panda-0"], dtype=np.float32).reshape(-1)
        right = np.asarray(action["panda-1"], dtype=np.float32).reshape(-1)
        if left.size < 8 or right.size < 8:
            return None
        return np.concatenate([left[:8], right[:8]]).astype(np.float32)

    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    arr = arr.reshape(-1)
    if arr.size < 16:
        return None
    return arr[:16].astype(np.float32)


def _trace_col(trace: np.ndarray, name: str) -> np.ndarray:
    return trace[:, int(np.where(ENV_TRACE_COLUMNS == name)[0][0])]


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _first_true_step(trace: np.ndarray, name: str) -> int | None:
    steps = _trace_col(trace, "step")
    values = _trace_col(trace, name)
    idx = np.where(np.isfinite(values) & (values > 0.5))[0]
    if idx.size == 0:
        return None
    return int(steps[idx[0]])


def summarize_trace(trace: np.ndarray) -> dict[str, Any]:
    barrier_z = _finite(_trace_col(trace, "barrier_z"))
    margin = _finite(_trace_col(trace, "success_margin"))
    left_dist = _finite(_trace_col(trace, "left_tcp_to_barrier"))
    right_dist = _finite(_trace_col(trace, "right_tcp_to_barrier"))
    left_target_dist = _finite(_trace_col(trace, "left_tcp_to_grasp_target"))
    right_target_dist = _finite(_trace_col(trace, "right_tcp_to_grasp_target"))
    left_grasp = _trace_col(trace, "left_grasping")
    right_grasp = _trace_col(trace, "right_grasping")

    def start_final_min_max(values: np.ndarray) -> dict[str, float | None]:
        if values.size == 0:
            return {"start": None, "final": None, "min": None, "max": None}
        return {
            "start": float(values[0]),
            "final": float(values[-1]),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    def min_final(values: np.ndarray) -> dict[str, float | None]:
        if values.size == 0:
            return {"min": None, "final": None}
        return {"min": float(values.min()), "final": float(values[-1])}

    return {
        "num_rows": int(trace.shape[0]),
        "num_steps": int(max(trace.shape[0] - 1, 0)),
        "barrier_z": start_final_min_max(barrier_z),
        "success_margin": start_final_min_max(margin),
        "left_tcp_to_barrier": min_final(left_dist),
        "right_tcp_to_barrier": min_final(right_dist),
        "left_tcp_to_grasp_target": min_final(left_target_dist),
        "right_tcp_to_grasp_target": min_final(right_target_dist),
        "left_grasp_count": int(np.sum(np.isfinite(left_grasp) & (left_grasp > 0.5))),
        "right_grasp_count": int(np.sum(np.isfinite(right_grasp) & (right_grasp > 0.5))),
        "left_first_grasp_step": _first_true_step(trace, "left_grasping"),
        "right_first_grasp_step": _first_true_step(trace, "right_grasping"),
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "nan" if value is None else f"{value:.{digits}f}"


def print_summary(summary: dict[str, Any]) -> None:
    bz = summary["barrier_z"]
    margin = summary["success_margin"]
    left = summary["left_tcp_to_barrier"]
    right = summary["right_tcp_to_barrier"]
    left_target = summary["left_tcp_to_grasp_target"]
    right_target = summary["right_tcp_to_grasp_target"]
    print(
        "expert barrier z: "
        f"start={_fmt(bz['start'])} final={_fmt(bz['final'])} "
        f"min={_fmt(bz['min'])} max={_fmt(bz['max'])}",
        flush=True,
    )
    print(
        "expert success margin: "
        f"start={_fmt(margin['start'])} final={_fmt(margin['final'])} "
        f"min={_fmt(margin['min'])} max={_fmt(margin['max'])}",
        flush=True,
    )
    print(
        "expert tcp->barrier: "
        f"left min={_fmt(left['min'])} final={_fmt(left['final'])} | "
        f"right min={_fmt(right['min'])} final={_fmt(right['final'])}",
        flush=True,
    )
    print(
        "expert grasping: "
        f"left_count={summary['left_grasp_count']} "
        f"left_first={summary['left_first_grasp_step']} "
        f"right_count={summary['right_grasp_count']} "
        f"right_first={summary['right_first_grasp_step']}",
        flush=True,
    )
    print(
        "expert tcp->grasp target: "
        f"left min={_fmt(left_target['min'])} final={_fmt(left_target['final'])} | "
        f"right min={_fmt(right_target['min'])} final={_fmt(right_target['final'])}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="LiftBarrier-rf")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", default=1000, type=int)
    ap.add_argument("--output-dir", default="/tmp/robofactory_expert_trace")
    ap.add_argument("--sim-backend", default="cpu")
    ap.add_argument("--shader", default="default")
    ap.add_argument("--obs-mode", default="none")
    ap.add_argument("--render-mode", default="rgb_array")
    args = ap.parse_args()

    import gymnasium as gym
    from robofactory import CONFIG_DIR
    from robofactory.planner.run import MP_SOLUTIONS

    if args.config is None:
        args.config = os.path.join(CONFIG_DIR, "table", "lift_barrier.yaml")
    if args.task not in MP_SOLUTIONS:
        raise ValueError(f"no RoboFactory motion-planning solution for {args.task}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Building expert env task={args.task} seed={args.seed}", flush=True)
    env = gym.make(
        args.task,
        config=args.config,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        reward_mode="dense",
        num_envs=1,
        parallel_in_single_scene=False,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend,
    )

    trace_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    step_count = 0
    original_reset = env.reset
    original_step = env.step

    def traced_reset(*reset_args, **reset_kwargs):
        nonlocal step_count
        result = original_reset(*reset_args, **reset_kwargs)
        step_count = 0
        trace_rows.append(collect_env_trace(env, 0, None, None))
        return result

    def traced_step(action):
        nonlocal step_count
        result = original_step(action)
        step_count += 1
        action16 = _flatten_bimanual_action(action)
        if action16 is not None:
            action_rows.append(action16.copy())
        _, _, _, _, info = result
        trace_rows.append(collect_env_trace(env, step_count, action16, info))
        return result

    env.reset = traced_reset
    env.step = traced_step

    # RoboFactory's RFSceneBuilder samples object poses with global np.random,
    # while env.reset(seed=...) only seeds ManiSkill's episode RNG.
    np.random.seed(args.seed)
    t0 = time.time()
    result = MP_SOLUTIONS[args.task](env, seed=args.seed, debug=False, vis=False)
    wall_s = time.time() - t0
    success = False
    if result != -1:
        try:
            success = bool(result[-1]["success"].item())
        except Exception:
            success = bool(result[-1].get("success", False))
    trace = np.stack(trace_rows) if trace_rows else np.zeros((0, len(ENV_TRACE_COLUMNS)), dtype=np.float32)
    exec_action = (
        np.stack(action_rows)
        if action_rows
        else np.zeros((0, 16), dtype=np.float32)
    )
    summary = summarize_trace(trace)
    summary.update({"seed": int(args.seed), "success": bool(success), "wall_s": round(wall_s, 1)})
    print(f"expert seed={args.seed} success={success} steps={exec_action.shape[0]} wall={wall_s:.1f}s", flush=True)
    print_summary(summary)

    npz_path = os.path.join(args.output_dir, f"expert_trace_seed{args.seed}.npz")
    json_path = os.path.join(args.output_dir, f"expert_trace_seed{args.seed}.json")
    np.savez_compressed(
        npz_path,
        seed=int(args.seed),
        success=bool(success),
        env_trace=trace,
        env_trace_columns=ENV_TRACE_COLUMNS,
        exec_action=exec_action,
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"wrote {npz_path}", flush=True)
    print(f"wrote {json_path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
