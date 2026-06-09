"""Replay LeRobot RoboFactory expert actions in the ManiSkill env.

This is an action-interface diagnostic, not a policy eval. It reads the
converted LeRobot parquet episodes, sends the stored absolute action targets
directly to RoboFactory's ``pd_joint_pos`` controller, and reports:

* reset qpos vs dataset state[0] alignment
* post-step qpos vs commanded joint target tracking
* post-step qpos vs dataset next state drift, when reset alignment is usable
* gripper close timing and task success

If reset alignment is poor, task success is not interpretable, because the
dataset action sequence is being replayed from a different initial state. The
controller tracking metrics remain useful either way.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robofactory.tasks import *  # noqa: F401,F403


ARM_DIM = 8
JOINT_DIMS = np.asarray([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14], dtype=np.int64)
GRIPPER_DIMS = np.asarray([7, 15], dtype=np.int64)


def _to_np(arr):
    if hasattr(arr, "cpu"):
        arr = arr.cpu().numpy()
    return np.asarray(arr).reshape(-1)


def extract_qpos(obs) -> np.ndarray:
    q0 = _to_np(obs["agent"]["panda-0"]["qpos"]).astype(np.float32)
    q1 = _to_np(obs["agent"]["panda-1"]["qpos"]).astype(np.float32)
    qpos = np.concatenate([q0[:ARM_DIM], q1[:ARM_DIM]]).astype(np.float32)
    if qpos.shape != (16,):
        raise ValueError(f"expected 16-D qpos, got {qpos.shape}")
    return qpos


def env_action_dict(action16: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "panda-0": np.asarray(action16[0:8], dtype=np.float32),
        "panda-1": np.asarray(action16[8:16], dtype=np.float32),
    }


def _bool_from(info_val) -> bool:
    if info_val is None:
        return False
    if hasattr(info_val, "item"):
        return bool(info_val.item())
    if hasattr(info_val, "any"):
        return bool(np.any(info_val))
    return bool(info_val)


def _stack_column(values: pd.Series, name: str) -> np.ndarray:
    arr = np.stack([np.asarray(v, dtype=np.float32) for v in values.to_list()])
    if arr.ndim != 2 or arr.shape[1] != 16:
        raise ValueError(f"{name} must be [T, 16], got {arr.shape}")
    return arr


def load_episode(root: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
    chunk = episode_index // 1000
    path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    return _stack_column(df["action"], "action"), _stack_column(
        df["observation.state"], "observation.state"
    )


def _safe_stats(values: list[float] | np.ndarray) -> dict[str, float]:
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


def replay_episode(
    env,
    root: Path,
    episode_index: int,
    seed: int | None,
    max_steps: int,
) -> dict[str, Any]:
    actions, states = load_episode(root, episode_index)
    raw_obs, _ = env.reset(seed=seed)
    reset_qpos = extract_qpos(raw_obs)
    reset_err = np.abs(reset_qpos - states[0])

    n_steps = min(max_steps, actions.shape[0])
    joint_target_err: list[float] = []
    joint_target_err_max: list[float] = []
    state_next_err: list[float] = []
    state_next_joint_err: list[float] = []
    cmd_delta_l1: list[float] = []
    actual_delta_l1: list[float] = []
    movement_ratio: list[float] = []
    success = False
    first_success_step = None
    final_qpos = reset_qpos.copy()

    for t in range(n_steps):
        pre_qpos = extract_qpos(raw_obs)
        action = actions[t].astype(np.float32, copy=False)
        raw_obs, reward, term, trunc, info = env.step(env_action_dict(action))
        post_qpos = extract_qpos(raw_obs)
        final_qpos = post_qpos

        target_err = np.abs(post_qpos[JOINT_DIMS] - action[JOINT_DIMS])
        joint_target_err.append(float(target_err.mean()))
        joint_target_err_max.append(float(target_err.max()))

        if t + 1 < states.shape[0]:
            next_err = np.abs(post_qpos - states[t + 1])
            state_next_err.append(float(next_err.mean()))
            state_next_joint_err.append(float(next_err[JOINT_DIMS].mean()))

        cmd = action[JOINT_DIMS] - pre_qpos[JOINT_DIMS]
        actual = post_qpos[JOINT_DIMS] - pre_qpos[JOINT_DIMS]
        cmd_delta_l1.append(float(np.abs(cmd).mean()))
        actual_delta_l1.append(float(np.abs(actual).mean()))
        cmd_norm = float(np.linalg.norm(cmd))
        actual_norm = float(np.linalg.norm(actual))
        movement_ratio.append(actual_norm / max(cmd_norm, 1e-8))

        if _bool_from(info.get("success", False)):
            success = True
            first_success_step = t + 1
            break
        if _bool_from(term) or _bool_from(trunc):
            break

    close_mask = actions[:n_steps, GRIPPER_DIMS] < 0.0
    episode = {
        "episode_index": int(episode_index),
        "seed": seed,
        "num_dataset_steps": int(actions.shape[0]),
        "num_replayed_steps": int(len(joint_target_err)),
        "success": bool(success),
        "first_success_step": first_success_step,
        "reset_l1_mean": float(reset_err.mean()),
        "reset_l1_max": float(reset_err.max()),
        "reset_joint_l1_mean": float(reset_err[JOINT_DIMS].mean()),
        "final_qpos": final_qpos.tolist(),
        "joint_target_l1": _safe_stats(joint_target_err),
        "joint_target_linf": _safe_stats(joint_target_err_max),
        "state_next_l1": _safe_stats(state_next_err),
        "state_next_joint_l1": _safe_stats(state_next_joint_err),
        "cmd_joint_delta_l1": _safe_stats(cmd_delta_l1),
        "actual_joint_delta_l1": _safe_stats(actual_delta_l1),
        "actual_to_cmd_joint_delta_ratio": _safe_stats(movement_ratio),
        "left_first_close_step": _first_index(close_mask[:, 0]) if close_mask.size else None,
        "right_first_close_step": _first_index(close_mask[:, 1]) if close_mask.size else None,
        "left_frac_close": float(close_mask[:, 0].mean()) if close_mask.size else 0.0,
        "right_frac_close": float(close_mask[:, 1].mean()) if close_mask.size else 0.0,
    }
    return episode


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        return {"num_episodes": 0, "success_rate": 0.0}
    ok = sum(1 for ep in episodes if ep["success"])

    def collect(path: tuple[str, ...]) -> list[float]:
        out = []
        for ep in episodes:
            cur: Any = ep
            for key in path:
                cur = cur[key]
            out.append(float(cur))
        return out

    return {
        "num_episodes": len(episodes),
        "success_count": ok,
        "success_rate": ok / len(episodes),
        "reset_l1_mean": _safe_stats(collect(("reset_l1_mean",))),
        "reset_l1_max": _safe_stats(collect(("reset_l1_max",))),
        "joint_target_l1_mean": _safe_stats(collect(("joint_target_l1", "mean"))),
        "joint_target_linf_max": _safe_stats(collect(("joint_target_linf", "max"))),
        "state_next_l1_mean": _safe_stats(collect(("state_next_l1", "mean"))),
        "actual_to_cmd_ratio_mean": _safe_stats(
            collect(("actual_to_cmd_joint_delta_ratio", "mean"))
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--task", default="LiftBarrier-rf")
    ap.add_argument("--config", default=None)
    ap.add_argument("--episode-start", type=int, default=0)
    ap.add_argument("--num-episodes", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument(
        "--seed-mode",
        choices=("seed-start", "episode-index", "none"),
        default="seed-start",
    )
    ap.add_argument("--max-steps", type=int, default=180)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import gymnasium as gym
    from robofactory import CONFIG_DIR

    data_root = Path(args.data_root)
    if args.config is None:
        args.config = os.path.join(CONFIG_DIR, "table", "lift_barrier.yaml")

    print(f"Building env task={args.task} config={args.config}", flush=True)
    env = gym.make(
        args.task,
        config=args.config,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend="cpu",
        enable_shadow=True,
        parallel_in_single_scene=False,
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
    )
    print("Env ready.", flush=True)

    episodes: list[dict[str, Any]] = []
    for i in range(args.num_episodes):
        episode_index = args.episode_start + i
        if args.seed_mode == "seed-start":
            seed = args.seed_start + i
        elif args.seed_mode == "episode-index":
            seed = episode_index
        else:
            seed = None
        ep = replay_episode(env, data_root, episode_index, seed, args.max_steps)
        episodes.append(ep)
        print(
            "episode={episode_index} seed={seed} success={success} "
            "steps={num_replayed_steps} reset_l1={reset_l1_mean:.4f} "
            "target_l1={target_l1:.4f} state_next_l1={state_l1:.4f} "
            "ratio={ratio:.3f}".format(
                episode_index=ep["episode_index"],
                seed=ep["seed"],
                success=ep["success"],
                num_replayed_steps=ep["num_replayed_steps"],
                reset_l1_mean=ep["reset_l1_mean"],
                target_l1=ep["joint_target_l1"]["mean"],
                state_l1=ep["state_next_l1"]["mean"],
                ratio=ep["actual_to_cmd_joint_delta_ratio"]["mean"],
            ),
            flush=True,
        )

    result = {
        "config": {
            "data_root": str(data_root),
            "task": args.task,
            "config": args.config,
            "episode_start": args.episode_start,
            "num_episodes": args.num_episodes,
            "seed_start": args.seed_start,
            "seed_mode": args.seed_mode,
            "max_steps": args.max_steps,
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


if __name__ == "__main__":
    main()
