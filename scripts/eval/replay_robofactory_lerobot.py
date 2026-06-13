"""Replay LeRobot RoboFactory actions in the RoboFactory env.

This is a ground-truth control diagnostic for the DreamZero closed-loop
RoboFactory eval path. It reads the converted LeRobot parquet action rows and
sends them directly to RoboFactory's ``pd_joint_pos`` controller, using the
same env construction and trace utilities as ``eval_robofactory_ws.py``.

If this succeeds with low reset qpos mismatch, the LeRobot action/state layout,
joint range, gripper convention, and env action adapter are likely sane. If it
fails, the model is not the first thing to debug.
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
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.eval_robofactory_ws import (  # noqa: E402
    ENV_TRACE_COLUMNS,
    collect_env_trace,
    env_action_dict,
    extract_obs,
)
from scripts.eval.trace_robofactory_expert import print_summary, summarize_trace  # noqa: E402


def _candidate_data_roots(path: Path) -> list[Path]:
    return [
        path,
        path / "LiftBarrier-rf-500",
        path / "LiftBarrier-rf" / "LiftBarrier-rf-500",
    ]


def resolve_data_root(path: str | os.PathLike[str]) -> Path:
    base = Path(path)
    for candidate in _candidate_data_roots(base):
        if (candidate / "meta" / "info.json").exists() and (candidate / "data").exists():
            return candidate
    matches = sorted(base.glob("*/meta/info.json")) if base.exists() else []
    for meta_path in matches:
        candidate = meta_path.parent.parent
        if (candidate / "data").exists():
            return candidate
    raise FileNotFoundError(f"could not find LeRobot dataset root under {base}")


def episode_parquet_path(data_root: Path, episode_index: int) -> Path:
    chunk_idx = episode_index // 1000
    path = data_root / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_index:06d}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _stack_column(df: pd.DataFrame, name: str) -> np.ndarray:
    values = [np.asarray(v, dtype=np.float32).reshape(-1) for v in df[name].to_list()]
    return np.stack(values).astype(np.float32)


def load_episode(data_root: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(episode_parquet_path(data_root, episode_index))
    actions = _stack_column(df, "action")
    states = _stack_column(df, "observation.state")
    if actions.ndim != 2 or states.ndim != 2:
        raise ValueError(f"unexpected action/state shapes: {actions.shape}, {states.shape}")
    if actions.shape[1] < 16 or states.shape[1] < 16:
        raise ValueError(f"expected at least 16 dims for 2 arms, got {actions.shape[1]}, {states.shape[1]}")
    return actions[:, :16], states[:, :16]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_source_episode_seeds(data_root: Path) -> dict[int, int]:
    """Read source RoboFactory episode seeds from converted LeRobot metadata."""
    seeds: dict[int, int] = {}
    meta_dir = data_root / "meta"
    for row in _load_jsonl(meta_dir / "robofactory_episode_metadata.jsonl"):
        if "episode_index" in row and "episode_seed" in row:
            seeds[int(row["episode_index"])] = int(row["episode_seed"])
    for row in _load_jsonl(meta_dir / "episodes.jsonl"):
        if "episode_index" in row and "source_episode_seed" in row:
            seeds[int(row["episode_index"])] = int(row["source_episode_seed"])
    return seeds


def _bool_from(info_val: Any) -> bool:
    if info_val is None:
        return False
    if hasattr(info_val, "item"):
        return bool(info_val.item())
    if hasattr(info_val, "any"):
        return bool(np.any(info_val))
    return bool(info_val)


def _reset_qpos16(env, seed: int) -> tuple[Any, np.ndarray]:
    # RoboFactory's scene builder uses global np.random for object pose
    # randomization, so env.reset(seed=...) alone is not enough for replayable
    # object initial states.
    np.random.seed(seed)
    raw_obs, _ = env.reset(seed=seed)
    _, _, _, qpos16 = extract_obs(raw_obs)
    return raw_obs, qpos16.astype(np.float32)


def choose_seed(env, candidate_seed: int, dataset_state0: np.ndarray, search_window: int) -> tuple[Any, int, np.ndarray, float]:
    lo = max(0, candidate_seed - search_window)
    hi = candidate_seed + search_window
    best: tuple[Any, int, np.ndarray, float] | None = None
    for seed in range(lo, hi + 1):
        raw_obs, qpos16 = _reset_qpos16(env, seed)
        mismatch = float(np.mean(np.abs(qpos16 - dataset_state0[:16])))
        if best is None or mismatch < best[3]:
            best = (raw_obs, seed, qpos16, mismatch)
    assert best is not None
    # Reset again so replay starts from the selected seed after any search resets.
    raw_obs, qpos16 = _reset_qpos16(env, best[1])
    mismatch = float(np.mean(np.abs(qpos16 - dataset_state0[:16])))
    return raw_obs, best[1], qpos16, mismatch


def replay_episode(
    env,
    actions16: np.ndarray,
    states16: np.ndarray,
    episode_index: int,
    candidate_seed: int,
    seed_search_window: int,
    max_steps: int,
    dump_dir: Path | None,
) -> dict[str, Any]:
    raw_obs, seed, qpos0, reset_mismatch = choose_seed(
        env,
        candidate_seed=candidate_seed,
        dataset_state0=states16[0],
        search_window=seed_search_window,
    )
    trace_rows = [collect_env_trace(env, 0, None, None)]
    exec_rows: list[np.ndarray] = []
    qpos_rows = [qpos0.copy()]
    success = False
    steps = 0
    t0 = time.time()

    limit = min(max_steps, actions16.shape[0])
    for idx in range(limit):
        action16 = actions16[idx].astype(np.float32, copy=True)
        raw_obs, _reward, term, trunc, info = env.step(env_action_dict(action16))
        steps += 1
        exec_rows.append(action16.copy())
        _, _, _, qpos16 = extract_obs(raw_obs)
        qpos_rows.append(qpos16.copy())
        trace_rows.append(collect_env_trace(env, steps, action16, info))
        if _bool_from(info.get("success", False)):
            success = True
            break
        if _bool_from(term) or _bool_from(trunc):
            break

    wall_s = time.time() - t0
    trace = np.stack(trace_rows).astype(np.float32)
    exec_action = (
        np.stack(exec_rows).astype(np.float32)
        if exec_rows
        else np.zeros((0, 16), dtype=np.float32)
    )
    obs_qpos = np.stack(qpos_rows).astype(np.float32)
    summary = summarize_trace(trace)
    summary.update(
        {
            "episode_index": int(episode_index),
            "candidate_seed": int(candidate_seed),
            "seed": int(seed),
            "success": bool(success),
            "steps": int(steps),
            "wall_s": round(wall_s, 1),
            "reset_qpos_l1_mean": reset_mismatch,
            "reset_qpos_l1_max": float(np.max(np.abs(qpos0 - states16[0]))),
        }
    )

    print(
        "replay "
        f"episode={episode_index} candidate_seed={candidate_seed} seed={seed} "
        f"success={success} steps={steps} wall={wall_s:.1f}s "
        f"reset_qpos_l1_mean={reset_mismatch:.6f} "
        f"reset_qpos_l1_max={summary['reset_qpos_l1_max']:.6f}",
        flush=True,
    )
    print_summary(summary)

    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dump_dir / f"replay_episode_{episode_index:06d}_seed{seed}.npz",
            episode_index=int(episode_index),
            seed=int(seed),
            success=bool(success),
            dataset_action=actions16.astype(np.float32),
            dataset_state=states16.astype(np.float32),
            exec_action=exec_action,
            obs_qpos=obs_qpos,
            env_trace=trace,
            env_trace_columns=ENV_TRACE_COLUMNS,
        )
        with (dump_dir / f"replay_episode_{episode_index:06d}_seed{seed}.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--task", default="LiftBarrier-rf")
    ap.add_argument("--config", default=None)
    ap.add_argument("--episode-start", type=int, default=0)
    ap.add_argument("--num-episodes", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument(
        "--seed-mode",
        choices=("episode-index", "seed-start", "source-episode-seed"),
        default="episode-index",
        help=(
            "episode-index uses seed=episode_index; seed-start uses seed_start+i; "
            "source-episode-seed uses RoboFactory episode_seed metadata written by "
            "scripts/data/robofactory_to_lerobot_v2.py."
        ),
    )
    ap.add_argument(
        "--seed-search-window",
        type=int,
        default=0,
        help="Search +/- this many seeds around the candidate by reset qpos L1.",
    )
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--sim-backend", default="cpu")
    ap.add_argument("--shader", default="default")
    args = ap.parse_args()

    import gymnasium as gym
    from robofactory import CONFIG_DIR

    data_root = resolve_data_root(args.data_root)
    if args.config is None:
        args.config = os.path.join(CONFIG_DIR, "table", "lift_barrier.yaml")
    print(f"Data root: {data_root}", flush=True)
    print(f"Config:    {args.config}", flush=True)
    print(f"Task:      {args.task}", flush=True)
    print(
        f"Episodes:  {args.episode_start}..{args.episode_start + args.num_episodes - 1}",
        flush=True,
    )

    env = gym.make(
        args.task,
        config=args.config,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=False,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
    )
    dump_dir = Path(args.output_dir) if args.output_dir else None
    source_seeds = load_source_episode_seeds(data_root)
    if args.seed_mode == "source-episode-seed":
        print(f"Loaded source episode seeds for {len(source_seeds)} episodes", flush=True)
    summaries = []
    try:
        for offset in range(args.num_episodes):
            episode_index = args.episode_start + offset
            if args.seed_mode == "episode-index":
                candidate_seed = episode_index
            elif args.seed_mode == "seed-start":
                candidate_seed = args.seed_start + offset
            else:
                if episode_index not in source_seeds:
                    raise KeyError(
                        f"episode {episode_index} has no source_episode_seed metadata under {data_root / 'meta'}"
                    )
                candidate_seed = source_seeds[episode_index]
            actions16, states16 = load_episode(data_root, episode_index)
            summaries.append(
                replay_episode(
                    env,
                    actions16=actions16,
                    states16=states16,
                    episode_index=episode_index,
                    candidate_seed=candidate_seed,
                    seed_search_window=max(0, args.seed_search_window),
                    max_steps=args.max_steps,
                    dump_dir=dump_dir,
                )
            )
    finally:
        env.close()

    ok = sum(1 for row in summaries if row.get("success"))
    total = len(summaries)
    rate = ok / total if total else 0.0
    reset_l1 = [row["reset_qpos_l1_mean"] for row in summaries]
    print(f"\nGT replay total: {ok}/{total} = {rate * 100:.1f}%", flush=True)
    if reset_l1:
        print(
            "reset_qpos_l1_mean: "
            f"mean={np.mean(reset_l1):.6f} min={np.min(reset_l1):.6f} max={np.max(reset_l1):.6f}",
            flush=True,
        )
    if dump_dir is not None:
        with (dump_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump({"results": summaries, "success_rate": rate}, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
