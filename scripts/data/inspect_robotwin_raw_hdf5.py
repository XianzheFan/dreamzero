"""Inspect raw RoboTwin HDF5 demonstrations before LeRobot conversion.

This validates the action source used by ``robotwin_to_lerobot_v2.py``:

    state[t]  = /joint_action at frame t
    action[t] = /joint_action at frame t + 1

RoboTwin's gripper convention is expected to be absolute ``0.0`` close and
``1.0`` open. The script is intentionally lightweight so it can run inside
OSMO data-preparation workflows before conversion starts.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ARM_DIMS = 7
STATE_DIM = 16
GRIPPER_DIMS = (7, 15)
JOINT_DIMS = tuple(dim for dim in range(STATE_DIM) if dim not in GRIPPER_DIMS)


def _episode_sort_key(path: Path) -> int:
    match = re.fullmatch(r"episode(\d+)\.hdf5", path.name)
    if not match:
        raise ValueError(f"unexpected episode filename: {path.name}")
    return int(match.group(1))


def list_episodes(episode_dir: Path) -> list[Path]:
    episodes = sorted(episode_dir.glob("episode*.hdf5"), key=_episode_sort_key)
    if not episodes:
        raise FileNotFoundError(f"no episode*.hdf5 files found under {episode_dir}")
    return episodes


def _as_column(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[1] != 1:
        raise ValueError(f"{name} must have shape [T] or [T, 1], got {arr.shape}")
    return arr


def read_joint_action_state(traj: h5py.File, *, require_vector: bool) -> np.ndarray:
    left_arm = np.asarray(traj["/joint_action/left_arm"][()], dtype=np.float32)
    right_arm = np.asarray(traj["/joint_action/right_arm"][()], dtype=np.float32)
    left_grip = _as_column(traj["/joint_action/left_gripper"][()], "left_gripper")
    right_grip = _as_column(traj["/joint_action/right_gripper"][()], "right_gripper")

    if left_arm.ndim != 2 or left_arm.shape[1] != ARM_DIMS:
        raise ValueError(f"left_arm must have shape [T, 7], got {left_arm.shape}")
    if right_arm.ndim != 2 or right_arm.shape[1] != ARM_DIMS:
        raise ValueError(f"right_arm must have shape [T, 7], got {right_arm.shape}")

    lengths = {left_arm.shape[0], right_arm.shape[0], left_grip.shape[0], right_grip.shape[0]}
    if len(lengths) != 1:
        raise ValueError(
            "joint_action component lengths differ: "
            f"left_arm={left_arm.shape[0]}, left_gripper={left_grip.shape[0]}, "
            f"right_arm={right_arm.shape[0]}, right_gripper={right_grip.shape[0]}"
        )

    state = np.concatenate([left_arm, left_grip, right_arm, right_grip], axis=1)
    if state.shape[1] != STATE_DIM:
        raise ValueError(f"concatenated joint_action state has shape {state.shape}")

    vector_path = "/joint_action/vector"
    if vector_path not in traj:
        if require_vector:
            raise ValueError(f"{vector_path} is missing")
        return state

    vector = np.asarray(traj[vector_path][()], dtype=np.float32)
    if vector.shape != state.shape:
        raise ValueError(
            f"{vector_path} shape {vector.shape} does not match named component "
            f"layout shape {state.shape}"
        )
    if not np.allclose(vector, state, atol=1e-5, rtol=1e-5):
        max_abs_diff = float(np.max(np.abs(vector - state)))
        raise ValueError(
            f"{vector_path} does not match [left_arm, left_gripper, right_arm, "
            f"right_gripper]; max_abs_diff={max_abs_diff:.6g}"
        )
    return state


def _value_summary(values: np.ndarray) -> dict[str, Any]:
    unique = np.unique(np.round(values.astype(np.float32), 4))
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "unique_rounded": unique[:16].astype(float).tolist(),
        "unique_rounded_count": int(len(unique)),
    }


def _assert_gripper_range(
    values: np.ndarray,
    dim: int,
    *,
    gripper_min: float,
    gripper_max: float,
    epsilon: float,
) -> None:
    value_min = float(values.min())
    value_max = float(values.max())
    if value_min < gripper_min - epsilon or value_max > gripper_max + epsilon:
        raise ValueError(
            f"gripper dim {dim} is outside expected range "
            f"[{gripper_min}, {gripper_max}]: min={value_min}, max={value_max}"
        )


def inspect_raw_episodes(
    episode_dir: Path,
    *,
    expected_episodes: int | None = None,
    sample_episodes: int = 5,
    gripper_dims: tuple[int, ...] = GRIPPER_DIMS,
    close_threshold: float = 0.5,
    gripper_min: float = 0.0,
    gripper_max: float = 1.0,
    gripper_range_epsilon: float = 1e-4,
    require_vector: bool = False,
    allow_static_gripper: bool = True,
) -> dict[str, Any]:
    episodes = list_episodes(episode_dir)
    if expected_episodes is not None and len(episodes) < expected_episodes:
        raise ValueError(
            f"raw dataset has {len(episodes)} episodes, expected at least {expected_episodes}"
        )

    if sample_episodes > 0:
        selected = episodes[:sample_episodes]
    else:
        selected = episodes

    observed_states: list[np.ndarray] = []
    implied_actions: list[np.ndarray] = []
    lengths: list[int] = []
    for path in selected:
        with h5py.File(path, "r") as traj:
            state_full = read_joint_action_state(traj, require_vector=require_vector)
        if state_full.shape[0] < 2:
            raise ValueError(f"{path} has fewer than 2 frames")
        observed_states.append(state_full[:-1])
        implied_actions.append(state_full[1:])
        lengths.append(int(state_full.shape[0] - 1))

    states = np.concatenate(observed_states, axis=0)
    actions = np.concatenate(implied_actions, axis=0)
    joint_delta = actions[:, JOINT_DIMS] - states[:, JOINT_DIMS]

    gripper_summary: dict[str, Any] = {}
    for dim in gripper_dims:
        if dim < 0 or dim >= STATE_DIM:
            raise ValueError(f"gripper dim {dim} outside state/action dim {STATE_DIM}")
        state_values = states[:, dim]
        action_values = actions[:, dim]
        _assert_gripper_range(
            state_values,
            dim,
            gripper_min=gripper_min,
            gripper_max=gripper_max,
            epsilon=gripper_range_epsilon,
        )
        _assert_gripper_range(
            action_values,
            dim,
            gripper_min=gripper_min,
            gripper_max=gripper_max,
            epsilon=gripper_range_epsilon,
        )
        action_summary = _value_summary(action_values)
        action_summary["close_fraction_raw_lt_threshold"] = float(
            np.mean(action_values < close_threshold)
        )
        action_summary["state_summary"] = _value_summary(state_values)
        if not allow_static_gripper and action_summary["unique_rounded_count"] < 2:
            raise ValueError(f"gripper dim {dim} is static: {action_summary}")
        gripper_summary[f"dim_{dim}"] = action_summary

    summary = {
        "episode_dir": str(episode_dir),
        "episode_count": int(len(episodes)),
        "sample_episode_count": int(len(selected)),
        "sample_frames_after_shift": int(actions.shape[0]),
        "sample_episode_lengths_after_shift": lengths,
        "state_action_layout": (
            "[left_arm_joint0..6, left_gripper, right_arm_joint0..6, right_gripper]"
        ),
        "implied_action": "absolute next-frame joint_action state",
        "gripper_convention": "0.0 close, 1.0 open",
        "gripper_summary": gripper_summary,
        "joint_delta_summary": {
            "abs_mean": float(np.mean(np.abs(joint_delta))),
            "abs_p95": float(np.quantile(np.abs(joint_delta), 0.95)),
            "abs_max": float(np.max(np.abs(joint_delta))),
        },
    }
    return summary


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims:
        raise ValueError("expected at least one gripper dim")
    return dims


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument("--sample-episodes", type=int, default=5)
    parser.add_argument("--gripper-dims", type=_parse_int_tuple, default=GRIPPER_DIMS)
    parser.add_argument("--close-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=1.0)
    parser.add_argument("--gripper-range-epsilon", type=float, default=1e-4)
    parser.add_argument("--require-vector", action="store_true")
    parser.add_argument("--allow-static-gripper", action="store_true")
    args = parser.parse_args()

    summary = inspect_raw_episodes(
        args.episode_dir,
        expected_episodes=args.expected_episodes,
        sample_episodes=args.sample_episodes,
        gripper_dims=args.gripper_dims,
        close_threshold=args.close_threshold,
        gripper_min=args.gripper_min,
        gripper_max=args.gripper_max,
        gripper_range_epsilon=args.gripper_range_epsilon,
        require_vector=args.require_vector,
        allow_static_gripper=args.allow_static_gripper,
    )
    print("RoboTwin raw HDF5 inspection passed:")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
