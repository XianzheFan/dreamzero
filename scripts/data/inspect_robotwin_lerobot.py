"""Inspect a converted RoboTwin LeRobot v2 dataset before DreamZero training.

The check is intentionally lightweight and reads only metadata plus parquet
action/state rows. It verifies the bimanual 16-dim layout and prints raw
gripper statistics so OSMO training logs prove that the run used the intended
post-training data and that gripper commands are present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required metadata file: {path}")
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required metadata file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_int_list(value: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims:
        raise ValueError("expected at least one integer dimension")
    if len(set(dims)) != len(dims):
        raise ValueError(f"duplicate dimensions: {dims}")
    return dims


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    if chunks_size <= 0:
        raise ValueError(f"chunks_size must be positive, got {chunks_size}")
    return episode_index // chunks_size


def _episode_data_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    data_path = info.get("data_path")
    if not isinstance(data_path, str):
        raise ValueError("meta/info.json is missing string data_path")
    chunks_size = int(info.get("chunks_size", 1000))
    rel = data_path.format(
        episode_chunk=_episode_chunk(episode_index, chunks_size),
        episode_index=episode_index,
    )
    return root / rel


def _stack_column(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df:
        raise ValueError(f"parquet file is missing column {column!r}")
    values = df[column].to_numpy()
    if len(values) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack([np.asarray(value, dtype=np.float32) for value in values], axis=0)


def _feature_dim(info: dict[str, Any], key: str) -> int:
    try:
        shape = info["features"][key]["shape"]
    except KeyError as exc:
        raise ValueError(f"meta/info.json is missing feature {key!r}") from exc
    if not isinstance(shape, list) or len(shape) != 1:
        raise ValueError(f"feature {key!r} must have 1-D shape, got {shape!r}")
    return int(shape[0])


def _stats_vector(stats: dict[str, Any], key: str, stat_name: str, expected_dim: int) -> np.ndarray:
    try:
        values = stats[key][stat_name]
    except KeyError as exc:
        raise ValueError(f"meta/stats.json is missing {key}.{stat_name}") from exc
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != (expected_dim,):
        raise ValueError(
            f"meta/stats.json {key}.{stat_name} shape {arr.shape} "
            f"does not match expected dim {(expected_dim,)}"
        )
    return arr


def _normalize_q99_value(value: float, q01: float, q99: float) -> float | None:
    if q01 == q99:
        return None
    return float(np.clip(2.0 * (value - q01) / (q99 - q01) - 1.0, -1.0, 1.0))


def _assert_modality_layout(modality: dict[str, Any]) -> None:
    expected = {
        ("action", "panda0_joint_pos"): (0, 7),
        ("action", "panda0_gripper_pos"): (7, 8),
        ("action", "panda1_joint_pos"): (8, 15),
        ("action", "panda1_gripper_pos"): (15, 16),
        ("state", "panda0_joint_pos"): (0, 7),
        ("state", "panda0_gripper_pos"): (7, 8),
        ("state", "panda1_joint_pos"): (8, 15),
        ("state", "panda1_gripper_pos"): (15, 16),
    }
    for (section, name), (start, end) in expected.items():
        try:
            entry = modality[section][name]
        except KeyError as exc:
            raise ValueError(f"meta/modality.json is missing {section}.{name}") from exc
        got = (int(entry.get("start")), int(entry.get("end")))
        if got != (start, end):
            raise ValueError(
                f"unexpected {section}.{name} layout: got {got}, expected {(start, end)}"
            )


def inspect_dataset(
    root: Path,
    expected_episodes: int | None,
    expected_action_dim: int,
    expected_state_dim: int,
    gripper_dims: tuple[int, ...],
    close_threshold: float,
    gripper_min: float,
    gripper_max: float,
    gripper_range_epsilon: float,
    sample_episodes: int,
    allow_static_gripper: bool,
    expected_embodiment_tag: str = "robotwin",
    legacy_embodiment_tags: tuple[str, ...] = ("robofactory",),
) -> dict[str, Any]:
    root = root.resolve()
    info = _load_json(root / "meta/info.json")
    modality = _load_json(root / "meta/modality.json")
    embodiment = _load_json(root / "meta/embodiment.json")
    stats = _load_json(root / "meta/stats.json")
    episodes = _load_jsonl(root / "meta/episodes.jsonl")
    _assert_modality_layout(modality)

    embodiment_tag = str(embodiment.get("embodiment_tag", ""))
    allowed_tags = {expected_embodiment_tag, *legacy_embodiment_tags}
    if embodiment_tag not in allowed_tags:
        raise ValueError(
            f"unexpected embodiment_tag {embodiment_tag!r}; expected "
            f"{expected_embodiment_tag!r}"
            + (
                f" or legacy {sorted(legacy_embodiment_tags)!r}"
                if legacy_embodiment_tags
                else ""
            )
        )

    if expected_episodes is not None and len(episodes) < expected_episodes:
        raise ValueError(
            f"dataset has {len(episodes)} episodes, expected at least {expected_episodes}"
        )

    action_dim = _feature_dim(info, "action")
    state_dim = _feature_dim(info, "observation.state")
    if action_dim != expected_action_dim:
        raise ValueError(f"action dim mismatch: got {action_dim}, expected {expected_action_dim}")
    if state_dim != expected_state_dim:
        raise ValueError(f"state dim mismatch: got {state_dim}, expected {expected_state_dim}")
    action_q01 = _stats_vector(stats, "action", "q01", action_dim)
    action_q99 = _stats_vector(stats, "action", "q99", action_dim)

    bad_dims = [dim for dim in gripper_dims if dim < 0 or dim >= action_dim]
    if bad_dims:
        raise ValueError(f"gripper dims outside action dim {action_dim}: {bad_dims}")

    selected = episodes
    if sample_episodes > 0:
        selected = episodes[:sample_episodes]

    action_rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    frame_count = 0
    for episode in selected:
        episode_index = int(episode["episode_index"])
        parquet_path = _episode_data_path(root, info, episode_index)
        if not parquet_path.is_file():
            raise FileNotFoundError(f"missing episode parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        action = _stack_column(df, "action")
        state = _stack_column(df, "observation.state")
        if action.ndim != 2 or action.shape[1] != action_dim:
            raise ValueError(
                f"{parquet_path} action shape {action.shape} does not match dim {action_dim}"
            )
        if state.ndim != 2 or state.shape[1] != state_dim:
            raise ValueError(
                f"{parquet_path} state shape {state.shape} does not match dim {state_dim}"
            )
        action_rows.append(action)
        state_rows.append(state)
        frame_count += int(action.shape[0])

    if not action_rows:
        raise ValueError("dataset contains no selected episodes")

    actions = np.concatenate(action_rows, axis=0)
    states = np.concatenate(state_rows, axis=0)
    gripper_summary: dict[str, Any] = {}
    for dim in gripper_dims:
        values = actions[:, dim].astype(np.float32)
        rounded_unique = np.unique(np.round(values, 4))
        close_fraction = float(np.mean(values < close_threshold))
        value_min = float(values.min())
        value_max = float(values.max())
        if value_min < gripper_min - gripper_range_epsilon or value_max > gripper_max + gripper_range_epsilon:
            raise ValueError(
                f"gripper dim {dim} is outside expected Robotwin range "
                f"[{gripper_min}, {gripper_max}]: min={value_min}, max={value_max}"
            )
        q01 = float(action_q01[dim])
        q99 = float(action_q99[dim])
        summary = {
            "min": value_min,
            "max": value_max,
            "mean": float(values.mean()),
            "std": float(values.std()),
            "q01": q01,
            "q99": q99,
            "close_fraction_raw_lt_threshold": close_fraction,
            "normalized_close_threshold_q99": _normalize_q99_value(
                close_threshold,
                q01,
                q99,
            ),
            "normalized_min_q99": _normalize_q99_value(value_min, q01, q99),
            "normalized_max_q99": _normalize_q99_value(value_max, q01, q99),
            "unique_rounded": rounded_unique[:16].astype(float).tolist(),
            "unique_rounded_count": int(len(rounded_unique)),
        }
        if not allow_static_gripper and summary["unique_rounded_count"] < 2:
            raise ValueError(f"gripper dim {dim} is static: {summary}")
        if not allow_static_gripper and (close_fraction <= 0.0 or close_fraction >= 1.0):
            raise ValueError(f"gripper dim {dim} has no open/close mix: {summary}")
        gripper_summary[f"dim_{dim}"] = summary

    state_gripper_summary = {
        f"dim_{dim}": {
            "min": float(states[:, dim].min()),
            "max": float(states[:, dim].max()),
            "mean": float(states[:, dim].mean()),
            "std": float(states[:, dim].std()),
        }
        for dim in gripper_dims
    }

    return {
        "root": str(root),
        "episodes": len(episodes),
        "selected_episodes": len(selected),
        "total_frames_meta": int(info.get("total_frames", -1)),
        "frames_read": frame_count,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "embodiment_tag": embodiment_tag,
        "expected_embodiment_tag": expected_embodiment_tag,
        "legacy_embodiment_tags": list(legacy_embodiment_tags),
        "gripper_dims": list(gripper_dims),
        "raw_close_threshold": close_threshold,
        "expected_gripper_range": [gripper_min, gripper_max],
        "gripper_range_epsilon": gripper_range_epsilon,
        "action_gripper": gripper_summary,
        "state_gripper": state_gripper_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument("--expected-action-dim", type=int, default=16)
    parser.add_argument("--expected-state-dim", type=int, default=16)
    parser.add_argument("--gripper-dims", type=_parse_int_list, default=(7, 15))
    parser.add_argument("--close-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=1.0)
    parser.add_argument("--gripper-range-epsilon", type=float, default=1e-4)
    parser.add_argument(
        "--sample-episodes",
        type=int,
        default=0,
        help="Read only the first N episodes; 0 reads all episodes.",
    )
    parser.add_argument("--allow-static-gripper", action="store_true")
    parser.add_argument("--expected-embodiment-tag", default="robotwin")
    parser.add_argument("--legacy-embodiment-tags", default="robofactory")
    args = parser.parse_args()

    summary = inspect_dataset(
        root=args.root,
        expected_episodes=args.expected_episodes,
        expected_action_dim=args.expected_action_dim,
        expected_state_dim=args.expected_state_dim,
        gripper_dims=args.gripper_dims,
        close_threshold=args.close_threshold,
        gripper_min=args.gripper_min,
        gripper_max=args.gripper_max,
        gripper_range_epsilon=args.gripper_range_epsilon,
        sample_episodes=args.sample_episodes,
        allow_static_gripper=args.allow_static_gripper,
        expected_embodiment_tag=args.expected_embodiment_tag,
        legacy_embodiment_tags=tuple(
            part.strip() for part in args.legacy_embodiment_tags.split(",") if part.strip()
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
