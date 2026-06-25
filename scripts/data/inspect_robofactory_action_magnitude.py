"""Inspect RoboFactory LeRobot action magnitude and normalization.

This is a diagnostic-only script. It reads a converted LeRobot v2 dataset and
reports the action magnitude distributions needed to diagnose under-commanding:

* absolute target-current joint deltas at each row;
* horizon target-current joint deltas, matching DreamZero's action chunks;
* optional q99 normalize->denormalize round-trip checks from metadata stats;
* gripper close timing distribution.

It does not modify the dataset.
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
        raise FileNotFoundError(f"missing required JSON file: {path}")
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def _parse_int_list(value: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims:
        raise ValueError("expected at least one integer dimension")
    if len(set(dims)) != len(dims):
        raise ValueError(f"duplicate dimensions: {dims}")
    return dims


def _summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def _per_dim_summary(values: np.ndarray, dims: list[int]) -> list[dict[str, Any]]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        arr = arr.reshape(-1, len(dims))
    rows: list[dict[str, Any]] = []
    for i, dim in enumerate(dims):
        item = _summary(arr[:, i])
        item["dim"] = int(dim)
        rows.append(item)
    return rows


def _derive_joint_spans(modality: dict[str, Any], gripper_dims: tuple[int, ...]) -> list[tuple[str, int, int]]:
    action_meta = modality.get("action", {})
    spans: list[tuple[str, int, int]] = []
    for name, meta in action_meta.items():
        if "joint" not in name:
            continue
        start = int(meta["start"])
        end = int(meta["end"])
        spans.append((name, start, end))
    if spans:
        return sorted(spans, key=lambda x: x[1])

    # Fallback for datasets without split modality names.
    action_dim = max(int(meta["end"]) for meta in action_meta.values())
    grippers = set(gripper_dims)
    dims = [dim for dim in range(action_dim) if dim not in grippers]
    grouped: list[tuple[str, int, int]] = []
    start = None
    prev = None
    for dim in dims:
        if start is None:
            start = dim
        elif prev is not None and dim != prev + 1:
            grouped.append((f"joint_{start}_{prev + 1}", start, prev + 1))
            start = dim
        prev = dim
    if start is not None and prev is not None:
        grouped.append((f"joint_{start}_{prev + 1}", start, prev + 1))
    return grouped


def _spans_to_dims(spans: list[tuple[str, int, int]]) -> list[int]:
    dims: list[int] = []
    for _, start, end in spans:
        dims.extend(range(start, end))
    return dims


def _normalize_q99(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    q01 = np.asarray(q01, dtype=np.float64)
    q99 = np.asarray(q99, dtype=np.float64)
    denom = q99 - q01
    mask = denom != 0
    normalized = np.zeros_like(x, dtype=np.float64)
    normalized[..., mask] = 2.0 * (x[..., mask] - q01[mask]) / denom[mask] - 1.0
    normalized[..., ~mask] = x[..., ~mask]
    clipped = np.clip(normalized, -1.0, 1.0)
    return normalized, clipped


def _denormalize_q99(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    q01 = np.asarray(q01, dtype=np.float64)
    q99 = np.asarray(q99, dtype=np.float64)
    return (x + 1.0) / 2.0 * (q99 - q01) + q01


def _roundtrip_summary(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> dict[str, Any]:
    raw_norm, clipped_norm = _normalize_q99(x, q01, q99)
    recovered = _denormalize_q99(clipped_norm, q01, q99)
    err = np.abs(recovered - x)
    clipped = np.abs(raw_norm) > 1.0
    return {
        "values": _summary(np.abs(x)),
        "normalized_abs": _summary(np.abs(raw_norm)),
        "clip_fraction": float(np.mean(clipped)) if clipped.size else 0.0,
        "roundtrip_abs_error": _summary(err),
    }


def _stats_vector(stats: dict[str, Any], key: str, stat_name: str, dim: int) -> np.ndarray | None:
    try:
        values = stats[key][stat_name]
    except KeyError:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (dim,):
        raise ValueError(f"stats {key}.{stat_name} shape {arr.shape}, expected {(dim,)}")
    return arr


def _relative_stats_for_span(
    relative_stats: dict[str, Any],
    name: str,
    width: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    stats = relative_stats.get(name)
    if stats is None:
        return None
    q01 = np.asarray(stats.get("q01"), dtype=np.float64)
    q99 = np.asarray(stats.get("q99"), dtype=np.float64)
    if q01.shape != (width,) or q99.shape != (width,):
        raise ValueError(
            f"relative stats for {name} have q01/q99 shapes {q01.shape}/{q99.shape}, "
            f"expected {(width,)}"
        )
    return q01, q99


def inspect_action_magnitude(
    root: Path,
    horizon: int,
    sample_episodes: int,
    gripper_dims: tuple[int, ...],
    close_threshold: float,
) -> dict[str, Any]:
    root = root.resolve()
    info = _load_json(root / "meta/info.json")
    modality = _load_json(root / "meta/modality.json")
    stats = _load_json(root / "meta/stats.json")
    relative_stats_path = root / "meta/relative_stats_dreamzero.json"
    relative_stats = _load_json(relative_stats_path) if relative_stats_path.is_file() else {}
    episodes = _load_jsonl(root / "meta/episodes.jsonl")
    selected = episodes[:sample_episodes] if sample_episodes > 0 else episodes

    action_dim = int(info["features"]["action"]["shape"][0])
    state_dim = int(info["features"]["observation.state"]["shape"][0])
    joint_spans = _derive_joint_spans(modality, gripper_dims)
    joint_dims = _spans_to_dims(joint_spans)

    action_q01 = _stats_vector(stats, "action", "q01", action_dim)
    action_q99 = _stats_vector(stats, "action", "q99", action_dim)
    state_q01 = _stats_vector(stats, "observation.state", "q01", state_dim)
    state_q99 = _stats_vector(stats, "observation.state", "q99", state_dim)

    one_step_abs: list[np.ndarray] = []
    horizon_abs: list[np.ndarray] = []
    horizon_by_offset: list[list[np.ndarray]] = [[] for _ in range(horizon)]
    action_joint_values: list[np.ndarray] = []
    state_joint_values: list[np.ndarray] = []
    action_all_values: list[np.ndarray] = []
    state_all_values: list[np.ndarray] = []
    gripper_values: list[np.ndarray] = []
    first_close_offsets: list[int] = []

    relative_by_span: dict[str, list[np.ndarray]] = {name: [] for name, _, _ in joint_spans}

    for episode in selected:
        episode_index = int(episode["episode_index"])
        parquet_path = _episode_data_path(root, info, episode_index)
        if not parquet_path.is_file():
            raise FileNotFoundError(f"missing episode parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        action = _stack_column(df, "action")
        state = _stack_column(df, "observation.state")

        action_all_values.append(action)
        state_all_values.append(state)
        action_joint_values.append(action[:, joint_dims])
        state_joint_values.append(state[:, joint_dims])
        gripper_values.append(action[:, list(gripper_dims)])

        delta = action[:, joint_dims] - state[:, joint_dims]
        one_step_abs.append(np.abs(delta))

        close_mask = np.any(action[:, list(gripper_dims)] < close_threshold, axis=1)
        close_indices = np.flatnonzero(close_mask)
        if close_indices.size:
            first_close_offsets.append(int(close_indices[0]))

        usable = max(0, len(action) - horizon)
        for start in range(usable):
            ref = state[start, joint_dims]
            chunk = action[start : start + horizon, joint_dims] - ref
            abs_chunk = np.abs(chunk)
            horizon_abs.append(abs_chunk)
            for offset in range(horizon):
                horizon_by_offset[offset].append(abs_chunk[offset])

        for name, start_dim, end_dim in joint_spans:
            usable = max(0, len(action) - horizon)
            for start in range(usable):
                ref = state[start, start_dim:end_dim]
                chunk = action[start : start + horizon, start_dim:end_dim] - ref
                relative_by_span[name].append(chunk)

    one_step = np.concatenate(one_step_abs, axis=0) if one_step_abs else np.zeros((0, len(joint_dims)))
    horizon_values = (
        np.concatenate([x.reshape(-1, len(joint_dims)) for x in horizon_abs], axis=0)
        if horizon_abs
        else np.zeros((0, len(joint_dims)))
    )
    action_joints = np.concatenate(action_joint_values, axis=0)
    state_joints = np.concatenate(state_joint_values, axis=0)
    action_all = np.concatenate(action_all_values, axis=0)
    state_all = np.concatenate(state_all_values, axis=0)
    grippers = np.concatenate(gripper_values, axis=0)

    per_offset = []
    for offset, chunks in enumerate(horizon_by_offset):
        values = np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.float32)
        item = _summary(values)
        item["offset"] = offset
        per_offset.append(item)

    absolute_roundtrip = None
    if action_q01 is not None and action_q99 is not None:
        absolute_roundtrip = _roundtrip_summary(action_all, action_q01, action_q99)

    state_roundtrip = None
    if state_q01 is not None and state_q99 is not None:
        state_roundtrip = _roundtrip_summary(state_all, state_q01, state_q99)

    relative_roundtrip: dict[str, Any] = {}
    for name, arrays in relative_by_span.items():
        if not arrays:
            continue
        values = np.concatenate([x.reshape(-1, x.shape[-1]) for x in arrays], axis=0)
        rel_stats = _relative_stats_for_span(relative_stats, name, values.shape[-1])
        entry: dict[str, Any] = {
            "target_current_abs": _summary(np.abs(values)),
            "target_current_abs_per_dim": _per_dim_summary(
                np.abs(values),
                list(range(values.shape[-1])),
            ),
        }
        if rel_stats is not None:
            q01, q99 = rel_stats
            entry["q99_roundtrip"] = _roundtrip_summary(values, q01, q99)
        else:
            entry["q99_roundtrip"] = None
        relative_roundtrip[name] = entry

    return {
        "root": str(root),
        "episodes_total": len(episodes),
        "episodes_read": len(selected),
        "horizon": horizon,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "joint_spans": [
            {"name": name, "start": start, "end": end} for name, start, end in joint_spans
        ],
        "joint_dims": joint_dims,
        "gripper_dims": list(gripper_dims),
        "one_step_target_current_abs": _summary(one_step),
        "one_step_target_current_abs_per_dim": _per_dim_summary(one_step, joint_dims),
        "horizon_target_current_abs": _summary(horizon_values),
        "horizon_target_current_abs_per_dim": _per_dim_summary(horizon_values, joint_dims),
        "horizon_target_current_abs_by_offset": per_offset,
        "action_joint_abs_values": _summary(np.abs(action_joints)),
        "state_joint_abs_values": _summary(np.abs(state_joints)),
        "gripper": {
            "close_threshold": close_threshold,
            "close_fraction_any_arm": float(np.mean(np.any(grippers < close_threshold, axis=1))),
            "first_close_step": _summary(np.asarray(first_close_offsets, dtype=np.float64)),
            "values": _summary(grippers),
        },
        "absolute_action_q99_roundtrip": absolute_roundtrip,
        "state_q99_roundtrip": state_roundtrip,
        "relative_stats_path": str(relative_stats_path),
        "relative_stats_present": bool(relative_stats),
        "relative_action_q99_roundtrip": relative_roundtrip,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument(
        "--sample-episodes",
        type=int,
        default=0,
        help="Read only the first N episodes; 0 reads all episodes.",
    )
    parser.add_argument("--gripper-dims", type=_parse_int_list, default=(7, 15))
    parser.add_argument("--close-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the full JSON payload to stdout.",
    )
    args = parser.parse_args()

    summary = inspect_action_magnitude(
        root=args.root,
        horizon=args.horizon,
        sample_episodes=args.sample_episodes,
        gripper_dims=args.gripper_dims,
        close_threshold=args.close_threshold,
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    if not args.quiet:
        print(text)


if __name__ == "__main__":
    main()
