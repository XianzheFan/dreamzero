"""Compare LiftBarrier model action dumps against dataset action targets.

This diagnostic reads RoboFactory closed-loop ``action_dump/episode_*.npz``
artifacts and the dataset-side JSON emitted by
``scripts/data/inspect_robofactory_action_magnitude.py``. It reports p50/p95
model action magnitudes next to the dataset one-step and horizon target
distributions so under-commanding can be quantified without watching videos.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


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


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return float(num / den)


def _find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"no {pattern} under {root}")
    return matches[0]


def _variant_name(path: Path) -> str:
    if path.parent.name == "action_dump":
        return path.parent.parent.name
    return path.parent.name


def _replan_from_variant(name: str) -> int | None:
    match = re.search(r"_rp(\d+)_", name)
    return int(match.group(1)) if match else None


def _short_variant(name: str) -> str:
    pieces: list[str] = []
    for token in name.split("_"):
        if token.startswith("rp") or token in {"raw", "smooth"}:
            pieces.append(token)
        elif token.startswith("jscale"):
            pieces.append(token)
        elif token.startswith("clip"):
            pieces.append(token)
    return "_".join(pieces) or name


def _joint_dims(action_dim: int, gripper_dims: tuple[int, ...]) -> list[int]:
    grippers = set(gripper_dims)
    return [idx for idx in range(action_dim) if idx not in grippers]


def collect_npz_stats(
    paths: list[Path],
    *,
    gripper_dims: tuple[int, ...] = (7, 15),
) -> dict[str, Any]:
    first_cmd_abs: list[np.ndarray] = []
    chunk_abs: list[np.ndarray] = []
    by_offset: list[list[np.ndarray]] | None = None
    exec_step_abs: list[np.ndarray] = []
    pred_step_abs: list[np.ndarray] = []
    episodes: list[str] = []
    chunk_len = 0
    action_dim = 0

    for path in paths:
        data = np.load(path, allow_pickle=True)
        if "pred_chunk" not in data or "obs_qpos" not in data:
            continue
        pred_chunk = np.asarray(data["pred_chunk"], dtype=np.float32)
        obs_qpos = np.asarray(data["obs_qpos"], dtype=np.float32)
        if pred_chunk.ndim != 3 or obs_qpos.ndim != 2:
            continue
        n_infer = min(pred_chunk.shape[0], obs_qpos.shape[0])
        if n_infer == 0:
            continue
        pred_chunk = pred_chunk[:n_infer]
        obs_qpos = obs_qpos[:n_infer]
        action_dim = int(pred_chunk.shape[-1])
        joint_dims = _joint_dims(action_dim, gripper_dims)
        current = obs_qpos[:, joint_dims]
        pred_joints = pred_chunk[:, :, joint_dims]
        abs_chunk = np.abs(pred_joints - current[:, None, :])

        first_cmd_abs.append(abs_chunk[:, 0, :])
        chunk_abs.append(abs_chunk.reshape(-1, len(joint_dims)))
        if by_offset is None:
            by_offset = [[] for _ in range(pred_chunk.shape[1])]
        for offset in range(min(len(by_offset), pred_chunk.shape[1])):
            by_offset[offset].append(abs_chunk[:, offset, :].reshape(-1))
        if pred_chunk.shape[1] > 1:
            pred_step_abs.append(
                np.abs(np.diff(pred_joints, axis=1)).reshape(-1, len(joint_dims))
            )
        if "exec_action" in data:
            exec_action = np.asarray(data["exec_action"], dtype=np.float32)
            if (
                exec_action.ndim == 2
                and exec_action.shape[0] > 1
                and exec_action.shape[1] >= action_dim
            ):
                exec_step_abs.append(np.abs(np.diff(exec_action[:, joint_dims], axis=0)))
        episodes.append(str(path))
        chunk_len = int(pred_chunk.shape[1])

    if not episodes:
        raise ValueError("no usable action dump npz files")

    offset_summaries: list[dict[str, Any]] = []
    if by_offset is not None:
        for offset, arrays in enumerate(by_offset):
            values = np.concatenate(arrays, axis=0) if arrays else np.asarray([])
            item = _summary(values)
            item["offset"] = offset
            offset_summaries.append(item)

    return {
        "episodes": len(episodes),
        "chunk_len": chunk_len,
        "action_dim": action_dim,
        "first_cmd_target_current_abs": _summary(np.concatenate(first_cmd_abs, axis=0)),
        "pred_chunk_target_current_abs": _summary(np.concatenate(chunk_abs, axis=0)),
        "pred_chunk_target_current_abs_by_offset": offset_summaries,
        "pred_chunk_step_abs": (
            _summary(np.concatenate(pred_step_abs, axis=0))
            if pred_step_abs
            else {"count": 0}
        ),
        "exec_step_abs": (
            _summary(np.concatenate(exec_step_abs, axis=0))
            if exec_step_abs
            else {"count": 0}
        ),
        "files": episodes,
    }


def _compare_to_dataset(model: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    data_one = dataset["one_step_target_current_abs"]
    data_horizon = dataset["horizon_target_current_abs"]
    data_offsets = dataset.get("horizon_target_current_abs_by_offset", [])
    model_offsets = model.get("pred_chunk_target_current_abs_by_offset", [])
    last_data = data_offsets[-1] if data_offsets else {}
    last_model = model_offsets[-1] if model_offsets else {}
    first = model["first_cmd_target_current_abs"]
    chunk = model["pred_chunk_target_current_abs"]
    return {
        "first_cmd_p50_over_data_one_step_p50": _ratio(first.get("p50"), data_one.get("p50")),
        "first_cmd_p95_over_data_one_step_p95": _ratio(first.get("p95"), data_one.get("p95")),
        "chunk_p50_over_data_horizon_p50": _ratio(chunk.get("p50"), data_horizon.get("p50")),
        "chunk_p95_over_data_horizon_p95": _ratio(chunk.get("p95"), data_horizon.get("p95")),
        "last_offset_p50_over_data_last_offset_p50": _ratio(
            last_model.get("p50"), last_data.get("p50")
        ),
        "last_offset_p95_over_data_last_offset_p95": _ratio(
            last_model.get("p95"), last_data.get("p95")
        ),
    }


def compare_model_data_action_magnitude(
    *,
    eval_root: Path,
    data_stats_root: Path,
    gripper_dims: tuple[int, ...] = (7, 15),
) -> dict[str, Any]:
    data_stats_path = _find_one(data_stats_root, "liftbarrier_action_magnitude.json")
    data_ref = json.loads(data_stats_path.read_text())

    grouped: dict[str, list[Path]] = {}
    for path in sorted(eval_root.rglob("episode_*.npz")):
        if path.parent.name != "action_dump":
            continue
        grouped.setdefault(_variant_name(path), []).append(path)
    if not grouped:
        raise FileNotFoundError(f"no action_dump episode npz files under {eval_root}")

    variants: list[dict[str, Any]] = []
    for name, paths in sorted(grouped.items()):
        stats = collect_npz_stats(paths, gripper_dims=gripper_dims)
        stats["variant"] = name
        stats["short_variant"] = _short_variant(name)
        stats["replan"] = _replan_from_variant(name)
        stats["comparison_to_dataset"] = _compare_to_dataset(stats, data_ref)
        variants.append(stats)

    return {
        "eval_root": str(eval_root),
        "data_stats_path": str(data_stats_path),
        "dataset": {
            "episodes_read": data_ref.get("episodes_read"),
            "one_step_target_current_abs": data_ref.get("one_step_target_current_abs"),
            "horizon_target_current_abs": data_ref.get("horizon_target_current_abs"),
            "horizon_target_current_abs_by_offset_last": (
                data_ref.get("horizon_target_current_abs_by_offset", [{}])[-1]
            ),
            "relative_stats_present": data_ref.get("relative_stats_present"),
        },
        "variants": variants,
    }


def format_summary(payload: dict[str, Any]) -> str:
    dataset = payload["dataset"]
    data_one = dataset["one_step_target_current_abs"]
    data_horizon = dataset["horizon_target_current_abs"]
    data_last = dataset["horizon_target_current_abs_by_offset_last"]
    lines = [
        f"data_stats_path={payload['data_stats_path']}",
        f"dataset_episodes={dataset.get('episodes_read')}",
        f"dataset_one_step_p50={data_one['p50']:.6f}",
        f"dataset_one_step_p95={data_one['p95']:.6f}",
        f"dataset_horizon_p50={data_horizon['p50']:.6f}",
        f"dataset_horizon_p95={data_horizon['p95']:.6f}",
        f"dataset_last_offset_p50={data_last['p50']:.6f}",
        f"dataset_last_offset_p95={data_last['p95']:.6f}",
        "",
        "\t".join(
            [
                "variant",
                "replan",
                "episodes",
                "first_p50",
                "first_p95",
                "chunk_p50",
                "chunk_p95",
                "last_p50",
                "last_p95",
                "exec_step_p50",
                "exec_step_p95",
                "chunk_p95_over_data",
                "last_p95_over_data",
            ]
        ),
    ]
    for variant in payload["variants"]:
        first = variant["first_cmd_target_current_abs"]
        chunk = variant["pred_chunk_target_current_abs"]
        offsets = variant["pred_chunk_target_current_abs_by_offset"]
        last = offsets[-1] if offsets else {}
        exec_step = variant["exec_step_abs"]
        comp = variant["comparison_to_dataset"]
        lines.append(
            "\t".join(
                [
                    variant["short_variant"],
                    str(variant.get("replan")),
                    str(variant["episodes"]),
                    f"{first.get('p50', float('nan')):.6f}",
                    f"{first.get('p95', float('nan')):.6f}",
                    f"{chunk.get('p50', float('nan')):.6f}",
                    f"{chunk.get('p95', float('nan')):.6f}",
                    f"{last.get('p50', float('nan')):.6f}",
                    f"{last.get('p95', float('nan')):.6f}",
                    f"{exec_step.get('p50', float('nan')):.6f}",
                    f"{exec_step.get('p95', float('nan')):.6f}",
                    f"{(comp.get('chunk_p95_over_data_horizon_p95') or float('nan')):.3f}",
                    f"{(comp.get('last_offset_p95_over_data_last_offset_p95') or float('nan')):.3f}",
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _parse_gripper_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims:
        raise ValueError("expected at least one gripper dimension")
    return dims


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--data-stats-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--gripper-dims", type=_parse_gripper_dims, default=(7, 15))
    args = parser.parse_args()

    payload = compare_model_data_action_magnitude(
        eval_root=args.eval_root,
        data_stats_root=args.data_stats_root,
        gripper_dims=args.gripper_dims,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.summary.write_text(format_summary(payload))
    print("=== summary.txt ===")
    print(args.summary.read_text())


if __name__ == "__main__":
    main()
