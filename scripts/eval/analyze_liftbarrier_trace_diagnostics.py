"""Analyze LiftBarrier TCP target traces and gripper timing from eval dumps.

``eval_robofactory_ws.py --dump-actions`` writes ``episode_<seed>.npz`` files
with an ``env_trace`` array and matching ``env_trace_columns`` labels. This
diagnostic turns those traces into the checks needed for the LiftBarrier 0%
teacher failure:

* whether TCP-to-grasp-target distances decrease and then plateau;
* whether the gripper close command happens while either TCP is still far from
  its target;
* whether actual grasping ever happens after the close command.

It is intentionally offline/read-only. It does not change training, eval, or
policy inference.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


TARGET_COLS = {
    "left": "left_tcp_to_grasp_target",
    "right": "right_tcp_to_grasp_target",
}
CMD_COLS = {
    "left": "cmd_left_gripper",
    "right": "cmd_right_gripper",
}
GRASP_COLS = {
    "left": "left_grasping",
    "right": "right_grasping",
}


def _summary(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
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
        "max": float(np.max(arr)),
    }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _npz_string(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return ""


def _columns(data: np.lib.npyio.NpzFile) -> dict[str, int]:
    if "env_trace_columns" not in data:
        raise KeyError("missing env_trace_columns")
    labels = [str(x) for x in np.asarray(data["env_trace_columns"]).reshape(-1)]
    return {label: idx for idx, label in enumerate(labels)}


def _finite_series(trace: np.ndarray, columns: dict[str, int], key: str) -> tuple[np.ndarray, np.ndarray]:
    steps = trace[:, columns["step"]].astype(np.float64)
    values = trace[:, columns[key]].astype(np.float64)
    mask = np.isfinite(steps) & np.isfinite(values)
    return steps[mask], values[mask]


def _first_true_step(
    trace: np.ndarray,
    columns: dict[str, int],
    key: str,
    predicate,
) -> int | None:
    if key not in columns:
        return None
    steps, values = _finite_series(trace, columns, key)
    if values.size == 0:
        return None
    idx = np.where(predicate(values))[0]
    if idx.size == 0:
        return None
    return int(round(float(steps[idx[0]])))


def _value_at_or_before(steps: np.ndarray, values: np.ndarray, step: int | None) -> float | None:
    if step is None or values.size == 0:
        return None
    idx = np.where(steps <= step)[0]
    if idx.size == 0:
        return None
    return _safe_float(values[idx[-1]])


def _value_at_or_after(steps: np.ndarray, values: np.ndarray, step: int | None) -> float | None:
    if step is None or values.size == 0:
        return None
    idx = np.where(steps >= step)[0]
    if idx.size == 0:
        return None
    return _safe_float(values[idx[0]])


def _variant_name(path: Path) -> str:
    if path.parent.name == "action_dump":
        return path.parent.parent.name
    return path.parent.name


def _replan_from_variant(name: str) -> int | None:
    match = re.search(r"_rp(\d+)_", name)
    return int(match.group(1)) if match else None


def _target_trace_metrics(
    trace: np.ndarray,
    columns: dict[str, int],
    arm: str,
    *,
    step_marks: tuple[int, ...],
    contact_threshold: float,
    close_threshold: float,
    decisive_threshold: float,
    tail_window: int,
) -> dict[str, Any]:
    target_key = TARGET_COLS[arm]
    cmd_key = CMD_COLS[arm]
    grasp_key = GRASP_COLS[arm]
    steps, dist = _finite_series(trace, columns, target_key)
    if dist.size == 0:
        return {"available": False}

    min_idx = int(np.argmin(dist))
    final_step = int(round(float(steps[-1])))
    final_value = _safe_float(dist[-1])
    tail_start = max(0, final_step - tail_window)
    tail = dist[steps >= tail_start]
    first_close_trace_step = _first_true_step(
        trace, columns, cmd_key, lambda values: values < close_threshold
    )
    first_decisive_trace_step = _first_true_step(
        trace, columns, cmd_key, lambda values: values < decisive_threshold
    )
    first_grasp_step = _first_true_step(
        trace, columns, grasp_key, lambda values: values > 0.5
    )

    close_action_step = (
        first_decisive_trace_step - 1 if first_decisive_trace_step is not None else None
    )
    before_close = _value_at_or_before(steps, dist, close_action_step)
    after_close = _value_at_or_after(steps, dist, first_decisive_trace_step)

    curve = {
        str(mark): _value_at_or_before(steps, dist, mark)
        for mark in step_marks
    }
    first_below_contact = None
    below = np.where(dist <= contact_threshold)[0]
    if below.size:
        first_below_contact = int(round(float(steps[below[0]])))

    start_value = _safe_float(dist[0])
    min_value = _safe_float(dist[min_idx])
    last_tail_mean = _safe_float(np.mean(tail)) if tail.size else None
    return {
        "available": True,
        "start": start_value,
        "min": min_value,
        "min_step": int(round(float(steps[min_idx]))),
        "final": final_value,
        "last_window_mean": last_tail_mean,
        "final_minus_min": (
            None if final_value is None or min_value is None else final_value - min_value
        ),
        "start_to_min_improvement": (
            None if start_value is None or min_value is None else start_value - min_value
        ),
        "first_step_below_contact_threshold": first_below_contact,
        "first_close_trace_step": first_close_trace_step,
        "first_decisive_close_trace_step": first_decisive_trace_step,
        "first_decisive_close_action_step": close_action_step,
        "target_dist_before_decisive_close": before_close,
        "target_dist_after_decisive_close": after_close,
        "decisive_close_before_contact": (
            None if before_close is None else bool(before_close > contact_threshold)
        ),
        "first_grasp_step": first_grasp_step,
        "curve": curve,
    }


def analyze_episode(
    path: Path,
    *,
    step_marks: tuple[int, ...],
    contact_threshold: float,
    close_threshold: float,
    decisive_threshold: float,
    tail_window: int,
) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    if "env_trace" not in data:
        raise KeyError(f"{path} missing env_trace")
    trace = np.asarray(data["env_trace"], dtype=np.float64)
    if trace.ndim != 2:
        raise ValueError(f"{path} env_trace must be 2-D, got {trace.shape}")
    columns = _columns(data)
    required = {"step", *TARGET_COLS.values(), *CMD_COLS.values(), *GRASP_COLS.values()}
    missing = sorted(required.difference(columns))
    if missing:
        raise KeyError(f"{path} missing env_trace columns: {missing}")

    success = bool(np.asarray(data["success"]).reshape(-1)[0]) if "success" in data else False
    episode = {
        "file": str(path),
        "seed": int(np.asarray(data["seed"]).reshape(-1)[0]) if "seed" in data else None,
        "success": success,
        "action_representation": _npz_string(data["action_representation"])
        if "action_representation" in data
        else "",
        "variant": _variant_name(path),
        "replan": _replan_from_variant(_variant_name(path)),
        "arms": {},
    }
    for arm in ("left", "right"):
        episode["arms"][arm] = _target_trace_metrics(
            trace,
            columns,
            arm,
            step_marks=step_marks,
            contact_threshold=contact_threshold,
            close_threshold=close_threshold,
            decisive_threshold=decisive_threshold,
            tail_window=tail_window,
        )
    return episode


def _aggregate_arm(episodes: list[dict[str, Any]], arm: str, step_marks: tuple[int, ...]) -> dict[str, Any]:
    metrics = [episode["arms"][arm] for episode in episodes if episode["arms"][arm]["available"]]
    out: dict[str, Any] = {"episodes": len(metrics)}
    scalar_keys = [
        "start",
        "min",
        "final",
        "last_window_mean",
        "final_minus_min",
        "start_to_min_improvement",
        "target_dist_before_decisive_close",
        "target_dist_after_decisive_close",
    ]
    for key in scalar_keys:
        out[key] = _summary([m[key] for m in metrics if m.get(key) is not None])
    for key in [
        "min_step",
        "first_decisive_close_trace_step",
        "first_decisive_close_action_step",
        "first_grasp_step",
        "first_step_below_contact_threshold",
    ]:
        out[key] = _summary([m[key] for m in metrics if m.get(key) is not None])
    out["decisive_close_count"] = sum(
        int(m.get("first_decisive_close_trace_step") is not None) for m in metrics
    )
    out["decisive_close_before_contact_count"] = sum(
        int(m.get("decisive_close_before_contact") is True) for m in metrics
    )
    out["grasp_count"] = sum(int(m.get("first_grasp_step") is not None) for m in metrics)
    out["contact_reached_count"] = sum(
        int(m.get("first_step_below_contact_threshold") is not None) for m in metrics
    )
    out["curve"] = {
        str(mark): _summary(
            [
                m["curve"][str(mark)]
                for m in metrics
                if m.get("curve", {}).get(str(mark)) is not None
            ]
        )
        for mark in step_marks
    }
    return out


def aggregate_variant(
    name: str,
    episodes: list[dict[str, Any]],
    *,
    step_marks: tuple[int, ...],
    contact_threshold: float,
) -> dict[str, Any]:
    return {
        "variant": name,
        "replan": _replan_from_variant(name),
        "episodes": len(episodes),
        "success_count": sum(int(e["success"]) for e in episodes),
        "contact_threshold": contact_threshold,
        "left": _aggregate_arm(episodes, "left", step_marks),
        "right": _aggregate_arm(episodes, "right", step_marks),
        "episode_metrics": episodes,
    }


def analyze_root(
    eval_root: Path,
    *,
    step_marks: tuple[int, ...],
    contact_threshold: float,
    close_threshold: float,
    decisive_threshold: float,
    tail_window: int,
) -> dict[str, Any]:
    grouped: dict[str, list[Path]] = {}
    for path in sorted(eval_root.rglob("episode_*.npz")):
        if path.parent.name != "action_dump":
            continue
        grouped.setdefault(_variant_name(path), []).append(path)
    if not grouped:
        raise FileNotFoundError(f"no action_dump episode npz files under {eval_root}")

    variants = []
    for name, paths in sorted(grouped.items()):
        episodes = [
            analyze_episode(
                path,
                step_marks=step_marks,
                contact_threshold=contact_threshold,
                close_threshold=close_threshold,
                decisive_threshold=decisive_threshold,
                tail_window=tail_window,
            )
            for path in paths
        ]
        variants.append(
            aggregate_variant(
                name,
                episodes,
                step_marks=step_marks,
                contact_threshold=contact_threshold,
            )
        )
    return {
        "eval_root": str(eval_root),
        "step_marks": list(step_marks),
        "contact_threshold": contact_threshold,
        "close_threshold": close_threshold,
        "decisive_threshold": decisive_threshold,
        "tail_window": tail_window,
        "variants": variants,
    }


def write_summary(result: dict[str, Any], path: Path) -> None:
    lines = [
        f"eval_root={result['eval_root']}",
        f"contact_threshold={result['contact_threshold']}",
        f"decisive_threshold={result['decisive_threshold']}",
        "",
        "\t".join(
            [
                "variant",
                "replan",
                "episodes",
                "success",
                "left_min_p50",
                "right_min_p50",
                "left_final_p50",
                "right_final_p50",
                "left_before_close_p50",
                "right_before_close_p50",
                "left_close_far_eps",
                "right_close_far_eps",
                "left_grasp_eps",
                "right_grasp_eps",
            ]
        ),
    ]
    for variant in result["variants"]:
        left = variant["left"]
        right = variant["right"]

        def p50(arm: dict[str, Any], key: str) -> str:
            value = arm.get(key, {}).get("p50")
            return "nan" if value is None else f"{value:.6f}"

        lines.append(
            "\t".join(
                [
                    variant["variant"],
                    str(variant["replan"]),
                    str(variant["episodes"]),
                    str(variant["success_count"]),
                    p50(left, "min"),
                    p50(right, "min"),
                    p50(left, "final"),
                    p50(right, "final"),
                    p50(left, "target_dist_before_decisive_close"),
                    p50(right, "target_dist_before_decisive_close"),
                    str(left["decisive_close_before_contact_count"]),
                    str(right["decisive_close_before_contact_count"]),
                    str(left["grasp_count"]),
                    str(right["grasp_count"]),
                ]
            )
        )
        lines.append("curve_step\tleft_p50\tright_p50")
        for mark in result["step_marks"]:
            left_value = left["curve"][str(mark)].get("p50")
            right_value = right["curve"][str(mark)].get("p50")
            lines.append(
                "\t".join(
                    [
                        str(mark),
                        "nan" if left_value is None else f"{left_value:.6f}",
                        "nan" if right_value is None else f"{right_value:.6f}",
                    ]
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_step_marks(value: str) -> tuple[int, ...]:
    marks = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not marks:
        raise argparse.ArgumentTypeError("empty step mark list")
    if any(mark < 0 for mark in marks):
        raise argparse.ArgumentTypeError("step marks must be non-negative")
    return tuple(dict.fromkeys(marks))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--step-marks",
        type=_parse_step_marks,
        default=_parse_step_marks("0,10,20,30,40,50,52,60,75,100,150,200,250,300"),
    )
    parser.add_argument("--contact-threshold", type=float, default=0.05)
    parser.add_argument("--close-threshold", type=float, default=0.0)
    parser.add_argument("--decisive-threshold", type=float, default=-0.5)
    parser.add_argument("--tail-window", type=int, default=50)
    args = parser.parse_args()

    result = analyze_root(
        args.eval_root,
        step_marks=args.step_marks,
        contact_threshold=args.contact_threshold,
        close_threshold=args.close_threshold,
        decisive_threshold=args.decisive_threshold,
        tail_window=args.tail_window,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(result, args.summary)
    print(args.summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
