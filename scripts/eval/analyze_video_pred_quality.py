"""Analyze DreamZero predicted-video diagnostic dumps.

``bimanual_policy_server.py --save-video-pred`` writes one ``manifest.jsonl``
per session directory. Each manifest row lists predicted wrist videos, observed
conditioning-window videos, comparison panels, and rollout metadata. This helper
turns those artifacts into numeric quality diagnostics so closed-loop runs can
be compared without relying only on subjective video inspection.

The metrics are intentionally model-agnostic:

* temporal absolute differences identify frozen or flickery predictions;
* Laplacian variance is a simple blur/sharpness proxy;
* saturation/black/white fractions catch decode or color-range failures;
* optional pred-vs-observed MAE compares each predicted agent video with the
  matching observed wrist window when available.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _stats(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {
            "mean": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "mean": _safe_float(values.mean()),
        "p10": _safe_float(np.quantile(values, 0.10)),
        "p50": _safe_float(np.quantile(values, 0.50)),
        "p90": _safe_float(np.quantile(values, 0.90)),
        "p95": _safe_float(np.quantile(values, 0.95)),
        "max": _safe_float(values.max()),
    }


def _mean_or_none(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return None
    return _safe_float(values.mean())


def _gray(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected [T,H,W,3] RGB frames, got {frames.shape}")
    return (
        0.299 * frames[..., 0]
        + 0.587 * frames[..., 1]
        + 0.114 * frames[..., 2]
    )


def _resize_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.uint8)
    if frame.shape[:2] == (height, width):
        return frame
    try:
        import cv2

        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    except Exception:
        y_idx = np.linspace(0, frame.shape[0] - 1, height).round().astype(np.int64)
        x_idx = np.linspace(0, frame.shape[1] - 1, width).round().astype(np.int64)
        return frame[np.ix_(y_idx, x_idx)]


def _laplacian_variance(frames: np.ndarray) -> np.ndarray:
    gray = _gray(frames)
    try:
        import cv2

        return np.asarray(
            [
                cv2.Laplacian(frame.astype(np.float32), cv2.CV_32F).var()
                for frame in gray
            ],
            dtype=np.float32,
        )
    except Exception:
        if gray.shape[1] < 3 or gray.shape[2] < 3:
            return np.zeros((gray.shape[0],), dtype=np.float32)
        lap = (
            -4.0 * gray[:, 1:-1, 1:-1]
            + gray[:, :-2, 1:-1]
            + gray[:, 2:, 1:-1]
            + gray[:, 1:-1, :-2]
            + gray[:, 1:-1, 2:]
        )
        return lap.reshape(lap.shape[0], -1).var(axis=1)


def video_metrics(
    frames: np.ndarray, *, freeze_diff_threshold: float = 1.0
) -> dict[str, Any]:
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected [T,H,W,3] RGB frames, got {frames.shape}")
    gray = _gray(frames)
    temporal_absdiff = (
        np.abs(np.diff(frames.astype(np.float32), axis=0)).mean(axis=(1, 2, 3))
        if frames.shape[0] > 1
        else np.zeros((0,), dtype=np.float32)
    )
    spatial_std = gray.reshape(gray.shape[0], -1).std(axis=1)
    lap_var = _laplacian_variance(frames)
    black_frac = (frames <= 2).mean(axis=(1, 2, 3))
    white_frac = (frames >= 253).mean(axis=(1, 2, 3))
    saturation_frac = black_frac + white_frac
    return {
        "frame_count": int(frames.shape[0]),
        "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
        "mean_luma": _safe_float(gray.mean()),
        "std_luma": _safe_float(gray.std()),
        "spatial_std": _stats(spatial_std),
        "temporal_absdiff": _stats(temporal_absdiff),
        "temporal_freeze_frac": (
            _safe_float((temporal_absdiff < freeze_diff_threshold).mean())
            if temporal_absdiff.size
            else None
        ),
        "laplacian_var": _stats(lap_var),
        "saturation_frac": _stats(saturation_frac),
        "black_frac": _stats(black_frac),
        "white_frac": _stats(white_frac),
    }


def compare_videos(
    pred_frames: np.ndarray, observed_frames: np.ndarray
) -> dict[str, Any]:
    pred_frames = np.asarray(pred_frames, dtype=np.uint8)
    observed_frames = np.asarray(observed_frames, dtype=np.uint8)
    if pred_frames.ndim != 4 or observed_frames.ndim != 4:
        raise ValueError(
            "expected [T,H,W,3] videos, got "
            f"pred={pred_frames.shape} observed={observed_frames.shape}"
        )
    if pred_frames.shape[-1] != 3 or observed_frames.shape[-1] != 3:
        raise ValueError(
            f"expected RGB videos, got pred={pred_frames.shape} observed={observed_frames.shape}"
        )
    n = min(pred_frames.shape[0], observed_frames.shape[0])
    if n <= 0:
        return {"frame_count": 0, "mae_rgb": None, "mae_luma": None}
    height, width = pred_frames.shape[1], pred_frames.shape[2]
    obs = np.stack(
        [_resize_frame(frame, height, width) for frame in observed_frames[:n]],
        axis=0,
    )
    pred = pred_frames[:n].astype(np.float32)
    obs_f = obs.astype(np.float32)
    luma_diff = np.abs(_gray(pred_frames[:n]) - _gray(obs))
    return {
        "frame_count": int(n),
        "mae_rgb": _safe_float(np.abs(pred - obs_f).mean()),
        "mae_luma": _safe_float(luma_diff.mean()),
        "first_frame_mae_rgb": _safe_float(np.abs(pred[0] - obs_f[0]).mean()),
        "last_frame_mae_rgb": _safe_float(np.abs(pred[-1] - obs_f[-1]).mean()),
    }


def read_video(path: Path, *, max_frames: int | None = None) -> np.ndarray:
    path = Path(path)
    frames = []
    try:
        import av

        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
                if max_frames is not None and len(frames) >= max_frames:
                    break
    except Exception:
        import cv2

        cap = cv2.VideoCapture(str(path))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame[:, :, ::-1].copy())
                if max_frames is not None and len(frames) >= max_frames:
                    break
        finally:
            cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(frames, axis=0).astype(np.uint8)


def _load_manifest_rows(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for manifest in sorted(root.rglob("manifest.jsonl")):
        for line_no, line in enumerate(
            manifest.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            if not line.strip():
                continue
            try:
                rows.append((manifest.parent, json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def _fallback_video_rows(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    pred_files: list[Path] = []
    for path in sorted(root.rglob("*.mp4")):
        rel_parts = path.relative_to(root).parts
        if any(part in {"observed", "comparison"} for part in rel_parts):
            continue
        if "_compare" in path.name or "_observed_" in path.name:
            continue
        pred_files.append(path)
    if not pred_files:
        return []
    by_dir: dict[Path, list[str]] = {}
    for path in pred_files:
        by_dir.setdefault(path.parent, []).append(path.name)
    return [
        (directory, {"pred_files": sorted(files), "observed_files": []})
        for directory, files in sorted(by_dir.items())
    ]


def _agent_id_from_name(name: str) -> int | None:
    match = re.search(r"_agent(\d+)(?:\.|_)", Path(name).name)
    return int(match.group(1)) if match else None


def _observed_for_agent(
    session_dir: Path,
    observed_files: Iterable[str],
    agent_id: int | None,
) -> tuple[Path | None, str | None]:
    observed_paths = [session_dir / rel for rel in observed_files]
    if agent_id is not None:
        needle = f"_observed_agent{agent_id}.mp4"
        for path in observed_paths:
            if path.name.endswith(needle):
                return path, f"agent{agent_id}"
    for path in observed_paths:
        if path.name.endswith("_observed_global.mp4"):
            return path, "global"
    return (observed_paths[0], "observed") if observed_paths else (None, None)


def _risk_flags(metrics: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    temporal = metrics["temporal_absdiff"]
    lap = metrics["laplacian_var"]
    sat = metrics["saturation_frac"]
    freeze = metrics.get("temporal_freeze_frac")
    if freeze is not None and freeze > 0.80:
        flags.append("mostly_frozen")
    if temporal["p95"] is not None and temporal["p95"] > 45.0:
        flags.append("high_temporal_flicker")
    if lap["p10"] is not None and lap["p10"] < 20.0:
        flags.append("low_sharpness")
    if sat["mean"] is not None and sat["mean"] > 0.20:
        flags.append("high_saturation")
    if metrics["std_luma"] is not None and metrics["std_luma"] < 5.0:
        flags.append("low_dynamic_range")
    return flags


def analyze_video_tree(
    root: Path, *, max_frames: int | None = None
) -> dict[str, Any]:
    root = Path(root)
    manifest_rows = _load_manifest_rows(root)
    rows = manifest_rows if manifest_rows else _fallback_video_rows(root)
    videos = []
    failures = []
    for session_dir, entry in rows:
        pred_files = entry.get("pred_files") or []
        observed_files = entry.get("observed_files") or []
        for rel_pred in pred_files:
            pred_path = session_dir / rel_pred
            agent_id = _agent_id_from_name(rel_pred)
            try:
                pred_frames = read_video(pred_path, max_frames=max_frames)
                metrics = video_metrics(pred_frames)
                observed_path, observed_kind = _observed_for_agent(
                    session_dir,
                    observed_files,
                    agent_id,
                )
                comparison = None
                if observed_path is not None and observed_path.exists():
                    observed_frames = read_video(observed_path, max_frames=max_frames)
                    comparison = compare_videos(pred_frames, observed_frames)
                videos.append(
                    {
                        "path": pred_path.relative_to(root).as_posix(),
                        "session_dir": session_dir.relative_to(root).as_posix(),
                        "agent_id": agent_id,
                        "infer_idx": entry.get("infer_idx"),
                        "env_step": entry.get("env_step"),
                        "replan_every": entry.get("replan_every"),
                        "chunk_start_index": entry.get("chunk_start_index"),
                        "reset_causal_state_each_infer": entry.get(
                            "reset_causal_state_each_infer"
                        ),
                        "video_pred_rollout_mode": entry.get("video_pred_rollout_mode"),
                        "last_video_pred_rollout_mode": entry.get(
                            "last_video_pred_rollout_mode"
                        ),
                        "mai_rolling_noise": entry.get("mai_rolling_noise"),
                        "noise_draw_counts": entry.get("noise_draw_counts"),
                        "observed_path": (
                            observed_path.relative_to(root).as_posix()
                            if observed_path is not None and observed_path.exists()
                            else None
                        ),
                        "observed_kind": observed_kind,
                        "metrics": metrics,
                        "pred_vs_observed": comparison,
                        "risk_flags": _risk_flags(metrics),
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "path": pred_path.relative_to(root).as_posix()
                        if pred_path.exists()
                        else str(pred_path),
                        "error": str(exc),
                    }
                )
    return {
        "root": str(root),
        "videos": videos,
        "failures": failures,
        "summary": _aggregate(videos),
    }


def _aggregate(videos: list[dict[str, Any]]) -> dict[str, Any]:
    if not videos:
        return {"video_count": 0}

    def collect(path: tuple[str, ...]) -> np.ndarray:
        values = []
        for row in videos:
            value: Any = row
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if value is not None:
                values.append(float(value))
        return np.asarray(values, dtype=np.float32)

    flag_counts: dict[str, int] = {}
    for row in videos:
        for flag in row["risk_flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    return {
        "video_count": len(videos),
        "videos_with_flags": sum(1 for row in videos if row["risk_flags"]),
        "flag_counts": flag_counts,
        "temporal_absdiff_mean": _mean_or_none(
            collect(("metrics", "temporal_absdiff", "mean"))
        ),
        "temporal_absdiff_p95_mean": _mean_or_none(
            collect(("metrics", "temporal_absdiff", "p95"))
        ),
        "temporal_freeze_frac_mean": _mean_or_none(
            collect(("metrics", "temporal_freeze_frac"))
        ),
        "laplacian_var_mean": _mean_or_none(
            collect(("metrics", "laplacian_var", "mean"))
        ),
        "saturation_frac_mean": _mean_or_none(
            collect(("metrics", "saturation_frac", "mean"))
        ),
        "pred_vs_observed_mae_rgb_mean": _mean_or_none(
            collect(("pred_vs_observed", "mae_rgb"))
        ),
    }


def write_text_report(payload: dict[str, Any], path: Path) -> None:
    summary = payload["summary"]
    lines = [
        f"root: {payload['root']}",
        f"videos: {summary.get('video_count', 0)}",
        f"videos_with_flags: {summary.get('videos_with_flags', 0)}",
        f"flag_counts: {summary.get('flag_counts', {})}",
        "aggregate:",
        f"  temporal_absdiff_mean: {summary.get('temporal_absdiff_mean')}",
        f"  temporal_absdiff_p95_mean: {summary.get('temporal_absdiff_p95_mean')}",
        f"  temporal_freeze_frac_mean: {summary.get('temporal_freeze_frac_mean')}",
        f"  laplacian_var_mean: {summary.get('laplacian_var_mean')}",
        f"  saturation_frac_mean: {summary.get('saturation_frac_mean')}",
        f"  pred_vs_observed_mae_rgb_mean: {summary.get('pred_vs_observed_mae_rgb_mean')}",
        "",
        "per video:",
    ]
    for row in payload["videos"]:
        metrics = row["metrics"]
        comparison = row.get("pred_vs_observed") or {}
        lines.append(
            "  "
            f"{row['path']} agent={row.get('agent_id')} env_step={row.get('env_step')} "
            f"temporal_mean={metrics['temporal_absdiff']['mean']} "
            f"temporal_p95={metrics['temporal_absdiff']['p95']} "
            f"freeze={metrics['temporal_freeze_frac']} "
            f"lap_mean={metrics['laplacian_var']['mean']} "
            f"sat_mean={metrics['saturation_frac']['mean']} "
            f"obs_mae={comparison.get('mae_rgb')} "
            f"flags={row['risk_flags']}"
        )
    if payload["failures"]:
        lines.extend(["", "decode failures:"])
        for failure in payload["failures"]:
            lines.append(f"  {failure['path']}: {failure['error']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_contact_sheet(
    payload: dict[str, Any], path: Path, *, max_videos: int = 16
) -> None:
    if not payload["videos"]:
        return
    from PIL import Image, ImageDraw

    thumbs = []
    root = Path(payload["root"])
    for row in payload["videos"][:max_videos]:
        frames = read_video(root / row["path"])
        idxs = [0, frames.shape[0] // 2, frames.shape[0] - 1]
        strip = np.concatenate([frames[i] for i in idxs], axis=1)
        image = Image.fromarray(strip).resize((360, 120))
        canvas = Image.new("RGB", (360, 148), (255, 255, 255))
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        label = (
            f"{Path(row['path']).name} flags={','.join(row['risk_flags']) or 'none'}"
        )
        draw.text((4, 124), label[:74], fill=(0, 0, 0))
        thumbs.append(canvas)
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 360, rows * 148), (240, 240, 240))
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * 360, (i // cols) * 148))
    sheet.save(path, quality=90)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Directory containing video_pred session dumps.")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-txt", default=None)
    ap.add_argument("--contact-sheet", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    payload = analyze_video_tree(Path(args.root), max_frames=args.max_frames)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.output_txt:
        write_text_report(payload, Path(args.output_txt))
    if args.contact_sheet:
        write_contact_sheet(payload, Path(args.contact_sheet))


if __name__ == "__main__":
    main()
