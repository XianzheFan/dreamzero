"""Summarize DreamZero predicted-video diagnostics.

The policy server's ``--save-video-pred`` output contains decoded future
predictions, observed conditioning windows, optional conditioning decodes, and
comparison videos. This script computes lightweight per-file quality metrics and
can write a contact sheet for quick visual inspection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size == 0 or b.size == 0:
        return None
    if float(a.std()) < 1e-6 or float(b.std()) < 1e-6:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _read_video(path: Path, max_frames: int = 96) -> np.ndarray:
    import av

    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= max_frames:
                break
    if not frames:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    return np.stack(frames, axis=0).astype(np.uint8, copy=False)


def _category(path: Path) -> str:
    parts = set(path.parts)
    name = path.name
    if "comparison" in parts:
        return "comparison"
    if "conditioning" in parts:
        if "clean_x" in name:
            return "conditioning_clean_x"
        if "y_latent" in name:
            return "conditioning_y_latent"
        if "observed" in name:
            return "conditioning_observed"
        return "conditioning"
    if "observed" in parts or "_observed_" in name:
        return "observed"
    if name.endswith(".mp4") and "_agent" in name:
        return "pred"
    return "other"


def analyze_video_file(path: Path, root: Path, max_frames: int = 96) -> dict[str, Any]:
    frames = _read_video(path, max_frames=max_frames)
    stat: dict[str, Any] = {
        "path": str(path),
        "relpath": str(path.relative_to(root)),
        "name": path.name,
        "category": _category(path),
        "size": int(path.stat().st_size),
        "frames_read": int(frames.shape[0]),
        "shape": [int(v) for v in frames.shape],
    }
    if frames.size == 0:
        stat.update(
            {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "h_neighbor_corr": None,
                "v_neighbor_corr": None,
                "temporal_corr": None,
                "temporal_absdiff_mean": None,
                "first_last_absdiff_mean": None,
                "low_spatial_corr": True,
            }
        )
        return stat

    arr = frames.astype(np.float32)
    h_corr = _safe_corr(arr[:, :, :-1, :], arr[:, :, 1:, :])
    v_corr = _safe_corr(arr[:, :-1, :, :], arr[:, 1:, :, :])
    temporal_corr = None
    temporal_absdiff_mean = None
    first_last_absdiff_mean = None
    if arr.shape[0] > 1:
        temporal_corr = _safe_corr(arr[:-1], arr[1:])
        temporal_absdiff_mean = float(np.abs(arr[1:] - arr[:-1]).mean())
        first_last_absdiff_mean = float(np.abs(arr[-1] - arr[0]).mean())

    low_spatial_corr = bool(
        (h_corr is not None and h_corr < 0.65)
        or (v_corr is not None and v_corr < 0.65)
    )
    stat.update(
        {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "h_neighbor_corr": h_corr,
            "v_neighbor_corr": v_corr,
            "temporal_corr": temporal_corr,
            "temporal_absdiff_mean": temporal_absdiff_mean,
            "first_last_absdiff_mean": first_last_absdiff_mean,
            "low_spatial_corr": low_spatial_corr,
        }
    )
    return stat


def _aggregate(files: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, dict[str, Any]] = {}
    for item in files:
        cat = str(item["category"])
        entry = by_category.setdefault(
            cat,
            {
                "count": 0,
                "low_spatial_corr_count": 0,
                "mean_std": [],
                "mean_temporal_absdiff": [],
                "mean_first_last_absdiff": [],
            },
        )
        entry["count"] += 1
        if item.get("low_spatial_corr"):
            entry["low_spatial_corr_count"] += 1
        for dst_key, src_key in (
            ("mean_std", "std"),
            ("mean_temporal_absdiff", "temporal_absdiff_mean"),
            ("mean_first_last_absdiff", "first_last_absdiff_mean"),
        ):
            value = item.get(src_key)
            if value is not None:
                entry[dst_key].append(float(value))

    for entry in by_category.values():
        for key in ("mean_std", "mean_temporal_absdiff", "mean_first_last_absdiff"):
            values = entry[key]
            entry[key] = float(np.mean(values)) if values else None
    return {
        "num_files": len(files),
        "by_category": by_category,
    }


def _select_contact_files(files: list[dict[str, Any]], max_videos: int) -> list[Path]:
    priority = {
        "comparison": 0,
        "pred": 1,
        "observed": 2,
        "conditioning_clean_x": 3,
        "conditioning_y_latent": 4,
    }
    ordered = sorted(
        files,
        key=lambda item: (
            priority.get(str(item["category"]), 99),
            str(item["relpath"]),
        ),
    )
    return [Path(item["path"]) for item in ordered[:max_videos]]


def write_contact_sheet(
    files: list[dict[str, Any]],
    output: Path,
    *,
    max_videos: int = 18,
    thumb_height: int = 150,
) -> None:
    from PIL import Image, ImageDraw

    selected = _select_contact_files(files, max_videos=max_videos)
    if not selected:
        return

    rows: list[list[Image.Image]] = []
    label_h = 24
    for path in selected:
        frames = _read_video(path, max_frames=96)
        if frames.size == 0:
            continue
        indices = [0, frames.shape[0] // 2, frames.shape[0] - 1]
        row: list[Image.Image] = []
        for idx in indices:
            im = Image.fromarray(frames[idx])
            scale = thumb_height / max(1, im.height)
            im = im.resize(
                (max(1, int(im.width * scale)), thumb_height),
                Image.Resampling.BILINEAR,
            )
            canvas = Image.new("RGB", (im.width, thumb_height + label_h), (12, 12, 12))
            canvas.paste(im, (0, label_h))
            draw = ImageDraw.Draw(canvas)
            label = f"{path.parent.name}/{path.name} f{idx}"
            draw.text((4, 4), label[:90], fill=(255, 255, 255))
            row.append(canvas)
        rows.append(row)

    if not rows:
        return
    cols = 3
    cell_w = max(im.width for row in rows for im in row)
    cell_h = thumb_height + label_h
    sheet = Image.new("RGB", (cell_w * cols, cell_h * len(rows)), (0, 0, 0))
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            sheet.paste(im, (c * cell_w, r * cell_h))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def analyze_video_pred_dir(
    video_pred_dir: Path,
    *,
    max_frames: int = 96,
    max_files: int | None = None,
) -> dict[str, Any]:
    root = Path(video_pred_dir)
    files = sorted(root.rglob("*.mp4"))
    if max_files is not None:
        files = files[:max_files]
    items = [analyze_video_file(path, root, max_frames=max_frames) for path in files]
    return {
        "video_pred_dir": str(root),
        "summary": _aggregate(items),
        "files": items,
    }


def _write_text_summary(report: dict[str, Any], path: Path) -> None:
    lines = [
        f"video_pred_dir: {report['video_pred_dir']}",
        f"num_files: {report['summary']['num_files']}",
    ]
    for category, stats in sorted(report["summary"]["by_category"].items()):
        lines.append(
            f"{category}: count={stats['count']} "
            f"low_spatial_corr={stats['low_spatial_corr_count']} "
            f"mean_temporal_absdiff={stats['mean_temporal_absdiff']} "
            f"mean_first_last_absdiff={stats['mean_first_last_absdiff']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_pred_dir")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-txt", default=None)
    ap.add_argument("--contact-sheet", default=None)
    ap.add_argument("--max-frames", type=int, default=96)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--contact-max-videos", type=int, default=18)
    args = ap.parse_args()

    report = analyze_video_pred_dir(
        Path(args.video_pred_dir),
        max_frames=args.max_frames,
        max_files=args.max_files,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.output_txt:
        _write_text_summary(report, Path(args.output_txt))
    if args.contact_sheet:
        write_contact_sheet(
            report["files"],
            Path(args.contact_sheet),
            max_videos=args.contact_max_videos,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
