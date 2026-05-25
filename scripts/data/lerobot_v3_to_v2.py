"""Convert a LeRobot v3 dataset to the v2 layout dreamzero expects.

Inputs (v3):
    data/chunk-NNN/file-MMM.parquet           (multiple episodes per file,
                                                distinguished by episode_index
                                                column)
    videos/<video_key>/chunk-NNN/file-MMM.mp4 (chunked videos)
    meta/info.json
    meta/tasks.parquet
    meta/episodes/chunk-NNN/file-MMM.parquet

Outputs (v2, dreamzero-expected):
    data/chunk-000/episode_{ep:06d}.parquet
    videos/chunk-000/<video_key>/episode_{ep:06d}.mp4
    meta/info.json
    meta/tasks.jsonl
    meta/episodes.jsonl

The conversion uses the per-episode timestamps recorded in the v3
``meta/episodes/...`` parquet to slice the chunked videos with ffmpeg.
Audio is dropped, video stream is re-encoded with libx264 to guarantee
that the timestamp boundaries become keyframes (AV1 chunks don't have
keyframes at every episode boundary, so ``-c:v copy`` cannot be used).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _load_data_parquet(src: Path, chunk_idx: int, file_idx: int, cache: dict) -> pd.DataFrame:
    key = (chunk_idx, file_idx)
    if key not in cache:
        p = src / f"data/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet"
        cache[key] = pd.read_parquet(p)
    return cache[key]


def _slice_video(
    src_mp4: Path,
    dst_mp4: Path,
    t0: float,
    t1: float,
    *,
    re_encode: bool = True,
    crf: int = 18,
) -> None:
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{t0:.6f}",
        "-to", f"{t1:.6f}",
        "-i", str(src_mp4),
        "-an",  # drop audio
    ]
    if re_encode:
        cmd += [
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
        ]
    else:
        cmd += ["-c:v", "copy"]
    cmd.append(str(dst_mp4))
    subprocess.run(cmd, check=True)


def convert(
    src: Path,
    dst: Path,
    *,
    re_encode: bool = True,
    crf: int = 18,
    overwrite: bool = False,
) -> None:
    if dst.exists():
        if not overwrite:
            sys.exit(f"Output {dst} exists; pass --overwrite to wipe.")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    info = json.loads((src / "meta/info.json").read_text())
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    print(f"video keys: {video_keys}")

    ep_meta = pd.read_parquet(src / "meta/episodes/chunk-000/file-000.parquet")
    print(f"episodes: {len(ep_meta)}")
    tasks_pq = pd.read_parquet(src / "meta/tasks.parquet")
    print(f"tasks parquet columns: {list(tasks_pq.columns)}")

    data_cache: dict = {}
    episodes_jsonl: list[dict] = []

    for _, row in ep_meta.iterrows():
        ep_idx = int(row.episode_index)
        length = int(row.length)
        print(f"  ep {ep_idx:3d}: length={length}")

        # 1) parquet
        d_chunk = int(row["data/chunk_index"])
        d_file = int(row["data/file_index"])
        big_df = _load_data_parquet(src, d_chunk, d_file, data_cache)
        ep_df = big_df[big_df.episode_index == ep_idx].copy()
        # Some dreamzero readers expect frame_index to start at 0; reset.
        ep_df = ep_df.sort_values("frame_index").reset_index(drop=True)
        # If frame_index already starts at 0 we keep it; otherwise normalise.
        if ep_df.frame_index.iloc[0] != 0:
            ep_df["frame_index"] = ep_df["frame_index"] - ep_df["frame_index"].iloc[0]
        # ``index`` is the global frame index across the dataset; preserve.
        assert len(ep_df) == length, (
            f"ep {ep_idx} expected {length} rows, got {len(ep_df)}"
        )
        out_pq = dst / f"data/chunk-000/episode_{ep_idx:06d}.parquet"
        out_pq.parent.mkdir(parents=True, exist_ok=True)
        ep_df.to_parquet(out_pq, index=False)

        # 2) videos
        for vk in video_keys:
            v_chunk = int(row[f"videos/{vk}/chunk_index"])
            v_file = int(row[f"videos/{vk}/file_index"])
            t0 = float(row[f"videos/{vk}/from_timestamp"])
            t1 = float(row[f"videos/{vk}/to_timestamp"])
            src_mp4 = src / f"videos/{vk}/chunk-{v_chunk:03d}/file-{v_file:03d}.mp4"
            dst_mp4 = dst / f"videos/chunk-000/{vk}/episode_{ep_idx:06d}.mp4"
            _slice_video(src_mp4, dst_mp4, t0, t1, re_encode=re_encode, crf=crf)

        # 3) episodes.jsonl entry
        tasks_field = row["tasks"]
        if hasattr(tasks_field, "tolist"):
            tasks_list = tasks_field.tolist()
        elif isinstance(tasks_field, (list, tuple)):
            tasks_list = list(tasks_field)
        else:
            tasks_list = [tasks_field]
        episodes_jsonl.append({
            "episode_index": ep_idx,
            "tasks": tasks_list,
            "length": length,
        })

    # 4) meta/episodes.jsonl
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    with (dst / "meta/episodes.jsonl").open("w") as f:
        for entry in episodes_jsonl:
            f.write(json.dumps(entry) + "\n")

    # 5) meta/tasks.jsonl. v3 tasks.parquet is a row-indexed DataFrame
    # ``{task: <task_string>, task_index: <int>}`` per row in the modern
    # LeRobot release; older v3 dumps store it indexed by task_string
    # with task_index as the column. Handle both.
    tasks_list: list[dict] = []
    if "task" in tasks_pq.columns and "task_index" in tasks_pq.columns:
        for _, row in tasks_pq.iterrows():
            tasks_list.append({
                "task_index": int(row.task_index),
                "task": str(row.task),
            })
    elif "task_index" in tasks_pq.columns:
        # Index is the task string, single column is the index value.
        for task_str, idx in tasks_pq["task_index"].items():
            tasks_list.append({"task_index": int(idx), "task": str(task_str)})
    else:
        raise ValueError(f"Unrecognised tasks.parquet schema: {list(tasks_pq.columns)}")
    tasks_list.sort(key=lambda d: d["task_index"])
    with (dst / "meta/tasks.jsonl").open("w") as f:
        for entry in tasks_list:
            f.write(json.dumps(entry) + "\n")

    # 6) meta/info.json (v2 path templates)
    v2_info = dict(info)
    v2_info["codebase_version"] = "v2.0"
    v2_info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    v2_info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    v2_info["total_episodes"] = len(episodes_jsonl)
    v2_info["chunks_size"] = 1000
    for stale in ("data_files_size_in_mb", "video_files_size_in_mb", "splits"):
        v2_info.pop(stale, None)
    (dst / "meta/info.json").write_text(json.dumps(v2_info, indent=2))

    print(f"\nDone: {len(episodes_jsonl)} episodes written to {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--copy", action="store_true",
                    help="Use ffmpeg -c:v copy (no re-encode). Faster but "
                         "may produce truncated clips if keyframes don't "
                         "align with episode boundaries.")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    convert(args.src, args.dst,
            re_encode=not args.copy, crf=args.crf, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
