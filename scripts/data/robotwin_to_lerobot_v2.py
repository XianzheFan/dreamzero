"""Convert a RoboTwin 2.0 franka-panda demonstration set into a LeRobot v2
layout consumable by the DreamZero ``multi-agent`` branch under the
existing ``robofactory`` embodiment tag (both expose two Panda arms with
the same 7 joints + 1 gripper per arm).

Source layout (one episode per .hdf5, produced by RoboTwin's
``script/collect_data.py``)::

    RoboTwin/data/{task}/{config}/data/episode{i}.hdf5

Each episode contains::

    /joint_action/left_arm        [T, 7]      panda joint positions
    /joint_action/left_gripper    [T] or [T, 1]
    /joint_action/right_arm       [T, 7]
    /joint_action/right_gripper   [T] or [T, 1]
    /joint_action/vector          [T, 16]     concat in left-arm,
                                              left-gripper, right-arm,
                                              right-gripper order
    /observation/head_camera/rgb     [T] uint8 (JPEG bytes per frame)
    /observation/left_camera/rgb     [T] uint8
    /observation/right_camera/rgb    [T] uint8

Output (LeRobot v2 schema, reusing the existing two-Panda RoboFactory
modality names so the bimanual DreamZero transform can be reused)::

    {out_dir}/
        data/chunk-000/episode_000000.parquet
        videos/chunk-000/observation.images.global/episode_000000.mp4
        videos/chunk-000/observation.images.agent0/episode_000000.mp4
        videos/chunk-000/observation.images.agent1/episode_000000.mp4
        meta/{info,modality,episodes,tasks,stats,embodiment}.{json,jsonl}
        meta/step_filter.jsonl
        meta/relative_stats_dreamzero.json

State / action layout (16 dims total, matches ``robofactory``)::

    [left_arm_joint(0:7), left_gripper(7:8),
     right_arm_joint(8:15), right_gripper(15:16)]

Actions are stored as absolute next-step qpos targets. DreamZero's
``relative_action`` training path then converts the joint targets to
``target_qpos - current_qpos`` on the fly and normalizes those relative
joint offsets, matching the DROID-style pretraining convention. Grippers
remain absolute 0/1 targets.

Idle anchors are excluded through ``meta/step_filter.jsonl`` instead of
deleting frames. This keeps videos/parquets simple while matching the
DreamZero loader's existing step-filter contract. The relative action
statistics are computed from the same kept anchors, so normalization and
training samples stay aligned.

Usage::

    python scripts/data/robotwin_to_lerobot_v2.py \\
        --episode-dir /path/to/RoboTwin/data/beat_block_hammer/demo_randomized/data \\
        --out /path/to/lerobot_v2/beat_block_hammer-rt \\
        --task "beat the red block with the hammer" \\
        --num-episodes 10
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

# Avoid OpenBLAS/OpenMP import-time thread storms on login nodes with a
# tight RLIMIT_NPROC. Video encoding/decoding below is explicitly
# single-threaded, so this does not reduce useful converter parallelism.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import av
import cv2
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

FPS = 20
CHUNK_SIZE = 1000

CAMERAS = {
    # LeRobot key -> HDF5 camera group under /observation/{name}/rgb.
    # The "global" view is the head camera (shared by both agents); each
    # arm gets its own wrist camera as its per-agent view.
    "observation.images.global": "head_camera",
    "observation.images.agent0": "left_camera",
    "observation.images.agent1": "right_camera",
}

ARM_STATE_DIM = 8   # 7 joints + 1 gripper finger
ARM_ACTION_DIM = 8
STATE_DIM = 2 * ARM_STATE_DIM  # 16
ACTION_DIM = 2 * ARM_ACTION_DIM  # 16
JOINT_INDICES = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14])
RELATIVE_ACTION_SLICES = {
    "panda0_joint_pos": slice(0, 7),
    "panda1_joint_pos": slice(8, 15),
}


def _decode_jpeg_stream(rgb_dataset: h5py.Dataset, T: int) -> np.ndarray:
    """Decode a RoboTwin ``/observation/{cam}/rgb`` stream of JPEG bytes
    into an [T, H, W, 3] uint8 RGB tensor.
    """
    frames: list[np.ndarray] = []
    for i in range(T):
        buf = rgb_dataset[i]
        if isinstance(buf, np.ndarray):
            buf = buf.tobytes()
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to decode frame {i} from {rgb_dataset.name}")
        # cv2 gives BGR; convert to RGB.
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0)


def encode_video(frames: np.ndarray, output_path: Path, fps: int) -> None:
    """RGB uint8 [T, H, W, 3] -> h264 mp4."""
    options = {
        "threads": "1",
        "thread_type": "slice",
        "preset": "ultrafast",
        "tune": "zerolatency",
        "crf": "23",
    }
    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("h264", rate=fps, options=options)
    stream.width = frames.shape[2]
    stream.height = frames.shape[1]
    stream.pix_fmt = "yuv420p"

    video_frame = av.VideoFrame(width=stream.width, height=stream.height, format="rgb24")
    frame_array = video_frame.to_ndarray(format="rgb24")
    for frame in frames:
        frame_array[:] = frame
        for packet in stream.encode(video_frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()


def _read_state_and_action(traj: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(state[T, 16], action[T-1, 16])``.

    ``action[t]`` is the absolute next-step qpos target. The DreamZero
    dataset loader applies ``relative_action`` for the joint keys during
    training, so the raw LeRobot rows stay in the same form as DROID:
    future target action plus current observation state.
    """
    left_arm = traj["/joint_action/left_arm"][()].astype(np.float32)
    right_arm = traj["/joint_action/right_arm"][()].astype(np.float32)
    left_grip = np.asarray(traj["/joint_action/left_gripper"][()], dtype=np.float32)
    right_grip = np.asarray(traj["/joint_action/right_gripper"][()], dtype=np.float32)

    if left_grip.ndim == 1:
        left_grip = left_grip[:, None]
    if right_grip.ndim == 1:
        right_grip = right_grip[:, None]

    assert left_arm.shape[1] == 7, f"expected 7 joints, got {left_arm.shape}"
    assert right_arm.shape[1] == 7
    assert left_grip.shape[1] == 1, f"expected scalar gripper, got {left_grip.shape}"
    assert right_grip.shape[1] == 1

    T = left_arm.shape[0]
    state = np.concatenate([left_arm, left_grip, right_arm, right_grip], axis=1)
    assert state.shape == (T, STATE_DIM)

    # Absolute next-step qpos target. Joint targets are converted to
    # relative offsets by the DreamZero dataset loader when
    # ``relative_action`` is enabled; grippers are kept absolute.
    action = state[1:].copy()
    return state[:-1], action                      # T-1 rows each


def _next_step_joint_motion_scores(
    state: np.ndarray,
    action: np.ndarray,
) -> np.ndarray:
    """Score each anchor by its next-step joint displacement."""
    scores = np.zeros(len(state), dtype=np.float32)
    joint_delta = action[:, JOINT_INDICES] - state[:, JOINT_INDICES]
    scores[: len(joint_delta)] = np.linalg.norm(joint_delta, axis=1)
    return scores


def _step_filter_and_relative_samples(
    state: np.ndarray,
    action: np.ndarray,
    action_horizon: int,
    idle_threshold: float,
) -> tuple[list[int], dict[str, np.ndarray], dict[str, float]]:
    """Return excluded anchor indices and relative joint samples to normalize.

    ``step_filter.jsonl`` stores indices to remove. A row is treated as idle
    when its next-step joint target stays within ``idle_threshold`` L2
    distance from the anchor state. Relative stats are pooled over the same
    kept anchors and the full action horizon.
    """
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")

    scores = _next_step_joint_motion_scores(state, action)
    filtered = np.flatnonzero(scores <= idle_threshold).astype(np.int64)

    # Keep at least one valid anchor so tiny smoke datasets remain usable.
    if len(filtered) == len(state) and len(state) > 0:
        filtered = filtered[filtered != int(np.argmax(scores))]

    filtered_set = set(int(i) for i in filtered.tolist())
    usable_length = max(0, len(action) - action_horizon + 1)
    rel_samples: dict[str, list[np.ndarray]] = {
        key: [] for key in RELATIVE_ACTION_SLICES
    }

    kept_full_horizon = 0
    for anchor_idx in range(usable_length):
        if anchor_idx in filtered_set:
            continue
        kept_full_horizon += 1
        for key, slc in RELATIVE_ACTION_SLICES.items():
            rel = (
                action[anchor_idx : anchor_idx + action_horizon, slc]
                - state[anchor_idx, slc]
            )
            rel_samples[key].append(rel.astype(np.float32, copy=False))

    rel_arrays = {
        key: (
            np.concatenate(chunks, axis=0)
            if chunks
            else np.empty((0, slc.stop - slc.start), dtype=np.float32)
        )
        for key, chunks in rel_samples.items()
        for slc in [RELATIVE_ACTION_SLICES[key]]
    }
    summary = {
        "num_rows": float(len(state)),
        "filtered_rows": float(len(filtered)),
        "kept_rows": float(len(state) - len(filtered)),
        "kept_full_horizon_anchors": float(kept_full_horizon),
        "idle_threshold": float(idle_threshold),
        "motion_score_p50": float(np.quantile(scores, 0.50)) if len(scores) else 0.0,
        "motion_score_p95": float(np.quantile(scores, 0.95)) if len(scores) else 0.0,
    }
    return filtered.tolist(), rel_arrays, summary


def _per_dim_stats(arr: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std": (arr.std(axis=0) + 1e-8).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


def convert_episode(
    traj_path: Path,
    episode_index: int,
    task_index: int,
    out_root: Path,
    cumulative_index: int,
    action_horizon: int,
    idle_threshold: float,
) -> tuple[
    int,
    np.ndarray,
    np.ndarray,
    tuple[int, int],
    list[int],
    dict[str, np.ndarray],
    dict[str, float],
]:
    """Write one .parquet + N .mp4 files for one episode."""
    with h5py.File(traj_path, "r") as traj:
        state, action = _read_state_and_action(traj)
        T = state.shape[0]
        step_filter, relative_samples, filter_summary = _step_filter_and_relative_samples(
            state=state,
            action=action,
            action_horizon=action_horizon,
            idle_threshold=idle_threshold,
        )

        chunk_idx = episode_index // CHUNK_SIZE
        chunk_dir = f"chunk-{chunk_idx:03d}"

        df = pd.DataFrame(
            {
                "action": list(action),
                "observation.state": list(state),
                "timestamp": np.arange(T, dtype=np.float32) / FPS,
                "frame_index": np.arange(T, dtype=np.int64),
                "episode_index": np.full(T, episode_index, dtype=np.int64),
                "index": np.arange(cumulative_index, cumulative_index + T, dtype=np.int64),
                "task_index": np.full(T, task_index, dtype=np.int64),
            }
        )
        data_dir = out_root / "data" / chunk_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False),
            data_dir / f"episode_{episode_index:06d}.parquet",
        )

        sample_hw: tuple[int, int] | None = None
        for video_key, cam_name in CAMERAS.items():
            rgb_ds = traj[f"/observation/{cam_name}/rgb"]
            # The first T+1 obs frames map onto the T state rows.
            rgb = _decode_jpeg_stream(rgb_ds, T)
            if sample_hw is None:
                sample_hw = (rgb.shape[1], rgb.shape[2])
            video_dir = out_root / "videos" / chunk_dir / video_key
            video_dir.mkdir(parents=True, exist_ok=True)
            encode_video(rgb, video_dir / f"episode_{episode_index:06d}.mp4", FPS)

    assert sample_hw is not None
    return T, action, state, sample_hw, step_filter, relative_samples, filter_summary


def write_meta(
    out_root: Path,
    task_text: str,
    num_episodes: int,
    total_frames: int,
    episode_lengths: list[int],
    sample_video_hw: tuple[int, int],
    actions: list[np.ndarray],
    states: list[np.ndarray],
    step_filters: list[list[int]],
    relative_samples: dict[str, list[np.ndarray]],
    idle_filter_summaries: list[dict[str, float]],
    action_horizon: int,
) -> None:
    meta = out_root / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    h, w = sample_video_hw
    video_feature_template = {
        "dtype": "video",
        "shape": [h, w, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": h,
            "video.width": w,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": FPS,
            "video.channels": 3,
            "has_audio": False,
        },
    }

    # Reuse the robofactory state/action naming so the existing modality
    # config (state.panda0_joint_pos, etc.) lines up dimension-for-
    # dimension. The training pipeline doesn't care that the actual robot
    # is a RoboTwin franka-panda rather than a RoboFactory franka-panda;
    # what matters is the 7+1 per-arm split.
    state_names = [
        *(f"panda0_joint_{i}.pos" for i in range(7)),
        "panda0_gripper.pos",
        *(f"panda1_joint_{i}.pos" for i in range(7)),
        "panda1_gripper.pos",
    ]
    info = {
        "codebase_version": "v2.0",
        "robot_type": "bi_panda_robotwin",
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float32", "names": state_names, "shape": [ACTION_DIM]},
            "observation.state": {"dtype": "float32", "names": state_names, "shape": [STATE_DIM]},
            "observation.images.global": video_feature_template,
            "observation.images.agent0": video_feature_template,
            "observation.images.agent1": video_feature_template,
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2))

    modality = {
        "state": {
            "panda0_joint_pos": {
                "original_key": "observation.state",
                "start": 0, "end": 7,
                "rotation_type": None, "absolute": True, "dtype": "float32", "range": None,
            },
            "panda0_gripper_pos": {
                "original_key": "observation.state",
                "start": 7, "end": 8,
                "rotation_type": None, "absolute": True, "dtype": "float32", "range": None,
            },
            "panda1_joint_pos": {
                "original_key": "observation.state",
                "start": 8, "end": 15,
                "rotation_type": None, "absolute": True, "dtype": "float32", "range": None,
            },
            "panda1_gripper_pos": {
                "original_key": "observation.state",
                "start": 15, "end": 16,
                "rotation_type": None, "absolute": True, "dtype": "float32", "range": None,
            },
        },
        "action": {
            "panda0_joint_pos": {
                "original_key": "action",
                "start": 0, "end": 7,
                "rotation_type": None, "absolute": True, "dtype": "float32", "range": None,
            },
            "panda0_gripper_pos": {
                "original_key": "action",
                "start": 7, "end": 8,
                "rotation_type": None, "absolute": True, "dtype": "float32", "range": None,
            },
            "panda1_joint_pos": {
                "original_key": "action",
                "start": 8, "end": 15,
                "rotation_type": None, "absolute": True, "dtype": "float32", "range": None,
            },
            "panda1_gripper_pos": {
                "original_key": "action",
                "start": 15, "end": 16,
                "rotation_type": None, "absolute": True, "dtype": "float32", "range": None,
            },
        },
        "video": {
            "global_camera-images-rgb": {"original_key": "observation.images.global"},
            "agent0_camera-images-rgb": {"original_key": "observation.images.agent0"},
            "agent1_camera-images-rgb": {"original_key": "observation.images.agent1"},
        },
        "annotation": {"task": {"original_key": "task_index"}},
    }
    (meta / "modality.json").write_text(json.dumps(modality, indent=2))

    with (meta / "episodes.jsonl").open("w") as f:
        for ep_idx, length in enumerate(episode_lengths):
            f.write(json.dumps({
                "episode_index": ep_idx,
                "tasks": [task_text],
                "length": length,
            }) + "\n")

    with (meta / "tasks.jsonl").open("w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_text}) + "\n")

    with (meta / "step_filter.jsonl").open("w") as f:
        for ep_idx, step_indices in enumerate(step_filters):
            f.write(json.dumps({
                "episode_index": ep_idx,
                "step_indices": step_indices,
                "reason": "idle_next_step_joint_motion",
            }) + "\n")

    (meta / "embodiment.json").write_text(json.dumps({
        "robot_type": "bi_panda_robotwin",
        # Reuse the existing robofactory embodiment tag so we don't need
        # to wire up a new entry in embodiment_tags.py / base_48.yaml.
        "embodiment_tag": "robofactory",
    }, indent=2))

    all_actions = np.concatenate(actions, axis=0)
    all_states = np.concatenate(states, axis=0)

    stats = {
        "observation.state": _per_dim_stats(all_states.astype(np.float32)),
        "action": _per_dim_stats(all_actions.astype(np.float32)),
        "timestamp": _per_dim_stats(np.array([[0.0]], dtype=np.float32)),
    }
    (meta / "stats.json").write_text(json.dumps(stats, indent=2))

    relative_stats = {}
    for key, chunks in relative_samples.items():
        if not chunks:
            continue
        arr = np.concatenate(chunks, axis=0).astype(np.float32)
        relative_stats[key] = _per_dim_stats(arr)
    if not relative_stats:
        raise RuntimeError("idle filter removed all full-horizon relative action samples")
    (meta / "relative_stats_dreamzero.json").write_text(
        json.dumps(relative_stats, indent=2)
    )
    horizon_stats = meta / "relative_horizon_stats_dreamzero.json"
    if horizon_stats.exists():
        horizon_stats.unlink()

    filter_summary = {
        "action_horizon": action_horizon,
        "num_episodes": num_episodes,
        "total_rows": int(sum(item["num_rows"] for item in idle_filter_summaries)),
        "filtered_rows": int(sum(item["filtered_rows"] for item in idle_filter_summaries)),
        "kept_rows": int(sum(item["kept_rows"] for item in idle_filter_summaries)),
        "kept_full_horizon_anchors": int(
            sum(item["kept_full_horizon_anchors"] for item in idle_filter_summaries)
        ),
        "idle_threshold": idle_filter_summaries[0]["idle_threshold"]
        if idle_filter_summaries
        else None,
        "motion_score_p50_mean": float(
            np.mean([item["motion_score_p50"] for item in idle_filter_summaries])
        ) if idle_filter_summaries else 0.0,
        "motion_score_p95_mean": float(
            np.mean([item["motion_score_p95"] for item in idle_filter_summaries])
        ) if idle_filter_summaries else 0.0,
    }
    (meta / "idle_filter_summary.json").write_text(
        json.dumps(filter_summary, indent=2)
    )


def _list_episodes(episode_dir: Path) -> list[Path]:
    files = sorted(episode_dir.glob("episode*.hdf5"),
                   key=lambda p: int(p.stem.replace("episode", "")))
    if not files:
        raise SystemExit(f"No episode*.hdf5 found under {episode_dir}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", required=True, type=Path,
                        help="Path to RoboTwin {task}/{config}/data directory")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output directory for LeRobot v2 dataset")
    parser.add_argument("--task", required=True, type=str,
                        help="Natural-language task description")
    parser.add_argument("--num-episodes", type=int, default=-1,
                        help="Number of episodes to convert (-1 = all)")
    parser.add_argument("--action-horizon", type=int, default=24,
                        help="Action horizon used for idle filtering and relative stats")
    parser.add_argument("--idle-filter-threshold", type=float, default=1e-3,
                        help="Filter anchors whose future joint-motion L2 max is <= this")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    episode_files = _list_episodes(args.episode_dir)
    if args.num_episodes > 0:
        episode_files = episode_files[: args.num_episodes]

    episode_lengths: list[int] = []
    actions_buf: list[np.ndarray] = []
    states_buf: list[np.ndarray] = []
    step_filters: list[list[int]] = []
    relative_samples: dict[str, list[np.ndarray]] = {
        key: [] for key in RELATIVE_ACTION_SLICES
    }
    idle_filter_summaries: list[dict[str, float]] = []
    cumulative = 0
    sample_hw: tuple[int, int] | None = None

    for ep_idx, fp in enumerate(tqdm(episode_files, desc="episodes")):
        T, action, state, hw, step_filter, rel_samples, filter_summary = convert_episode(
            traj_path=fp,
            episode_index=ep_idx,
            task_index=0,
            out_root=args.out,
            cumulative_index=cumulative,
            action_horizon=args.action_horizon,
            idle_threshold=args.idle_filter_threshold,
        )
        if sample_hw is None:
            sample_hw = hw
        cumulative += T
        episode_lengths.append(T)
        actions_buf.append(action)
        states_buf.append(state)
        step_filters.append(step_filter)
        idle_filter_summaries.append(filter_summary)
        for key, arr in rel_samples.items():
            if len(arr):
                relative_samples[key].append(arr)

    assert sample_hw is not None
    write_meta(
        out_root=args.out,
        task_text=args.task,
        num_episodes=len(episode_lengths),
        total_frames=cumulative,
        episode_lengths=episode_lengths,
        sample_video_hw=sample_hw,
        actions=actions_buf,
        states=states_buf,
        step_filters=step_filters,
        relative_samples=relative_samples,
        idle_filter_summaries=idle_filter_summaries,
        action_horizon=args.action_horizon,
    )
    filtered = sum(len(indices) for indices in step_filters)
    kept = cumulative - filtered
    print(
        f"Done. {len(episode_lengths)} episodes / {cumulative} frames "
        f"({kept} kept anchors, {filtered} idle-filtered) -> {args.out}"
    )


if __name__ == "__main__":
    main()
