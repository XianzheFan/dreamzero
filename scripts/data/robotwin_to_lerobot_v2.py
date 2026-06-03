"""Convert a RoboTwin 2.0 franka-panda demonstration set into a LeRobot v2
layout consumable by the DreamZero ``multi-agent`` branch under the dedicated
``robotwin`` embodiment tag (two Panda/Franka arms with 7 joints + 1 gripper
per arm).

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

Output (LeRobot v2 schema, using Panda-style modality names under the
RoboTwin-specific transform namespace)::

    {out_dir}/
        data/chunk-000/episode_000000.parquet
        videos/chunk-000/observation.images.global/episode_000000.mp4
        videos/chunk-000/observation.images.agent0/episode_000000.mp4
        videos/chunk-000/observation.images.agent1/episode_000000.mp4
        meta/{info,modality,episodes,tasks,stats,embodiment}.{json,jsonl}

State / action layout (16 dims total)::

    [left_arm_joint(0:7), left_gripper(7:8),
     right_arm_joint(8:15), right_gripper(15:16)]

Actions are stored as absolute next-step qpos targets. DreamZero's
``relative_action`` training path then converts the joint targets to
``target_qpos - current_qpos`` on the fly and normalizes those relative
joint offsets, matching the DROID-style pretraining convention. Grippers
remain absolute 0/1 targets.

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


def _validate_joint_action_vector(traj: h5py.File, state: np.ndarray) -> None:
    """Check RoboTwin's canonical joint-action vector when present.

    The converter intentionally writes ``state`` from the named component
    datasets so the layout is explicit. RoboTwin also stores the same value
    as ``/joint_action/vector``; validating it catches silent left/right or
    gripper ordering drift in upstream collection code.
    """
    vector_path = "/joint_action/vector"
    if vector_path not in traj:
        return

    vector = np.asarray(traj[vector_path][()], dtype=np.float32)
    if vector.shape != state.shape:
        raise ValueError(
            f"{vector_path} shape {vector.shape} does not match concatenated "
            f"joint_action shape {state.shape}"
        )

    if not np.allclose(vector, state, atol=1e-5, rtol=1e-5):
        max_abs_diff = float(np.max(np.abs(vector - state)))
        raise ValueError(
            f"{vector_path} does not match concatenated left_arm, left_gripper, "
            f"right_arm, right_gripper layout; max_abs_diff={max_abs_diff:.6g}"
        )


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
    _validate_joint_action_vector(traj, state)

    # Absolute next-step qpos target. Joint targets are converted to
    # relative offsets by the DreamZero dataset loader when
    # ``relative_action`` is enabled; grippers are kept absolute.
    action = state[1:].copy()
    return state[:-1], action                      # T-1 rows each


def convert_episode(
    traj_path: Path,
    episode_index: int,
    task_index: int,
    out_root: Path,
    cumulative_index: int,
) -> tuple[int, np.ndarray, np.ndarray, tuple[int, int]]:
    """Write one .parquet + N .mp4 files for one episode."""
    with h5py.File(traj_path, "r") as traj:
        state, action = _read_state_and_action(traj)
        T = state.shape[0]

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
    return T, action, state, sample_hw


def write_meta(
    out_root: Path,
    task_text: str,
    num_episodes: int,
    total_frames: int,
    episode_lengths: list[int],
    sample_video_hw: tuple[int, int],
    actions: list[np.ndarray],
    states: list[np.ndarray],
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

    # Keep Panda-style state/action naming so the tensor layout is explicit,
    # but use the dedicated robotwin embodiment tag/config. RoboTwin and
    # RoboFactory have compatible 7+1 per-arm slices but different simulator
    # gripper semantics, so their config namespaces stay separate.
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

    (meta / "embodiment.json").write_text(json.dumps({
        "robot_type": "bi_panda_robotwin",
        "embodiment_tag": "robotwin",
    }, indent=2))

    all_actions = np.concatenate(actions, axis=0)
    all_states = np.concatenate(states, axis=0)

    def per_dim_stats(arr: np.ndarray) -> dict[str, list[float]]:
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": (arr.std(axis=0) + 1e-8).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }

    stats = {
        "observation.state": per_dim_stats(all_states.astype(np.float32)),
        "action": per_dim_stats(all_actions.astype(np.float32)),
        "timestamp": per_dim_stats(np.array([[0.0]], dtype=np.float32)),
    }
    (meta / "stats.json").write_text(json.dumps(stats, indent=2))

    # If the user reuses an output directory that was previously converted
    # with a different action convention, force DreamZero to recalculate
    # relative stats from the new absolute action rows on the next train.
    for stale_name in (
        "relative_stats_dreamzero.json",
        "relative_horizon_stats_dreamzero.json",
    ):
        stale_path = meta / stale_name
        if stale_path.exists():
            stale_path.unlink()


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
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    episode_files = _list_episodes(args.episode_dir)
    if args.num_episodes > 0:
        episode_files = episode_files[: args.num_episodes]

    episode_lengths: list[int] = []
    actions_buf: list[np.ndarray] = []
    states_buf: list[np.ndarray] = []
    cumulative = 0
    sample_hw: tuple[int, int] | None = None

    for ep_idx, fp in enumerate(tqdm(episode_files, desc="episodes")):
        T, action, state, hw = convert_episode(
            traj_path=fp,
            episode_index=ep_idx,
            task_index=0,
            out_root=args.out,
            cumulative_index=cumulative,
        )
        if sample_hw is None:
            sample_hw = hw
        cumulative += T
        episode_lengths.append(T)
        actions_buf.append(action)
        states_buf.append(state)

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
    )
    print(f"Done. {len(episode_lengths)} episodes / {cumulative} frames -> {args.out}")


if __name__ == "__main__":
    main()
