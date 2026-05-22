"""Convert a RoboFactory ManiSkill .h5 demonstration set into a
LeRobot v2 layout that the DreamZero ``multi-agent`` branch can consume
via the ``robofactory`` embodiment tag.

Source layout (one task per .h5, as produced by RoboFactory's
``script/generate_data.py``)::

    data/h5_data/{task}.h5
    data/h5_data/{task}.json

Each trajectory in the .h5 has::

    obs/agent/panda-0/qpos             [T,   9]    (7 arm + 2 finger)
    obs/agent/panda-1/qpos             [T,   9]
    obs/sensor_data/head_camera_agent0/rgb  [T, H, W, 3]  uint8
    obs/sensor_data/head_camera_agent1/rgb  [T, H, W, 3]
    obs/sensor_data/head_camera_global/rgb  [T, H, W, 3]
    actions/panda-0                    [T-1, 8]    (7 joint deltas + 1 gripper)
    actions/panda-1                    [T-1, 8]

Output (the LeRobot v2 schema expected by
``ShardedLeRobotSubLangSingleActionChunkDatasetDROID``)::

    {out_dir}/
        data/chunk-000/episode_000000.parquet ...
        videos/chunk-000/observation.images.global/episode_000000.mp4
        videos/chunk-000/observation.images.agent0/episode_000000.mp4
        videos/chunk-000/observation.images.agent1/episode_000000.mp4
        meta/{info,modality,episodes,tasks,stats,embodiment}.{json,jsonl}

State / action concat layout (per row, 16 dims total)::

    [panda0_joint(0:7), panda0_gripper(7:8), panda1_joint(8:15), panda1_gripper(15:16)]

Usage::

    python scripts/data/robofactory_to_lerobot_v2.py \\
        --h5 /path/to/RoboFactory/robofactory/data/h5_data/LiftBarrier-rf.h5 \\
        --out /path/to/lerobot_v2/LiftBarrier-rf \\
        --task "the two robot arms lift the barrier together" \\
        --num-episodes 150
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

FPS = 20
CHUNK_SIZE = 1000
CAMERAS = {
    "observation.images.global": "head_camera_global",
    "observation.images.agent0": "head_camera_agent0",
    "observation.images.agent1": "head_camera_agent1",
}
# Per-arm state slice: qpos[:8] = 7 arm joints + 1 finger joint.
ARM_STATE_DIM = 8
ARM_ACTION_DIM = 8
STATE_DIM = 2 * ARM_STATE_DIM  # 16
ACTION_DIM = 2 * ARM_ACTION_DIM  # 16


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


def _per_arm_state(traj: h5py.Group, arm: str) -> np.ndarray:
    """qpos[:, :8] for the given arm; first 7 are joints, dim 7 is one
    gripper finger (the two fingers mirror each other so one is enough)."""
    qpos = traj[f"obs/agent/{arm}/qpos"][:]  # [T, 9]
    return qpos[:, :ARM_STATE_DIM].astype(np.float32)


def _per_arm_action(traj: h5py.Group, arm: str) -> np.ndarray:
    """actions/{arm} -> [T-1, 8] (joint deltas + gripper)."""
    return traj[f"actions/{arm}"][:].astype(np.float32)


def convert_episode(
    traj: h5py.Group,
    episode_index: int,
    task_index: int,
    out_root: Path,
    cumulative_index: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Write one .parquet + N .mp4 files for one trajectory.

    Returns ``(length, action_buffer, state_buffer)`` so callers can
    accumulate global stats and the index counter.
    """
    panda0_state = _per_arm_state(traj, "panda-0")
    panda1_state = _per_arm_state(traj, "panda-1")
    panda0_action = _per_arm_action(traj, "panda-0")
    panda1_action = _per_arm_action(traj, "panda-1")

    # ManiSkill emits one extra obs at the terminal step (no paired
    # action). Trim to min length so each row has a real action.
    T = min(len(panda0_state), len(panda0_action))
    state = np.concatenate([panda0_state[:T], panda1_state[:T]], axis=1)  # [T, 16]
    action = np.concatenate([panda0_action[:T], panda1_action[:T]], axis=1)  # [T, 16]

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
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                   data_dir / f"episode_{episode_index:06d}.parquet")

    for video_key, h5_cam in CAMERAS.items():
        rgb = traj[f"obs/sensor_data/{h5_cam}/rgb"][:T]  # [T, H, W, 3] uint8
        video_dir = out_root / "videos" / chunk_dir / video_key
        video_dir.mkdir(parents=True, exist_ok=True)
        encode_video(rgb, video_dir / f"episode_{episode_index:06d}.mp4", FPS)

    return T, action, state


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

    state_names = [
        *(f"panda0_joint_{i}.pos" for i in range(7)),
        "panda0_gripper.pos",
        *(f"panda1_joint_{i}.pos" for i in range(7)),
        "panda1_gripper.pos",
    ]
    info = {
        "codebase_version": "v2.0",
        "robot_type": "bi_panda_robofactory",
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
                "rotation_type": None, "absolute": False, "dtype": "float32", "range": None,
            },
            "panda0_gripper_pos": {
                "original_key": "action",
                "start": 7, "end": 8,
                "rotation_type": None, "absolute": True, "dtype": "float32", "range": None,
            },
            "panda1_joint_pos": {
                "original_key": "action",
                "start": 8, "end": 15,
                "rotation_type": None, "absolute": False, "dtype": "float32", "range": None,
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
        "robot_type": "bi_panda_robofactory",
        "embodiment_tag": "robofactory",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True, type=Path,
                        help="Path to RoboFactory .h5 file (e.g. data/h5_data/LiftBarrier-rf.h5)")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output directory for LeRobot v2 dataset")
    parser.add_argument("--task", required=True, type=str,
                        help="Natural-language task description (used by tasks.jsonl)")
    parser.add_argument("--num-episodes", type=int, default=-1,
                        help="Number of episodes to convert (-1 for all)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.h5, "r") as f:
        # Sort by integer suffix so episode_000000 corresponds to traj_0.
        traj_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
        if args.num_episodes > 0:
            traj_keys = traj_keys[: args.num_episodes]

        episode_lengths: list[int] = []
        actions_buf: list[np.ndarray] = []
        states_buf: list[np.ndarray] = []
        cumulative = 0
        sample_hw: tuple[int, int] | None = None

        for ep_idx, key in enumerate(tqdm(traj_keys, desc="episodes")):
            traj = f[key]
            if sample_hw is None:
                rgb_shape = traj[f"obs/sensor_data/head_camera_global/rgb"].shape
                sample_hw = (rgb_shape[1], rgb_shape[2])
            length, action, state = convert_episode(
                traj=traj,
                episode_index=ep_idx,
                task_index=0,
                out_root=args.out,
                cumulative_index=cumulative,
            )
            cumulative += length
            episode_lengths.append(length)
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
