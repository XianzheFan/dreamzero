"""Convert a RoboFactory ManiSkill .h5 demonstration set into a
LeRobot v2 layout that the DreamZero ``multi-agent`` branch can consume
via the ``robofactory`` embodiment tag.

The script auto-detects the agent count from the .h5 (counts
``obs/agent/panda-N`` subgroups), so the same code path handles 2-arm,
3-arm and 4-arm RoboFactory tasks.

Source layout (one task per .h5, as produced by RoboFactory's
``script/generate_data.py``). Each trajectory has, for ``N``
agents 0..N-1::

    obs/agent/panda-i/qpos                       [T,   9]    (7 arm + 2 finger)
    obs/sensor_data/head_camera_global/rgb       [T, H, W, 3]  uint8
    obs/sensor_data/head_camera_agent{i}/rgb     [T, H, W, 3]
    actions/panda-i                              [T-1, 8]    (7 absolute joint targets + 1 gripper cmd)

Output (the LeRobot v2 schema expected by
``ShardedLeRobotSubLangSingleActionChunkDatasetDROID``)::

    {out_dir}/
        data/chunk-000/episode_000000.parquet ...
        videos/chunk-000/observation.images.global/episode_000000.mp4
        videos/chunk-000/observation.images.agent{i}/episode_000000.mp4
        meta/{info,modality,episodes,tasks,stats,embodiment}.{json,jsonl}

State / action concat layout (per row, ``num_arms * 8`` dims total)::

    [panda0_joint(0:7), panda0_gripper(7:8),
     panda1_joint(8:15), panda1_gripper(15:16),
     ... up to N ...]

Actions are stored as absolute controller targets, matching DreamZero's
raw LeRobot action convention. During training the dataset config enables
``relative_action`` for joint keys, so the loader converts joint targets
to ``target_joint - current_qpos`` on the fly while grippers remain
absolute commands.

Usage::

    python scripts/data/robofactory_to_lerobot_v2.py \\
        --h5 /path/to/RoboFactory/robofactory/data/h5_data/TakePhoto-rf.h5 \\
        --out /path/to/lerobot_v2/TakePhoto-rf \\
        --task "the four robot arms cooperate to take a photo" \\
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
# Per-arm state slice: qpos[:8] = 7 arm joints + 1 finger joint.
ARM_STATE_DIM = 8
ARM_ACTION_DIM = 8


def detect_num_arms(traj: h5py.Group) -> int:
    """Count ``obs/agent/panda-N`` groups in a trajectory."""
    agents = traj["obs/agent"]
    arms = [k for k in agents.keys() if k.startswith("panda-")]
    return len(arms)


def cameras_for_num_arms(num_arms: int) -> dict[str, str]:
    """LeRobot key -> h5 sensor key mapping for ``num_arms`` agents."""
    cams = {"observation.images.global": "head_camera_global"}
    for n in range(num_arms):
        cams[f"observation.images.agent{n}"] = f"head_camera_agent{n}"
    return cams


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
    """Raw ``actions/{arm}`` -> [T-1, 8].

    RoboFactory stores controller commands as absolute joint targets plus
    a gripper command. The LeRobot row keeps that raw convention; the
    DreamZero dataset loader converts joint targets to relative offsets
    when ``relative_action`` is enabled.
    """
    return traj[f"actions/{arm}"][:].astype(np.float32)


def convert_episode(
    traj: h5py.Group,
    episode_index: int,
    task_index: int,
    out_root: Path,
    cumulative_index: int,
    num_arms: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Write one .parquet + N .mp4 files for one trajectory.

    Returns ``(length, action_buffer, state_buffer)`` so callers can
    accumulate global stats and the index counter.
    """
    per_arm_state = [_per_arm_state(traj, f"panda-{n}") for n in range(num_arms)]
    per_arm_action = [_per_arm_action(traj, f"panda-{n}") for n in range(num_arms)]

    # ManiSkill emits one extra obs at the terminal step (no paired
    # action). Trim to min length so each row has a real action.
    T = min(
        min(len(s) for s in per_arm_state),
        min(len(a) for a in per_arm_action),
    )
    state = np.concatenate([s[:T] for s in per_arm_state], axis=1)
    action = np.concatenate([a[:T] for a in per_arm_action], axis=1)

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

    cameras = cameras_for_num_arms(num_arms)
    for video_key, h5_cam in cameras.items():
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
    num_arms: int,
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

    state_dim = num_arms * ARM_STATE_DIM
    action_dim = num_arms * ARM_ACTION_DIM
    state_names: list[str] = []
    for n in range(num_arms):
        state_names.extend(f"panda{n}_joint_{i}.pos" for i in range(7))
        state_names.append(f"panda{n}_gripper.pos")

    features = {
        "action": {"dtype": "float32", "names": state_names, "shape": [action_dim]},
        "observation.state": {"dtype": "float32", "names": state_names, "shape": [state_dim]},
        "observation.images.global": video_feature_template,
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for n in range(num_arms):
        features[f"observation.images.agent{n}"] = video_feature_template

    info = {
        "codebase_version": "v2.0",
        "robot_type": f"{num_arms}_panda_robofactory",
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2))

    state_modality: dict[str, dict] = {}
    action_modality: dict[str, dict] = {}
    video_modality: dict[str, dict] = {
        "global_camera-images-rgb": {"original_key": "observation.images.global"},
    }
    for n in range(num_arms):
        s_start = n * ARM_STATE_DIM
        a_start = n * ARM_ACTION_DIM
        state_modality[f"panda{n}_joint_pos"] = {
            "original_key": "observation.state",
            "start": s_start, "end": s_start + 7,
            "rotation_type": None, "absolute": True,
            "dtype": "float32", "range": None,
        }
        state_modality[f"panda{n}_gripper_pos"] = {
            "original_key": "observation.state",
            "start": s_start + 7, "end": s_start + 8,
            "rotation_type": None, "absolute": True,
            "dtype": "float32", "range": None,
        }
        action_modality[f"panda{n}_joint_pos"] = {
            "original_key": "action",
            "start": a_start, "end": a_start + 7,
            # Stored as absolute target qpos. DreamZero's
            # ``relative_action`` training path turns this into
            # target_joint - current_qpos for Franka joint keys.
            "rotation_type": None, "absolute": True,
            "dtype": "float32", "range": None,
        }
        action_modality[f"panda{n}_gripper_pos"] = {
            "original_key": "action",
            "start": a_start + 7, "end": a_start + 8,
            "rotation_type": None, "absolute": True,
            "dtype": "float32", "range": None,
        }
        video_modality[f"agent{n}_camera-images-rgb"] = {
            "original_key": f"observation.images.agent{n}",
        }

    modality = {
        "state": state_modality,
        "action": action_modality,
        "video": video_modality,
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
        "robot_type": f"{num_arms}_panda_robofactory",
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

    # If an output directory is reused after changing action rows, force
    # DreamZero to recalculate relative stats from the current absolute
    # action targets on the next train.
    for stale_name in (
        "relative_stats_dreamzero.json",
        "relative_horizon_stats_dreamzero.json",
    ):
        stale_path = meta / stale_name
        if stale_path.exists():
            stale_path.unlink()


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
    parser.add_argument("--num-arms", type=int, default=None,
                        help="Optional sanity check: assert detected arm count matches.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.h5, "r") as f:
        # Sort by integer suffix so episode_000000 corresponds to traj_0.
        traj_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
        if args.num_episodes > 0:
            traj_keys = traj_keys[: args.num_episodes]

        # Detect arm count from the first trajectory. RoboFactory's
        # generator pins this per-task, so the first traj is authoritative.
        num_arms = detect_num_arms(f[traj_keys[0]])
        print(f"Detected num_arms = {num_arms}")
        if args.num_arms is not None and args.num_arms != num_arms:
            raise ValueError(
                f"--num-arms={args.num_arms} disagrees with .h5 ({num_arms})."
            )

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
                num_arms=num_arms,
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
        num_arms=num_arms,
    )
    print(f"Done. {len(episode_lengths)} episodes / {cumulative} frames -> {args.out}")


if __name__ == "__main__":
    main()
