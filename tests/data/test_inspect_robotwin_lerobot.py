import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_inspector():
    pytest.importorskip("pandas")
    module_name = "inspect_robotwin_lerobot_for_test"
    module_path = REPO_ROOT / "scripts/data/inspect_robotwin_lerobot.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_fake_dataset(
    root: Path,
    *,
    episodes: int = 2,
    static_gripper: bool = False,
    embodiment_tag: str = "robotwin",
):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    meta = root / "meta"
    meta.mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)

    info = {
        "total_episodes": episodes,
        "total_frames": episodes * 4,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "action": {"dtype": "float32", "names": None, "shape": [16]},
            "observation.state": {"dtype": "float32", "names": None, "shape": [16]},
        },
    }
    (meta / "info.json").write_text(json.dumps(info))
    modality = {
        "state": {
            "panda0_joint_pos": {"start": 0, "end": 7},
            "panda0_gripper_pos": {"start": 7, "end": 8},
            "panda1_joint_pos": {"start": 8, "end": 15},
            "panda1_gripper_pos": {"start": 15, "end": 16},
        },
        "action": {
            "panda0_joint_pos": {"start": 0, "end": 7},
            "panda0_gripper_pos": {"start": 7, "end": 8},
            "panda1_joint_pos": {"start": 8, "end": 15},
            "panda1_gripper_pos": {"start": 15, "end": 16},
        },
    }
    (meta / "modality.json").write_text(json.dumps(modality))
    (meta / "embodiment.json").write_text(
        json.dumps({"robot_type": "bi_panda_robotwin", "embodiment_tag": embodiment_tag})
    )

    with (meta / "episodes.jsonl").open("w") as f:
        for episode_index in range(episodes):
            f.write(json.dumps({"episode_index": episode_index, "length": 4}) + "\n")

    for episode_index in range(episodes):
        action = np.zeros((4, 16), dtype=np.float32)
        state = np.zeros((4, 16), dtype=np.float32)
        if static_gripper:
            action[:, [7, 15]] = 1.0
            state[:, [7, 15]] = 1.0
        else:
            action[:, 7] = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
            action[:, 15] = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
            state[:, [7, 15]] = action[:, [7, 15]]
        df = pd.DataFrame(
            {
                "action": list(action),
                "observation.state": list(state),
            }
        )
        df.to_parquet(root / f"data/chunk-000/episode_{episode_index:06d}.parquet")

    all_actions = []
    all_states = []
    for episode_index in range(episodes):
        action = np.zeros((4, 16), dtype=np.float32)
        state = np.zeros((4, 16), dtype=np.float32)
        if static_gripper:
            action[:, [7, 15]] = 1.0
            state[:, [7, 15]] = 1.0
        else:
            action[:, 7] = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
            action[:, 15] = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
            state[:, [7, 15]] = action[:, [7, 15]]
        all_actions.append(action)
        all_states.append(state)

    def _stats(array: np.ndarray):
        return {
            "mean": array.mean(axis=0).tolist(),
            "std": (array.std(axis=0) + 1e-8).tolist(),
            "min": array.min(axis=0).tolist(),
            "max": array.max(axis=0).tolist(),
            "q01": np.quantile(array, 0.01, axis=0).tolist(),
            "q99": np.quantile(array, 0.99, axis=0).tolist(),
        }

    (meta / "stats.json").write_text(
        json.dumps(
            {
                "action": _stats(np.concatenate(all_actions, axis=0)),
                "observation.state": _stats(np.concatenate(all_states, axis=0)),
            }
        )
    )


def test_inspect_robotwin_lerobot_reports_gripper_stats(tmp_path):
    inspector = _load_inspector()
    _write_fake_dataset(tmp_path)

    summary = inspector.inspect_dataset(
        root=tmp_path,
        expected_episodes=2,
        expected_action_dim=16,
        expected_state_dim=16,
        gripper_dims=(7, 15),
        close_threshold=0.5,
        gripper_min=0.0,
        gripper_max=1.0,
        gripper_range_epsilon=1e-4,
        sample_episodes=0,
        allow_static_gripper=False,
    )

    assert summary["episodes"] == 2
    assert summary["frames_read"] == 8
    assert summary["embodiment_tag"] == "robotwin"
    assert summary["gripper_dims"] == [7, 15]
    assert summary["action_gripper"]["dim_7"]["close_fraction_raw_lt_threshold"] == 0.5
    assert summary["action_gripper"]["dim_15"]["close_fraction_raw_lt_threshold"] == 0.5
    assert summary["expected_gripper_range"] == [0.0, 1.0]
    assert summary["action_gripper"]["dim_7"]["q01"] == 0.0
    assert summary["action_gripper"]["dim_7"]["q99"] == 1.0
    assert summary["action_gripper"]["dim_7"]["normalized_close_threshold_q99"] == 0.0
    assert summary["action_gripper"]["dim_15"]["normalized_min_q99"] == -1.0
    assert summary["action_gripper"]["dim_15"]["normalized_max_q99"] == 1.0


def test_inspect_robotwin_lerobot_fails_when_dataset_too_small(tmp_path):
    inspector = _load_inspector()
    _write_fake_dataset(tmp_path, episodes=1)

    with pytest.raises(ValueError, match="expected at least 2"):
        inspector.inspect_dataset(
            root=tmp_path,
            expected_episodes=2,
            expected_action_dim=16,
            expected_state_dim=16,
            gripper_dims=(7, 15),
            close_threshold=0.5,
            gripper_min=0.0,
            gripper_max=1.0,
            gripper_range_epsilon=1e-4,
            sample_episodes=0,
            allow_static_gripper=False,
        )


def test_inspect_robotwin_lerobot_accepts_legacy_robofactory_tag(tmp_path):
    inspector = _load_inspector()
    _write_fake_dataset(tmp_path, embodiment_tag="robofactory")

    summary = inspector.inspect_dataset(
        root=tmp_path,
        expected_episodes=2,
        expected_action_dim=16,
        expected_state_dim=16,
        gripper_dims=(7, 15),
        close_threshold=0.5,
        gripper_min=0.0,
        gripper_max=1.0,
        gripper_range_epsilon=1e-4,
        sample_episodes=1,
        allow_static_gripper=False,
    )

    assert summary["embodiment_tag"] == "robofactory"


def test_inspect_robotwin_lerobot_rejects_non_robotwin_gripper_range(tmp_path):
    pd = pytest.importorskip("pandas")
    inspector = _load_inspector()
    _write_fake_dataset(tmp_path)
    parquet = tmp_path / "data/chunk-000/episode_000000.parquet"
    df = pd.read_parquet(parquet)
    actions = [np.asarray(value, dtype=np.float32).copy() for value in df["action"]]
    actions[0][7] = -1.0
    df["action"] = actions
    df.to_parquet(parquet)

    with pytest.raises(ValueError, match="outside expected Robotwin range"):
        inspector.inspect_dataset(
            root=tmp_path,
            expected_episodes=2,
            expected_action_dim=16,
            expected_state_dim=16,
            gripper_dims=(7, 15),
            close_threshold=0.5,
            gripper_min=0.0,
            gripper_max=1.0,
            gripper_range_epsilon=1e-4,
            sample_episodes=0,
            allow_static_gripper=False,
        )


def test_inspect_robotwin_lerobot_fails_on_static_gripper(tmp_path):
    inspector = _load_inspector()
    _write_fake_dataset(tmp_path, static_gripper=True)

    with pytest.raises(ValueError, match="static|open/close"):
        inspector.inspect_dataset(
            root=tmp_path,
            expected_episodes=2,
            expected_action_dim=16,
            expected_state_dim=16,
            gripper_dims=(7, 15),
            close_threshold=0.5,
            gripper_min=0.0,
            gripper_max=1.0,
            gripper_range_epsilon=1e-4,
            sample_episodes=0,
            allow_static_gripper=False,
        )
