import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_VECTOR = object()


def _load_inspector():
    pytest.importorskip("h5py")

    module_name = "inspect_robotwin_raw_hdf5_for_test"
    module_path = _REPO_ROOT / "scripts" / "data" / "inspect_robotwin_raw_hdf5.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _joint_components(T=5):
    left_arm = np.arange(T * 7, dtype=np.float32).reshape(T, 7) * 0.01
    right_arm = 10.0 + np.arange(T * 7, dtype=np.float32).reshape(T, 7) * 0.01
    left_grip = np.array([1.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)[:T]
    right_grip = np.array([1.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)[:T]
    vector = np.concatenate(
        [left_arm, left_grip[:, None], right_arm, right_grip[:, None]],
        axis=1,
    )
    return left_arm, left_grip, right_arm, right_grip, vector


def _write_episode_hdf5(path, *, vector_override=_DEFAULT_VECTOR, left_grip_override=None):
    h5py = pytest.importorskip("h5py")
    left_arm, left_grip, right_arm, right_grip, vector = _joint_components()
    if left_grip_override is not None:
        left_grip = np.asarray(left_grip_override, dtype=np.float32)
        vector = np.concatenate(
            [left_arm, left_grip[:, None], right_arm, right_grip[:, None]],
            axis=1,
        )
    if vector_override is not _DEFAULT_VECTOR:
        vector = vector_override

    with h5py.File(path, "w") as f:
        group = f.create_group("joint_action")
        group.create_dataset("left_arm", data=left_arm)
        group.create_dataset("left_gripper", data=left_grip)
        group.create_dataset("right_arm", data=right_arm)
        group.create_dataset("right_gripper", data=right_grip)
        if vector is not None:
            group.create_dataset("vector", data=vector)


def test_inspect_raw_episodes_reports_next_step_action_and_gripper_stats(tmp_path):
    inspector = _load_inspector()
    _write_episode_hdf5(tmp_path / "episode0.hdf5")
    _write_episode_hdf5(tmp_path / "episode1.hdf5")

    summary = inspector.inspect_raw_episodes(
        tmp_path,
        expected_episodes=2,
        sample_episodes=1,
        require_vector=True,
        allow_static_gripper=False,
    )

    assert summary["episode_count"] == 2
    assert summary["sample_episode_count"] == 1
    assert summary["sample_frames_after_shift"] == 4
    assert summary["implied_action"] == "absolute next-frame joint_action state"
    assert summary["gripper_convention"] == "0.0 close, 1.0 open"
    assert summary["gripper_summary"]["dim_7"]["unique_rounded_count"] == 2
    assert summary["gripper_summary"]["dim_15"]["unique_rounded_count"] == 2
    assert summary["joint_delta_summary"]["abs_max"] > 0.0


def test_inspect_raw_episodes_rejects_vector_layout_mismatch(tmp_path):
    inspector = _load_inspector()
    _, _, _, _, vector = _joint_components()
    vector[:, [7, 15]] = vector[:, [15, 7]]
    _write_episode_hdf5(tmp_path / "episode0.hdf5", vector_override=vector)

    with pytest.raises(ValueError, match="/joint_action/vector"):
        inspector.inspect_raw_episodes(tmp_path, require_vector=True)


def test_inspect_raw_episodes_rejects_gripper_outside_robotwin_range(tmp_path):
    inspector = _load_inspector()
    _write_episode_hdf5(
        tmp_path / "episode0.hdf5",
        left_grip_override=np.array([1.0, 0.5, -1.0, 0.0, 1.0], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="outside expected range"):
        inspector.inspect_raw_episodes(tmp_path, require_vector=True)
