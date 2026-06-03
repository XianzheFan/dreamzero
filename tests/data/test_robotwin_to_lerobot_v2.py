import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_VECTOR = object()


def _load_converter():
    for dep in ("av", "cv2", "h5py", "pandas", "pyarrow"):
        pytest.importorskip(dep)

    module_name = "robotwin_to_lerobot_v2_for_test"
    module_path = _REPO_ROOT / "scripts" / "data" / "robotwin_to_lerobot_v2.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _joint_components(T=4):
    left_arm = np.arange(T * 7, dtype=np.float32).reshape(T, 7) * 0.01
    right_arm = 10.0 + np.arange(T * 7, dtype=np.float32).reshape(T, 7) * 0.01
    left_grip = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)[:T]
    right_grip = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)[:T]
    vector = np.concatenate(
        [left_arm, left_grip[:, None], right_arm, right_grip[:, None]],
        axis=1,
    )
    return left_arm, left_grip, right_arm, right_grip, vector


def _write_episode_hdf5(path, *, vector_override=_DEFAULT_VECTOR):
    h5py = pytest.importorskip("h5py")
    left_arm, left_grip, right_arm, right_grip, vector = _joint_components()
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
    return np.concatenate(
        [left_arm, left_grip[:, None], right_arm, right_grip[:, None]],
        axis=1,
    )


def test_read_state_and_action_uses_absolute_next_step_qpos_layout(tmp_path):
    h5py = pytest.importorskip("h5py")
    converter = _load_converter()
    expected = _write_episode_hdf5(tmp_path / "episode0.hdf5")

    with h5py.File(tmp_path / "episode0.hdf5", "r") as traj:
        state, action = converter._read_state_and_action(traj)

    np.testing.assert_allclose(state, expected[:-1])
    np.testing.assert_allclose(action, expected[1:])


def test_joint_action_vector_mismatch_raises(tmp_path):
    h5py = pytest.importorskip("h5py")
    converter = _load_converter()
    _, _, _, _, vector = _joint_components()
    vector[:, [7, 15]] = vector[:, [15, 7]]
    _write_episode_hdf5(tmp_path / "episode0.hdf5", vector_override=vector)

    with h5py.File(tmp_path / "episode0.hdf5", "r") as traj:
        with pytest.raises(ValueError, match="/joint_action/vector"):
            converter._read_state_and_action(traj)


def test_missing_joint_action_vector_is_allowed(tmp_path):
    h5py = pytest.importorskip("h5py")
    converter = _load_converter()
    expected = _write_episode_hdf5(tmp_path / "episode0.hdf5", vector_override=None)

    with h5py.File(tmp_path / "episode0.hdf5", "r") as traj:
        state, action = converter._read_state_and_action(traj)

    np.testing.assert_allclose(state, expected[:-1])
    np.testing.assert_allclose(action, expected[1:])
