import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _install_import_stub(name: str, **attrs) -> None:
    if name in sys.modules:
        return
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, AttributeError):
        spec = None
    if spec is not None:
        return
    module = types.ModuleType(name)
    for attr_name, attr_value in attrs.items():
        setattr(module, attr_name, attr_value)
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, "__path__", getattr(parent, "__path__", []))
            setattr(parent, child_name, module)
    sys.modules[name] = module


def _load_converter():
    # ``write_meta`` only needs json/numpy/pathlib, but the converter imports
    # video/parquet dependencies at module import time. Stub them so this
    # metadata regression test stays lightweight.
    _install_import_stub("av")
    _install_import_stub("h5py")
    _install_import_stub("pandas")
    _install_import_stub("pyarrow")
    _install_import_stub("pyarrow.parquet")
    _install_import_stub("tqdm", tqdm=lambda iterable=None, **_: iterable)
    module_name = "robofactory_to_lerobot_v2_for_test"
    module_path = _REPO_ROOT / "scripts" / "data" / "robofactory_to_lerobot_v2.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _sample_state_action():
    state = np.zeros((4, 16), dtype=np.float32)
    action = np.zeros((4, 16), dtype=np.float32)
    state[:, 0:7] = np.linspace(0.0, 0.6, 7, dtype=np.float32)
    state[:, 8:15] = np.linspace(1.0, 1.6, 7, dtype=np.float32)
    state[:, 7] = [0.04, 0.04, 0.0, 0.0]
    state[:, 15] = [0.0, 0.04, 0.04, 0.0]
    action[:, 0:7] = state[:, 0:7] + 0.01
    action[:, 8:15] = state[:, 8:15] - 0.01
    action[:, 7] = [1.0, -1.0, -1.0, 1.0]
    action[:, 15] = [-1.0, 1.0, -1.0, 1.0]
    return state, action


def test_write_meta_uses_robofactory_tag_and_gripper_slices(tmp_path):
    converter = _load_converter()
    state, action = _sample_state_action()

    converter.write_meta(
        out_root=tmp_path,
        task_text="lift the barrier",
        num_episodes=1,
        total_frames=state.shape[0],
        episode_lengths=[state.shape[0]],
        sample_video_hw=(240, 320),
        actions=[action],
        states=[state],
        num_arms=2,
    )

    embodiment = json.loads((tmp_path / "meta" / "embodiment.json").read_text())
    modality = json.loads((tmp_path / "meta" / "modality.json").read_text())
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    stats = json.loads((tmp_path / "meta" / "stats.json").read_text())

    assert embodiment == {
        "robot_type": "2_panda_robofactory",
        "embodiment_tag": "robofactory",
    }
    assert "robotwin" not in json.dumps(embodiment)
    assert info["features"]["action"]["shape"] == [16]
    assert info["features"]["observation.state"]["shape"] == [16]
    assert info["features"]["action"]["names"][7] == "panda0_gripper.pos"
    assert info["features"]["action"]["names"][15] == "panda1_gripper.pos"

    assert modality["action"]["panda0_joint_pos"]["start"] == 0
    assert modality["action"]["panda0_joint_pos"]["end"] == 7
    assert modality["action"]["panda0_gripper_pos"]["start"] == 7
    assert modality["action"]["panda0_gripper_pos"]["end"] == 8
    assert modality["action"]["panda1_joint_pos"]["start"] == 8
    assert modality["action"]["panda1_joint_pos"]["end"] == 15
    assert modality["action"]["panda1_gripper_pos"]["start"] == 15
    assert modality["action"]["panda1_gripper_pos"]["end"] == 16
    assert modality["video"]["global_camera-images-rgb"]["original_key"] == (
        "observation.images.global"
    )
    assert modality["video"]["agent0_camera-images-rgb"]["original_key"] == (
        "observation.images.agent0"
    )
    assert modality["video"]["agent1_camera-images-rgb"]["original_key"] == (
        "observation.images.agent1"
    )

    assert stats["action"]["min"][7] == -1.0
    assert stats["action"]["max"][7] == 1.0
    assert stats["action"]["min"][15] == -1.0
    assert stats["action"]["max"][15] == 1.0
