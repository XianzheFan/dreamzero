import importlib.util
from pathlib import Path

import numpy as np


def _load_analyzer_module():
    path = Path(__file__).resolve().parents[2] / "scripts/eval/analyze_gripper_dump.py"
    spec = importlib.util.spec_from_file_location("analyze_gripper_dump_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_env_trace_debug_summarizes_liftbarrier_contact_metrics():
    mod = _load_analyzer_module()
    columns = (
        "step",
        "barrier_z",
        "success_margin",
        "left_tcp_to_barrier",
        "right_tcp_to_barrier",
        "left_tcp_to_grasp_target",
        "right_tcp_to_grasp_target",
        "left_grasping",
        "right_grasping",
    )
    trace = np.asarray(
        [
            [0, 0.20, 0.05, 0.30, 0.40, 0.20, 0.25, 0, 0],
            [5, 0.28, 0.13, 0.08, 0.15, 0.03, 0.04, 1, 0],
            [6, 0.31, 0.16, 0.07, 0.12, 0.02, 0.03, 1, 1],
        ],
        dtype=np.float32,
    )

    debug = mod._env_trace_debug(trace, columns)

    np.testing.assert_allclose(debug["barrier_z"]["start"], 0.20)
    np.testing.assert_allclose(debug["barrier_z"]["final"], 0.31)
    np.testing.assert_allclose(debug["success_margin"]["max"], 0.16)
    assert debug["barrier_z"]["max_step"] == 6
    assert debug["success_margin"]["max_step"] == 6
    np.testing.assert_allclose(debug["left_tcp_to_barrier"]["min"], 0.07)
    assert debug["left_tcp_to_barrier"]["min_step"] == 6
    np.testing.assert_allclose(debug["right_tcp_to_barrier"]["final"], 0.12)
    np.testing.assert_allclose(debug["left_tcp_to_grasp_target"]["min"], 0.02)
    assert debug["left_tcp_to_grasp_target"]["min_step"] == 6
    np.testing.assert_allclose(debug["right_tcp_to_grasp_target"]["min"], 0.03)
    assert debug["right_tcp_to_grasp_target"]["min_step"] == 6
    assert debug["left_grasp_count"] == 2
    assert debug["right_grasp_count"] == 1
    assert debug["left_first_grasp_step"] == 5
    assert debug["right_first_grasp_step"] == 6


def test_analyze_episode_uses_executed_chunk_start_for_first_cmd_delta(tmp_path):
    mod = _load_analyzer_module()
    path = tmp_path / "episode_1000.npz"
    action_dim = 16
    pred_chunk = np.zeros((2, 4, action_dim), dtype=np.float32)
    obs_qpos = np.zeros((2, action_dim), dtype=np.float32)
    joint_dims = [dim for dim in range(action_dim) if dim not in (7, 15)]

    pred_chunk[:, 0, joint_dims] = 100.0
    pred_chunk[0, 2, joint_dims] = 0.25
    pred_chunk[1, 1, joint_dims] = 0.75
    np.savez_compressed(
        path,
        seed=np.asarray(1000, dtype=np.int32),
        success=np.asarray(False),
        exec_action=np.zeros((3, action_dim), dtype=np.float32),
        pred_chunk=pred_chunk,
        obs_qpos=obs_qpos,
        infer_step=np.asarray([0, 24], dtype=np.int32),
        exec_chunk_start_index=np.asarray([2, 1], dtype=np.int32),
        exec_chunk_stop_index=np.asarray([4, 3], dtype=np.int32),
    )

    episode = mod.analyze_episode(
        str(path),
        close_threshold=0.0,
        decisive_threshold=-0.2,
        print_profiles=False,
        num_arms=None,
        arm_dim=8,
        gripper_offset=7,
        explicit_gripper_dims=None,
        custom_arm_labels=None,
    )

    np.testing.assert_allclose(episode["first_cmd_delta_mean"], 0.5)
    np.testing.assert_allclose(episode["first_cmd_delta_max"], 0.75)
    assert episode["first_cmd_chunk_index_min"] == 1
    assert episode["first_cmd_chunk_index_max"] == 2
