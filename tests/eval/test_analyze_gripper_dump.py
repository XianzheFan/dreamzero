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
