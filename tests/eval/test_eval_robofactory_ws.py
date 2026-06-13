import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _load_eval_module(monkeypatch):
    robofactory = types.ModuleType("robofactory")
    tasks = types.ModuleType("robofactory.tasks")
    monkeypatch.setitem(sys.modules, "robofactory", robofactory)
    monkeypatch.setitem(sys.modules, "robofactory.tasks", tasks)

    path = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "eval_robofactory_ws.py"
    spec = importlib.util.spec_from_file_location("eval_robofactory_ws_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_close_after_step_does_not_force_open_before_threshold(monkeypatch):
    mod = _load_eval_module(monkeypatch)
    action = np.zeros(16, dtype=np.float32)
    action[7] = -1.0
    action[15] = -1.0

    out = mod.apply_gripper_override(
        action,
        step=10,
        mode="close-after-step",
        close_after_step=40,
        open_value=1.0,
        close_value=-1.0,
    )

    np.testing.assert_allclose(out[[7, 15]], [-1.0, -1.0])


def test_open_then_close_after_step_forces_schedule(monkeypatch):
    mod = _load_eval_module(monkeypatch)
    action = np.zeros(16, dtype=np.float32)
    action[7] = -1.0
    action[15] = -1.0

    early = mod.apply_gripper_override(
        action,
        step=10,
        mode="open-then-close-after-step",
        close_after_step=40,
        open_value=1.0,
        close_value=-1.0,
    )
    late = mod.apply_gripper_override(
        action,
        step=40,
        mode="open-then-close-after-step",
        close_after_step=40,
        open_value=1.0,
        close_value=-1.0,
    )

    np.testing.assert_allclose(early[[7, 15]], [1.0, 1.0])
    np.testing.assert_allclose(late[[7, 15]], [-1.0, -1.0])


def test_scale_joint_targets_preserves_grippers(monkeypatch):
    mod = _load_eval_module(monkeypatch)
    qpos = np.arange(16, dtype=np.float32)
    action = qpos.copy()
    action[0:7] += 0.1
    action[8:15] -= 0.2
    action[7] = -1.0
    action[15] = 1.0

    out = mod.scale_joint_targets(action, qpos, scale=2.0)

    np.testing.assert_allclose(out[0:7], qpos[0:7] + 0.2, atol=1e-6)
    np.testing.assert_allclose(out[8:15], qpos[8:15] - 0.4, atol=1e-6)
    np.testing.assert_allclose(out[[7, 15]], [-1.0, 1.0])


def test_scale_joint_targets_one_is_noop(monkeypatch):
    mod = _load_eval_module(monkeypatch)
    qpos = np.arange(16, dtype=np.float32)
    action = qpos + 0.5

    out = mod.scale_joint_targets(action, qpos, scale=1.0)

    np.testing.assert_allclose(out, action)


def test_scale_joint_targets_clip_limits_joint_delta_only(monkeypatch):
    mod = _load_eval_module(monkeypatch)
    qpos = np.zeros(16, dtype=np.float32)
    action = np.zeros(16, dtype=np.float32)
    action[0:7] = 1.0
    action[8:15] = -1.0
    action[7] = -1.0
    action[15] = 1.0

    out = mod.scale_joint_targets(action, qpos, scale=2.0, clip=0.25)

    np.testing.assert_allclose(out[0:7], 0.25, atol=1e-6)
    np.testing.assert_allclose(out[8:15], -0.25, atol=1e-6)
    np.testing.assert_allclose(out[[7, 15]], [-1.0, 1.0])


def test_scale_joint_targets_accepts_per_arm_scale(monkeypatch):
    mod = _load_eval_module(monkeypatch)
    qpos = np.zeros(16, dtype=np.float32)
    action = np.zeros(16, dtype=np.float32)
    action[0:7] = 0.1
    action[8:15] = -0.1
    action[7] = -1.0
    action[15] = 1.0

    out = mod.scale_joint_targets(
        action,
        qpos,
        scale=2.0,
        left_scale=3.0,
        right_scale=5.0,
    )

    np.testing.assert_allclose(out[0:7], 0.3, atol=1e-6)
    np.testing.assert_allclose(out[8:15], -0.5, atol=1e-6)
    np.testing.assert_allclose(out[[7, 15]], [-1.0, 1.0])


def test_prepare_env_action_target_scales_absolute_chunk_from_infer_qpos(monkeypatch):
    mod = _load_eval_module(monkeypatch)
    infer_qpos = np.zeros(16, dtype=np.float32)
    rolling_qpos = np.zeros(16, dtype=np.float32)
    rolling_qpos[0:7] = 0.8
    rolling_qpos[8:15] = -0.8
    action = np.zeros(16, dtype=np.float32)
    action[0:7] = 0.2
    action[8:15] = -0.3
    action[7] = -1.0
    action[15] = 1.0

    out = mod.prepare_env_action_target(
        action,
        rolling_qpos,
        infer_qpos,
        action_representation="absolute_qpos",
        joint_target_scale=2.0,
    )

    np.testing.assert_allclose(out[0:7], 0.4, atol=1e-6)
    np.testing.assert_allclose(out[8:15], -0.6, atol=1e-6)
    np.testing.assert_allclose(out[[7, 15]], [-1.0, 1.0])


def test_prepare_env_action_target_scales_legacy_delta_from_rolling_qpos(monkeypatch):
    mod = _load_eval_module(monkeypatch)
    infer_qpos = np.zeros(16, dtype=np.float32)
    rolling_qpos = np.zeros(16, dtype=np.float32)
    rolling_qpos[0:7] = 0.8
    rolling_qpos[8:15] = -0.8
    action = np.zeros(16, dtype=np.float32)
    action[0:7] = 0.2
    action[8:15] = -0.3
    action[7] = -1.0
    action[15] = 1.0

    out = mod.prepare_env_action_target(
        action,
        rolling_qpos,
        infer_qpos,
        action_representation="robotwin_delta",
        joint_target_scale=2.0,
    )

    np.testing.assert_allclose(out[0:7], 1.2, atol=1e-6)
    np.testing.assert_allclose(out[8:15], -1.4, atol=1e-6)
    np.testing.assert_allclose(out[[7, 15]], [-1.0, 1.0])


def test_collect_env_trace_records_barrier_tcp_and_gripper(monkeypatch):
    mod = _load_eval_module(monkeypatch)

    class Pose:
        def __init__(self, p):
            self._p = np.asarray(p, dtype=np.float32)
            self.p = self._p[None, :]

        def to_transformation_matrix(self):
            matrix = np.eye(4, dtype=np.float32)
            matrix[:3, 3] = self._p
            return matrix[None, ...]

    class Body:
        def __init__(self, p):
            self.pose = Pose(p)

    class Agent:
        def __init__(self, robot_p, tcp_p, grasping):
            self.robot = Body(robot_p)
            self.tcp = Body(tcp_p)
            self._grasping = grasping

        def is_grasping(self, _actor):
            return np.asarray([self._grasping])

    class MultiAgent:
        def __init__(self, agents):
            self.agents = agents

    class Env:
        @property
        def unwrapped(self):
            return self

    env = Env()
    env.barrier = Body([0.1, 0.2, 0.25])
    contact0 = np.eye(4, dtype=np.float32)
    contact1 = np.eye(4, dtype=np.float32)
    contact2 = np.eye(4, dtype=np.float32)
    contact1[:3, 3] = [0.0, 0.0, 0.05]
    contact2[:3, 3] = [0.0, 0.1, 0.0]
    env.annotation_data = {
        "barrier": {
            "scale": [1.0, 1.0, 1.0],
            "contact_points_pose": [contact0, contact1, contact2],
        }
    }
    env.agent = MultiAgent(
        [
            Agent([0.0, 0.0, 0.0], [0.1, 0.2, 0.30], True),
            Agent([0.0, 0.0, 0.0], [0.1, 0.3, 0.25], False),
        ]
    )
    action = np.zeros(16, dtype=np.float32)
    action[7] = 1.0
    action[15] = -1.0

    trace = mod.collect_env_trace(env, step=12, action16=action, info={"success": np.asarray([False])})
    columns = list(mod.ENV_TRACE_COLUMNS)
    values = dict(zip(columns, trace))

    assert values["step"] == 12.0
    assert values["barrier_z"] == np.float32(0.25)
    assert values["success"] == 0.0
    assert values["left_grasping"] == 1.0
    assert values["right_grasping"] == 0.0
    assert values["cmd_left_gripper"] == 1.0
    assert values["cmd_right_gripper"] == -1.0
    np.testing.assert_allclose(values["success_margin"], 0.10, atol=1e-6)
    np.testing.assert_allclose(values["left_tcp_to_barrier"], 0.05, atol=1e-6)
    np.testing.assert_allclose(values["right_tcp_to_barrier"], 0.10, atol=1e-6)
    np.testing.assert_allclose(values["left_tcp_to_grasp_target"], 0.0, atol=1e-6)
    np.testing.assert_allclose(values["right_tcp_to_grasp_target"], 0.0, atol=1e-6)
