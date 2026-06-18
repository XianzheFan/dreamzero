import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _load_eval_module():
    robofactory_mod = types.ModuleType("robofactory")
    tasks_mod = types.ModuleType("robofactory.tasks")
    tasks_mod.__all__ = []
    sys.modules.setdefault("robofactory", robofactory_mod)
    sys.modules.setdefault("robofactory.tasks", tasks_mod)

    path = Path(__file__).resolve().parents[2] / "scripts/eval/eval_robofactory_ws.py"
    spec = importlib.util.spec_from_file_location("eval_robofactory_ws_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_joint_delta_output_clip_applies_after_scale():
    mod = _load_eval_module()
    ref = np.zeros(16, dtype=np.float32)
    action = np.zeros(16, dtype=np.float32)
    action[:7] = 0.2
    action[8:15] = -0.2

    out = mod.scale_joint_target_delta(
        action,
        ref,
        scale=3.0,
        clip=None,
        output_clip=0.25,
    )

    np.testing.assert_allclose(out[:7], 0.25)
    np.testing.assert_allclose(out[8:15], -0.25)


def test_joint_delta_input_clip_still_applies_before_scale():
    mod = _load_eval_module()
    ref = np.zeros(16, dtype=np.float32)
    action = np.zeros(16, dtype=np.float32)
    action[:7] = 0.2

    out = mod.scale_joint_target_delta(
        action,
        ref,
        scale=3.0,
        clip=0.1,
        output_clip=None,
    )

    np.testing.assert_allclose(out[:7], 0.3)


def test_joint_delta_scaling_leaves_grippers_unchanged():
    mod = _load_eval_module()
    ref = np.zeros(16, dtype=np.float32)
    action = np.ones(16, dtype=np.float32) * 0.5
    action[7] = 1.0
    action[15] = -1.0

    out = mod.scale_joint_target_delta(
        action,
        ref,
        scale=4.0,
        clip=0.2,
        output_clip=0.3,
    )

    assert out[7] == 1.0
    assert out[15] == -1.0


def test_policy_close_latch_holds_each_gripper_independently():
    mod = _load_eval_module()
    action = np.ones(16, dtype=np.float32)
    latch = {"left": False, "right": False}

    out = mod.apply_gripper_override(
        action,
        step=0,
        mode="policy-close-latch",
        close_after_step=0,
        open_value=1.0,
        close_value=-1.0,
        latch_state=latch,
        policy_close_threshold=0.0,
    )
    assert out[7] == 1.0
    assert out[15] == 1.0
    assert latch == {"left": False, "right": False}

    action[7] = -0.2
    out = mod.apply_gripper_override(
        action,
        step=1,
        mode="policy-close-latch",
        close_after_step=0,
        open_value=1.0,
        close_value=-1.0,
        latch_state=latch,
        policy_close_threshold=0.0,
    )
    assert out[7] == -1.0
    assert out[15] == 1.0
    assert latch == {"left": True, "right": False}

    action[7] = 1.0
    action[15] = -0.3
    out = mod.apply_gripper_override(
        action,
        step=2,
        mode="policy-close-latch",
        close_after_step=0,
        open_value=1.0,
        close_value=-1.0,
        latch_state=latch,
        policy_close_threshold=0.0,
    )
    assert out[7] == -1.0
    assert out[15] == -1.0
    assert latch == {"left": True, "right": True}


def test_policy_close_latch_requires_state():
    mod = _load_eval_module()
    action = np.ones(16, dtype=np.float32)

    try:
        mod.apply_gripper_override(
            action,
            step=0,
            mode="policy-close-latch",
            close_after_step=0,
            open_value=1.0,
            close_value=-1.0,
        )
    except ValueError as exc:
        assert "latch_state" in str(exc)
    else:
        raise AssertionError("policy-close-latch should require latch_state")


def test_joint_delta_per_arm_scale_overrides_global_scale():
    mod = _load_eval_module()
    ref = np.zeros(16, dtype=np.float32)
    action = np.zeros(16, dtype=np.float32)
    action[:7] = 0.1
    action[8:15] = 0.1

    out = mod.scale_joint_target_delta(
        action,
        ref,
        scale=2.0,
        clip=None,
        output_clip=None,
        right_scale=5.0,
    )

    np.testing.assert_allclose(out[:7], 0.2)
    np.testing.assert_allclose(out[8:15], 0.5)


def test_joint_target_slew_rate_caps_arm_targets_only():
    mod = _load_eval_module()
    previous = np.zeros(16, dtype=np.float32)
    action = np.ones(16, dtype=np.float32)
    action[:7] = 0.8
    action[8:15] = -0.9
    action[7] = 1.0
    action[15] = -1.0

    out = mod.limit_joint_target_slew(action, previous, max_delta=0.25)

    np.testing.assert_allclose(out[:7], 0.25)
    np.testing.assert_allclose(out[8:15], -0.25)
    assert out[7] == 1.0
    assert out[15] == -1.0


def test_joint_target_slew_rate_disabled_returns_input():
    mod = _load_eval_module()
    previous = np.zeros(16, dtype=np.float32)
    action = np.arange(16, dtype=np.float32)

    out = mod.limit_joint_target_slew(action, previous, max_delta=None)

    assert out is action


def test_absolute_qpos_ignores_joint_delta_controls_by_default():
    mod = _load_eval_module()

    resolved = mod.resolve_joint_delta_controls(
        "absolute_qpos",
        scale=12.0,
        clip=0.2,
        output_clip=1.5,
        left_scale=4.0,
        right_scale=8.0,
        allow_absolute_scale=False,
    )

    assert resolved == (1.0, None, None, None, None, True)


def test_absolute_qpos_can_opt_into_joint_delta_controls_for_diagnostics():
    mod = _load_eval_module()

    resolved = mod.resolve_joint_delta_controls(
        "absolute_qpos",
        scale=12.0,
        clip=0.2,
        output_clip=1.5,
        left_scale=4.0,
        right_scale=8.0,
        allow_absolute_scale=True,
    )

    assert resolved == (12.0, 0.2, 1.5, 4.0, 8.0, False)


def test_legacy_delta_representation_keeps_joint_delta_controls():
    mod = _load_eval_module()

    resolved = mod.resolve_joint_delta_controls(
        "robotwin_delta",
        scale=3.0,
        clip=None,
        output_clip=0.5,
        left_scale=None,
        right_scale=4.0,
        allow_absolute_scale=False,
    )

    assert resolved == (3.0, None, 0.5, None, 4.0, False)


def _fake_obs(qpos0, qpos1):
    return {
        "agent": {
            "panda-0": {"qpos": np.asarray(qpos0, dtype=np.float32)},
            "panda-1": {"qpos": np.asarray(qpos1, dtype=np.float32)},
        }
    }


def test_observed_reference_uses_post_step_qpos():
    mod = _load_eval_module()
    commanded = np.arange(16, dtype=np.float32)
    obs_after = _fake_obs(
        np.arange(9, dtype=np.float32) + 100.0,
        np.arange(9, dtype=np.float32) + 200.0,
    )

    out = mod.update_loop_qpos_after_step(
        obs_after,
        commanded,
        joint_delta_scale_reference="observed",
    )

    np.testing.assert_allclose(out[:8], np.arange(8, dtype=np.float32) + 100.0)
    np.testing.assert_allclose(out[8:16], np.arange(8, dtype=np.float32) + 200.0)


def test_commanded_reference_modes_keep_commanded_target():
    mod = _load_eval_module()
    commanded = np.arange(16, dtype=np.float32)
    obs_after = _fake_obs(
        np.arange(9, dtype=np.float32) + 100.0,
        np.arange(9, dtype=np.float32) + 200.0,
    )

    for mode in ("previous", "chunk"):
        out = mod.update_loop_qpos_after_step(
            obs_after,
            commanded,
            joint_delta_scale_reference=mode,
        )
        np.testing.assert_allclose(out, commanded)
        assert out is not commanded


class _Pose:
    def __init__(self, p):
        self.p = np.asarray(p, dtype=np.float32)

    def to_transformation_matrix(self):
        mat = np.eye(4, dtype=np.float32)
        mat[:3, 3] = self.p
        return mat


class _Actor:
    def __init__(self, p):
        self.pose = _Pose(p)


class _Agent:
    def __init__(self, robot_p, tcp_p, grasping):
        self.robot = _Actor(robot_p)
        self.tcp = _Actor(tcp_p)
        self._grasping = grasping

    def is_grasping(self, _actor):
        return self._grasping


def test_collect_env_trace_records_liftbarrier_geometry():
    mod = _load_eval_module()
    contact0 = np.eye(4, dtype=np.float32)
    contact1 = np.eye(4, dtype=np.float32)
    contact2 = np.eye(4, dtype=np.float32)
    contact1[:3, 3] = [0.05, 0.0, 0.0]
    contact2[:3, 3] = [0.0, 0.10, 0.0]
    root = types.SimpleNamespace(
        barrier=_Actor([0.0, 0.0, 0.25]),
        annotation_data={
            "barrier": {
                "contact_points_pose": [contact0, contact1, contact2],
                "scale": 1.0,
            }
        },
    )
    left = _Agent([0.0, 0.0, 0.0], [0.05, 0.0, 0.25], True)
    right = _Agent([0.0, 0.0, 0.0], [0.0, 0.10, 0.25], False)
    root.agent = types.SimpleNamespace(agents=[left, right])
    env = types.SimpleNamespace(unwrapped=root)
    action = np.zeros(16, dtype=np.float32)
    action[7] = -1.0
    action[15] = 1.0

    trace = mod.collect_env_trace(env, 5, action, {"success": True})
    columns = tuple(str(x) for x in mod.ENV_TRACE_COLUMNS)

    assert trace[columns.index("step")] == 5.0
    assert trace[columns.index("success")] == 1.0
    np.testing.assert_allclose(trace[columns.index("barrier_z")], 0.25)
    np.testing.assert_allclose(trace[columns.index("success_margin")], 0.10)
    np.testing.assert_allclose(trace[columns.index("left_tcp_to_barrier")], 0.05)
    np.testing.assert_allclose(trace[columns.index("right_tcp_to_barrier")], 0.10)
    assert trace[columns.index("left_grasping")] == 1.0
    assert trace[columns.index("right_grasping")] == 0.0
    assert trace[columns.index("cmd_left_gripper")] == -1.0
    assert trace[columns.index("cmd_right_gripper")] == 1.0
    np.testing.assert_allclose(trace[columns.index("left_tcp_to_grasp_target")], 0.0)
    np.testing.assert_allclose(trace[columns.index("right_tcp_to_grasp_target")], 0.0)


def test_liftbarrier_strict_success_requires_bilateral_grasp_history():
    mod = _load_eval_module()
    columns = tuple(str(x) for x in mod.ENV_TRACE_COLUMNS)
    trace = np.zeros(len(columns), dtype=np.float32)
    trace[columns.index("success")] = 1.0
    trace[columns.index("left_grasping")] = 1.0
    trace[columns.index("right_grasping")] = 0.0

    counts = {"left": 0, "right": 0}
    mod.update_liftbarrier_grasp_counts(counts, trace)

    assert counts == {"left": 1, "right": 0}
    assert not mod.liftbarrier_strict_success(trace, counts, min_grasp_count=1)

    trace[columns.index("left_grasping")] = 0.0
    trace[columns.index("right_grasping")] = 1.0
    mod.update_liftbarrier_grasp_counts(counts, trace)

    assert counts == {"left": 1, "right": 1}
    assert mod.liftbarrier_strict_success(trace, counts, min_grasp_count=1)


def test_liftbarrier_strict_success_still_requires_sim_success():
    mod = _load_eval_module()
    columns = tuple(str(x) for x in mod.ENV_TRACE_COLUMNS)
    trace = np.zeros(len(columns), dtype=np.float32)
    trace[columns.index("success")] = 0.0

    assert not mod.liftbarrier_strict_success(
        trace,
        {"left": 5, "right": 5},
        min_grasp_count=1,
    )
