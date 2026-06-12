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
