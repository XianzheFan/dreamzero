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
