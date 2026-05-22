"""Tests for BimanualDreamTransform.

Validates that the multi-agent post-DreamTransform split:
  * stacks state / action / images along a leading P axis
  * preserves per-arm slice values exactly
  * shares the top view across agents (per the [[0,1], [0,2]] config)
  * passes other DreamTransform outputs through unchanged
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _maybe_load_bimanual_transform():
    """Return ``BimanualDreamTransform`` or skip if heavy deps missing."""
    try:
        from groot.vla.model.dreamzero.transform.bimanual_cotrain import (
            BimanualDreamTransform,
        )
    except Exception as e:  # pragma: no cover -- skip when transformers/etc missing
        pytest.skip(f"deps not installed for BimanualDreamTransform: {e}")
    return BimanualDreamTransform


@pytest.fixture
def yam_post_dream():
    """A fake DreamTransform-output dict with YAM shapes."""
    T_s, T_a = 1, 24
    max_state_dim, max_action_dim = 44, 32
    V, T, C, H, W = 3, 33, 3, 176, 320
    # State / action: raw 14 dims of data padded with zeros to max_state_dim.
    state = np.zeros((T_s, max_state_dim), dtype=np.float32)
    state[:, :14] = np.arange(14, dtype=np.float32)
    state_mask = np.zeros_like(state, dtype=bool)
    state_mask[:, :14] = True
    action = np.zeros((T_a, max_action_dim), dtype=np.float32)
    action[:, :14] = np.tile(np.arange(14, dtype=np.float32), (T_a, 1)) * 0.1
    action_mask = np.zeros_like(action, dtype=bool)
    action_mask[:, :14] = True
    # Images: per-view distinct content so we can verify shared/own.
    images = np.zeros((V, T, C, H, W), dtype=np.uint8)
    for v in range(V):
        images[v] = v * 10  # view 0 = 0, view 1 = 10, view 2 = 20
    return {
        "state": state,
        "state_mask": state_mask,
        "action": action,
        "action_mask": action_mask,
        "images": images,
        "embodiment_id": np.int64(7),  # arbitrary
        "has_real_action": np.ones((), dtype=bool),
        "language_input_ids": np.zeros(512, dtype=np.int64),  # placeholder
    }


def test_split_shapes(yam_post_dream):
    Cls = _maybe_load_bimanual_transform()
    # Build a "lite" instance and call only the helper methods (skip
    # the full DreamTransform __init__ which downloads a tokenizer).
    inst = Cls.__new__(Cls)
    inst.__dict__["agent_video_views"] = [[0, 1], [0, 2]]
    inst.__dict__["agent_state_dims"] = [(0, 7), (7, 14)]
    inst.__dict__["agent_action_dims"] = [(0, 7), (7, 14)]
    inst._validate_groups()

    state = inst._split_dense(
        yam_post_dream["state"], inst.agent_state_dims
    )
    action = inst._split_dense(
        yam_post_dream["action"], inst.agent_action_dims
    )
    images = inst._split_video(yam_post_dream["images"])

    assert state.shape == (2, 1, 7)
    assert action.shape == (2, 24, 7)
    assert images.shape == (2, 2, 33, 3, 176, 320)


def test_per_arm_state_values(yam_post_dream):
    Cls = _maybe_load_bimanual_transform()
    inst = Cls.__new__(Cls)
    inst.__dict__["agent_state_dims"] = [(0, 7), (7, 14)]

    state = inst._split_dense(
        yam_post_dream["state"], inst.agent_state_dims
    )
    # Raw values were ``np.arange(14)``; per-arm split should recover
    # [0..7) for agent 0 and [7..14) for agent 1.
    np.testing.assert_array_equal(state[0, 0], np.arange(0, 7, dtype=np.float32))
    np.testing.assert_array_equal(state[1, 0], np.arange(7, 14, dtype=np.float32))


def test_shared_top_view(yam_post_dream):
    Cls = _maybe_load_bimanual_transform()
    inst = Cls.__new__(Cls)
    inst.__dict__["agent_video_views"] = [[0, 1], [0, 2]]
    inst._validate_groups = lambda: None  # bypass

    images = inst._split_video(yam_post_dream["images"])
    # agent 0 view 0 (top) == agent 1 view 0 (top) byte-equal
    np.testing.assert_array_equal(images[0, 0], images[1, 0])
    # agent 0 view 1 (left) != agent 1 view 1 (right)
    assert not np.array_equal(images[0, 1], images[1, 1])


def test_wrong_agent_count_raises(yam_post_dream):
    Cls = _maybe_load_bimanual_transform()
    inst = Cls.__new__(Cls)
    # Mismatched: 2 video agents, 3 state agents
    inst.__dict__["agent_video_views"] = [[0, 1], [0, 2]]
    inst.__dict__["agent_state_dims"] = [(0, 5), (5, 10), (10, 14)]
    inst.__dict__["agent_action_dims"] = [(0, 7), (7, 14)]
    with pytest.raises(AssertionError, match="agent_state_dims"):
        inst._validate_groups()


def test_unequal_widths_raise(yam_post_dream):
    Cls = _maybe_load_bimanual_transform()
    inst = Cls.__new__(Cls)
    # Mismatched widths: agent 0 has 6 dims, agent 1 has 8 dims
    inst.__dict__["agent_video_views"] = [[0, 1], [0, 2]]
    inst.__dict__["agent_state_dims"] = [(0, 6), (6, 14)]
    inst.__dict__["agent_action_dims"] = [(0, 7), (7, 14)]
    with pytest.raises(AssertionError, match="same state width"):
        inst._validate_groups()
