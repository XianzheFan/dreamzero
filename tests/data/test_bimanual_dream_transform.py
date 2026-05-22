"""Tests for BimanualDreamTransform.

Validates that the multi-agent post-DreamTransform split:
  * stacks state / action along a leading P axis
  * per-agent V-tiles videos into ``[P, T, 2H, 2W, C]`` via the
    overridden ``_prepare_video`` + ``_apply_vlm_processing``
  * preserves per-arm slice values exactly
  * shares the top view across agents (per the [[0,1], [0,2]] config)
  * rejects malformed agent groupings
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


def _make_inst(Cls, *, views=((0, 1), (0, 2)),
               state_dims=((0, 7), (7, 14)),
               action_dims=((0, 7), (7, 14))):
    """Build a lite instance via ``__new__`` (skip the full pydantic init
    that pulls in a tokenizer)."""
    inst = Cls.__new__(Cls)
    inst.__dict__["agent_video_views"] = [list(v) for v in views]
    inst.__dict__["agent_state_dims"] = [tuple(s) for s in state_dims]
    inst.__dict__["agent_action_dims"] = [tuple(a) for a in action_dims]
    return inst


@pytest.fixture
def yam_post_dream():
    """A fake DreamTransform-output dict with YAM shapes."""
    T_s, T_a = 1, 24
    max_state_dim, max_action_dim = 44, 32
    V, T, C, H, W = 3, 33, 3, 176, 320
    state = np.zeros((T_s, max_state_dim), dtype=np.float32)
    state[:, :14] = np.arange(14, dtype=np.float32)
    state_mask = np.zeros_like(state, dtype=bool)
    state_mask[:, :14] = True
    action = np.zeros((T_a, max_action_dim), dtype=np.float32)
    action[:, :14] = np.tile(np.arange(14, dtype=np.float32), (T_a, 1)) * 0.1
    action_mask = np.zeros_like(action, dtype=bool)
    action_mask[:, :14] = True
    # Raw video layout from the loader: [T, V, H, W, C].
    video = np.zeros((T, V, H, W, C), dtype=np.uint8)
    for v in range(V):
        video[:, v, ...] = v * 10  # view 0 = 0, view 1 = 10, view 2 = 20
    return {
        "state": state,
        "state_mask": state_mask,
        "action": action,
        "action_mask": action_mask,
        "video": video,
        "embodiment_id": np.int64(7),
        "has_real_action": np.ones((), dtype=bool),
        "language_input_ids": np.zeros(512, dtype=np.int64),
    }


def test_per_agent_video_tile_shape(yam_post_dream):
    """``_prepare_video`` emits ``[P, T, C, 2H, 2W]`` per-agent tiles."""
    Cls = _maybe_load_bimanual_transform()
    inst = _make_inst(Cls)

    images = inst._prepare_video({"video": yam_post_dream["video"]})
    # YAM: V=3, T=33, C=3, H=176, W=320 -> per agent 2x2 tile is 352x640.
    assert images.shape == (2, 33, 3, 352, 640), images.shape


def test_vlm_processing_preserves_p_axis(yam_post_dream):
    """``_apply_vlm_processing`` must not collapse P into T."""
    Cls = _maybe_load_bimanual_transform()
    inst = _make_inst(Cls)

    tiled = inst._prepare_video({"video": yam_post_dream["video"]})
    out = inst._apply_vlm_processing({"images": tiled, "language": "test"})
    # Output is C-last with the P axis preserved up front: [P, T, H, W, C].
    assert out["images"].shape == (2, 33, 352, 640, 3), out["images"].shape
    assert out["text"] == "test"


def test_state_split_shapes(yam_post_dream):
    Cls = _maybe_load_bimanual_transform()
    inst = _make_inst(Cls)
    state = inst._split_dense(yam_post_dream["state"], inst.agent_state_dims)
    action = inst._split_dense(yam_post_dream["action"], inst.agent_action_dims)
    assert state.shape == (2, 1, 7)
    assert action.shape == (2, 24, 7)


def test_per_arm_state_values(yam_post_dream):
    Cls = _maybe_load_bimanual_transform()
    inst = _make_inst(Cls)
    state = inst._split_dense(yam_post_dream["state"], inst.agent_state_dims)
    # Raw values were ``np.arange(14)``; per-arm split recovers
    # [0..7) for agent 0 and [7..14) for agent 1.
    np.testing.assert_array_equal(state[0, 0], np.arange(0, 7, dtype=np.float32))
    np.testing.assert_array_equal(state[1, 0], np.arange(7, 14, dtype=np.float32))


def test_shared_top_view_in_tile(yam_post_dream):
    """With ``[[0, 1], [0, 2]]`` the top-view slot must be byte-equal
    across both agents' tiles, but their per-agent wrist slot must differ.
    """
    Cls = _maybe_load_bimanual_transform()
    inst = _make_inst(Cls)
    tiled = inst._prepare_video({"video": yam_post_dream["video"]})  # [P, T, C, 2H, 2W]

    H = yam_post_dream["video"].shape[2]
    W = yam_post_dream["video"].shape[3]
    # Top-left slot (= view index 0 for both agents) -- shared.
    np.testing.assert_array_equal(
        tiled[0, :, :, :H, :W], tiled[1, :, :, :H, :W],
    )
    # Bottom-left slot (= view index 1 for agent 0, view 2 for agent 1) -- differs.
    assert not np.array_equal(
        tiled[0, :, :, H:, :W], tiled[1, :, :, H:, :W],
    )


def test_top_right_slot_zero_when_two_views(yam_post_dream):
    """V_per_agent=2 uses slots 0 (TL) and 1 (BL); TR/BR remain zero."""
    Cls = _maybe_load_bimanual_transform()
    inst = _make_inst(Cls)
    tiled = inst._prepare_video({"video": yam_post_dream["video"]})

    H = yam_post_dream["video"].shape[2]
    W = yam_post_dream["video"].shape[3]
    # Right half must be all-zero for both agents.
    np.testing.assert_array_equal(tiled[:, :, :, :, W:], 0)


def test_wrong_agent_count_raises(yam_post_dream):
    Cls = _maybe_load_bimanual_transform()
    inst = _make_inst(
        Cls,
        views=((0, 1), (0, 2)),
        state_dims=((0, 5), (5, 10), (10, 14)),
        action_dims=((0, 7), (7, 14)),
    )
    with pytest.raises(AssertionError, match="agent_state_dims"):
        inst._validate_groups()


def test_unequal_widths_raise(yam_post_dream):
    Cls = _maybe_load_bimanual_transform()
    inst = _make_inst(
        Cls,
        views=((0, 1), (0, 2)),
        state_dims=((0, 6), (6, 14)),
        action_dims=((0, 7), (7, 14)),
    )
    with pytest.raises(AssertionError, match="same state width"):
        inst._validate_groups()


def test_too_many_views_per_agent_raises():
    Cls = _maybe_load_bimanual_transform()
    inst = _make_inst(
        Cls,
        views=((0, 1, 2, 3, 4), (5, 6, 7, 8, 9)),
        state_dims=((0, 7), (7, 14)),
        action_dims=((0, 7), (7, 14)),
    )
    with pytest.raises(AssertionError, match="At most 4 views"):
        inst._validate_groups()
