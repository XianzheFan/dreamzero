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
               action_dims=((0, 7), (7, 14)),
               global_views=None,
               global_condition_mode="full",
               state_pad_dim=None,
               action_pad_dim=None):
    """Build a lite instance via ``__new__`` (skip the full pydantic init
    that pulls in a tokenizer)."""
    inst = Cls.__new__(Cls)
    inst.__dict__["agent_video_views"] = [list(v) for v in views]
    inst.__dict__["global_views"] = (
        None if global_views is None else list(global_views)
    )
    inst.__dict__["global_condition_mode"] = global_condition_mode
    inst.__dict__["agent_state_dims"] = [tuple(s) for s in state_dims]
    inst.__dict__["agent_action_dims"] = [tuple(a) for a in action_dims]
    inst.__dict__["agent_state_pad_dim"] = state_pad_dim
    inst.__dict__["agent_action_pad_dim"] = action_pad_dim
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


def test_eight_dim_per_arm_split_preserves_gripper_values_and_masks():
    Cls = _maybe_load_bimanual_transform()
    inst = _make_inst(
        Cls,
        state_dims=((0, 8), (8, 16)),
        action_dims=((0, 8), (8, 16)),
    )
    state = np.zeros((1, 16), dtype=np.float32)
    state[0, 0:7] = np.arange(7, dtype=np.float32)
    state[0, 7] = 0.25
    state[0, 8:15] = np.arange(10, 17, dtype=np.float32)
    state[0, 15] = 0.75
    action = np.zeros((24, 16), dtype=np.float32)
    action[:, 0:7] = 0.1
    action[:, 7] = 0.0
    action[:, 8:15] = 0.2
    action[:, 15] = 1.0
    state_mask = np.ones_like(state, dtype=bool)
    action_mask = np.ones_like(action, dtype=bool)

    split_state = inst._split_dense(state, inst.agent_state_dims)
    split_action = inst._split_dense(action, inst.agent_action_dims)
    split_state_mask = inst._split_dense(state_mask, inst.agent_state_dims)
    split_action_mask = inst._split_dense(action_mask, inst.agent_action_dims)

    assert split_state.shape == (2, 1, 8)
    assert split_action.shape == (2, 24, 8)
    np.testing.assert_allclose(split_state[0, 0, 7], 0.25)
    np.testing.assert_allclose(split_state[1, 0, 7], 0.75)
    np.testing.assert_allclose(split_action[0, :, 7], np.zeros(24))
    np.testing.assert_allclose(split_action[1, :, 7], np.ones(24))
    assert split_state_mask[:, :, 7].all()
    assert split_action_mask[:, :, 7].all()


def test_droid_width_per_arm_split_pads_each_agent_independently():
    Cls = _maybe_load_bimanual_transform()
    inst = _make_inst(
        Cls,
        state_dims=((0, 8), (8, 16)),
        action_dims=((0, 8), (8, 16)),
        state_pad_dim=64,
        action_pad_dim=32,
    )
    state = np.zeros((1, 16), dtype=np.float32)
    state[0, 0:8] = np.arange(8, dtype=np.float32)
    state[0, 8:16] = np.arange(10, 18, dtype=np.float32)
    action = np.zeros((24, 16), dtype=np.float32)
    action[:, 0:8] = np.arange(8, dtype=np.float32)
    action[:, 8:16] = np.arange(10, 18, dtype=np.float32)

    split_state = inst._split_dense(
        state, inst.agent_state_dims, pad_dim=inst.agent_state_pad_dim
    )
    split_action = inst._split_dense(
        action, inst.agent_action_dims, pad_dim=inst.agent_action_pad_dim
    )
    split_state_mask = inst._split_mask_from_raw(
        state, inst.agent_state_dims, pad_dim=inst.agent_state_pad_dim
    )
    split_action_mask = inst._split_mask_from_raw(
        action, inst.agent_action_dims, pad_dim=inst.agent_action_pad_dim
    )

    assert split_state.shape == (2, 1, 64)
    assert split_action.shape == (2, 24, 32)
    np.testing.assert_array_equal(split_state[0, 0, :8], np.arange(8))
    np.testing.assert_array_equal(split_state[1, 0, :8], np.arange(10, 18))
    np.testing.assert_array_equal(split_state[:, :, 8:], 0.0)
    np.testing.assert_array_equal(
        split_action[0, :, :8], np.tile(np.arange(8, dtype=np.float32), (24, 1))
    )
    np.testing.assert_array_equal(
        split_action[1, :, :8],
        np.tile(np.arange(10, 18, dtype=np.float32), (24, 1)),
    )
    np.testing.assert_array_equal(split_action[:, :, 8:], 0.0)
    assert split_state_mask[:, :, :8].all()
    assert not split_state_mask[:, :, 8:].any()
    assert split_action_mask[:, :, :8].all()
    assert not split_action_mask[:, :, 8:].any()


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


def test_shared_global_current_repeat_removes_future_frames(yam_post_dream):
    Cls = _maybe_load_bimanual_transform()
    video = yam_post_dream["video"].copy()
    # Make the global view time-varying so future leakage would be visible.
    for t in range(video.shape[0]):
        video[t, 0, ...] = t
    inst = _make_inst(
        Cls,
        views=((1,), (2,)),
        global_views=(0,),
        global_condition_mode="current_repeat",
    )

    global_video = inst._prepare_global_video({"video": video})

    assert global_video.shape == (33, 176, 320, 3)
    # Every frame in the clean global stream must be the current frame
    # (delta index 0), not the future frames 1..24.
    np.testing.assert_array_equal(
        global_video,
        np.repeat(global_video[0:1], global_video.shape[0], axis=0),
    )
    np.testing.assert_array_equal(global_video[0], video[0, 0])


def test_shared_global_three_agents_keep_wrist_streams_native_size(yam_post_dream):
    Cls = _maybe_load_bimanual_transform()
    video = np.concatenate(
        [yam_post_dream["video"], np.full_like(yam_post_dream["video"][:, :1], 30)],
        axis=1,
    )
    inst = _make_inst(
        Cls,
        views=((1,), (2,), (3,)),
        state_dims=((0, 8), (8, 16), (16, 24)),
        action_dims=((0, 8), (8, 16), (16, 24)),
        global_views=(0,),
        global_condition_mode="current_repeat",
    )

    images = inst._prepare_video({"video": video})
    global_video = inst._prepare_global_video({"video": video})

    assert images.shape == (3, 33, 3, 176, 320)
    assert global_video.shape == (33, 176, 320, 3)
    # The shared scene view is factored out; per-agent streams are only
    # wrist views 1/2/3 and therefore differ from the global view 0.
    assert int(images[0, 0, 0, 0, 0]) == 10
    assert int(images[1, 0, 0, 0, 0]) == 20
    assert int(images[2, 0, 0, 0, 0]) == 30
    assert int(global_video[0, 0, 0, 0]) == 0


def test_shared_global_full_mode_preserves_window(yam_post_dream):
    Cls = _maybe_load_bimanual_transform()
    video = yam_post_dream["video"].copy()
    for t in range(video.shape[0]):
        video[t, 0, ...] = t
    inst = _make_inst(
        Cls,
        views=((1,), (2,)),
        global_views=(0,),
        global_condition_mode="full",
    )

    global_video = inst._prepare_global_video({"video": video})

    assert int(global_video[0, 0, 0, 0]) == 0
    assert int(global_video[-1, 0, 0, 0]) == video.shape[0] - 1
