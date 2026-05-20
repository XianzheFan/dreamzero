"""Smoke tests for :class:`MultiAgentStackTransform`.

The transform module is loaded by path to avoid pulling in the heavy
``groot.vla.data.transform`` package init (transformers, albumentations, ...).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASE_PATH = _REPO_ROOT / "groot/vla/data/transform/base.py"
_SCHEMA_PATH = _REPO_ROOT / "groot/vla/data/schema/__init__.py"
_MODULE_PATH = _REPO_ROOT / "groot/vla/data/transform/multi_agent.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load just the dependencies multi_agent needs, in order, so we don't trigger
# the parent package's __init__.py.
_load("groot.vla.data.schema", _SCHEMA_PATH)
_load("groot.vla.data.transform.base", _BASE_PATH)
_mod = _load("groot.vla.data.transform.multi_agent", _MODULE_PATH)
MultiAgentStackTransform = _mod.MultiAgentStackTransform


def _make_sample(T=4, V=4, H=8, W=8, C=3, D_state=10, D_action=14):
    return {
        "video": np.random.rand(T, V, H, W, C).astype(np.float32),
        "state": torch.randn(T, D_state),
        "action": torch.randn(T, D_action),
    }


def test_single_agent_default_inserts_p1_axis():
    transform = MultiAgentStackTransform()
    sample = _make_sample()
    video_before = sample["video"].copy()
    state_before = sample["state"].clone()
    action_before = sample["action"].clone()

    out = transform.apply(dict(sample))

    assert out["num_agents"] == 1
    assert out["video"].shape == (1, *video_before.shape)
    assert out["state"].shape == (1, *state_before.shape)
    assert out["action"].shape == (1, *action_before.shape)

    np.testing.assert_array_equal(out["video"][0], video_before)
    torch.testing.assert_close(out["state"][0], state_before)
    torch.testing.assert_close(out["action"][0], action_before)


def test_multi_agent_splits_video_views():
    transform = MultiAgentStackTransform(
        agent_video_views=[[0, 1], [2, 3]],
    )
    sample = _make_sample(V=4)
    video_before = sample["video"].copy()

    out = transform.apply(dict(sample))

    assert out["num_agents"] == 2
    assert out["video"].shape == (2, 4, 2, 8, 8, 3)
    np.testing.assert_array_equal(out["video"][0], video_before[:, [0, 1]])
    np.testing.assert_array_equal(out["video"][1], video_before[:, [2, 3]])
    # state / action still get the P=1 unsqueeze in the no-group path.
    assert out["state"].shape[0] == 1
    assert out["action"].shape[0] == 1


def test_multi_agent_splits_state_and_action():
    transform = MultiAgentStackTransform(
        agent_state_dims=[(0, 5), (5, 10)],
        agent_action_dims=[(0, 7), (7, 14)],
    )
    sample = _make_sample(D_state=10, D_action=14)
    state_before = sample["state"].clone()
    action_before = sample["action"].clone()

    out = transform.apply(dict(sample))

    assert out["num_agents"] == 2
    assert out["state"].shape == (2, 4, 5)
    assert out["action"].shape == (2, 4, 7)
    torch.testing.assert_close(out["state"][0], state_before[:, 0:5])
    torch.testing.assert_close(out["state"][1], state_before[:, 5:10])
    torch.testing.assert_close(out["action"][0], action_before[:, 0:7])
    torch.testing.assert_close(out["action"][1], action_before[:, 7:14])


def test_multi_agent_full_bimanual_layout():
    transform = MultiAgentStackTransform(
        agent_video_views=[[0, 1], [2, 3]],
        agent_state_dims=[(0, 5), (5, 10)],
        agent_action_dims=[(0, 7), (7, 14)],
    )
    sample = _make_sample()

    out = transform.apply(dict(sample))

    assert out["num_agents"] == 2
    assert out["video"].shape == (2, 4, 2, 8, 8, 3)
    assert out["state"].shape == (2, 4, 5)
    assert out["action"].shape == (2, 4, 7)


def test_inconsistent_agent_counts_raise():
    transform = MultiAgentStackTransform(
        agent_video_views=[[0, 1], [2, 3]],
        agent_state_dims=[(0, 5)],  # only 1 agent declared here
    )
    with pytest.raises(AssertionError, match="same number of agents"):
        transform.apply(_make_sample())


def test_unequal_per_agent_widths_raise():
    transform = MultiAgentStackTransform(
        agent_video_views=[[0, 1], [2]],
    )
    with pytest.raises(AssertionError, match="same number of views"):
        transform.apply(_make_sample())


def test_unapply_single_agent_roundtrips():
    transform = MultiAgentStackTransform()
    sample = _make_sample()
    video_before = sample["video"].copy()
    state_before = sample["state"].clone()

    out = transform.apply(dict(sample))
    restored = transform.unapply(out)

    np.testing.assert_array_equal(restored["video"], video_before)
    torch.testing.assert_close(restored["state"], state_before)


def test_unapply_multi_agent_roundtrips_state():
    transform = MultiAgentStackTransform(
        agent_state_dims=[(0, 5), (5, 10)],
        agent_action_dims=[(0, 7), (7, 14)],
    )
    sample = _make_sample()
    state_before = sample["state"].clone()
    action_before = sample["action"].clone()

    out = transform.apply(dict(sample))
    restored = transform.unapply(out)

    torch.testing.assert_close(restored["state"], state_before)
    torch.testing.assert_close(restored["action"], action_before)
