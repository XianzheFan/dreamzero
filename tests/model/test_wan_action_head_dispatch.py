"""WANPolicyHead P-axis detection / dispatch tests.

Validates:
  * ``_detect_multi_agent`` recognises the leading P axis on
    ``state`` / ``action``;
  * ``forward`` routes ``[B, P, ...]`` inputs to ``_forward_multi_agent``;
  * ``forward`` keeps the single-agent path for ``[B, T, D]`` inputs.

The multi-agent body itself is exercised end-to-end (with a real
diffusion model + scheduler + VAE-shaped inputs) in
``test_wan_action_head_multi_agent.py``.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


def _maybe_load_head():
    try:
        from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import (
            WANPolicyHead,
        )
    except Exception as e:  # pragma: no cover
        pytest.skip(f"deps not installed: {e}")
    return WANPolicyHead


def test_detect_multi_agent_state_p():
    Cls = _maybe_load_head()
    # Use __new__ to skip the heavy __init__.
    inst = Cls.__new__(Cls)
    # Multi-agent: state [B=1, P=2, T_s=1, D=7]
    af = BatchFeature(data={
        "state":  torch.zeros(1, 2, 1, 7),
        "action": torch.zeros(1, 2, 24, 7),
    })
    assert inst._detect_multi_agent(af) == 2


def test_detect_single_agent_returns_none():
    Cls = _maybe_load_head()
    inst = Cls.__new__(Cls)
    # Single-agent: state [B=1, T_s=1, D=44]
    af = BatchFeature(data={
        "state":  torch.zeros(1, 1, 44),
        "action": torch.zeros(1, 24, 32),
    })
    assert inst._detect_multi_agent(af) is None


def test_forward_routes_to_multi_agent():
    """``forward`` must call ``_forward_multi_agent`` when a P axis is
    present on ``state`` / ``action``. We monkey-patch the heavy
    multi-agent body to a sentinel so we can assert dispatch without
    instantiating the full WANPolicyHead.
    """
    Cls = _maybe_load_head()
    inst = Cls.__new__(Cls)
    af = BatchFeature(data={
        "state":  torch.zeros(1, 2, 1, 7),
        "action": torch.zeros(1, 2, 24, 7),
    })

    sentinel = BatchFeature(data={"loss": torch.tensor(1.234)})
    called = {}

    def _stub(self, backbone_output, action_input, num_agents):
        called["P"] = num_agents
        called["backbone_output"] = backbone_output
        called["action_input"] = action_input
        return sentinel

    # Bind the stub as an instance attribute. forward() calls
    # ``self._forward_multi_agent(...)`` so an attribute lookup hits the
    # stub before the class-level method.
    inst._forward_multi_agent = _stub.__get__(inst, Cls)
    out = inst.forward(BatchFeature(data={}), af)
    assert out is sentinel
    assert called["P"] == 2
    assert called["action_input"] is af


def test_detect_three_agents():
    Cls = _maybe_load_head()
    inst = Cls.__new__(Cls)
    af = BatchFeature(data={
        "state":  torch.zeros(2, 3, 1, 5),
        "action": torch.zeros(2, 3, 24, 5),
    })
    assert inst._detect_multi_agent(af) == 3
