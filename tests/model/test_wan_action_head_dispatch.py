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


def test_multi_agent_conditioning_uses_encode_image_latent():
    Cls = _maybe_load_head()
    inst = Cls.__new__(Cls)
    inst.model = type("M", (), {"model_type": "i2v"})()
    inst.image_encoder = object()
    inst.vae = object()
    inst._device = "cpu"

    videos = torch.zeros(1, 2, 3, 5, 8, 8)
    latents = torch.zeros(1, 2, 16, 3, 4, 4)
    clean_from_image = torch.full((2, 16, 1, 4, 4), 7.0)

    def fake_encode_image(image, num_frames, height, width):
        assert tuple(image.shape) == (2, 1, 3, 8, 8)
        return (
            torch.zeros(2, 4),
            torch.zeros(2, 20, 3, 4, 4),
            clean_from_image,
        )

    inst.encode_image = fake_encode_image

    _, _, clean_x = inst._prepare_multi_agent_i2v_conditioning(
        videos=videos,
        latents=latents,
        condition_frame_index=0,
    )

    assert clean_x.shape == latents.shape
    assert torch.all(clean_x == 7.0)
    assert inst._mai_clean_video_cond_source == "encode_image"


def test_multi_agent_rolling_noise_preserve_reset_keeps_stream(monkeypatch):
    Cls = _maybe_load_head()
    monkeypatch.setenv("MAI_ROLLING_NOISE", "1")

    def make_inst():
        inst = Cls.__new__(Cls)
        inst.seed = 123
        inst.model = object()
        inst.current_start_frame = 5
        inst.kv_cache1 = "cache"
        inst.kv_cache_neg = "cache"
        inst.crossattn_cache = "cache"
        inst.crossattn_cache_neg = "cache"
        inst.clip_feas = "clip"
        inst.ys = "ys"
        inst.language = torch.ones(1)
        inst._ma_cached_token_agent_id = "agent"
        inst._ma_cached_token_agent_id_neg = "agent-neg"
        inst._ma_cached_until_frame = 5
        return inst

    continuous = make_inst()
    continuous._generate_multi_agent_sequence_noise(
        (2, 3), device="cpu", dtype=torch.float32, stream="causal_video"
    )
    expected_second = continuous._generate_multi_agent_sequence_noise(
        (2, 3), device="cpu", dtype=torch.float32, stream="causal_video"
    )

    preserved = make_inst()
    preserved._generate_multi_agent_sequence_noise(
        (2, 3), device="cpu", dtype=torch.float32, stream="causal_video"
    )
    preserved.reset_causal_state(preserve_rollout_noise=True)
    after_preserve = preserved._generate_multi_agent_sequence_noise(
        (2, 3), device="cpu", dtype=torch.float32, stream="causal_video"
    )

    reset = make_inst()
    first = reset._generate_multi_agent_sequence_noise(
        (2, 3), device="cpu", dtype=torch.float32, stream="causal_video"
    )
    reset.reset_causal_state()
    after_full_reset = reset._generate_multi_agent_sequence_noise(
        (2, 3), device="cpu", dtype=torch.float32, stream="causal_video"
    )

    assert torch.equal(after_preserve, expected_second)
    assert torch.equal(after_full_reset, first)
    assert preserved.language is not None
    assert reset.language is None
