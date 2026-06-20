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
from types import SimpleNamespace

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


def test_reset_causal_state_clears_multi_agent_cache_progress():
    Cls = _maybe_load_head()
    inst = Cls.__new__(Cls)
    inst.kv_cache1 = object()
    inst.kv_cache_neg = object()
    inst.crossattn_cache = object()
    inst.crossattn_cache_neg = object()
    inst.clip_feas = object()
    inst.ys = object()
    inst.current_start_frame = 17
    inst._ma_cached_token_agent_id = torch.ones(3)
    inst._ma_cached_token_agent_id_neg = torch.ones(3)
    inst._ma_cached_until_frame = 17
    inst._ma_noise_generators = {"causal_video": object()}
    inst._ma_noise_generator_devices = {"causal_video": "cpu"}
    inst._ma_noise_draw_counts = {"causal_video": 3}
    inst.language = torch.ones(1)
    inst.model = SimpleNamespace(_cached_token_agent_id=torch.ones(3))

    inst.reset_causal_state()

    assert inst.kv_cache1 is None
    assert inst.kv_cache_neg is None
    assert inst.crossattn_cache is None
    assert inst.crossattn_cache_neg is None
    assert inst.clip_feas is None
    assert inst.ys is None
    assert inst.current_start_frame == 0
    assert inst._ma_cached_token_agent_id is None
    assert inst._ma_cached_token_agent_id_neg is None
    assert inst._ma_cached_until_frame == 0
    assert inst._ma_noise_generators == {}
    assert inst._ma_noise_generator_devices == {}
    assert inst._ma_noise_draw_counts == {}
    assert inst.language is None
    assert inst.model._cached_token_agent_id is None


def test_reset_causal_state_can_preserve_rollout_noise():
    Cls = _maybe_load_head()
    inst = Cls.__new__(Cls)
    generator = object()
    inst.kv_cache1 = object()
    inst.kv_cache_neg = object()
    inst.crossattn_cache = object()
    inst.crossattn_cache_neg = object()
    inst.clip_feas = object()
    inst.ys = object()
    inst.current_start_frame = 17
    inst._ma_cached_token_agent_id = torch.ones(3)
    inst._ma_cached_token_agent_id_neg = torch.ones(3)
    inst._ma_cached_until_frame = 17
    inst._ma_noise_generators = {"causal_video": generator}
    inst._ma_noise_generator_devices = {"causal_video": "cpu"}
    inst._ma_noise_draw_counts = {"causal_video": 3}
    language = torch.ones(1)
    inst.language = language
    inst.model = SimpleNamespace(_cached_token_agent_id=torch.ones(3))

    inst.reset_causal_state(preserve_rollout_noise=True)

    assert inst.kv_cache1 is None
    assert inst.kv_cache_neg is None
    assert inst.crossattn_cache is None
    assert inst.crossattn_cache_neg is None
    assert inst.current_start_frame == 0
    assert inst._ma_cached_token_agent_id is None
    assert inst._ma_cached_token_agent_id_neg is None
    assert inst._ma_cached_until_frame == 0
    assert inst._ma_noise_generators == {"causal_video": generator}
    assert inst._ma_noise_generator_devices == {"causal_video": "cpu"}
    assert inst._ma_noise_draw_counts == {"causal_video": 3}
    assert inst.language is language
    assert inst.model._cached_token_agent_id is None


def test_multi_agent_rolling_noise_advances_within_episode(monkeypatch):
    Cls = _maybe_load_head()
    inst = Cls.__new__(Cls)
    inst.seed = 123
    monkeypatch.delenv("MAI_ROLLING_NOISE", raising=False)

    first = inst._generate_multi_agent_sequence_noise(
        (2, 3),
        device="cpu",
        dtype=torch.float32,
        stream="causal_action",
    )
    second = inst._generate_multi_agent_sequence_noise(
        (2, 3),
        device="cpu",
        dtype=torch.float32,
        stream="causal_action",
    )

    assert not torch.equal(first, second)
    assert inst._ma_noise_draw_counts["causal_action"] == 2


def test_multi_agent_rolling_noise_can_restore_fixed_seed_mode(monkeypatch):
    Cls = _maybe_load_head()
    inst = Cls.__new__(Cls)
    inst.seed = 123
    monkeypatch.setenv("MAI_ROLLING_NOISE", "0")

    first = inst._generate_multi_agent_sequence_noise(
        (2, 3),
        device="cpu",
        dtype=torch.float32,
        stream="causal_action",
    )
    second = inst._generate_multi_agent_sequence_noise(
        (2, 3),
        device="cpu",
        dtype=torch.float32,
        stream="causal_action",
    )

    torch.testing.assert_close(first, second)
    assert not hasattr(inst, "_ma_noise_draw_counts")


def test_multi_agent_noise_streams_use_distinct_seeds(monkeypatch):
    Cls = _maybe_load_head()
    inst = Cls.__new__(Cls)
    inst.seed = 123
    monkeypatch.delenv("MAI_ROLLING_NOISE", raising=False)

    video = inst._generate_multi_agent_sequence_noise(
        (2, 3),
        device="cpu",
        dtype=torch.float32,
        stream="causal_video",
    )
    action = inst._generate_multi_agent_sequence_noise(
        (2, 3),
        device="cpu",
        dtype=torch.float32,
        stream="causal_action",
    )

    assert not torch.equal(video, action)


def test_multi_agent_i2v_clean_condition_uses_encode_image_latent():
    Cls = _maybe_load_head()
    inst = Cls.__new__(Cls)
    inst.model = SimpleNamespace(model_type="i2v")
    inst.image_encoder = object()
    inst.vae = object()
    inst._device = "cpu"

    videos = torch.zeros(1, 2, 3, 5, 4, 4)
    latents = torch.zeros(1, 2, 4, 3, 2, 2)
    latents[:, :, :, 0] = 11.0
    clean_image = torch.full((2, 4, 1, 2, 2), 7.0)

    def _fake_encode_image(image, num_frames, height, width):
        assert image.shape == (2, 1, 3, 4, 4)
        assert num_frames == 5
        assert height == 4
        assert width == 4
        clip = torch.ones(2, 6)
        y = torch.ones(2, 8, 3, 2, 2)
        return clip, y, clean_image

    inst.encode_image = _fake_encode_image

    clip, y, clean_x = inst._prepare_multi_agent_i2v_conditioning(
        videos=videos,
        latents=latents,
        condition_frame_index=0,
    )

    assert clip.shape == (1, 2, 6)
    assert y.shape == (1, 2, 8, 3, 2, 2)
    assert clean_x.shape == latents.shape
    torch.testing.assert_close(clean_x, torch.full_like(latents, 7.0))
    assert inst._mai_clean_video_cond_source == "encode_image"


def test_write_denoised_context_cache_default_on(monkeypatch):
    Cls = _maybe_load_head()
    inst = Cls.__new__(Cls)
    inst._ma_cached_until_frame = 0
    calls = []

    def _fake_run(**kwargs):
        calls.append(kwargs)
        return []

    inst._run_multi_agent_diffusion_steps = _fake_run
    monkeypatch.delenv("MAI_WRITE_DENOISED_CONTEXT_CACHE", raising=False)
    noisy_video = torch.randn(1, 2, requires_grad=True)
    noisy_action = torch.randn(1, 2, 3, 4, requires_grad=True)
    state_features = torch.randn(1, 2, 1, 5, requires_grad=True)
    embodiment_id = torch.tensor([0])
    latents = torch.zeros(1)
    y_source = torch.arange(6).reshape(1, 1, 6)
    clean_latents = torch.arange(24).reshape(1, 1, 2, 6, 1, 2)
    global_latents = torch.arange(10).reshape(1, 1, 10)

    wrote = inst._write_multi_agent_denoised_context_cache(
        noisy_video=noisy_video,
        B=1,
        block=2,
        latents=latents,
        prompt_embs=[torch.zeros(1, 1)],
        seq_len=8,
        noisy_action=noisy_action,
        state_features=state_features,
        embodiment_id=embodiment_id,
        y_source=y_source,
        clip_feature=torch.zeros(1, 1),
        kv_caches=[torch.empty(0)],
        crossattn_caches=[torch.empty(0)],
        current_start_frame=4,
        clean_latents=clean_latents,
        global_latents=global_latents,
    )

    assert wrote is True
    assert inst._ma_cached_until_frame == 6
    assert len(calls) == 1
    call = calls[0]
    assert call["noisy_input"].requires_grad is False
    assert call["action"].requires_grad is False
    assert call["state"].requires_grad is False
    torch.testing.assert_close(call["action"], noisy_action.detach())
    torch.testing.assert_close(call["state"], state_features.detach())
    assert call["embodiment_id"] is embodiment_id
    assert call["seq_len"] == 8
    assert call["kv_cache_metadata"] == {
        "start_frame": 4,
        "update_kv_cache": True,
    }
    torch.testing.assert_close(call["timestep"], torch.zeros(1, 2, dtype=torch.int64))
    torch.testing.assert_close(call["timestep_action"], torch.zeros(1, 3, dtype=torch.int64))
    torch.testing.assert_close(call["y"], y_source.narrow(2, 4, 2))
    torch.testing.assert_close(call["global_video"], global_latents.narrow(2, 4, 2))
    torch.testing.assert_close(call["clean_x"], clean_latents.narrow(3, 4, 2))


def test_write_denoised_context_cache_can_be_disabled(monkeypatch):
    Cls = _maybe_load_head()
    inst = Cls.__new__(Cls)
    inst._ma_cached_until_frame = 7
    calls = []
    inst._run_multi_agent_diffusion_steps = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setenv("MAI_WRITE_DENOISED_CONTEXT_CACHE", "0")

    wrote = inst._write_multi_agent_denoised_context_cache(
        noisy_video=torch.zeros(1),
        B=1,
        block=2,
        latents=torch.zeros(1),
        prompt_embs=[torch.zeros(1, 1)],
        seq_len=8,
        noisy_action=None,
        state_features=None,
        embodiment_id=None,
        y_source=None,
        clip_feature=None,
        kv_caches=[torch.empty(0)],
        crossattn_caches=[torch.empty(0)],
        current_start_frame=4,
        clean_latents=None,
        global_latents=None,
    )

    assert wrote is False
    assert calls == []
    assert inst._ma_cached_until_frame == 7
