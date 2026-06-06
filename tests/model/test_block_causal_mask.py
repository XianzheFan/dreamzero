"""Tests for the sparse-hub multi-agent attention mask.

The production mask keeps the hub-mediated topology:
same-agent tokens, hub tokens, and shared-global tokens are visible; direct
agent-to-agent attention remains masked. We intentionally do not compose an
extra block-causal time mask here. DreamZero's original training path is
bidirectional within each denoised video chunk, and the block-causal variant
introduced a strong periodic grid artifact in multi-agent predicted video.

We exercise:
  * forward shape/finite smoke for a multi-frame sparse-hub input;
  * same-agent future-frame changes can influence earlier frames inside the
    same denoise chunk, matching the single-agent train path;
  * cached streaming tokens participate in attention.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("ATTENTION_BACKEND", "torch")

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(scope="module")
def cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for sparse-hub mask smoke")


def _make_model(num_agents=2, num_frame_per_block=1, device="cuda"):
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
        CausalWanModel,
    )

    model = CausalWanModel(
        model_type="t2v",
        patch_size=(1, 2, 2),
        frame_seqlen=4,
        text_len=8,
        in_dim=8,
        dim=96,
        ffn_dim=192,
        freq_dim=32,
        text_dim=32,
        out_dim=8,
        num_heads=4,
        num_layers=2,
        num_frame_per_block=num_frame_per_block,
        action_dim=4,
        num_registers=2,
        max_state_dim=8,
        max_num_embodiments=1,
        hidden_size=64,
        num_action_per_block=1,
        num_state_per_block=1,
        concat_first_frame_latent=False,
        num_agents=num_agents,
        agent_dim=4,
        simplex_pool_size=2,
        num_hub_tokens=4,
    ).to(device).eval()
    model.init_weights()
    torch.nn.init.normal_(model.head.head.weight, mean=0.0, std=0.02)
    return model.to(dtype=torch.bfloat16)


def _make_inputs(B=1, P=2, T_a=4, D_a=4, F_lat=4, H=4, W=4, device="cuda"):
    seq_len = P * F_lat * (H // 2) * (W // 2)
    return dict(
        x=torch.randn(B, P, 8, F_lat, H, W, device=device, dtype=torch.bfloat16),
        timestep=torch.randint(0, 1000, (B, F_lat), device=device).float(),
        timestep_action=torch.randint(0, 1000, (B, T_a), device=device).float(),
        context=torch.randn(B, 8, 32, device=device, dtype=torch.bfloat16),
        seq_len=seq_len,
        action=torch.randn(B, P, T_a, D_a, device=device, dtype=torch.bfloat16),
    )


def test_forward_runs_with_sparse_hub_mask(cuda_available):
    """Sanity: forward still runs end-to-end with multi-block input
    (num_frame_per_block=1, F=4 -> 4 blocks)."""
    torch.manual_seed(0)
    model = _make_model(num_agents=2, num_frame_per_block=1)
    inputs = _make_inputs(F_lat=4)
    with torch.no_grad():
        video, action = model(**inputs)
    B, P, _, F_lat, H, W = inputs["x"].shape
    assert video.shape == (B, P, 8, F_lat, H, W)
    assert torch.isfinite(video).all()
    assert torch.isfinite(action).all()


def test_future_frame_swap_reaches_same_agent_current_frame(cuda_available):
    """Sparse-hub masking should not impose extra temporal causality.

    Swapping a later frame from the same agent is allowed to change the
    earliest-frame prediction inside the same denoise chunk, matching the
    original single-agent train path.
    """
    torch.manual_seed(0)
    device = "cuda"
    model = _make_model(num_agents=2, num_frame_per_block=1, device=device)
    inputs = _make_inputs(F_lat=4, device=device)

    # Make inputs identical across agents so simplex variation is the only
    # remaining per-agent signal; we still operate on agent 0 below.
    inputs["x"][:, 1] = inputs["x"][:, 0]

    with torch.no_grad():
        video_a, _ = model(**inputs)

    # Perturb only the LAST frame of agent 0 (and 1 for symmetry).
    inputs_b = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in inputs.items()}
    inputs_b["x"][:, :, :, -1] = torch.randn_like(inputs_b["x"][:, :, :, -1])
    with torch.no_grad():
        video_b, _ = model(**inputs_b)

    # Frame-0 prediction should change because same-agent future tokens
    # remain visible inside the current denoise chunk.
    frame0_a = video_a[:, :, :, 0]
    frame0_b = video_b[:, :, :, 0]
    diff = (frame0_a - frame0_b).abs().mean().float().item()
    assert diff > 1e-3, (
        f"Frame-0 prediction ignored a same-agent future-frame perturbation "
        f"(mean diff {diff:.3e}). The mask may be imposing temporal causality."
    )

    # Sanity: the LAST frame's prediction should have changed (we
    # actually modified that frame's input).
    frame_last_a = video_a[:, :, :, -1]
    frame_last_b = video_b[:, :, :, -1]
    diff_last = (frame_last_a - frame_last_b).abs().mean().float().item()
    assert diff_last > 1e-2, (
        "Frame-last prediction did NOT change under frame-last input "
        "perturbation -- test setup is bogus."
    )


def test_single_block_case_is_unconstrained(cuda_available):
    """A future-frame perturbation can reach the early frame prediction."""
    torch.manual_seed(0)
    device = "cuda"
    # F=2, num_frame_per_block=2 -> exactly 1 block.
    model = _make_model(num_agents=2, num_frame_per_block=2, device=device)
    inputs = _make_inputs(F_lat=2, device=device)

    with torch.no_grad():
        video_a, _ = model(**inputs)
    inputs_b = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in inputs.items()}
    inputs_b["x"][:, :, :, -1] = torch.randn_like(inputs_b["x"][:, :, :, -1])
    with torch.no_grad():
        video_b, _ = model(**inputs_b)

    # Frame 0 should differ because intra-block attention is allowed.
    diff_f0 = (video_a[:, :, :, 0] - video_b[:, :, :, 0]).abs().mean().float().item()
    assert diff_f0 > 1e-3, (
        "Single-block case should allow bidirectional in-block attention; "
        f"frame-0 didn't change ({diff_f0:.3e})."
    )


def test_streaming_cached_tokens_always_visible(cuda_available):
    """Cached tokens from past chunks must remain visible.

    A streaming call with a populated cache should produce a DIFFERENT
    output than the same call with that cache replaced by zeros, proving
    the cache K/V actually participated in attention.

    We run the warm-up + streaming sequence twice on separate model
    instances so each has its own session-tracked
    ``_cached_token_agent_id``.
    """
    torch.manual_seed(0)
    device = "cuda"
    inputs_warm = _make_inputs(F_lat=2, device=device)
    inputs_stream = _make_inputs(F_lat=2, device=device)

    def warm_then_stream(zero_the_cache: bool):
        m = _make_model(num_agents=2, num_frame_per_block=1, device=device)
        num_layers = len(m.blocks)
        with torch.no_grad():
            _, _, cache_after_warm = m(
                **inputs_warm,
                kv_cache=[None] * num_layers,
                crossattn_cache=[None] * num_layers,
                current_start_frame=0,
            )
            if zero_the_cache:
                cache_after_warm = [torch.zeros_like(c) for c in cache_after_warm]
            video, _, _ = m(
                **inputs_stream,
                kv_cache=cache_after_warm,
                crossattn_cache=[None] * num_layers,
                current_start_frame=2,
            )
        return video

    video_with_cache = warm_then_stream(zero_the_cache=False)
    video_with_zero_cache = warm_then_stream(zero_the_cache=True)

    diff = (video_with_cache - video_with_zero_cache).abs().mean().float().item()
    assert diff > 1e-3, (
        f"Streaming output ignored the populated cache (mean diff {diff:.3e}) "
        "-- sparse-hub masking may be incorrectly blocking cached tokens."
    )
