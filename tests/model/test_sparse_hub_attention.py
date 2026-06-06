"""Tests for Sparse Hub Attention (PR 4).

Covers:
* ``SimplexRotaryPositionEmbedding4D.hub_freqs`` shape + identity contract
  on the spatial and agent bands.
* The hub mask routes attention through hub tokens only (no direct
  agent-to-agent path).
* ``CausalWanModel`` builds learnable ``hub_tokens`` when
  ``num_agents>1`` and ``use_sparse_hub_attention=True``.
* Hub tokens accumulate gradient during a P=2 forward + backward.
* Toggling ``use_sparse_hub_attention=False`` reverts to PR-3a behaviour
  (no hub tokens, dense attention).
"""

import importlib.util
import os
import sys
from pathlib import Path

# Same dependency-light pattern as the other tests in this directory.
os.environ.setdefault("ATTENTION_BACKEND", "torch")

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

_SIMPLEX_PATH = _REPO_ROOT / "groot/vla/model/dreamzero/modules/simplex_rope.py"


def _load_simplex():
    spec = importlib.util.spec_from_file_location(
        "simplex_rope_hub_under_test", _SIMPLEX_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["simplex_rope_hub_under_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for hub attention smoke")


# ---------------------------------------------------------------------------
# hub_freqs unit tests
# ---------------------------------------------------------------------------


def test_hub_freqs_shape_real():
    _mod = _load_simplex()
    rope = _mod.SimplexRotaryPositionEmbedding4D(
        num_heads=4, head_dim=126, simplex_pool_size=4, agent_dim=16,
    )
    freqs = rope.hub_freqs(f=3, k_hub=8)
    # real-stacked: (2 * F * K, 1, head_dim/2)
    assert freqs.shape == (2 * 3 * 8, 1, 63)


def test_hub_freqs_shape_polar():
    _mod = _load_simplex()
    rope = _mod.SimplexRotaryPositionEmbedding4D(
        num_heads=4, head_dim=126, simplex_pool_size=4, agent_dim=16,
        polar_output=True,
    )
    freqs = rope.hub_freqs(f=3, k_hub=8)
    assert freqs.shape == (3 * 8, 1, 63)
    assert freqs.dtype.is_complex


def test_hub_freqs_agent_band_is_identity_rotation():
    """Hub tokens must use identity rotation on the agent + spatial bands."""
    _mod = _load_simplex()
    rope = _mod.SimplexRotaryPositionEmbedding4D(
        num_heads=4, head_dim=126, simplex_pool_size=4, agent_dim=16,
    )
    f, k_hub = 2, 4
    freqs = rope.hub_freqs(f=f, k_hub=k_hub)
    N = f * k_hub
    cos = freqs[:N]
    sin = freqs[N:]

    d_t_active_half = rope.d_t_active // 2
    d_t_full_half = rope.d_t_full // 2
    # Agent band slots (cos=1, sin=0) -> identity rotation.
    agent_cos = cos[:, 0, d_t_active_half:d_t_full_half]
    agent_sin = sin[:, 0, d_t_active_half:d_t_full_half]
    assert torch.equal(agent_cos, torch.ones_like(agent_cos))
    assert torch.equal(agent_sin, torch.zeros_like(agent_sin))
    # Spatial bands (everything after d_t_full_half) -> identity.
    spatial_cos = cos[:, 0, d_t_full_half:]
    spatial_sin = sin[:, 0, d_t_full_half:]
    assert torch.equal(spatial_cos, torch.ones_like(spatial_cos))
    assert torch.equal(spatial_sin, torch.zeros_like(spatial_sin))


def test_hub_freqs_share_frame_temporal_phase():
    """All K hub tokens of the same frame must share the temporal phase."""
    _mod = _load_simplex()
    rope = _mod.SimplexRotaryPositionEmbedding4D(
        num_heads=4, head_dim=126, simplex_pool_size=4, agent_dim=16,
    )
    f, k_hub = 3, 4
    freqs = rope.hub_freqs(f=f, k_hub=k_hub)
    N = f * k_hub
    cos = freqs[:N]
    d_t_active_half = rope.d_t_active // 2
    for f_idx in range(f):
        base = cos[f_idx * k_hub, 0, :d_t_active_half]
        for k_idx in range(1, k_hub):
            row = cos[f_idx * k_hub + k_idx, 0, :d_t_active_half]
            torch.testing.assert_close(row, base)


# ---------------------------------------------------------------------------
# CausalWanModel hub wiring
# ---------------------------------------------------------------------------


def _make_model(num_agents=2, num_hub_tokens=8, use_hub=True, device="cuda"):
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
        num_frame_per_block=1,
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
        num_hub_tokens=num_hub_tokens,
        use_sparse_hub_attention=use_hub,
    ).to(device).eval()
    model.init_weights()
    return model.to(dtype=torch.bfloat16)


def test_p1_has_no_hub_tokens(cuda_available):
    model = _make_model(num_agents=1)
    assert model.hub_tokens is None


def test_p2_with_hub_builds_param(cuda_available):
    model = _make_model(num_agents=2, num_hub_tokens=4)
    assert model.hub_tokens is not None
    assert model.hub_tokens.shape == (4, 96)


def test_p2_without_hub_falls_back_to_dense(cuda_available):
    model = _make_model(num_agents=2, use_hub=False)
    assert model.hub_tokens is None


def test_p2_hub_forward_shapes(cuda_available):
    torch.manual_seed(0)
    device = "cuda"
    model = _make_model(num_agents=2, num_hub_tokens=4, device=device)

    B, P, C_in, F_lat, H, W = 1, 2, 8, 2, 4, 4
    seq_len = P * F_lat * (H // 2) * (W // 2)

    x = torch.randn(B, P, C_in, F_lat, H, W, device=device, dtype=torch.bfloat16)
    timestep = torch.randint(0, 1000, (B, F_lat), device=device).float()
    context = torch.randn(B, 8, 32, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        video, action = model(
            x=x, timestep=timestep, context=context, seq_len=seq_len,
        )
    assert action is None
    assert video.shape == (B, P, 8, F_lat, H, W)
    assert torch.isfinite(video).all()


def test_hub_tokens_receive_gradient(cuda_available):
    """A backward pass through the multi-agent path must populate
    ``hub_tokens.grad`` -- i.e. the hubs are actually used."""
    torch.manual_seed(0)
    device = "cuda"
    model = _make_model(num_agents=2, num_hub_tokens=4, device=device)
    # We need gradients; init_weights zero-inits the head, which would zero
    # any video-side loss. Re-init the head to small random weights.
    torch.nn.init.normal_(model.head.head.weight, mean=0.0, std=0.02)
    model = model.train()

    B, P, C_in, F_lat, H, W = 1, 2, 8, 2, 4, 4
    seq_len = P * F_lat * (H // 2) * (W // 2)
    x = torch.randn(B, P, C_in, F_lat, H, W, device=device, dtype=torch.bfloat16)
    timestep = torch.randint(0, 1000, (B, F_lat), device=device).float()
    context = torch.randn(B, 8, 32, device=device, dtype=torch.bfloat16)

    video, _ = model(x=x, timestep=timestep, context=context, seq_len=seq_len)
    loss = video.float().pow(2).mean()
    loss.backward()

    assert model.hub_tokens.grad is not None
    assert torch.isfinite(model.hub_tokens.grad).all()
    assert (model.hub_tokens.grad.abs() > 0).any(), (
        "Hub tokens received an all-zero gradient -- attention may be "
        "ignoring them"
    )


def test_hub_mask_blocks_direct_cross_agent_path(cuda_available):
    """Independently verify the hub mask logic used inside the model.

    Two distinct agent tokens must NOT attend to each other directly;
    every cross-agent token pair must include at least one hub token.
    """
    P, L_per_agent, F_g, K_hub = 2, 8, 2, 4
    hub_token_count = F_g * K_hub

    agent_ids = torch.arange(P).repeat_interleave(L_per_agent)
    hub_ids = torch.full((hub_token_count,), P, dtype=torch.long)
    token_agent = torch.cat([agent_ids, hub_ids], dim=0)
    is_hub = token_agent == P
    same_agent = token_agent.unsqueeze(1) == token_agent.unsqueeze(0)
    mask = same_agent | is_hub.unsqueeze(1) | is_hub.unsqueeze(0)

    # Pick one token from each agent stream that is not a hub.
    a0 = 0
    a1 = L_per_agent  # first token of agent 1
    assert not mask[a0, a1], "agent 0 -> agent 1 must be masked out"
    assert not mask[a1, a0], "agent 1 -> agent 0 must be masked out"

    # Hub <-> anyone should be open.
    h0 = P * L_per_agent
    assert mask[a0, h0] and mask[h0, a0]
    assert mask[a1, h0] and mask[h0, a1]

    # Self-attention is always open.
    diag = mask.diagonal()
    assert diag.all()


def test_multi_agent_register_block_ids_align_with_future_video_blocks():
    """Video chunk i must be able to read action/state register chunk i.

    The first latent frame is an observed conditioning frame, so future
    block 0 starts at latent frame 1. This catches the regression where all
    registers were assigned to the last block and early video tokens could
    not attend to their own action/state conditioning.
    """
    pytest.importorskip("einops")
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
        CausalWanModel,
    )

    device = torch.device("cpu")
    video_ids = CausalWanModel._future_video_block_ids(
        num_frames=9,
        num_frame_per_block=2,
        start_frame=0,
        device=device,
    )
    clean_ids = CausalWanModel._clean_context_block_ids(
        num_frames=9,
        num_frame_per_block=2,
        start_frame=0,
        device=device,
    )
    register_ids = CausalWanModel._register_block_ids(
        num_action_tokens=96,
        num_state_tokens=4,
        num_action_per_block=24,
        num_state_per_block=1,
        start_frame=0,
        num_frame_per_block=2,
        device=device,
    )

    assert video_ids.tolist() == [-1, 0, 0, 1, 1, 2, 2, 3, 3]
    assert clean_ids.tolist() == [-1] * 9
    assert register_ids[:96].reshape(4, 24)[:, 0].tolist() == [0, 1, 2, 3]
    assert register_ids[96:].tolist() == [0, 1, 2, 3]

    # Future video block 0 can see action/state block 0, but not future
    # action/state blocks.
    q_block0 = video_ids[1]
    assert q_block0 >= register_ids[0]
    assert q_block0 >= register_ids[96]
    assert not bool(q_block0 >= register_ids[24])
    assert not bool(q_block0 >= register_ids[97])

    # Future video block 1 sees all clean current-observation context and
    # its own action/state chunk.
    q_block1 = video_ids[3]
    assert all(bool(q_block1 >= c) for c in clean_ids)
    assert q_block1 >= register_ids[24]


def test_multi_agent_clean_context_is_visible_after_causal_priming():
    """Causal start_frame=1 must still read the current observation.

    Closed-loop multi-agent inference primes frame 0 into the KV cache, then
    denoises future frames starting at latent frame 1. ``clean_x`` is the
    current observed frame repeated over the temporal window, so assigning it
    to future block 1 would make the first future block unable to attend to
    the current image condition.
    """
    pytest.importorskip("einops")
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
        CausalWanModel,
    )

    device = torch.device("cpu")
    video_ids = CausalWanModel._future_video_block_ids(
        num_frames=2,
        num_frame_per_block=2,
        start_frame=1,
        device=device,
    )
    clean_ids = CausalWanModel._clean_context_block_ids(
        num_frames=2,
        num_frame_per_block=2,
        start_frame=1,
        device=device,
    )

    assert video_ids.tolist() == [0, 0]
    assert clean_ids.tolist() == [-1, -1]
    assert all(bool(video_ids[0] >= c) for c in clean_ids)
