"""End-to-end forward smoke for the multi-agent branch (PR 3a).

Builds a tiny ``CausalWanModel`` with ``num_agents=2`` and runs a random
``[B, P, C, F, H, W]`` input through ``forward``. We verify only:

* P=1 instantiation keeps the byte-identical 3D path (``self.simplex_rope is None``).
* P=2 instantiation builds the simplex RoPE.
* P=2 forward produces a tensor of shape ``[B, P, C_out, F, H, W]``.
* Permuting agent slots in the input does not crash and yields a different
  but same-shaped output (simplex RoPE is *not* slot-blind by design when
  the *input data* is also permuted -- this just checks the wiring runs).
"""

import os
import sys
from pathlib import Path

# Force the pure-torch attention backend so the smoke does not require
# flash_attn (the default FA2 backend coerces internal compute to bfloat16
# regardless of the surrounding model dtype, which breaks Float-only Linear
# layers when flash_attn is not installed).
os.environ.setdefault("ATTENTION_BACKEND", "torch")

import pytest
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(scope="module")
def cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for CausalWanModel smoke")


def _make_model(num_agents: int, device="cuda"):
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
        CausalWanModel,
    )

    model = CausalWanModel(
        model_type="t2v",
        patch_size=(1, 2, 2),
        frame_seqlen=4,  # H_g * W_g per latent frame (=2*2)
        text_len=8,
        in_dim=8,
        dim=96,           # split into 4 heads of 24; head_dim 24 -> (8,8,8) base.
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
        # head_dim = dim/num_heads = 24; agent_dim must be even and leave
        # room in the temporal band. We use agent_dim=4 for V=2 testing.
        agent_dim=4,
        simplex_pool_size=2,
    ).to(device).eval()
    model.init_weights()
    # The bundled AttentionModule defaults to bfloat16 internally; bring the
    # rest of the model along so all Linear / Norm ops share that dtype.
    model = model.to(dtype=torch.bfloat16)
    return model


def test_shared_global_latent_uses_patch_embedding_latent_slice():
    pytest.importorskip("einops")
    pytest.importorskip("diffusers")
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
        CausalWanModel,
    )

    model = CausalWanModel(
        model_type="i2v",
        patch_size=(1, 2, 2),
        frame_seqlen=4,
        text_len=8,
        in_dim=12,
        dim=96,
        ffn_dim=192,
        freq_dim=32,
        text_dim=32,
        out_dim=8,
        num_heads=4,
        num_layers=1,
        num_frame_per_block=1,
        action_dim=4,
        num_registers=2,
        max_state_dim=8,
        max_num_embodiments=1,
        hidden_size=64,
        num_action_per_block=1,
        num_state_per_block=1,
        concat_first_frame_latent=True,
        num_agents=2,
        agent_dim=4,
        simplex_pool_size=2,
    ).eval()
    model.init_weights()
    x = torch.randn(1, 8, 2, 4, 4)

    projected = model._patch_embedding_latent_channels(x)
    expected = F.conv3d(
        x,
        model.patch_embedding.weight[:, : x.shape[1]],
        bias=model.patch_embedding.bias,
        stride=model.patch_embedding.stride,
        padding=model.patch_embedding.padding,
        dilation=model.patch_embedding.dilation,
        groups=model.patch_embedding.groups,
    )

    assert projected.shape == (1, model.dim, 2, 2, 2)
    torch.testing.assert_close(projected, expected)


def test_global_video_timestep_mode_config_validation():
    pytest.importorskip("einops")
    pytest.importorskip("diffusers")
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
        CausalWanModel,
    )

    kwargs = dict(
        model_type="i2v",
        patch_size=(1, 2, 2),
        frame_seqlen=4,
        text_len=8,
        in_dim=12,
        dim=96,
        ffn_dim=192,
        freq_dim=32,
        text_dim=32,
        out_dim=8,
        num_heads=4,
        num_layers=1,
        num_frame_per_block=1,
        action_dim=4,
        num_registers=2,
        max_state_dim=8,
        max_num_embodiments=1,
        hidden_size=64,
        num_action_per_block=1,
        num_state_per_block=1,
        concat_first_frame_latent=True,
        num_agents=2,
        agent_dim=4,
        simplex_pool_size=2,
    )

    model = CausalWanModel(**kwargs)
    assert model.global_video_timestep_mode == "video"
    assert model.global_video_attention_mode == "bidirectional"

    model = CausalWanModel(
        **kwargs,
        global_video_timestep_mode="clean",
        global_video_attention_mode="read_only",
    )
    assert model.global_video_timestep_mode == "clean"
    assert model.global_video_attention_mode == "read_only"

    with pytest.raises(ValueError, match="global_video_timestep_mode"):
        CausalWanModel(**kwargs, global_video_timestep_mode="future")
    with pytest.raises(ValueError, match="global_video_attention_mode"):
        CausalWanModel(**kwargs, global_video_attention_mode="future")


def test_p1_keeps_3d_path(cuda_available):
    model = _make_model(num_agents=1)
    assert model.simplex_rope is None, (
        "P=1 must skip simplex RoPE to stay byte-identical with the 3D path"
    )


def test_p2_builds_simplex_rope(cuda_available):
    model = _make_model(num_agents=2)
    assert model.simplex_rope is not None
    assert model.simplex_rope.simplex_pool_size == 2


def test_p2_forward_shapes(cuda_available):
    torch.manual_seed(0)
    device = "cuda"
    model = _make_model(num_agents=2, device=device)

    B, P, C_in, F_lat, H, W = 1, 2, 8, 2, 4, 4
    F_g, H_g, W_g = F_lat, H // 2, W // 2
    L_per_agent = F_g * H_g * W_g
    seq_len = P * L_per_agent

    x = torch.randn(B, P, C_in, F_lat, H, W, device=device, dtype=torch.bfloat16)
    timestep = torch.randint(0, 1000, (B, F_lat), device=device).float()
    context = torch.randn(B, 8, 32, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        video_pred, action_pred = model(
            x=x,
            timestep=timestep,
            context=context,
            seq_len=seq_len,
        )

    assert action_pred is None, "PR 3a video-only branch should not emit action pred"
    assert video_pred.shape == (B, P, 8, F_lat, H, W), (
        f"unexpected video pred shape: {tuple(video_pred.shape)}"
    )
    assert torch.isfinite(video_pred).all()


def test_p2_agent_perm_runs(cuda_available):
    """Different ``agent_perm`` must produce different simplex RoPE freqs.

    We compare the RoPE freqs directly rather than the model output:
    ``init_weights`` zero-initialises the prediction head on purpose, so the
    end-to-end output of a randomly-initialised model is identically zero
    and cannot distinguish anything.
    """
    torch.manual_seed(0)
    device = "cuda"
    model = _make_model(num_agents=2, device=device)

    F_g, H_g, W_g = 2, 2, 2
    freqs_a = model.simplex_rope(
        f=F_g, p=2, h=H_g, w=W_g,
        agent_perm=torch.tensor([0, 1], device=device, dtype=torch.long),
    )
    freqs_b = model.simplex_rope(
        f=F_g, p=2, h=H_g, w=W_g,
        agent_perm=torch.tensor([1, 0], device=device, dtype=torch.long),
    )
    assert freqs_a.shape == freqs_b.shape
    assert not torch.equal(freqs_a, freqs_b), (
        "Swapping agent_perm slots must yield a different RoPE freq tensor"
    )

    # Also verify the *forward call* with a perm runs without crashing.
    B, P, C_in, F_lat, H, W = 1, 2, 8, F_g, H_g * 2, W_g * 2
    seq_len = P * F_g * H_g * W_g
    x = torch.randn(B, P, C_in, F_lat, H, W, device=device, dtype=torch.bfloat16)
    timestep = torch.randint(0, 1000, (B, F_lat), device=device).float()
    context = torch.randn(B, 8, 32, device=device, dtype=torch.bfloat16)
    perm = torch.tensor([1, 0], device=device, dtype=torch.long)

    with torch.no_grad():
        video, _ = model(
            x=x, timestep=timestep, context=context, seq_len=seq_len, agent_perm=perm
        )
    assert video.shape == (B, P, 8, F_lat, H, W)
