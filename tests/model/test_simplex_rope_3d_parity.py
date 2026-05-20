"""Regression smoke: SimplexRotaryPositionEmbedding4D upper-band / spatial-band
output matches the existing 3D RoPE bit-for-bit.

This is the contract PR 3 will rely on when switching the model between the
3D and 4D RoPE at runtime: P=1 must not perturb the high-frequency temporal
slots or the spatial bands, only the low-frequency temporal slots that get
re-purposed for the agent axis.

We replicate the 3D freq computation from ``RotaryPositionEmbeddingNoPolarOp``
inline to avoid pulling the full ``wan_video_dit`` module (it depends on
diffusers / einops at import time, which are not installed in every env).
"""

import importlib.util
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "groot/vla/model/dreamzero/modules/simplex_rope.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_mod = _load("simplex_rope_under_test_parity", _MODULE_PATH)
SimplexRotaryPositionEmbedding4D = _mod.SimplexRotaryPositionEmbedding4D


def _reference_3d_freqs(f: int, h: int, w: int, head_dim: int):
    """Replicate ``RotaryPositionEmbeddingNoPolarOp.precompute_freqs_cis_3d``."""
    d_h = head_dim // 3
    d_w = head_dim // 3
    d_f = head_dim - 2 * (head_dim // 3)
    theta = 10000.0

    def _1d(dim, end):
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
        a = torch.outer(torch.arange(end).float(), inv)
        return torch.cos(a), torch.sin(a)

    f_cos, f_sin = _1d(d_f, f)
    h_cos, h_sin = _1d(d_h, h)
    w_cos, w_sin = _1d(d_w, w)

    cos = torch.cat(
        [
            f_cos[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            h_cos[:h].view(1, h, 1, -1).expand(f, h, w, -1),
            w_cos[:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1)
    sin = torch.cat(
        [
            f_sin[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            h_sin[:h].view(1, h, 1, -1).expand(f, h, w, -1),
            w_sin[:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1)
    return torch.cat([cos, sin], dim=0)


def test_simplex_p1_preserves_active_temporal_and_spatial():
    """At P=1 with reasonable agent_dim, the upper temporal + spatial slots
    must match the 3D reference. Only the lowest agent_dim/2 temporal complex
    slots diverge (those are now used by the agent axis).
    """
    head_dim = 126  # 42/42/42 split, matches Wan-style heads
    agent_dim = 12  # carve 12 temporal dims for agent band (6 complex slots)
    rope_4d = SimplexRotaryPositionEmbedding4D(
        num_heads=8,
        head_dim=head_dim,
        simplex_pool_size=4,
        agent_dim=agent_dim,
    )
    f, h, w = 6, 8, 8
    ref = _reference_3d_freqs(f, h, w, head_dim)
    new = rope_4d.forward(f=f, p=1, h=h, w=w)

    # Same row count: P=1 leaves token grid the same size as 3D.
    assert ref.shape == new.shape, f"shapes diverge: {ref.shape} vs {new.shape}"

    N = f * h * w
    d_t_full_half = rope_4d.d_t_full // 2
    d_t_active_half = rope_4d.d_t_active // 2
    d_h_half = head_dim // 3 // 2

    # Cos: rows [0, N). Sin: rows [N, 2N). Split is identical between ref & new.
    for offset in (0, N):
        ref_block = ref[offset : offset + N]
        new_block = new[offset : offset + N]
        # Upper temporal slots must be byte-identical.
        torch.testing.assert_close(
            new_block[:, 0, :d_t_active_half],
            ref_block[:, 0, :d_t_active_half],
            atol=0.0,
            rtol=0.0,
            msg="High-frequency temporal slots diverged under simplex 4D RoPE at P=1",
        )
        # Spatial bands must be byte-identical.
        torch.testing.assert_close(
            new_block[:, 0, d_t_full_half:],
            ref_block[:, 0, d_t_full_half:],
            atol=0.0,
            rtol=0.0,
            msg="Spatial bands diverged under simplex 4D RoPE at P=1",
        )


def test_simplex_p1_low_freq_diverges_only_in_agent_band():
    """Sanity check the *other* side of the contract: the simplex DOES alter
    the low-frequency temporal slots (that's where the agent band lives)."""
    head_dim = 126
    agent_dim = 16
    rope_4d = SimplexRotaryPositionEmbedding4D(
        num_heads=8, head_dim=head_dim, simplex_pool_size=4, agent_dim=agent_dim,
    )
    f, h, w = 4, 4, 4
    ref = _reference_3d_freqs(f, h, w, head_dim)
    new = rope_4d.forward(f=f, p=1, h=h, w=w)
    N = f * h * w
    d_t_active_half = rope_4d.d_t_active // 2
    d_t_full_half = rope_4d.d_t_full // 2

    # The agent slots (between d_t_active_half and d_t_full_half) should
    # differ from the 3D reference for at least one row.
    new_band = new[:N, 0, d_t_active_half:d_t_full_half]
    ref_band = ref[:N, 0, d_t_active_half:d_t_full_half]
    assert not torch.allclose(new_band, ref_band), (
        "Agent band should diverge from 3D-RoPE low-freq temporal; it didn't"
    )
