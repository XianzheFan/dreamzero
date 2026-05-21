"""Tests for ``simplex_rope``: vertex math and 4D RoPE freqs.

Loaded by path to avoid pulling the heavy ``groot.vla.model`` package init
(which depends on diffusers, einops, etc.).
"""

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "groot/vla/model/dreamzero/modules/simplex_rope.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_mod = _load("simplex_rope_under_test", _MODULE_PATH)
build_simplex_vertices = _mod.build_simplex_vertices
SimplexRotaryPositionEmbedding4D = _mod.SimplexRotaryPositionEmbedding4D


# ---------------------------------------------------------------------------
# Vertex math (Gamma-World Appendix B).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V,d", [(2, 4), (3, 8), (4, 16), (8, 32)])
def test_simplex_unit_norm(V, d):
    s = build_simplex_vertices(V, d)
    norms = torch.linalg.norm(s, dim=-1)
    torch.testing.assert_close(norms, torch.ones(V), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("V,d", [(2, 4), (3, 8), (4, 16), (8, 32)])
def test_simplex_pairwise_distance(V, d):
    s = build_simplex_vertices(V, d)
    expected = math.sqrt(2 * V / (V - 1))
    for v1 in range(V):
        for v2 in range(v1 + 1, V):
            dist = torch.linalg.norm(s[v1] - s[v2]).item()
            assert abs(dist - expected) < 1e-5, (
                f"V={V} d={d} v1={v1} v2={v2}: distance {dist} vs expected {expected}"
            )


@pytest.mark.parametrize("V,d", [(2, 4), (3, 8), (4, 16)])
def test_simplex_inner_product(V, d):
    s = build_simplex_vertices(V, d)
    expected = -1.0 / (V - 1)
    for v1 in range(V):
        for v2 in range(V):
            ip = (s[v1] * s[v2]).sum().item()
            if v1 == v2:
                assert abs(ip - 1.0) < 1e-5
            else:
                assert abs(ip - expected) < 1e-5


def test_simplex_rejects_V_below_2():
    with pytest.raises(ValueError, match="V >= 2"):
        build_simplex_vertices(1, 16)


def test_simplex_rejects_d_below_V():
    with pytest.raises(ValueError, match="d >= V"):
        build_simplex_vertices(4, 3)


# ---------------------------------------------------------------------------
# 4D RoPE class.
# ---------------------------------------------------------------------------


def _make_rope(head_dim=126, agent_dim=16, V=4, alpha=1.0):
    # head_dim is chosen so head_dim - 2*(head_dim//3) is divisible by 2
    # without surprises; 126 -> (42, 42, 42) split.
    return SimplexRotaryPositionEmbedding4D(
        num_heads=8,
        head_dim=head_dim,
        simplex_pool_size=V,
        agent_dim=agent_dim,
        alpha=alpha,
    )


def test_rope_output_shape():
    rope = _make_rope()
    freqs = rope.forward(f=3, p=2, h=4, w=5)
    N = 3 * 2 * 4 * 5
    assert freqs.shape == (2 * N, 1, rope.head_dim // 2)


def test_rope_dtype_and_device():
    rope = _make_rope()
    freqs = rope.forward(f=2, p=2, h=2, w=2)
    assert freqs.dtype == torch.float32
    assert freqs.device == torch.device("cpu")


def test_rope_p_exceeds_pool_raises():
    rope = _make_rope(V=4)
    with pytest.raises(ValueError, match="exceeds simplex_pool_size"):
        rope.forward(f=1, p=5, h=1, w=1)


def test_agent_perm_out_of_range_raises():
    rope = _make_rope(V=4)
    bad = torch.tensor([0, 99])
    with pytest.raises(ValueError, match="out-of-range vertex index"):
        rope.forward(f=1, p=2, h=1, w=1, agent_perm=bad)


def test_agent_permutation_swaps_per_agent_slices():
    """Swapping agent_perm should swap the corresponding rows in freqs."""
    rope = _make_rope(V=4)
    f, p, h, w = 1, 2, 1, 1
    freqs_a = rope.forward(f, p, h, w, agent_perm=torch.tensor([0, 1]))
    freqs_b = rope.forward(f, p, h, w, agent_perm=torch.tensor([1, 0]))

    # Layout is (P, F, H, W) C-order. With f=h=w=1, rows 0,1 are agent 0,1.
    # cos block lives in rows [0, N); sin block in [N, 2N) where N = P*F*H*W = 2.
    N = f * p * h * w
    cos_a = freqs_a[:N]
    cos_b = freqs_b[:N]
    sin_a = freqs_a[N:]
    sin_b = freqs_b[N:]

    torch.testing.assert_close(cos_a[0], cos_b[1])
    torch.testing.assert_close(cos_a[1], cos_b[0])
    torch.testing.assert_close(sin_a[0], sin_b[1])
    torch.testing.assert_close(sin_a[1], sin_b[0])


def test_active_temporal_band_independent_of_p():
    """The high-freq temporal slots must be the same across agents."""
    rope = _make_rope(V=4, agent_dim=16)
    f, p, h, w = 2, 3, 1, 1
    freqs = rope.forward(f, p, h, w)
    N = f * p * h * w

    # Channels are concatenated [t_band(d_t_full/2) | h(d_h/2) | w(d_w/2)].
    # Within t_band, the FIRST d_t_active/2 channels are the temporal slots
    # (shared across agents); the remaining agent_dim/2 channels are agent.
    d_t_active_half = rope.d_t_active // 2
    # Cos block rows: layout C-order (P, F, H, W). With h=w=1, row index is p_idx*F + f_idx.
    cos = freqs[:N]
    for f_idx in range(f):
        ref = cos[0 * f + f_idx, 0, :d_t_active_half]
        for p_idx in range(1, p):
            cur = cos[p_idx * f + f_idx, 0, :d_t_active_half]
            torch.testing.assert_close(cur, ref)


def test_agent_band_independent_of_t():
    """The simplex (low-temporal) slots must be the same across frames."""
    rope = _make_rope(V=4, agent_dim=16)
    f, p, h, w = 3, 2, 1, 1
    freqs = rope.forward(f, p, h, w)
    N = f * p * h * w
    d_t_active_half = rope.d_t_active // 2
    d_t_full_half = rope.d_t_full // 2

    cos = freqs[:N]
    for p_idx in range(p):
        ref = cos[p_idx * f + 0, 0, d_t_active_half:d_t_full_half]
        for f_idx in range(1, f):
            cur = cos[p_idx * f + f_idx, 0, d_t_active_half:d_t_full_half]
            torch.testing.assert_close(cur, ref)


def test_spatial_bands_match_standalone_3d_compute():
    """The h and w bands must match a direct 3D RoPE computation."""
    head_dim = 126
    agent_dim = 16
    rope = _make_rope(head_dim=head_dim, agent_dim=agent_dim, V=4)
    f, p, h, w = 1, 1, 4, 4
    freqs = rope.forward(f, p, h, w)
    N = f * p * h * w
    cos = freqs[:N]
    sin = freqs[N:]

    d_h = head_dim // 3
    d_w = head_dim // 3
    d_t_full_half = rope.d_t_full // 2
    d_h_half = d_h // 2
    d_w_half = d_w // 2

    theta = 10000.0
    inv_h = 1.0 / (theta ** (torch.arange(0, d_h, 2).float() / d_h))
    inv_w = 1.0 / (theta ** (torch.arange(0, d_w, 2).float() / d_w))
    h_angles = torch.outer(torch.arange(h).float(), inv_h)  # [H, d_h/2]
    w_angles = torch.outer(torch.arange(w).float(), inv_w)  # [W, d_w/2]

    # cos rows iterate (h, w) C-order with w innermost (since f=p=1).
    for hi in range(h):
        for wi in range(w):
            row = hi * w + wi
            h_part = cos[row, 0, d_t_full_half : d_t_full_half + d_h_half]
            w_part = cos[row, 0, d_t_full_half + d_h_half :]
            torch.testing.assert_close(h_part, torch.cos(h_angles[hi]))
            torch.testing.assert_close(w_part, torch.cos(w_angles[wi]))
            h_part_s = sin[row, 0, d_t_full_half : d_t_full_half + d_h_half]
            w_part_s = sin[row, 0, d_t_full_half + d_h_half :]
            torch.testing.assert_close(h_part_s, torch.sin(h_angles[hi]))
            torch.testing.assert_close(w_part_s, torch.sin(w_angles[wi]))


def test_simplex_phases_use_alpha():
    """alpha scales the simplex phase magnitude."""
    rope_a = _make_rope(V=4, agent_dim=16, alpha=1.0)
    rope_b = _make_rope(V=4, agent_dim=16, alpha=0.5)
    freqs_a = rope_a.forward(1, 2, 1, 1)
    freqs_b = rope_b.forward(1, 2, 1, 1)

    N = 1 * 2 * 1 * 1
    d_t_active_half = rope_a.d_t_active // 2
    d_t_full_half = rope_a.d_t_full // 2

    cos_a = freqs_a[:N, 0, d_t_active_half:d_t_full_half]
    cos_b = freqs_b[:N, 0, d_t_active_half:d_t_full_half]
    assert not torch.allclose(cos_a, cos_b), (
        "Different alpha values must produce different simplex phases"
    )


def test_polar_output_matches_no_polar():
    """polar_output=True must yield cos + i*sin of the no-polar output."""
    rope_real = _make_rope(V=4, agent_dim=16)
    rope_polar = SimplexRotaryPositionEmbedding4D(
        num_heads=8, head_dim=126, simplex_pool_size=4, agent_dim=16,
        polar_output=True,
    )
    f, p, h, w = 2, 3, 2, 2
    real_freqs = rope_real.forward(f, p, h, w)
    polar_freqs = rope_polar.forward(f, p, h, w)

    N = p * f * h * w
    assert polar_freqs.shape == (N, 1, 63), polar_freqs.shape
    assert polar_freqs.dtype.is_complex, polar_freqs.dtype

    # cos / sin packed into the complex tensor must match the real stacks.
    torch.testing.assert_close(polar_freqs.real, real_freqs[:N])
    torch.testing.assert_close(polar_freqs.imag, real_freqs[N:])


def test_layout_is_agent_major():
    """Sanity-check the (P, F, H, W) C-order. With f=2, p=3, the first F rows
    must belong to agent 0, the next F rows to agent 1, etc."""
    rope = _make_rope(V=4, agent_dim=16)
    f, p, h, w = 2, 3, 1, 1
    freqs = rope.forward(f, p, h, w)
    N = p * f * h * w
    d_t_active_half = rope.d_t_active // 2
    d_t_full_half = rope.d_t_full // 2

    cos = freqs[:N]
    # Within each agent's contiguous block, the agent-band slots must be constant.
    for p_idx in range(p):
        block = cos[p_idx * f : (p_idx + 1) * f, 0, d_t_active_half:d_t_full_half]
        for row in block[1:]:
            torch.testing.assert_close(row, block[0])

    # Across distinct agent blocks, the agent-band slots must differ.
    block_a = cos[0 * f, 0, d_t_active_half:d_t_full_half]
    block_b = cos[1 * f, 0, d_t_active_half:d_t_full_half]
    assert not torch.allclose(block_a, block_b)


def test_rope_rejects_unsupported_dims():
    with pytest.raises(ValueError, match="even"):
        SimplexRotaryPositionEmbedding4D(num_heads=8, head_dim=127, agent_dim=16)
    with pytest.raises(ValueError, match="even"):
        SimplexRotaryPositionEmbedding4D(num_heads=8, head_dim=128, agent_dim=15)
    with pytest.raises(ValueError, match="simplex_pool_size"):
        SimplexRotaryPositionEmbedding4D(
            num_heads=8, head_dim=128, simplex_pool_size=1, agent_dim=16
        )
    with pytest.raises(ValueError, match=r"agent_dim/2 \(2\)"):
        SimplexRotaryPositionEmbedding4D(
            num_heads=8, head_dim=128, simplex_pool_size=4, agent_dim=4
        )
