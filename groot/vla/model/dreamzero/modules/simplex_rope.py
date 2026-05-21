"""Simplex Rotary Agent Encoding (Gamma-World §3.3) and 4D RoPE.

This module provides a parameter-free way to encode agent identity along a new
rotary axis without breaking the permutation symmetry between agents. Agents
are placed at the vertices of a regular simplex in rotary angle space; the
pairwise angular distance is identical for every pair of distinct agents.

The 4D rotary layout is ``(t, p, h, w)``. To stay compatible with a 3D-RoPE
pretrained checkpoint, the agent band is allocated from the *low-frequency
end* of the temporal band (ReRoPE-style), leaving the high-frequency temporal
slots and the full spatial bands intact.

Math reference: Gamma-World paper, Appendix B.
"""

import math

import torch
import torch.nn as nn


def build_simplex_vertices(V: int, d: int) -> torch.Tensor:
    """Construct ``V`` regular-simplex vertices in :math:`\\mathbb{R}^d`.

    All vertices are unit norm and pairwise equidistant:

    * ``||s_v||_2 = 1`` for every ``v``
    * ``||s_v - s_w||_2^2 = 2V/(V-1)`` for every ``v != w``
    * ``<s_v, s_w> = -1/(V-1)`` for ``v != w``

    Construction follows Gamma-World Appendix B: take centered one-hot
    vectors ``\\bar s_v = e_v - 1/V \\cdot \\mathbf{1}`` in :math:`\\mathbb{R}^V`,
    normalise by ``sqrt(V/(V-1))``, then zero-pad to ``d`` dimensions. The
    zero padding preserves equidistance because every pairwise difference has
    the same non-zero coordinate pattern up to permutation.

    Args:
        V: Simplex pool size (number of vertices / max agent identities).
            Must be ``>= 2``.
        d: Embedding dimension. Must satisfy ``d >= V`` for the simple
            zero-padded construction used here.

    Returns:
        A ``(V, d)`` float tensor of vertex coordinates.
    """
    if V < 2:
        raise ValueError(f"Simplex needs V >= 2 vertices, got V={V}")
    if d < V:
        raise ValueError(
            f"build_simplex_vertices requires d >= V; got d={d}, V={V}. "
            f"Increase the agent rotary band width or shrink the simplex pool."
        )

    centered = torch.eye(V) - 1.0 / V  # row v = e_v - 1/V * 1, shape [V, V]
    vertices = math.sqrt(V / (V - 1)) * centered  # unit-norm, equidistant
    if d > V:
        pad = torch.zeros(V, d - V, dtype=vertices.dtype)
        vertices = torch.cat([vertices, pad], dim=-1)
    return vertices


class SimplexRotaryPositionEmbedding4D(nn.Module):
    """4D rotary position embedding over ``(t, p, h, w)``.

    The agent band of width ``agent_dim`` is carved out of the *low-frequency*
    end of the temporal band, so the spatial bands and the high-frequency
    temporal slots match the layout of the underlying 3D-RoPE checkpoint
    (ReRoPE-style allocation, Gamma-World §3.3 last paragraph).

    Frequency contract per token at coordinate ``(t, p, h, w)`` (channels are
    written left to right along ``head_dim``):

    * temporal active slots  (high freqs)  -- rotated by ``t * freq_t``
    * agent slots            (low temporal) -- rotated by simplex phase
                                                ``alpha * s_{perm(p)}``
    * h slots                              -- rotated by ``h * freq_h``
    * w slots                              -- rotated by ``w * freq_w``

    The returned freq tensor layout matches the existing 3D RoPE consumers
    (:func:`rope_apply_no_polar_op` / :func:`rope_apply_polar_op` in
    :mod:`wan_video_dit`): shape ``(2 * N, 1, head_dim/2)`` with cosine in
    the first ``N`` rows and sine in the second ``N`` rows. ``N`` indexes
    ``(t, p, h, w)`` in C-order with ``w`` innermost.

    Args:
        num_heads: Number of attention heads (kept for parity with the 3D
            RoPE class; unused in the freq computation itself).
        head_dim: Per-head dimension. Must be even.
        simplex_pool_size: Maximum number of distinct agent identities
            (``V`` in the paper). Defaults to ``4``, matching Gamma-World §D.
        agent_dim: Width of the agent rotary band, carved out of the temporal
            band. Must be even and ``<= temporal_dim``. Defaults to ``16``
            (i.e. ``agent_dim/2 = 8`` complex slots per vertex).
        alpha: Scale factor on the simplex phase (Equation 9 in the paper).
            Defaults to ``1.0``.
        end_t: Maximum supported temporal length for the precomputed table.
        end_hw: Maximum supported spatial length for the precomputed tables.
        theta: RoPE base. Defaults to the standard ``10000.0``.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        simplex_pool_size: int = 4,
        agent_dim: int = 16,
        alpha: float = 1.0,
        end_t: int = 1024,
        end_hw: int = 1024,
        theta: float = 10000.0,
        polar_output: bool = False,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        if agent_dim % 2 != 0:
            raise ValueError(f"agent_dim must be even, got {agent_dim}")
        if simplex_pool_size < 2:
            raise ValueError(f"simplex_pool_size must be >= 2, got {simplex_pool_size}")
        if agent_dim // 2 < simplex_pool_size:
            raise ValueError(
                f"agent_dim/2 ({agent_dim // 2}) must be >= simplex_pool_size "
                f"({simplex_pool_size}) for the zero-padded simplex construction"
            )

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.simplex_pool_size = simplex_pool_size
        self.agent_dim = agent_dim
        self.alpha = alpha
        self.polar_output = polar_output

        # 3D RoPE channel split, matching `precompute_freqs_cis_3d` in
        # :class:`RotaryPositionEmbeddingNoPolarOp` so spatial slots line up
        # exactly with the pretrained checkpoint.
        d_h = head_dim // 3
        d_w = head_dim // 3
        d_t_full = head_dim - d_h - d_w
        if agent_dim > d_t_full:
            raise ValueError(
                f"agent_dim ({agent_dim}) exceeds temporal band size "
                f"({d_t_full}); pick a smaller agent_dim or a larger head_dim."
            )

        self.d_t_full = d_t_full
        self.d_t_active = d_t_full - agent_dim  # post-ReRoPE temporal width
        self.d_h = d_h
        self.d_w = d_w

        # Per-axis frequency tables. The temporal table covers the full
        # ``d_t_full`` width using the *original* schedule; we'll select only
        # the high-freq prefix at runtime so the active temporal slots match
        # the 3D layout bit-for-bit.
        t_cos, t_sin = self._precompute_1d(d_t_full, end_t, theta)
        h_cos, h_sin = self._precompute_1d(d_h, end_hw, theta)
        w_cos, w_sin = self._precompute_1d(d_w, end_hw, theta)
        self.register_buffer("t_cos", t_cos, persistent=False)
        self.register_buffer("t_sin", t_sin, persistent=False)
        self.register_buffer("h_cos", h_cos, persistent=False)
        self.register_buffer("h_sin", h_sin, persistent=False)
        self.register_buffer("w_cos", w_cos, persistent=False)
        self.register_buffer("w_sin", w_sin, persistent=False)

        # Simplex phases per vertex, shape [V, agent_dim/2].
        simplex_phase = alpha * build_simplex_vertices(simplex_pool_size, agent_dim // 2)
        self.register_buffer("simplex_cos", torch.cos(simplex_phase), persistent=False)
        self.register_buffer("simplex_sin", torch.sin(simplex_phase), persistent=False)

    @staticmethod
    def _precompute_1d(dim: int, end: int, theta: float):
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim)
        )
        angles = torch.outer(torch.arange(end).float(), inv_freq)  # [end, dim/2]
        return torch.cos(angles), torch.sin(angles)

    def forward(
        self,
        f: int,
        p: int,
        h: int,
        w: int,
        agent_perm: torch.Tensor | None = None,
        start_frame: int = 0,
    ) -> torch.Tensor:
        """Build the freqs tensor for a ``(P, F, H, W)`` token grid.

        The output layout is **agent-major**, matching the Gamma-World
        "PTL agent tokens" convention (§3.3): each agent's ``(F, H, W)``
        token block is contiguous in the flattened sequence, so the model
        can sequence-concatenate agents as
        ``[agent_0_tokens, agent_1_tokens, ...]``.

        Args:
            f: Number of temporal positions.
            p: Number of active agents at runtime. Must be ``<= simplex_pool_size``.
            h: Number of vertical spatial positions.
            w: Number of horizontal spatial positions.
            agent_perm: Optional ``LongTensor[p]`` mapping each runtime agent
                slot to a simplex vertex index (``0 <= v < simplex_pool_size``).
                Defaults to the identity permutation ``[0, 1, ..., p-1]``.
            start_frame: Temporal index of the first frame in this call.
                Used for multi-call streaming inference so the new chunk's
                temporal RoPE positions continue the cached sequence.

        Returns:
            If ``polar_output`` is ``False`` (default): a
            ``(2 * P * F * H * W, 1, head_dim/2)`` float tensor with cos in
            rows ``[0, N)`` and sin in rows ``[N, 2N)``, matching the layout
            of :func:`rope_apply_no_polar_op`.

            If ``polar_output`` is ``True``: a
            ``(P * F * H * W, 1, head_dim/2)`` complex tensor where each
            element is ``cos + i*sin``, matching the layout of
            :func:`rope_apply_polar_op`.
        """
        if p > self.simplex_pool_size:
            raise ValueError(
                f"Active agents p={p} exceeds simplex_pool_size="
                f"{self.simplex_pool_size}"
            )

        device = self.t_cos.device
        if agent_perm is None:
            agent_perm = torch.arange(p, device=device)
        else:
            agent_perm = agent_perm.to(device=device, dtype=torch.long)
            if agent_perm.shape != (p,):
                raise ValueError(
                    f"agent_perm must have shape ({p},), got {tuple(agent_perm.shape)}"
                )
            if (agent_perm < 0).any() or (agent_perm >= self.simplex_pool_size).any():
                raise ValueError(
                    f"agent_perm contains out-of-range vertex index; valid "
                    f"range is [0, {self.simplex_pool_size})"
                )

        d_t_full_half = self.d_t_full // 2
        d_t_active_half = self.d_t_active // 2

        # Temporal slots: shape [F, d_t_full/2], offset by ``start_frame``
        # so streaming inference can continue a cached sequence.
        t_cos = self.t_cos[start_frame : start_frame + f]
        t_sin = self.t_sin[start_frame : start_frame + f]
        # Simplex slots for the active agents: shape [P, agent_dim/2]
        p_cos = self.simplex_cos[agent_perm]
        p_sin = self.simplex_sin[agent_perm]

        # Build the joint (F, P) temporal+agent band: high-freq slots use the
        # temporal phase (independent of p); low-freq slots use the simplex
        # phase (independent of t).
        upper_cos = t_cos[:, :d_t_active_half].unsqueeze(1).expand(f, p, d_t_active_half)
        upper_sin = t_sin[:, :d_t_active_half].unsqueeze(1).expand(f, p, d_t_active_half)
        lower_cos = p_cos.unsqueeze(0).expand(f, p, self.agent_dim // 2)
        lower_sin = p_sin.unsqueeze(0).expand(f, p, self.agent_dim // 2)
        tp_cos = torch.cat([upper_cos, lower_cos], dim=-1)  # [F, P, d_t_full/2]
        tp_sin = torch.cat([upper_sin, lower_sin], dim=-1)

        # Spatial bands.
        h_cos = self.h_cos[:h]  # [H, d_h/2]
        h_sin = self.h_sin[:h]
        w_cos = self.w_cos[:w]  # [W, d_w/2]
        w_sin = self.w_sin[:w]

        # Build freqs in (P, F, H, W) C-order so each agent's (F, H, W)
        # token block is contiguous in the flattened sequence. The temporal+
        # agent band ``tp_*`` is indexed (F, P, ...) above; transpose to
        # (P, F, ...) before broadcasting against spatial.
        # tp_*: [F, P, d_t_full/2] -> [P, F, d_t_full/2]
        tp_cos = tp_cos.transpose(0, 1).contiguous()
        tp_sin = tp_sin.transpose(0, 1).contiguous()

        # Broadcast everything to [P, F, H, W, head_dim/2] then flatten the
        # leading axes. W is innermost (C-order).
        freqs_cos = torch.cat(
            [
                tp_cos.view(p, f, 1, 1, -1).expand(p, f, h, w, -1),
                h_cos.view(1, 1, h, 1, -1).expand(p, f, h, w, -1),
                w_cos.view(1, 1, 1, w, -1).expand(p, f, h, w, -1),
            ],
            dim=-1,
        ).reshape(p * f * h * w, 1, -1)
        freqs_sin = torch.cat(
            [
                tp_sin.view(p, f, 1, 1, -1).expand(p, f, h, w, -1),
                h_sin.view(1, 1, h, 1, -1).expand(p, f, h, w, -1),
                w_sin.view(1, 1, 1, w, -1).expand(p, f, h, w, -1),
            ],
            dim=-1,
        ).reshape(p * f * h * w, 1, -1)

        if self.polar_output:
            # Pack cos / sin into a single complex tensor with shape
            # [P*F*H*W, 1, head_dim/2] -- matches rope_apply_polar_op.
            # ``torch.complex`` only supports Half / Float / Double, so we
            # promote bfloat16 buffers (which appear when the surrounding
            # model is cast to bf16) up to float32 here. Downstream
            # ``rope_apply_polar_op`` converts the *input* tensor to float64
            # before multiplying with these freqs, so this upcast does not
            # change end-to-end precision.
            if freqs_cos.dtype == torch.bfloat16:
                freqs_cos = freqs_cos.float()
                freqs_sin = freqs_sin.float()
            return torch.complex(freqs_cos, freqs_sin)
        return torch.cat([freqs_cos, freqs_sin], dim=0)

    def hub_freqs(self, f: int, k_hub: int, start_frame: int = 0) -> torch.Tensor:
        """Build RoPE freqs for ``f * k_hub`` hub tokens (Gamma-World §3.3).

        Hub tokens share the temporal phase of their associated frame and
        use identity rotation on the agent, height and width bands -- this
        keeps them temporally aligned while remaining neutral to agent
        identity and spatial position. The output layout is C-order
        ``(F, K)``: per frame, ``K`` hub tokens.

        Args:
            f: Number of latent frames.
            k_hub: Number of hub tokens per frame.
            start_frame: Temporal index of the first frame in this call
                (matches :meth:`forward`).

        Returns:
            Same layout as :meth:`forward` (real-stacked or complex,
            depending on ``polar_output``), with sequence length ``f*k_hub``.
        """
        d_t_full_half = self.d_t_full // 2
        agent_half = self.agent_dim // 2
        d_h_half = self.d_h // 2
        d_w_half = self.d_w // 2

        # Temporal slots: same as the agent path's high-frequency block,
        # then the agent-band slots are pinned to identity (cos=1, sin=0).
        t_cos = self.t_cos[start_frame : start_frame + f]  # [F, d_t_full/2]
        t_sin = self.t_sin[start_frame : start_frame + f]
        device = t_cos.device
        ident_cos = torch.ones(f, agent_half, device=device, dtype=t_cos.dtype)
        ident_sin = torch.zeros(f, agent_half, device=device, dtype=t_sin.dtype)

        # Splice: keep the upper d_t_active/2 temporal slots, replace the
        # lower agent_dim/2 slots with identity rotation.
        upper_cos = t_cos[:, : self.d_t_active // 2]
        upper_sin = t_sin[:, : self.d_t_active // 2]
        tp_cos = torch.cat([upper_cos, ident_cos], dim=-1)  # [F, d_t_full/2]
        tp_sin = torch.cat([upper_sin, ident_sin], dim=-1)

        # Spatial bands: identity rotation everywhere (hub tokens have no
        # spatial position).
        h_cos = torch.ones(d_h_half, device=device, dtype=t_cos.dtype)
        h_sin = torch.zeros(d_h_half, device=device, dtype=t_sin.dtype)
        w_cos = torch.ones(d_w_half, device=device, dtype=t_cos.dtype)
        w_sin = torch.zeros(d_w_half, device=device, dtype=t_sin.dtype)

        # Per-frame row: [t_band (d_t_full/2) | h_id (d_h/2) | w_id (d_w/2)].
        # Broadcast over k_hub copies per frame and over the head_dim/2 cat.
        per_frame_cos = torch.cat(
            [tp_cos, h_cos.expand(f, d_h_half), w_cos.expand(f, d_w_half)], dim=-1
        )  # [F, head_dim/2]
        per_frame_sin = torch.cat(
            [tp_sin, h_sin.expand(f, d_h_half), w_sin.expand(f, d_w_half)], dim=-1
        )

        freqs_cos = (
            per_frame_cos.unsqueeze(1).expand(f, k_hub, -1).reshape(f * k_hub, 1, -1)
        )
        freqs_sin = (
            per_frame_sin.unsqueeze(1).expand(f, k_hub, -1).reshape(f * k_hub, 1, -1)
        )

        if self.polar_output:
            if freqs_cos.dtype == torch.bfloat16:
                freqs_cos = freqs_cos.float()
                freqs_sin = freqs_sin.float()
            return torch.complex(freqs_cos, freqs_sin)
        return torch.cat([freqs_cos, freqs_sin], dim=0)

    def post_initialize(self):
        """Move precomputed buffers to CUDA, matching the existing RoPE API."""
        device = torch.device("cuda")
        for name in (
            "t_cos",
            "t_sin",
            "h_cos",
            "h_sin",
            "w_cos",
            "w_sin",
            "simplex_cos",
            "simplex_sin",
        ):
            setattr(self, name, getattr(self, name).to(device))
