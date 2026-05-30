"""Multi-agent ("bimanual") model_specific_transform.

Wraps :class:`DreamTransform` and adds an explicit ``P`` (agent) axis to
``state``, ``action``, their masks and ``images`` so the downstream
action head can route through the PR 1-8 multi-agent forward path.

YAM example -- left arm is agent 0, right arm is agent 1, top camera is
shared, each agent gets its own wrist::

    agent_video_views    = [[0, 1], [0, 2]]   # [top, left_wrist] / [top, right_wrist]
    agent_state_dims     = [(0, 7), (7, 14)]
    agent_action_dims    = [(0, 7), (7, 14)]

The transform output extends DreamTransform's keys with a leading
``P`` axis on these fields::

    state         [T_s, max_state_dim]   -> [P, T_s, dim_per_agent]
    state_mask    [T_s, max_state_dim]   -> [P, T_s, dim_per_agent]
    action        [T_a, max_action_dim]  -> [P, T_a, dim_per_agent]
    action_mask   [T_a, max_action_dim]  -> [P, T_a, dim_per_agent]
    images        [T, V, H, W, C]        -> [P, T, 2H, 2W, C]

Each agent's images carry that agent's views tiled into a single 2x2
grid (TL=v0, TR=v1, BL=v2, BR=v3, zeros for missing slots), matching
the single-agent ``DreamTransform._prepare_video`` layout. This is the
shape contract ``WANPolicyHead._forward_multi_agent`` expects: a single
image per agent, C-last, with a shared 2x2 grid scale so downstream
``target_video_height/width`` heuristics stay calibrated.

Plus a scalar ``num_agents`` key. All other DreamTransform outputs
(language tokens, embodiment_id, has_real_action, ...) pass through
unchanged.
"""

from typing import Any

import numpy as np
from einops import rearrange
from pydantic import Field

from groot.vla.model.dreamzero.transform.dreamzero_cotrain import DreamTransform


class BimanualDreamTransform(DreamTransform):
    """``DreamTransform`` that emits per-agent (P axis) tensors."""

    agent_video_views: list[list[int]] = Field(
        ...,
        description=(
            "Per-agent indices into the raw ``V`` axis of ``data['video']``. "
            "Two layouts are supported:\n\n"
            "(1) **Old / duplicated-global**: include the shared scene "
            "camera in every agent (e.g. ``[[0, 1], [0, 2]]`` -- top is "
            "duplicated). The transform 2x2-tiles each agent's views into "
            "a single image. ``global_views`` must be left ``None``.\n\n"
            "(2) **Shared-global**: pair with ``global_views`` to factor "
            "the shared scene camera out (e.g. ``global_views=[0]`` + "
            "``agent_video_views=[[1], [2]]``). The transform then emits "
            "per-agent wrist tiles **and** a separate ``video_global`` "
            "stream the downstream model treats as clean conditioning. "
            "Up to 4 views per agent in either layout."
        ),
    )
    global_views: list[int] | None = Field(
        default=None,
        description=(
            "Optional indices into the raw ``V`` axis for the **shared** "
            "scene cameras (encoded once, not per-agent). When set, those "
            "views are emitted under the ``video_global`` key in the "
            "transform output and the per-agent ``video`` key contains "
            "only the views listed in ``agent_video_views`` (no longer a "
            "tile of (global + wrist)). When ``None`` (default) the "
            "transform stays backward-compatible with the duplicated-"
            "global layout."
        ),
    )
    global_condition_mode: str = Field(
        default="full",
        description=(
            "How to build ``video_global`` when ``global_views`` is set. "
            "``full`` preserves the historical behavior and emits the "
            "full sampled global video window. ``current_repeat`` emits "
            "only the current global observation (delta index 0) repeated "
            "across the window, preventing future-frame leakage into the "
            "clean global-conditioning stream."
        ),
    )
    agent_state_dims: list[tuple[int, int]] = Field(
        ...,
        description=(
            "Per-agent ``(start, end)`` slices into the post-concat state "
            "dim. e.g. ``[(0, 7), (7, 14)]`` for YAM left/right arm."
        ),
    )
    agent_action_dims: list[tuple[int, int]] = Field(
        ...,
        description="Per-agent action dim slices (same shape rules as state).",
    )

    @property
    def num_agents(self) -> int:
        return len(self.agent_video_views)

    def _validate_groups(self) -> None:
        P = len(self.agent_video_views)
        assert len(self.agent_state_dims) == P, (
            f"agent_state_dims has {len(self.agent_state_dims)} agents, "
            f"expected {P}"
        )
        assert len(self.agent_action_dims) == P, (
            f"agent_action_dims has {len(self.agent_action_dims)} agents, "
            f"expected {P}"
        )
        view_widths = {len(v) for v in self.agent_video_views}
        assert len(view_widths) == 1, (
            f"All agents must have the same number of views; got "
            f"{[len(v) for v in self.agent_video_views]}"
        )
        max_views_per_agent = max(view_widths)
        assert max_views_per_agent <= 4, (
            f"At most 4 views per agent (2x2 tile slots); "
            f"got {max_views_per_agent}"
        )
        state_widths = {b - a for (a, b) in self.agent_state_dims}
        assert len(state_widths) == 1, (
            f"All agents must have the same state width; got "
            f"{[b - a for (a, b) in self.agent_state_dims]}"
        )
        action_widths = {b - a for (a, b) in self.agent_action_dims}
        assert len(action_widths) == 1, (
            f"All agents must have the same action width; got "
            f"{[b - a for (a, b) in self.agent_action_dims]}"
        )
        assert self.global_condition_mode in ("full", "current_repeat"), (
            "global_condition_mode must be 'full' or 'current_repeat'; "
            f"got {self.global_condition_mode!r}"
        )

    @staticmethod
    def _stack_agents(parts):
        """Stack along a new leading axis. Works for numpy or torch."""
        import torch
        if isinstance(parts[0], np.ndarray):
            return np.stack(parts, axis=0)
        if isinstance(parts[0], torch.Tensor):
            return torch.stack(parts, dim=0)
        raise TypeError(f"Cannot stack type {type(parts[0]).__name__}")

    @staticmethod
    def _tile_views_2x2(views: np.ndarray) -> np.ndarray:
        """Tile up to 4 per-agent views into a 2x2 grid.

        ``views``: ``[V_per_agent, T, C, H, W]`` -> ``[T, C, 2H, 2W]``.
        Slot assignment (mirrors single-agent ``DreamTransform._prepare_video``):
        TL=v0, BL=v1, TR=v2, BR=v3. Missing slots are zero-filled so the
        tile shape is invariant to ``V_per_agent``.
        """
        v, t, c, h, w = views.shape
        out = np.zeros((t, c, 2 * h, 2 * w), dtype=views.dtype)
        if v >= 1:
            out[:, :, :h, :w] = views[0]
        if v >= 2:
            out[:, :, h:, :w] = views[1]
        if v >= 3:
            out[:, :, :h, w:] = views[2]
        if v >= 4:
            out[:, :, h:, w:] = views[3]
        return out

    @staticmethod
    def _maybe_tile(views: np.ndarray) -> np.ndarray:
        """Return ``views[0]`` as-is when there is only one view, else
        delegate to :meth:`_tile_views_2x2`. The single-view shortcut
        avoids the 75% zero-padding cost of running 2x2 on a single
        view (which is the typical wrist-only / global-only layout in
        shared-global mode).
        """
        if views.shape[0] == 1:
            return views[0]  # [T, C, H, W]
        return BimanualDreamTransform._tile_views_2x2(views)

    def _split_dense(self, tensor, dims):
        """``[T, D]`` -> ``[P, T, D_per_agent]`` (also works for masks)."""
        per_agent = [tensor[:, a:b] for (a, b) in dims]
        return self._stack_agents(per_agent)

    def _prepare_global_video(self, data: dict) -> np.ndarray:
        """Build the shared global conditioning stream.

        Training video samples use delta indices starting at 0, so the
        current observation is frame 0 and later frames are future. In
        ``current_repeat`` mode we repeat that first frame over the whole
        global-conditioning window to keep train/eval causal while still
        matching the latent temporal shape expected by the DiT body.
        """
        assert self.global_views is not None
        raw = rearrange(data["video"], "t v h w c -> v t c h w")
        gv = raw[list(self.global_views)]                  # [V_g, T, C, H, W]
        global_video = self._maybe_tile(gv)                # [T, C, H, W] or [T, C, 2H, 2W]
        if self.global_condition_mode == "current_repeat":
            global_video = np.repeat(global_video[0:1], global_video.shape[0], axis=0)
        return rearrange(global_video, "t c h w -> t h w c").astype(np.uint8)

    def _prepare_video(self, data: dict):
        """Multi-agent override: per-agent video assembly.

        Two output shapes, gated by ``self.global_views``:

        * ``global_views is None`` (legacy 2x2-tile layout): each agent's
          views are tiled into a single ``[T, C, 2H, 2W]`` image and
          stacked along a new ``P`` axis -> ``[P, T, C, 2H, 2W]``. The
          shared scene camera is duplicated into every agent's tile.
        * ``global_views is not None`` (shared-global layout): the
          shared scene camera is factored out (see
          :meth:`apply_single`) and each agent gets ONLY its own wrist
          views. With the typical single wrist per agent we skip the
          tile entirely and return ``[P, T, C, H, W]``; with >=2
          per-agent views the same 2x2 tile rule applies.
        """
        self._validate_groups()
        # Raw layout: [T, V, H, W, C] (from LeRobot loader) -> [V, T, C, H, W].
        images = rearrange(data["video"], "t v h w c -> v t c h w")
        per_agent = []
        for view_idxs in self.agent_video_views:
            agent_views = images[list(view_idxs)]  # [V_per_agent, T, C, H, W]
            if self.global_views is None:
                # Legacy: always go through the 2x2 tile so per-agent
                # spatial shape stays calibrated against the pretrained
                # 2H x 2W token grid. This is the codepath the existing
                # checkpoints were trained against; do NOT change it.
                per_agent.append(self._tile_views_2x2(agent_views))
            else:
                # Shared-global: only tile when there's >=2 per-agent
                # views (the rare multi-wrist case); single-view wrist
                # stays at the camera's native H x W.
                per_agent.append(self._maybe_tile(agent_views))
        # [P, T, C, H_per, W_per] -- shape depends on the branch above.
        return np.stack(per_agent, axis=0)

    def _apply_vlm_processing(self, batch: dict) -> dict:
        """Multi-agent override: preserve the P axis (don't collapse with T).

        Parent's version does ``rearrange("v t c h w -> (t v) h w c")``
        which would fold the agent axis into time. We instead pass P
        through, emitting per-agent images as ``[P, T, H, W, C]``.
        """
        images = batch["images"]  # [P, T, C, H, W]
        np_images = rearrange(images, "p t c h w -> p t h w c")
        lang = batch.get("language")
        if isinstance(lang, (list, np.ndarray)):
            lang = lang[0]
        return {"images": np_images, "text": lang}

    def apply_single(self, data: dict) -> dict:
        out = super().apply_single(data)
        self._validate_groups()

        # State / action splits. Images already carry a P axis thanks to
        # the ``_prepare_video`` + ``_apply_vlm_processing`` overrides --
        # no further split here.
        if "state" in out:
            out["state"] = self._split_dense(out["state"], self.agent_state_dims)
        if "state_mask" in out:
            out["state_mask"] = self._split_dense(
                out["state_mask"], self.agent_state_dims
            )
        if "action" in out:
            out["action"] = self._split_dense(out["action"], self.agent_action_dims)
        if "action_mask" in out:
            out["action_mask"] = self._split_dense(
                out["action_mask"], self.agent_action_dims
            )
        if "lapa_action" in out:
            out["lapa_action"] = self._split_dense(
                out["lapa_action"], self.agent_action_dims
            )
        if "lapa_action_mask" in out:
            out["lapa_action_mask"] = self._split_dense(
                out["lapa_action_mask"], self.agent_action_dims
            )

        # Shared-global stream: factor the scene camera out of the
        # per-agent video and emit it once under ``video_global``. The
        # downstream multi-agent action head detects this key and routes
        # it through a single VAE encode + a P-less token block in the
        # DiT (see :meth:`_forward_multi_agent_body`). When global_views
        # is None we stay in the legacy duplicated-global layout and
        # this key is absent.
        if self.global_views is not None:
            assert all(
                v not in g
                for g in self.agent_video_views
                for v in self.global_views
            ), (
                "global_views must be disjoint from every agent's "
                "agent_video_views in shared-global layout; otherwise "
                "the shared camera ends up duplicated again."
            )
            # C-last, uint8 -- mirrors the per-agent ``images`` layout
            # so the action head can run them through the same VAE
            # normalize/encode helper.
            out["video_global"] = self._prepare_global_video(data)

        out["num_agents"] = np.int64(self.num_agents)
        return out
