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
            "e.g. ``[[0, 1], [0, 2]]`` for YAM = (top-shared, left-wrist) "
            "and (top-shared, right-wrist). At most 4 views per agent."
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

    def _split_dense(self, tensor, dims):
        """``[T, D]`` -> ``[P, T, D_per_agent]`` (also works for masks)."""
        per_agent = [tensor[:, a:b] for (a, b) in dims]
        return self._stack_agents(per_agent)

    def _prepare_video(self, data: dict):
        """Multi-agent override: per-agent V-tile, no global concat.

        Returns ``[P, T, C, 2H, 2W]`` -- per-agent 2x2-tiled images
        with a new leading P axis. Each agent's slot picks ``V_per_agent``
        views out of the raw ``V`` axis (via ``agent_video_views``) and
        tiles them into one image. P replaces the V dim that the parent
        would have collapsed via ``_apply_vlm_processing``.
        """
        self._validate_groups()
        # Raw layout: [T, V, H, W, C] (from LeRobot loader) -> [V, T, C, H, W].
        images = rearrange(data["video"], "t v h w c -> v t c h w")
        per_agent = []
        for view_idxs in self.agent_video_views:
            agent_views = images[list(view_idxs)]  # [V_per_agent, T, C, H, W]
            per_agent.append(self._tile_views_2x2(agent_views))
        # [P, T, C, 2H, 2W]
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

        out["num_agents"] = np.int64(self.num_agents)
        return out
