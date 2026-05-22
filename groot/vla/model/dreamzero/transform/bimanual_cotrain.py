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
``P`` axis on these fields:

    state         [T_s, max_state_dim]       -> [P, T_s, dim_per_agent]
    state_mask    [T_s, max_state_dim]       -> [P, T_s, dim_per_agent]
    action        [T_a, max_action_dim]      -> [P, T_a, dim_per_agent]
    action_mask   [T_a, max_action_dim]      -> [P, T_a, dim_per_agent]
    images        [V, T, C, H, W]            -> [P, V_per_agent, T, C, H, W]

Plus a scalar ``num_agents`` key. All other DreamTransform outputs
(language tokens, embodiment_id, has_real_action, ...) pass through
unchanged.

This is the data-side hook for full multi-agent training; the
``WANPolicyHead.forward`` needs to detect the P axis and route to
:meth:`CausalWanModel._forward_train_multi_agent`. See PR 9 for that.
"""

from typing import Any

import numpy as np
from pydantic import Field

from groot.vla.model.dreamzero.transform.dreamzero_cotrain import DreamTransform


class BimanualDreamTransform(DreamTransform):
    """``DreamTransform`` that emits per-agent (P axis) tensors."""

    agent_video_views: list[list[int]] = Field(
        ...,
        description=(
            "Per-agent indices into the post-concat ``V`` axis. "
            "e.g. ``[[0, 1], [0, 2]]`` for YAM = (top-shared, left-wrist) "
            "and (top-shared, right-wrist)."
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

    def _split_video(self, images):
        """``[V, T, C, H, W]`` -> ``[P, V_per_agent, T, C, H, W]``."""
        per_agent = [images[idxs] for idxs in self.agent_video_views]
        return self._stack_agents(per_agent)

    def _split_dense(self, tensor, dims):
        """``[T, D]`` -> ``[P, T, D_per_agent]`` (also works for masks)."""
        per_agent = [tensor[:, a:b] for (a, b) in dims]
        return self._stack_agents(per_agent)

    def apply_single(self, data: dict) -> dict:
        out = super().apply_single(data)
        self._validate_groups()

        # State / action.
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

        # Images (need to extract from VLM output then split). The parent
        # ``_apply_vlm_processing`` flattens to ``(t v) h w c``; we redo
        # the per-agent split on the *pre-flatten* layout produced by
        # ``_prepare_video``.
        if "images" in out:
            imgs = out["images"]
            if imgs.ndim == 5:
                # [V, T, C, H, W] -> [P, V_per_agent, T, C, H, W]
                out["images"] = self._split_video(imgs)
            elif imgs.ndim == 4:
                # VLM-flattened [(t v), h, w, c]: keep as-is, just record P.
                # Downstream needs to re-shape; for now we leave images flat
                # and rely on the multi-agent model's own per-agent encode.
                pass

        out["num_agents"] = np.int64(self.num_agents)
        return out
