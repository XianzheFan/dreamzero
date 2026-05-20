from typing import Any

import numpy as np
import torch
from pydantic import Field

from groot.vla.data.transform.base import InvertibleModalityTransform


class MultiAgentStackTransform(InvertibleModalityTransform):
    """Insert an explicit agent axis ``P`` after :class:`ConcatTransform`.

    The transform expects the data dict to already contain the concatenated
    ``video`` / ``state`` / ``action`` tensors produced by
    :class:`ConcatTransform`. It rearranges those tensors so that downstream
    model code can assume a uniform ``[P, T, ...]`` per-sample contract.

    Two modes:

    * **Single agent (default)**: no grouping fields are configured. The
      transform inserts a singleton ``P=1`` leading axis on each of
      ``video``, ``state`` and ``action``. This is a no-op augmentation of
      shape; numerical values are unchanged.

    * **Multi-agent**: configure ``agent_video_views``,
      ``agent_state_dims`` and/or ``agent_action_dims`` to split the
      concatenated tensors into per-agent slices and stack them along a
      new leading ``P`` axis. Slices must be disjoint within a tensor and
      the same length across agents (symmetric layout); use a separate
      role token to distinguish asymmetric roles.

    Post-transform shapes (per sample, before DataLoader collation):

    ``video``  : ``[P, T, V_per_agent, H, W, C]``
    ``state``  : ``[P, T, D_state_per_agent]``
    ``action`` : ``[P, T, D_action_per_agent]``
    """

    apply_to: list[str] = Field(
        default_factory=list,
        description="Unused; transform always operates on the concat keys.",
    )

    agent_video_views: list[list[int]] | None = Field(
        default=None,
        description=(
            "Per-agent indices into the post-concat ``V`` axis. "
            "e.g. ``[[0, 1], [2, 3]]`` for two agents with two views each. "
            "``None`` triggers single-agent (P=1) unsqueeze."
        ),
    )

    agent_state_dims: list[tuple[int, int]] | None = Field(
        default=None,
        description=(
            "Per-agent ``(start, end)`` slices into the post-concat state "
            "dim. ``None`` triggers single-agent (P=1) unsqueeze."
        ),
    )

    agent_action_dims: list[tuple[int, int]] | None = Field(
        default=None,
        description=(
            "Per-agent ``(start, end)`` slices into the post-concat action "
            "dim. ``None`` triggers single-agent (P=1) unsqueeze."
        ),
    )

    @property
    def num_agents(self) -> int:
        for groups in (
            self.agent_video_views,
            self.agent_state_dims,
            self.agent_action_dims,
        ):
            if groups is not None:
                return len(groups)
        return 1

    def _validate(self) -> None:
        configured = [
            g
            for g in (
                self.agent_video_views,
                self.agent_state_dims,
                self.agent_action_dims,
            )
            if g is not None
        ]
        if not configured:
            return
        P = len(configured[0])
        for g in configured:
            assert len(g) == P, (
                "All agent_*_dims / agent_video_views must declare the same "
                f"number of agents; got lengths {[len(x) for x in configured]}"
            )
        if self.agent_video_views is not None:
            widths = {len(idxs) for idxs in self.agent_video_views}
            assert len(widths) == 1, (
                "Each agent must have the same number of views; "
                f"got per-agent view counts {[len(i) for i in self.agent_video_views]}"
            )
        if self.agent_state_dims is not None:
            widths = {b - a for (a, b) in self.agent_state_dims}
            assert len(widths) == 1, (
                "Each agent must have the same state width; "
                f"got per-agent widths {[b - a for (a, b) in self.agent_state_dims]}"
            )
        if self.agent_action_dims is not None:
            widths = {b - a for (a, b) in self.agent_action_dims}
            assert len(widths) == 1, (
                "Each agent must have the same action width; "
                f"got per-agent widths {[b - a for (a, b) in self.agent_action_dims]}"
            )

    @staticmethod
    def _expand_axis0(x):
        if isinstance(x, torch.Tensor):
            return x.unsqueeze(0)
        return np.expand_dims(x, axis=0)

    @staticmethod
    def _stack_axis0(parts):
        if isinstance(parts[0], torch.Tensor):
            return torch.stack(parts, dim=0)
        return np.stack(parts, axis=0)

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        self._validate()
        P = self.num_agents

        if "video" in data:
            video = data["video"]  # [T, V, H, W, C]
            assert video.ndim == 5, (
                f"MultiAgentStackTransform expects post-concat video of shape "
                f"[T, V, H, W, C]; got {tuple(video.shape)}"
            )
            if self.agent_video_views is None:
                data["video"] = self._expand_axis0(video)  # [1, T, V, H, W, C]
            else:
                per_agent = [video[:, idxs, :, :, :] for idxs in self.agent_video_views]
                data["video"] = self._stack_axis0(per_agent)  # [P, T, V_p, H, W, C]

        if "state" in data:
            state = data["state"]  # [T, D]
            assert state.ndim == 2, (
                f"MultiAgentStackTransform expects post-concat state of shape "
                f"[T, D]; got {tuple(state.shape)}"
            )
            if self.agent_state_dims is None:
                data["state"] = self._expand_axis0(state)  # [1, T, D]
            else:
                per_agent = [state[:, a:b] for (a, b) in self.agent_state_dims]
                data["state"] = self._stack_axis0(per_agent)  # [P, T, D_p]

        if "action" in data:
            action = data["action"]  # [T, D]
            assert action.ndim == 2, (
                f"MultiAgentStackTransform expects post-concat action of shape "
                f"[T, D]; got {tuple(action.shape)}"
            )
            if self.agent_action_dims is None:
                data["action"] = self._expand_axis0(action)  # [1, T, D]
            else:
                per_agent = [action[:, a:b] for (a, b) in self.agent_action_dims]
                data["action"] = self._stack_axis0(per_agent)  # [P, T, D_p]

        data["num_agents"] = P
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Collapse the ``P`` axis back to the pre-transform layout.

        For single-agent mode (P=1) this is the inverse of ``apply``. For
        multi-agent mode it concatenates per-agent slices back into the
        original layout; this is mainly used for debugging / un-normalizing
        outputs, not for the training rollout path.
        """
        data.pop("num_agents", None)

        def _collapse(x, joiner_axis: int):
            if x.shape[0] == 1:
                if isinstance(x, torch.Tensor):
                    return x.squeeze(0)
                return np.squeeze(x, axis=0)
            parts = [x[p] for p in range(x.shape[0])]
            if isinstance(x, torch.Tensor):
                return torch.cat(parts, dim=joiner_axis - 1)
            return np.concatenate(parts, axis=joiner_axis - 1)

        if "video" in data:
            data["video"] = _collapse(data["video"], joiner_axis=2)  # cat on V
        if "state" in data:
            data["state"] = _collapse(data["state"], joiner_axis=2)  # cat on D
        if "action" in data:
            data["action"] = _collapse(data["action"], joiner_axis=2)  # cat on D
        return data
