"""WebSocket inference server for the multi-agent (bimanual) DreamZero VLA.

Splits the eval architecture in two so the **policy** and the
**simulator** can live in different Python / glibc environments:

* This server: runs in the dreamzero conda env (Python 3.11, glibc
  2.28+), holds the LoRA-fine-tuned VLA + its transforms in process,
  serves action chunks over a WebSocket.
* RoboTwin eval client (``policy/DreamZero/deploy_policy.py``): runs in
  the egl container (Ubuntu 18.04, glibc 2.27, Python 3.9), so the
  only heavy deps it needs are ``websockets``, ``msgpack``, ``numpy``.

Wire protocol (msgpack-numpy on each direction):

  Server -> client on connect (one frame)::
      {
        "num_agents": 2,
        "image_resolution": [H, W],
        "num_frames": 33,
        "action_horizon": 24,
        "action_dim": 16,
        "fps": 20,
      }

  Client -> server (``endpoint="reset"``)::
      {"endpoint": "reset", "session_id": str, "prompt": str}
      -> returns "reset successful"

  Client -> server (``endpoint="infer"``)::
      {
        "endpoint": "infer",
        "session_id": str,
        "qpos": np.ndarray [16] float32,
        "head_rgb": np.ndarray [H, W, 3] uint8,
        "left_rgb": np.ndarray [H, W, 3] uint8,
        "right_rgb": np.ndarray [H, W, 3] uint8,
        "prompt": str,  # optional override; reset's value used if absent
      }
      -> returns
      {
        "action_chunk": np.ndarray [T_a=24, 16] float32,
        # For checkpoints trained with DreamZero ``relative_action``, the
        # server adds the denormalized joint offsets back to the current
        # qpos and returns absolute qpos targets. Legacy checkpoints that
        # stored deltas directly still return delta-joint slots.
      }

Per-session video history is held server-side as a deque of
``num_frames`` length so the client only has to send a single new RGB
triple per inference call.

Usage::

    python -m eval_utils.bimanual_policy_server \\
        --ckpt-dir /lustre/.../checkpoints/robotwin_bimanual_smoke \\
        --ckpt-setting checkpoint-10 \\
        --port 5001
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import dataclasses
import gc
import json
import logging
import os
import sys
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Dict

import numpy as np
import websockets.asyncio.server
import websockets.frames


@dataclasses.dataclass
class BimanualServerConfig:
    num_agents: int = 2
    # Raw per-camera resolution the dataset / transform chain expects --
    # matches the LeRobot v2 mp4 dimensions written by
    # ``scripts/data/robofactory_to_lerobot_v2.py``. The chain handles its
    # own downstream Resize to the model's target.
    image_resolution: tuple[int, int] = (240, 320)   # (H, W)
    num_frames: int = 33
    action_horizon: int = 24
    action_dim: int = 16
    fps: int = 20
    return_action_debug: bool = False
    action_representation: str = "robotwin_delta"
    gripper_convention: str = "auto"
    gripper_close_value: float = 0.0
    gripper_open_value: float = 1.0
    shared_global_wrist_window_mode: str = "history-current-first"
    reset_causal_state_each_infer: bool = False
    video_pred_rollout_mode: str = "action"


_SHARED_GLOBAL_WRIST_WINDOW_MODES = (
    "repeat-current",
    "history-current-first",
    "history-chronological",
)

_VIDEO_PRED_ROLLOUT_MODES = ("action", "noncausal")
_MISSING = object()

_ACTION_HEAD_CONTROL_STATE_ATTRS = (
    "current_start_frame",
    "language",
    "kv_cache1",
    "kv_cache_neg",
    "crossattn_cache",
    "crossattn_cache_neg",
    "clip_feas",
    "ys",
    "_ma_cached_token_agent_id",
    "_ma_cached_token_agent_id_neg",
    "skip_countdown",
)


def _resolve_video_pred_rollout_mode(value: str | None) -> str:
    mode = str(value or "action").strip().lower()
    if mode not in _VIDEO_PRED_ROLLOUT_MODES:
        raise ValueError(
            "video_pred_rollout_mode must be one of "
            f"{_VIDEO_PRED_ROLLOUT_MODES}; got {value!r}"
        )
    return mode


def _make_packer():
    """Return (pack, unpack) callables backed by msgpack-numpy."""
    try:
        from openpi_client import msgpack_numpy as _mn  # type: ignore
        packer = _mn.Packer()
        return packer.pack, _mn.unpackb
    except Exception:
        import msgpack
        import msgpack_numpy
        msgpack_numpy.patch()
        return (
            lambda obj: msgpack.packb(obj, use_bin_type=True),
            lambda buf: msgpack.unpackb(buf, raw=False),
        )


def _configure_torch_dynamo_for_serving(torch_module) -> None:
    """Avoid default torch.compile recompile caps during long VLA serving."""
    dynamo = getattr(torch_module, "_dynamo", None)
    config = getattr(dynamo, "config", None)
    if config is None:
        return
    if hasattr(config, "cache_size_limit"):
        config.cache_size_limit = max(int(config.cache_size_limit), 1000)
    if hasattr(config, "recompile_limit"):
        config.recompile_limit = max(int(config.recompile_limit), 800)
    if hasattr(config, "accumulated_cache_size_limit"):
        config.accumulated_cache_size_limit = max(
            int(config.accumulated_cache_size_limit), 1000
        )
    if hasattr(config, "accumulated_recompile_limit"):
        config.accumulated_recompile_limit = max(
            int(config.accumulated_recompile_limit), 2000
        )


@dataclasses.dataclass
class DistributedServingContext:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    inference_parallel_size: int = 1
    device: str = "cpu"
    device_mesh: Any | None = None

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_leader(self) -> bool:
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def _resolve_inference_parallel_size(
    requested_size: int,
    world_size: int,
) -> int:
    """Resolve and validate the serving-time inference-parallel world size.

    The WAN action head currently implements two-way inference parallelism
    for CFG branches, not arbitrary tensor/FSDP sharding. Keep validation
    explicit so launching with 8 ranks fails fast instead of hanging in
    distributed collectives.
    """
    if requested_size < 0:
        raise ValueError(
            "inference_parallel_size must be >= 0; use 0 to infer from WORLD_SIZE"
        )
    resolved = world_size if requested_size == 0 else requested_size
    if resolved not in (1, 2):
        raise ValueError(
            "DreamZero websocket serving currently supports inference_parallel_size "
            f"1 or 2, got {resolved}. The existing WAN action head only supports "
            "two-way CFG inference parallelism; 8-way tensor/FSDP sharding needs "
            "a separate model-parallel implementation."
        )
    if world_size != resolved:
        raise ValueError(
            "Distributed serving launch mismatch: requested "
            f"inference_parallel_size={resolved}, but torch distributed WORLD_SIZE="
            f"{world_size}. Launch with torchrun --nproc_per_node={resolved}, or "
            "set --inference-parallel-size 1 for non-distributed serving."
        )
    return resolved


def _init_distributed_serving(requested_size: int) -> DistributedServingContext:
    import torch
    import torch.distributed as dist

    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", rank if world_size > 1 else 0)
    ip_size = _resolve_inference_parallel_size(requested_size, world_size)

    device = "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"

    if world_size == 1:
        return DistributedServingContext(
            rank=0,
            local_rank=local_rank,
            world_size=1,
            inference_parallel_size=ip_size,
            device=device,
            device_mesh=None,
        )

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this Python build")
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    from torch.distributed.device_mesh import init_device_mesh

    mesh_device = "cuda" if torch.cuda.is_available() else "cpu"
    device_mesh = init_device_mesh(
        mesh_device,
        (ip_size,),
        mesh_dim_names=("ip",),
    )
    return DistributedServingContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        inference_parallel_size=ip_size,
        device=device,
        device_mesh=device_mesh,
    )


SLICE_COMPATIBLE_PRETRAINED_KEYS = frozenset(
    {
        "action_head.model.action_decoder.layer2.W",
        "action_head.model.action_decoder.layer2.b",
        "action_head.model.action_encoder.W1.W",
        "action_head.model.patch_embedding.weight",
        "action_head.model.state_encoder.layer1.W",
    }
)

FINETUNE_PATCH_EMBEDDING_KEYS = frozenset(
    {
        "action_head.model.patch_embedding.weight",
    }
)


def _slice_copy_pretrained_tensor(key: str, ckpt_tensor, model_tensor):
    """Match training-time partial-load for known architecture deltas."""
    if key not in SLICE_COMPATIBLE_PRETRAINED_KEYS:
        return None
    if ckpt_tensor.ndim != model_tensor.ndim:
        return None

    adapted_tensor = model_tensor.detach().clone()
    copy_slices = tuple(
        slice(0, min(ckpt_size, model_size))
        for ckpt_size, model_size in zip(ckpt_tensor.shape, model_tensor.shape)
    )
    adapted_tensor[copy_slices].copy_(
        ckpt_tensor[copy_slices].to(
            device=adapted_tensor.device,
            dtype=adapted_tensor.dtype,
        )
    )
    return adapted_tensor


def _log_shape_mismatches(
    stage: str,
    sliced_mismatched: dict[str, tuple],
    dropped_mismatched: dict[str, tuple],
    *,
    dropped_reason: str,
) -> None:
    if sliced_mismatched:
        logging.info(
            "%s: slice-copied %d shape-mismatched tensor(s) from overlapping dimensions.",
            stage,
            len(sliced_mismatched),
        )
        for k, (ckpt_shape, model_shape) in list(sliced_mismatched.items())[:10]:
            logging.info(
                "%s: slice-copied %s ckpt=%s -> model=%s",
                stage,
                k,
                ckpt_shape,
                model_shape,
            )
    if dropped_mismatched:
        logging.info(
            "%s: dropped %d shape-mismatched tensor(s) (%s).",
            stage,
            len(dropped_mismatched),
            dropped_reason,
        )
        for k, (ckpt_shape, model_shape) in list(dropped_mismatched.items())[:10]:
            logging.info(
                "%s: dropped %s ckpt=%s -> model=%s",
                stage,
                k,
                ckpt_shape,
                model_shape,
            )


def _filter_shape_mismatches_for_load(
    sd: dict,
    model_state: dict,
    *,
    drop_mismatched_keys: frozenset[str] = frozenset(),
) -> tuple[dict, dict[str, tuple], dict[str, tuple]]:
    kept = {}
    dropped_mismatched: dict[str, tuple] = {}
    sliced_mismatched: dict[str, tuple] = {}
    for k, v in sd.items():
        ref = model_state.get(k)
        if ref is not None and tuple(ref.shape) != tuple(v.shape):
            if k in drop_mismatched_keys:
                dropped_mismatched[k] = (tuple(v.shape), tuple(ref.shape))
                continue
            adapted = _slice_copy_pretrained_tensor(k, v, ref)
            if adapted is not None:
                sliced_mismatched[k] = (tuple(v.shape), tuple(ref.shape))
                kept[k] = adapted
            else:
                dropped_mismatched[k] = (tuple(v.shape), tuple(ref.shape))
            continue
        kept[k] = v
    return kept, sliced_mismatched, dropped_mismatched


class BimanualPolicy:
    """Loads the LoRA-fine-tuned VLA + the bimanual_cotrain transform,
    exposes ``infer(obs)`` and ``reset(info)``.

    Mirrors ``policy/DreamZero/deploy_policy.py::DreamZeroBimanualPolicy``
    on the server side: encodes the obs into the P-axis tensors the model
    was trained on, then inverse-normalizes the predicted deltas back to
    physical units. Per-session rolling video window lives here so the
    client stays lightweight.
    """

    def __init__(
        self,
        ckpt_dir: Path,
        ckpt_setting: str,
        # Raw per-camera resolution the transform chain expects -- matches
        # the LeRobot v2 mp4 dimensions for RoboFactory bimanual. The
        # in-chain Resize handles downsampling to the model's target.
        image_h: int = 240,
        image_w: int = 320,
        model_image_h: int | None = None,
        model_image_w: int | None = None,
        num_frames: int = 33,
        action_horizon: int = 24,
        action_dim: int = 16,
        save_video_pred: bool = False,
        video_pred_dir: str | None = None,
        video_pred_rollout_mode: str | None = None,
        return_action_debug: bool = False,
        prompt_override: str | None = None,
        gripper_binarize_threshold: float | None = None,
        gripper_override: str = "none",
        gripper_close_after_infer: int = 0,
        gripper_close_value: float | None = None,
        gripper_force_open_until_infer: int | None = None,
        gripper_convention: str = "auto",
        shared_global_wrist_window_mode: str | None = None,
        reset_causal_state_each_infer: bool | None = None,
        device: str | None = None,
        device_mesh: Any | None = None,
    ):
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_setting = ckpt_setting
        self.image_h = image_h
        self.image_w = image_w
        self.model_image_h = model_image_h
        self.model_image_w = model_image_w
        self.num_frames = num_frames
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.save_video_pred = save_video_pred
        self.video_pred_dir = Path(video_pred_dir) if video_pred_dir else None
        if video_pred_rollout_mode is None:
            video_pred_rollout_mode = os.environ.get(
                "DREAMZERO_VIDEO_PRED_ROLLOUT_MODE",
                "action",
            )
        self.video_pred_rollout_mode = _resolve_video_pred_rollout_mode(
            video_pred_rollout_mode
        )
        self.return_action_debug = return_action_debug
        self.prompt_override = (
            prompt_override
            if prompt_override is not None
            else os.environ.get("DREAMZERO_PROMPT_OVERRIDE", "")
        ).strip()
        if gripper_binarize_threshold is None:
            gripper_threshold_env = os.environ.get(
                "DREAMZERO_GRIPPER_BINARIZE_THRESHOLD", ""
            ).strip()
            gripper_binarize_threshold = (
                float(gripper_threshold_env) if gripper_threshold_env else None
            )
        if gripper_binarize_threshold is not None and not (
            -1.0 <= gripper_binarize_threshold <= 1.0
        ):
            raise ValueError(
                "gripper_binarize_threshold must be in [-1, 1], got "
                f"{gripper_binarize_threshold}"
            )
        self.gripper_binarize_threshold = gripper_binarize_threshold
        self.gripper_convention = str(gripper_convention or "auto").strip().lower()
        if self.gripper_convention not in ("auto", "robotwin", "robofactory"):
            raise ValueError(
                "gripper_convention must be one of auto, robotwin, robofactory; "
                f"got {gripper_convention!r}"
            )
        self.gripper_override = str(gripper_override or "none").strip().lower()
        if self.gripper_override not in ("none", "close-after-infer", "close-all"):
            raise ValueError(
                "gripper_override must be one of none, close-after-infer, close-all; "
                f"got {gripper_override!r}"
            )
        self.gripper_close_after_infer = int(gripper_close_after_infer or 0)
        self.gripper_close_value = (
            None if gripper_close_value is None else float(gripper_close_value)
        )
        if self.gripper_close_value is not None and not (
            -1.0 <= self.gripper_close_value <= 1.0
        ):
            raise ValueError(
                "gripper_close_value must be in [-1, 1], got "
                f"{self.gripper_close_value}"
            )
        if gripper_force_open_until_infer is None:
            gripper_force_open_until_infer = int(
                os.environ.get("DREAMZERO_GRIPPER_FORCE_OPEN_UNTIL_INFER", "0") or 0
            )
        self.gripper_force_open_until_infer = int(gripper_force_open_until_infer or 0)
        if self.gripper_force_open_until_infer < 0:
            raise ValueError(
                "gripper_force_open_until_infer must be >= 0, got "
                f"{self.gripper_force_open_until_infer}"
            )
        if shared_global_wrist_window_mode is None:
            shared_global_wrist_window_mode = os.environ.get(
                "DREAMZERO_SHARED_GLOBAL_WRIST_WINDOW_MODE",
                "history-current-first",
            )
        self.shared_global_wrist_window_mode = str(
            shared_global_wrist_window_mode or "history-current-first"
        ).strip().lower()
        if self.shared_global_wrist_window_mode not in _SHARED_GLOBAL_WRIST_WINDOW_MODES:
            raise ValueError(
                "shared_global_wrist_window_mode must be one of "
                f"{_SHARED_GLOBAL_WRIST_WINDOW_MODES}; got "
                f"{shared_global_wrist_window_mode!r}"
            )
        if reset_causal_state_each_infer is None:
            reset_causal_state_each_infer = self._parse_bool(
                os.environ.get("DREAMZERO_RESET_CAUSAL_STATE_EACH_INFER", "0")
            )
        self.reset_causal_state_each_infer = bool(reset_causal_state_each_infer)
        self._last_action_debug: dict[str, np.ndarray] = {}
        self._relative_action = False
        self._relative_action_per_horizon = False
        self._relative_action_keys: set[str] = set()
        self.action_representation = "robotwin_delta"
        self._requested_device = device
        self._device_mesh = device_mesh

        self._sessions: dict[str, dict] = {}
        self._load()
        logging.info(
            "Gripper convention resolved to %s (close=%s open=%s)",
            self._resolved_gripper_convention(),
            self._gripper_close_target(),
            self._gripper_open_target(),
        )
        logging.info(
            "Shared-global wrist window mode: %s",
            self.shared_global_wrist_window_mode,
        )
        logging.info(
            "Reset causal state each infer: %s",
            self.reset_causal_state_each_infer,
        )
        logging.info(
            "Predicted-video rollout mode: %s",
            self.video_pred_rollout_mode,
        )

    def _effective_prompt(self, prompt: str | None) -> str:
        if self.prompt_override:
            return self.prompt_override
        return prompt or ""

    def _sync_runtime_shape_from_config(self) -> None:
        """Keep eval request shapes aligned with the checkpoint config."""
        for attr, cfg_key in (
            ("action_horizon", "action_horizon"),
            ("num_frames", "num_frames"),
        ):
            cfg_value = self._cfg.get(cfg_key, None)
            if cfg_value is None:
                continue
            cfg_value = int(cfg_value)
            current = int(getattr(self, attr))
            if current != cfg_value:
                logging.warning(
                    "Overriding eval %s=%d from checkpoint config %s=%d",
                    attr,
                    current,
                    cfg_key,
                    cfg_value,
                )
                setattr(self, attr, cfg_value)

    def _apply_eval_config_overrides(self) -> None:
        """Apply diagnostic-only config overrides before model instantiation."""
        diffusion_cfgs = self._diffusion_model_cfgs()

        disable_hub = self._env_bool("DREAMZERO_DISABLE_MULTI_AGENT_HUB")
        if disable_hub:
            if not diffusion_cfgs:
                logging.warning(
                    "DREAMZERO_DISABLE_MULTI_AGENT_HUB=1 requested, but no "
                    "action_head_cfg was found in the resolved checkpoint config"
                )
            else:
                old_values = []
                for diffusion_cfg in diffusion_cfgs:
                    old_values.append(
                        (
                            diffusion_cfg.get("num_hub_tokens", None),
                            diffusion_cfg.get("use_sparse_hub_attention", None),
                        )
                    )
                    diffusion_cfg.num_hub_tokens = 0
                    diffusion_cfg.use_sparse_hub_attention = False
                logging.warning(
                    "DREAMZERO_DISABLE_MULTI_AGENT_HUB=1: overriding "
                    "%d diffusion_model_cfg copy/copies to num_hub_tokens=0 and "
                    "use_sparse_hub_attention=False for eval; old values=%s",
                    len(diffusion_cfgs),
                    old_values,
                )

        eval_in_dim = os.environ.get("DREAMZERO_EVAL_DIFFUSION_IN_DIM", "").strip()
        eval_concat = os.environ.get(
            "DREAMZERO_EVAL_CONCAT_FIRST_FRAME_LATENT", ""
        ).strip()
        if not eval_in_dim and not eval_concat:
            return
        if not diffusion_cfgs:
            logging.warning(
                "DreamZero eval diffusion-structure override requested, but no "
                "action_head_cfg was found in the resolved checkpoint config"
            )
            return

        updates = []
        for diffusion_cfg in diffusion_cfgs:
            if eval_in_dim:
                old = diffusion_cfg.get("in_dim", None)
                new = int(eval_in_dim)
                if old != new:
                    diffusion_cfg.in_dim = new
                    updates.append(("in_dim", old, new))
            if eval_concat:
                old = diffusion_cfg.get("concat_first_frame_latent", None)
                new = self._parse_bool(eval_concat)
                if old != new:
                    diffusion_cfg.concat_first_frame_latent = new
                    updates.append(("concat_first_frame_latent", old, new))
        if updates:
            logging.warning(
                "Applying DreamZero eval diffusion-structure override to %d "
                "config copy/copies: %s",
                len(diffusion_cfgs),
                ", ".join(f"{name}:{old}->{new}" for name, old, new in updates),
            )

    def _diffusion_model_cfgs(self) -> list:
        diffusion_cfgs = []
        try:
            if "action_head_cfg" in self._cfg:
                diffusion_cfgs.append(
                    self._cfg.action_head_cfg.config.diffusion_model_cfg
                )
        except Exception:
            pass
        try:
            if "action_head_cfg" in self._cfg.get("model", {}):
                diffusion_cfgs.append(
                    self._cfg.model.action_head_cfg.config.diffusion_model_cfg
                )
        except Exception:
            pass
        try:
            model_config = self._cfg.get("model", {}).get("config", {})
            if "action_head_cfg" in model_config:
                diffusion_cfgs.append(
                    model_config.action_head_cfg.config.diffusion_model_cfg
                )
        except Exception:
            pass
        return diffusion_cfgs

    @staticmethod
    def _parse_bool(value: str) -> bool:
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"invalid boolean value: {value!r}")

    @classmethod
    def _env_bool(cls, name: str) -> bool:
        value = os.environ.get(name, "0")
        return cls._parse_bool(value)

    def _model_resize_resolution(self) -> tuple[int, int] | None:
        if self.model_image_h is None and self.model_image_w is None:
            return None
        if self.model_image_h is None or self.model_image_w is None:
            raise ValueError(
                "model_image_h and model_image_w must be set together; "
                f"got {self.model_image_h=} {self.model_image_w=}"
            )
        return int(self.model_image_h), int(self.model_image_w)

    def _apply_model_resolution_overrides(self) -> None:
        """Optionally override model-side resize while preserving raw metadata.

        ``image_h``/``image_w`` are the raw simulator/client/server image
        contract and should match the checkpoint metadata. This optional
        model-side override only changes the checkpoint's in-chain resize
        target, leaving VideoToTensor's raw-resolution check untouched.
        """
        from omegaconf import OmegaConf

        resolution = self._model_resize_resolution()
        if resolution is None:
            return
        target_h, target_w = resolution
        updates: list[tuple[str, object, object]] = []

        def _update_existing(path: str, value: int, *, skip_none: bool = False) -> None:
            old = OmegaConf.select(self._cfg, path, default=None)
            if old is None and skip_none:
                return
            if old is None and path not in ("image_resolution_height", "image_resolution_width"):
                return
            try:
                old_int = None if old is None else int(old)
            except (TypeError, ValueError):
                old_int = old
            if old_int == value:
                return
            OmegaConf.update(self._cfg, path, value, merge=False, force_add=False)
            updates.append((path, old, value))

        _update_existing("image_resolution_height", target_h)
        _update_existing("image_resolution_width", target_w)

        # Only sync action-head explicit target fields when the checkpoint
        # already has them; adding new fields can change older checkpoints in
        # ways that are harder to reason about.
        for prefix in (
            "action_head_cfg.config",
            "model.action_head_cfg.config",
            "model.config.action_head_cfg.config",
        ):
            _update_existing(
                f"{prefix}.target_video_height", target_h, skip_none=True
            )
            _update_existing(
                f"{prefix}.target_video_width", target_w, skip_none=True
            )

        if updates:
            logging.warning(
                "Applying model-side eval resize override HxW=%dx%d: %s",
                target_h,
                target_w,
                ", ".join(f"{path}:{old}->{new}" for path, old, new in updates),
            )

    def _set_transform_resize_resolution(self, transform: object) -> None:
        resolution = self._model_resize_resolution()
        if resolution is None:
            return
        target_h, target_w = resolution
        changed = []
        stack = [transform]
        while stack:
            t = stack.pop(0)
            if t is None:
                continue
            stack[0:0] = list(getattr(t, "transforms", []))
            if t.__class__.__name__ != "VideoResize":
                continue
            old = (getattr(t, "height", None), getattr(t, "width", None))
            if old == (target_h, target_w):
                continue
            setattr(t, "height", target_h)
            setattr(t, "width", target_w)
            changed.append((old, (target_h, target_w)))
        if changed:
            logging.warning(
                "Overriding %d VideoResize transform(s) to HxW=%dx%d",
                len(changed),
                target_h,
                target_w,
            )

    def _load(self) -> None:
        import torch
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        from safetensors.torch import load_file

        _configure_torch_dynamo_for_serving(torch)

        exp_cfg_dir = self.ckpt_dir / self.ckpt_setting / "experiment_cfg"
        if not (exp_cfg_dir / "conf.yaml").is_file():
            exp_cfg_dir = self.ckpt_dir / "experiment_cfg"
        cfg_path = exp_cfg_dir / "conf.yaml"
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"No resolved Hydra config under {self.ckpt_dir}; expected "
                f"{cfg_path} or {self.ckpt_dir / 'experiment_cfg' / 'conf.yaml'}"
            )
        self._cfg = OmegaConf.load(str(cfg_path))
        self._apply_eval_config_overrides()
        self._sync_runtime_shape_from_config()
        self._apply_model_resolution_overrides()
        rel_keys = self._cfg.get("relative_action_keys", []) or []
        self._relative_action = bool(self._cfg.get("relative_action", False))
        self._relative_action_per_horizon = bool(
            self._cfg.get("relative_action_per_horizon", False)
        )
        self._relative_action_keys = {str(k) for k in list(rel_keys)}
        self.action_representation = (
            "absolute_qpos"
            if self._uses_anchor_relative_actions()
            else "robotwin_delta"
        )

        meta_path = exp_cfg_dir / "metadata.json"
        if meta_path.is_file():
            with open(meta_path) as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {}
            logging.warning("metadata.json missing; outputs will not be denormalized")

        # PR 11 load sequence (mirrors groot/vla/experiment/base.py::create_model
        # and eval_utils/offline_eval_bimanual.py::load_model). Skipping any
        # of the four steps below leaves the text encoder / VAE / base WAN
        # body at random init or drops the LoRA-wrapped fine-tune weights
        # silently -- both produce a server that loads "successfully" with
        # zero unexpected keys (strict=False) but predicts garbage actions.
        logging.info("Step 1/4: instantiate(cfg.model)")
        model = instantiate(self._cfg.model)

        pretrained_path = self._cfg.get("pretrained_model_path", None)
        if pretrained_path is None:
            raise ValueError(
                "cfg.pretrained_model_path is required (base WAN body + text "
                "encoder live there); the fine-tune ckpt only holds LoRA "
                "deltas. Offline inference cannot proceed without it."
            )
        pretrained_dir = Path(pretrained_path)
        logging.info(
            "Step 2/4: load pretrained base shards from %s", pretrained_dir
        )
        model_state = model.state_dict()

        step2_sliced: dict[str, tuple] = {}
        step2_dropped: dict[str, tuple] = {}

        pretrained_index = pretrained_dir / "model.safetensors.index.json"
        if pretrained_index.is_file():
            with open(pretrained_index) as f:
                p_index = json.load(f)
            for shard_file in sorted(set(p_index["weight_map"].values())):
                shard_sd = load_file(str(pretrained_dir / shard_file))
                shard_sd, sliced, dropped = _filter_shape_mismatches_for_load(
                    shard_sd, model_state
                )
                step2_sliced.update(sliced)
                step2_dropped.update(dropped)
                model.load_state_dict(shard_sd, strict=False)
                del shard_sd
                gc.collect()
        else:
            pretrained_safe = pretrained_dir / "model.safetensors"
            if not pretrained_safe.is_file():
                raise FileNotFoundError(
                    f"No model.safetensors[.index.json] under {pretrained_dir}"
                )
            sd, step2_sliced, step2_dropped = _filter_shape_mismatches_for_load(
                load_file(str(pretrained_safe)),
                model_state,
            )
            model.load_state_dict(sd, strict=False)
        _log_shape_mismatches(
            "Step 2/4",
            step2_sliced,
            step2_dropped,
            dropped_reason="expected for multi-agent deltas vs DROID",
        )

        if (
            hasattr(model, "action_head")
            and hasattr(model.action_head, "inject_lora_after_loading")
            and getattr(model.action_head.config, "defer_lora_injection", False)
        ):
            logging.info("Step 3/4: inject_lora_after_loading()")
            model.action_head.inject_lora_after_loading()

        logging.info(
            "Step 4/4: load fine-tune LoRA ckpt from %s/%s",
            self.ckpt_dir, self.ckpt_setting,
        )
        drop_finetune_mismatched_keys = frozenset()
        if self._env_bool("DREAMZERO_EVAL_SKIP_FINETUNE_PATCH_EMBEDDING_MISMATCH"):
            drop_finetune_mismatched_keys = FINETUNE_PATCH_EMBEDDING_KEYS
            logging.warning(
                "DREAMZERO_EVAL_SKIP_FINETUNE_PATCH_EMBEDDING_MISMATCH=1: "
                "dropping mismatched fine-tune patch_embedding tensors so the "
                "DROID I2V 36-channel patch embedding stays intact."
            )
        finetune_model_state = model.state_dict()
        step4_sliced: dict[str, tuple] = {}
        step4_dropped: dict[str, tuple] = {}
        weight_path = self.ckpt_dir / self.ckpt_setting / "model.safetensors"
        index_path = (
            self.ckpt_dir / self.ckpt_setting / "model.safetensors.index.json"
        )
        if index_path.is_file():
            with open(index_path) as f:
                index = json.load(f)
            for shard_file in sorted(set(index["weight_map"].values())):
                shard_state_dict = load_file(
                    str(self.ckpt_dir / self.ckpt_setting / shard_file)
                )
                shard_state_dict, sliced, dropped = _filter_shape_mismatches_for_load(
                    shard_state_dict,
                    finetune_model_state,
                    drop_mismatched_keys=drop_finetune_mismatched_keys,
                )
                step4_sliced.update(sliced)
                step4_dropped.update(dropped)
                model.load_state_dict(shard_state_dict, strict=False)
                del shard_state_dict
                gc.collect()
        elif weight_path.is_file():
            state_dict = load_file(str(weight_path))
            state_dict, step4_sliced, step4_dropped = _filter_shape_mismatches_for_load(
                state_dict,
                finetune_model_state,
                drop_mismatched_keys=drop_finetune_mismatched_keys,
            )
            model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(
                f"No model.safetensors[.index.json] under "
                f"{self.ckpt_dir / self.ckpt_setting}"
            )
        _log_shape_mismatches(
            "Step 4/4",
            step4_sliced,
            step4_dropped,
            dropped_reason="fine-tune checkpoint architecture differs from eval model",
        )

        device = self._requested_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._device = device
        self._dtype = (
            torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        )
        model = model.to(device=device, dtype=self._dtype)
        if self._device_mesh is not None:
            if not hasattr(model, "parallelize"):
                raise RuntimeError(
                    "Distributed serving requested, but loaded VLA model does not "
                    "provide parallelize(device_mesh)."
                )
            model.parallelize(self._device_mesh)
            logging.info(
                "Enabled inference-parallel WAN serving on device mesh %s",
                self._device_mesh,
            )
        model.eval()
        self._model = model
        logging.info("VLA loaded onto %s in %s", device, self._dtype)

        try:
            transforms = instantiate(self._cfg.transforms)
            metadata_tag = self._metadata_tag()
            if metadata_tag in transforms:
                transform_tag = metadata_tag
            elif "robofactory" in transforms:
                transform_tag = "robofactory"
                if metadata_tag is not None:
                    logging.warning(
                        "metadata tag %s has no matching transform; falling back to robofactory",
                        metadata_tag,
                    )
            elif "robotwin" in transforms:
                transform_tag = "robotwin"
                if metadata_tag is not None:
                    logging.warning(
                        "metadata tag %s has no matching transform; falling back to robotwin",
                        metadata_tag,
                    )
            else:
                raise KeyError(
                    f"No supported bimanual transform found. Available transforms: {list(transforms.keys())}"
                )
            self._transform = transforms[transform_tag]
            self._set_transform_resize_resolution(self._transform)
            # The transform pipeline needs normalization stats + modality
            # metadata before it can be applied. Mirrors sim_policy.py:365.
            from groot.vla.data.schema.lerobot import DatasetMetadata

            if metadata_tag is not None:
                metadata = DatasetMetadata.model_validate(self._metadata[metadata_tag])
                self._transform.set_metadata(metadata)
                logging.info(
                    "Bimanual transform ready (transform=%s metadata=%s)",
                    transform_tag,
                    metadata_tag,
                )
            else:
                logging.warning(
                    "metadata.json lacks 'robotwin' or legacy 'robofactory' key — "
                    "transform will fail on first infer(). Found keys: %s",
                    list(self._metadata.keys()),
                )
        except Exception:
            self._transform = None
            logging.exception(
                "bimanual transforms failed to instantiate — server will "
                "still start, but infer() will raise."
            )

    def _metadata_tag(self) -> str | None:
        if "robotwin" in self._metadata:
            return "robotwin"
        if "robofactory" in self._metadata:
            return "robofactory"
        return None

    # ----- session bookkeeping ------------------------------------------
    def _session(self, session_id: str) -> dict:
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "history": deque(maxlen=self.num_frames),
                "prompt": "",
                "infer_idx": 0,
            }
        return self._sessions[session_id]

    def reset(self, info: dict) -> str:
        sid = info.get("session_id", "")
        prompt = self._effective_prompt(info.get("prompt", ""))
        sess = self._session(sid)
        sess["history"].clear()
        sess["prompt"] = prompt
        sess["infer_idx"] = 0
        self._reset_action_head_causal_state("episode reset")
        return "reset successful"

    def _reset_action_head_causal_state(self, reason: str) -> bool:
        model = getattr(self, "_model", None)
        action_head = getattr(model, "action_head", None)
        if action_head is not None and hasattr(action_head, "reset_causal_state"):
            action_head.reset_causal_state()
            logging.debug("Reset action-head causal state for %s", reason)
            return True
        return False

    def _maybe_reset_action_head_causal_state_for_infer(self) -> bool:
        if not self.reset_causal_state_each_infer:
            return False
        return self._reset_action_head_causal_state("infer")

    def _uses_shared_global(self) -> bool:
        """Whether the loaded transform emits ``video_global``.

        The outer object is a ComposedModalityTransform; the inner
        BimanualDreamTransform carries ``global_views`` when the
        checkpoint was trained with the shared-global layout.
        """
        for t in getattr(self._transform, "transforms", []):
            if getattr(t, "global_views", None) is not None:
                return True
        return False

    def _iter_transform_tree(self):
        stack = [self._transform]
        while stack:
            transform = stack.pop(0)
            if transform is None:
                continue
            yield transform
            stack[0:0] = list(getattr(transform, "transforms", []))

    @staticmethod
    def _transform_emits_actions(transform: object) -> bool:
        return all(
            callable(getattr(transform, attr, None))
            for attr in ("_prepare_action", "_prepare_state", "_prepare_video")
        )

    def _set_eval_inference_transform_modes(self) -> list[tuple[object, bool]]:
        """Use deterministic eval preprocessing while still emitting actions.

        DreamTransform only includes action/action_mask when ``training`` is
        true, but setting the entire composed transform to train also enables
        random crop, color jitter, and language dropout. Closed-loop eval needs
        deterministic video/state preprocessing; only the model-specific
        DreamTransform should be in train mode for the placeholder action path.
        """
        saved_modes: list[tuple[object, bool]] = []
        for transform in self._iter_transform_tree():
            if not hasattr(transform, "training"):
                continue
            saved_modes.append((transform, bool(getattr(transform, "training"))))
            setattr(transform, "training", self._transform_emits_actions(transform))
        return saved_modes

    @staticmethod
    def _restore_transform_modes(saved_modes: list[tuple[object, bool]]) -> None:
        for transform, training in saved_modes:
            setattr(transform, "training", training)

    def _build_video_windows(
        self,
        history: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return global/left/right video windows for the transform input.

        Shared-global training uses ``global_condition_mode=current_repeat`` for
        the global stream: frame 0 is the current observation and the action head
        uses frame 0 for per-agent I2V conditioning whenever ``video_global`` is
        present. The wrist stream is configurable for eval diagnostics:

        * repeat-current: repeat the current wrist frames, matching the
          historical closed-loop server behavior.
        * history-current-first: put the current wrist frame at index 0, then
          append the rest of the rolling history.
        * history-chronological: keep the rolling history unchanged.

        Legacy non-shared-global checkpoints keep the rolling history window.
        """
        global_history = np.stack([h for (h, _, _) in history], axis=0)
        agent0_history = np.stack([l for (_, l, _) in history], axis=0)
        agent1_history = np.stack([r for (_, _, r) in history], axis=0)
        if self._uses_shared_global():
            current_global, current_agent0, current_agent1 = history[-1]
            global_history = np.repeat(current_global[None], self.num_frames, axis=0)
            mode = getattr(
                self,
                "shared_global_wrist_window_mode",
                "repeat-current",
            )
            if mode == "repeat-current":
                agent0_history = np.repeat(
                    current_agent0[None], self.num_frames, axis=0
                )
                agent1_history = np.repeat(
                    current_agent1[None], self.num_frames, axis=0
                )
            elif mode == "history-current-first":
                agent0_history = np.concatenate(
                    [agent0_history[-1:], agent0_history[:-1]], axis=0
                )
                agent1_history = np.concatenate(
                    [agent1_history[-1:], agent1_history[:-1]], axis=0
                )
            elif mode == "history-chronological":
                pass
            else:
                raise ValueError(
                    "shared_global_wrist_window_mode must be one of "
                    f"{_SHARED_GLOBAL_WRIST_WINDOW_MODES}; got {mode!r}"
                )
        return global_history, agent0_history, agent1_history

    # ----- inference ----------------------------------------------------
    def infer(self, obs: dict) -> dict:
        """Pre-Concat dotted-key batch construction (PR 11+).

        Builds the schema the dataset loader emits at training time
        (one key per camera, per per-arm state segment, plus
        placeholder per-arm action segments) so the full
        ``ComposedModalityTransform`` chain (q99 normalize + Concat
        + BimanualDreamTransform) runs end-to-end. The model's
        ``get_action`` returns ``action_pred[B, P, T_a, D_per_arm]``
        which we then unpack per-arm and denormalize.
        """
        import cv2
        import torch

        if self._transform is None:
            raise RuntimeError(
                "Bimanual transform did not instantiate at load time. Check "
                "the saved experiment_cfg/conf.yaml against this dreamzero "
                "branch."
            )

        sid = obs.get("session_id", "")
        sess = self._session(sid)
        if obs.get("prompt"):
            sess["prompt"] = self._effective_prompt(obs["prompt"])
        if "step" in obs and obs["step"] is not None:
            sess["last_env_step"] = int(obs["step"])
        else:
            sess["last_env_step"] = None
        if obs.get("replan_every") is not None:
            sess["last_replan_every"] = int(obs["replan_every"])
        if obs.get("chunk_start_index") is not None:
            sess["last_chunk_start_index"] = int(obs["chunk_start_index"])
        self._maybe_reset_action_head_causal_state_for_infer()

        qpos = np.asarray(obs["qpos"], dtype=np.float32).reshape(-1)
        assert qpos.shape == (16,), f"need 16-dim qpos, got {qpos.shape}"

        head = np.asarray(obs["head_rgb"], dtype=np.uint8)
        lft = np.asarray(obs["left_rgb"], dtype=np.uint8)
        rgt = np.asarray(obs["right_rgb"], dtype=np.uint8)
        H, W = self.image_h, self.image_w
        head = cv2.resize(head, (W, H))
        lft = cv2.resize(lft, (W, H))
        rgt = cv2.resize(rgt, (W, H))

        # Rolling history of RAW (no V-tile) per-camera frames;
        # left-pad with the first observation so we always have
        # ``num_frames`` of context.
        sess["history"].append((head, lft, rgt))
        history = list(sess["history"])
        while len(history) < self.num_frames:
            history.insert(0, history[0])
        history = history[-self.num_frames:]
        # [T, H, W, 3] uint8 per camera. head=global, lft=agent0, rgt=agent1
        # (matches the LeRobot v2 camera naming written by
        # ``scripts/data/robofactory_to_lerobot_v2.py``).
        global_video, agent0_video, agent1_video = self._build_video_windows(history)
        self._record_observed_video_debug(
            sess,
            global_video=global_video,
            agent0_video=agent0_video,
            agent1_video=agent1_video,
        )

        # Per-arm state slices (T_s=1, current step only).
        T_s = 1
        T_a = self.action_horizon
        prompt = self._effective_prompt(sess.get("prompt", ""))

        # ``action.*`` keys are required by ``StateActionTransform`` /
        # ``ConcatTransform`` even at inference time -- the model
        # ignores their values and starts denoising from random noise
        # (see ``WANPolicyHead._get_action_multi_agent``). Zeros are
        # safe placeholders here; they only have to satisfy the
        # downstream shape contract.
        batch = {
            "video.global_camera-images-rgb": global_video,
            "video.agent0_camera-images-rgb": agent0_video,
            "video.agent1_camera-images-rgb": agent1_video,
            "state.panda0_joint_pos":    qpos[0:7].reshape(T_s, 7).copy(),
            "state.panda0_gripper_pos":  qpos[7:8].reshape(T_s, 1).copy(),
            "state.panda1_joint_pos":    qpos[8:15].reshape(T_s, 7).copy(),
            "state.panda1_gripper_pos":  qpos[15:16].reshape(T_s, 1).copy(),
            "action.panda0_joint_pos":   np.zeros((T_a, 7), dtype=np.float32),
            "action.panda0_gripper_pos": np.zeros((T_a, 1), dtype=np.float32),
            "action.panda1_joint_pos":   np.zeros((T_a, 7), dtype=np.float32),
            "action.panda1_gripper_pos": np.zeros((T_a, 1), dtype=np.float32),
            "annotation.task": prompt,
        }

        with torch.inference_mode():
            # DreamTransform.apply_single has a ``if self.training:`` gate
            # that drops ``action`` / ``action_mask`` / ``has_real_action``
            # in eval mode, but the multi-agent inference path needs those
            # tensors for shape. Keep video/state preprocessing deterministic
            # by only enabling training mode on the model-specific transform;
            # leave random crop, color jitter, perturb/dropout transforms off.
            saved_transform_modes = self._set_eval_inference_transform_modes()
            try:
                normalized_inputs = self._transform.apply(batch)
            finally:
                self._restore_transform_modes(saved_transform_modes)

            # ``text`` and ``text_negative`` come out as raw Python strings
            # (the collator usually tokenizes; we don't use a collator).
            # Tokenize manually with the BimanualDreamTransform's tokenizer
            # (the outer ComposedModalityTransform doesn't expose it; we
            # find it on the inner model_specific_transform).
            tok = None
            for t in getattr(self._transform, "transforms", []):
                if hasattr(t, "tokenizer"):
                    tok = t.tokenizer
                    break
            if tok is None:
                raise RuntimeError(
                    "No tokenizer found on any sub-transform; cannot "
                    "tokenize text for inference."
                )
            for str_key, ids_key, mask_key in [
                ("text", "text", "text_attention_mask"),
                ("text_negative", "text_negative", "text_attention_mask_negative"),
            ]:
                if str_key in normalized_inputs and isinstance(
                    normalized_inputs[str_key], str
                ):
                    text_val = normalized_inputs[str_key]
                    ids, mask = tok(
                        text_val, return_mask=True, add_special_tokens=True
                    )
                    # HuggingfaceTokenizer wraps the single string into a
                    # 1-element list, so ids/mask come out as (1, seq_len)
                    # with a leading batch dim. Drop it so the generic
                    # ``.unsqueeze(0)`` below adds it back uniformly.
                    if ids.dim() == 2 and ids.shape[0] == 1:
                        ids = ids.squeeze(0)
                        mask = mask.squeeze(0)
                    normalized_inputs[ids_key] = ids
                    normalized_inputs[mask_key] = mask
            # ``transform.apply`` runs unbatched (single-sample mode).
            # Add a leading B=1 dim and move tensors to device. The dict
            # also contains scalar/numpy ints (e.g. ``num_agents`` set by
            # BimanualDreamTransform as np.int64) -- wrap those as 1-D
            # tensors so ``prepare_input``'s tree.map_structure can
            # ``torch.is_floating_point`` them without crashing.
            inputs_gpu = {}
            for k, v in normalized_inputs.items():
                if isinstance(v, torch.Tensor):
                    t = v
                elif isinstance(v, np.ndarray):
                    t = torch.from_numpy(v)
                elif isinstance(
                    v, (int, float, bool, np.integer, np.floating, np.bool_)
                ):
                    t = torch.as_tensor(v)
                else:
                    # Drop strings / unknown types: ``annotation.task`` is
                    # left in the post-transform dict as a raw str, but
                    # the model's ``prepare_input`` does
                    # ``tree.map_structure(torch.is_floating_point, ...)``
                    # which only accepts Tensors. Tokenized output already
                    # lives under ``text`` / ``text_attention_mask``.
                    continue
                if t.is_floating_point():
                    t = t.to(self._device, dtype=self._dtype)
                else:
                    t = t.to(self._device)
                inputs_gpu[k] = t.unsqueeze(0)

            outputs = self._model.get_action(inputs_gpu)

        flat_action = self._denorm_action(outputs, qpos)
        self._apply_gripper_force_open(sess, flat_action)
        self._apply_gripper_override(sess, flat_action)
        self._last_action_debug["action_physical_final"] = flat_action.copy()
        self._log_action_summary(sess, sid, flat_action)

        if self.save_video_pred:
            try:
                if self.video_pred_rollout_mode == "noncausal":
                    self._run_noncausal_video_pred_rollout(inputs_gpu)
                self._dump_video_pred(sess, sid)
                self._dump_conditioning_pred(sess, sid)
            except Exception:
                logging.exception("save_video_pred failed; continuing without")

        reply = {"action_chunk": flat_action.astype(np.float32)}
        if self.return_action_debug:
            reply.update(
                {
                    key: value.astype(np.float32)
                    for key, value in self._last_action_debug.items()
                }
            )
        sess["infer_idx"] = sess.get("infer_idx", 0) + 1
        return reply

    @staticmethod
    def _video_pred_context(sess: dict) -> tuple[int, int | None, str]:
        infer_idx = int(sess.get("infer_idx", 0))
        raw_env_step = sess.get("last_env_step")
        env_step = None if raw_env_step is None else int(raw_env_step)
        if env_step is None:
            prefix = f"infer{infer_idx:04d}_envunknown"
        else:
            prefix = f"infer{infer_idx:04d}_env{env_step:04d}"
        return infer_idx, env_step, prefix

    @staticmethod
    def _append_video_pred_manifest(out_dir: Path, entry: dict[str, Any]) -> None:
        manifest_path = out_dir / "manifest.jsonl"
        with manifest_path.open("a") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

    @staticmethod
    def _clone_action_head_state_value(value: Any) -> Any:
        import torch

        torch_tensor = getattr(torch, "Tensor", ())
        if torch_tensor and isinstance(value, torch_tensor):
            return value.detach().clone()
        if isinstance(value, list):
            return [BimanualPolicy._clone_action_head_state_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(BimanualPolicy._clone_action_head_state_value(v) for v in value)
        if isinstance(value, dict):
            return {
                copy.deepcopy(k): BimanualPolicy._clone_action_head_state_value(v)
                for k, v in value.items()
            }
        try:
            return copy.deepcopy(value)
        except Exception:
            return value

    def _snapshot_action_head_control_state(self) -> dict[str, Any]:
        action_head = getattr(getattr(self, "_model", None), "action_head", None)
        if action_head is None:
            return {}
        snapshot: dict[str, Any] = {}
        for attr in _ACTION_HEAD_CONTROL_STATE_ATTRS:
            if hasattr(action_head, attr):
                snapshot[attr] = self._clone_action_head_state_value(getattr(action_head, attr))
            else:
                snapshot[attr] = _MISSING
        model = getattr(action_head, "model", None)
        if model is not None:
            if hasattr(model, "_cached_token_agent_id"):
                snapshot["model._cached_token_agent_id"] = self._clone_action_head_state_value(
                    getattr(model, "_cached_token_agent_id")
                )
            else:
                snapshot["model._cached_token_agent_id"] = _MISSING
        return snapshot

    def _restore_action_head_control_state(self, snapshot: dict[str, Any]) -> None:
        if not snapshot:
            return
        action_head = getattr(getattr(self, "_model", None), "action_head", None)
        if action_head is None:
            return
        for attr in _ACTION_HEAD_CONTROL_STATE_ATTRS:
            value = snapshot.get(attr, _MISSING)
            if value is _MISSING:
                if hasattr(action_head, attr):
                    delattr(action_head, attr)
                continue
            setattr(action_head, attr, value)
        if "model._cached_token_agent_id" in snapshot:
            model = getattr(action_head, "model", None)
            if model is not None:
                value = snapshot["model._cached_token_agent_id"]
                if value is _MISSING:
                    if hasattr(model, "_cached_token_agent_id"):
                        delattr(model, "_cached_token_agent_id")
                else:
                    setattr(model, "_cached_token_agent_id", value)

    def _run_noncausal_video_pred_rollout(self, inputs_gpu: dict[str, Any]) -> None:
        """Refresh ``_last_video_pred`` with noncausal flowmatch for diagnostics.

        The control action has already been computed before this runs; this
        second pass only makes the saved pred video legible when causal/unipc
        inference is the rollout mode used for action generation.
        """
        import torch

        control_state = self._snapshot_action_head_control_state()
        old_causal = os.environ.get("MAI_USE_CAUSAL_INFERENCE")
        os.environ["MAI_USE_CAUSAL_INFERENCE"] = "0"
        logging.info(
            "Running noncausal predicted-video diagnostic rollout; control "
            "action remains from the primary action rollout"
        )
        try:
            with torch.inference_mode():
                self._model.get_action(inputs_gpu)
        finally:
            if old_causal is None:
                os.environ.pop("MAI_USE_CAUSAL_INFERENCE", None)
            else:
                os.environ["MAI_USE_CAUSAL_INFERENCE"] = old_causal
            self._restore_action_head_control_state(control_state)

    def _dump_video_pred(self, sess: dict, sid: str) -> None:
        """VAE-decode the action_head's last denoised video latents and
        write one mp4 per agent. Called from infer() when save_video_pred
        is on. Adds ~5-15s per call (heavy VAE decode); diagnostic only.
        """
        action_head = self._model.action_head
        latents = getattr(action_head, "_last_video_pred", None)
        if latents is None:
            logging.warning("action_head._last_video_pred missing; "
                            "make sure _get_action_multi_agent stashes it")
            return
        logging.info(
            "video_pred runtime: latent_shape=%s current_start_frame=%s "
            "num_frame_per_block=%s num_inference_steps=%s causal=%s "
            "anchor_i2v=%s scheduler=%s",
            tuple(latents.shape),
            getattr(action_head, "current_start_frame", None),
            getattr(action_head, "num_frame_per_block", None),
            getattr(action_head, "_mai_num_inference_steps", None),
            os.environ.get("MAI_USE_CAUSAL_INFERENCE", "1"),
            getattr(action_head, "_mai_anchor_i2v_first_frame", None),
            getattr(action_head, "_mai_causal_scheduler", None),
        )
        frames = self._decode_latent_video(latents)

        out_dir = self.video_pred_dir or (self.ckpt_dir / "video_pred")
        out_dir = Path(out_dir) / f"session_{sid[:12]}"
        out_dir.mkdir(parents=True, exist_ok=True)
        infer_idx, env_step, prefix = self._video_pred_context(sess)
        P, T, H, W = frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3]
        pred_files = self._write_decoded_video_set(frames, out_dir, prefix)
        observed_files: list[str] = []
        comparison_files: list[str] = []
        observed_videos = sess.get("last_observed_video_debug")
        if isinstance(observed_videos, dict):
            observed_dir = out_dir / "observed"
            observed_files = [
                str(Path("observed") / name)
                for name in self._write_observed_video_windows(
                    observed_videos,
                    observed_dir,
                    prefix,
                )
            ]
            comparison_files = [
                str(Path("comparison") / name)
                for name in self._write_pred_observed_comparison(
                    pred_frames=frames,
                    observed_videos=observed_videos,
                    out_dir=out_dir / "comparison",
                    prefix=prefix,
                )
            ]
        self._append_video_pred_manifest(
            out_dir,
            {
                "infer_idx": infer_idx,
                "env_step": env_step,
                "session_id_prefix": sid[:12],
                "latent_shape": list(latents.shape),
                "decoded_shape": [int(P), int(T), int(H), int(W), 3],
                "pred_files": pred_files,
                "observed_files": observed_files,
                "comparison_files": comparison_files,
                "replan_every": sess.get("last_replan_every"),
                "chunk_start_index": sess.get("last_chunk_start_index"),
                "shared_global_wrist_window_mode": self.shared_global_wrist_window_mode,
                "reset_causal_state_each_infer": self.reset_causal_state_each_infer,
                "video_pred_rollout_mode": self.video_pred_rollout_mode,
                "pred_video_semantics": (
                    "decoded denoised wrist-future latents; comparison panels use "
                    "the observed conditioning-window frame at the same displayed "
                    "index, clamped to the final observed frame"
                ),
            },
        )
        logging.info(
            "wrote predicted video: infer_idx=%d env_step=%s session=%s dir=%s",
            infer_idx,
            "unknown" if env_step is None else env_step,
            sid[:12],
            out_dir,
        )

    def _decode_latent_video(self, latents):
        """Decode VAE latents in ``[B, P, C, F, H, W]`` layout."""
        import torch

        action_head = self._model.action_head
        B, P, C_lat, F_lat, H_lat, W_lat = latents.shape
        lat_bp = latents.reshape(B * P, C_lat, F_lat, H_lat, W_lat)
        with torch.inference_mode():
            frames = action_head.vae.decode(
                lat_bp.to(self._device, dtype=self._dtype),
                tiled=action_head.tiled,
                tile_size=(action_head.tile_size_height,
                           action_head.tile_size_width),
                tile_stride=(action_head.tile_stride_height,
                             action_head.tile_stride_width),
            )
        frames = frames.float()
        frames = ((frames + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        frames = frames.cpu().numpy()
        return frames.transpose(0, 2, 3, 4, 1).reshape(
            B, P, -1, frames.shape[3], frames.shape[4], 3
        )[0]

    def _write_decoded_video_set(self, frames, out_dir: Path, prefix: str) -> list[str]:
        import av

        out_dir.mkdir(parents=True, exist_ok=True)
        P, T, H, W = frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3]
        written: list[str] = []
        for p in range(P):
            out_name = f"{prefix}_agent{p}.mp4"
            out_path = out_dir / out_name
            with av.open(str(out_path), mode="w") as container:
                stream = container.add_stream("h264", rate=20)
                stream.width = W
                stream.height = H
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "23"}
                for t in range(T):
                    av_frame = av.VideoFrame.from_ndarray(frames[p, t], format="rgb24")
                    for packet in stream.encode(av_frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            written.append(out_name)
        return written

    def _current_observed_frame_index(self, frames: np.ndarray) -> int:
        if frames.shape[0] <= 1:
            return 0
        if self._uses_shared_global():
            mode = getattr(
                self,
                "shared_global_wrist_window_mode",
                "repeat-current",
            )
            return frames.shape[0] - 1 if mode == "history-chronological" else 0
        return frames.shape[0] - 1

    @staticmethod
    def _observed_frame_index_for_pred_step(frames: np.ndarray, pred_step: int) -> int:
        if frames.shape[0] <= 1:
            return 0
        return min(max(int(pred_step), 0), frames.shape[0] - 1)

    @staticmethod
    def _resize_rgb_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.shape[:2] == (height, width):
            return frame.copy()
        import cv2

        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _label_rgb_frame(frame: np.ndarray, label: str) -> np.ndarray:
        import cv2

        out = np.asarray(frame, dtype=np.uint8).copy()
        cv2.rectangle(out, (0, 0), (min(out.shape[1], 178), 24), (0, 0, 0), -1)
        cv2.putText(
            out,
            label,
            (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return out

    def _write_pred_observed_comparison(
        self,
        *,
        pred_frames: np.ndarray,
        observed_videos: dict[str, np.ndarray],
        out_dir: Path,
        prefix: str,
    ) -> list[str]:
        import av

        pred_frames = np.asarray(pred_frames, dtype=np.uint8)
        if pred_frames.ndim != 5 or pred_frames.shape[-1] != 3:
            logging.warning(
                "Skipping pred/observed comparison with unexpected pred shape %s",
                tuple(pred_frames.shape),
            )
            return []
        if "global" not in observed_videos:
            return []

        out_dir.mkdir(parents=True, exist_ok=True)
        P, T, H, W = (
            pred_frames.shape[0],
            pred_frames.shape[1],
            pred_frames.shape[2],
            pred_frames.shape[3],
        )
        global_video = np.asarray(observed_videos["global"], dtype=np.uint8)
        if global_video.ndim != 4 or global_video.shape[-1] != 3:
            return []
        written: list[str] = []
        for p in range(P):
            wrist_name = f"agent{p}"
            wrist_video = np.asarray(observed_videos.get(wrist_name), dtype=np.uint8)
            if wrist_video.ndim != 4 or wrist_video.shape[-1] != 3:
                logging.warning(
                    "Skipping pred/observed comparison for %s with observed shape %s",
                    wrist_name,
                    tuple(wrist_video.shape),
                )
                continue
            out_name = f"{prefix}_agent{p}_compare.mp4"
            out_path = out_dir / out_name
            with av.open(str(out_path), mode="w") as container:
                stream = container.add_stream("h264", rate=20)
                stream.width = W * 3
                stream.height = H
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "23"}
                for t in range(T):
                    global_idx = self._observed_frame_index_for_pred_step(global_video, t)
                    wrist_idx = self._observed_frame_index_for_pred_step(wrist_video, t)
                    global_frame = self._resize_rgb_frame(global_video[global_idx], H, W)
                    wrist_frame = self._resize_rgb_frame(wrist_video[wrist_idx], H, W)
                    global_panel = self._label_rgb_frame(
                        global_frame,
                        f"global obs[{global_idx}]",
                    )
                    wrist_panel = self._label_rgb_frame(
                        wrist_frame,
                        f"{wrist_name} obs[{wrist_idx}]",
                    )
                    pred_panel = self._label_rgb_frame(
                        pred_frames[p, t],
                        f"{wrist_name} pred",
                    )
                    canvas = np.concatenate(
                        [global_panel, wrist_panel, pred_panel],
                        axis=1,
                    )
                    av_frame = av.VideoFrame.from_ndarray(canvas, format="rgb24")
                    for packet in stream.encode(av_frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            written.append(out_name)
        return written

    @staticmethod
    def _record_observed_video_debug(
        sess: dict,
        *,
        global_video: np.ndarray,
        agent0_video: np.ndarray,
        agent1_video: np.ndarray,
    ) -> None:
        """Keep a copy of the exact RGB windows fed to the transform."""
        sess["last_observed_video_debug"] = {
            "global": np.asarray(global_video, dtype=np.uint8).copy(),
            "agent0": np.asarray(agent0_video, dtype=np.uint8).copy(),
            "agent1": np.asarray(agent1_video, dtype=np.uint8).copy(),
        }

    def _write_observed_video_windows(
        self,
        videos: dict[str, np.ndarray],
        out_dir: Path,
        prefix: str,
    ) -> list[str]:
        import av

        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        for name, frames in videos.items():
            frames = np.asarray(frames, dtype=np.uint8)
            if frames.ndim != 4 or frames.shape[-1] != 3:
                logging.warning(
                    "Skipping observed video %s with unexpected shape %s",
                    name,
                    tuple(frames.shape),
                )
                continue
            T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
            out_name = f"{prefix}_observed_{name}.mp4"
            out_path = out_dir / out_name
            with av.open(str(out_path), mode="w") as container:
                stream = container.add_stream("h264", rate=20)
                stream.width = W
                stream.height = H
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "23"}
                for t in range(T):
                    av_frame = av.VideoFrame.from_ndarray(frames[t], format="rgb24")
                    for packet in stream.encode(av_frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            written.append(out_name)
        return written

    def _dump_conditioning_pred(self, sess: dict, sid: str) -> None:
        """Decode the I2V conditioning latents once for debugging.

        This checks whether the VAE encode/decode path for the current
        observation is sane. If these files are structured while
        ``stepXXXX_agent*.mp4`` is noise, the denoising/video-conditioning
        path is the problem, not the VAE diagnostic decode path.
        """
        infer_idx, env_step, prefix = self._video_pred_context(sess)
        if infer_idx != 0 or sess.get("condition_debug_dumped", False):
            return

        action_head = self._model.action_head
        out_dir = self.video_pred_dir or (self.ckpt_dir / "video_pred")
        out_dir = Path(out_dir) / f"session_{sid[:12]}" / "conditioning"

        observed_videos = sess.get("last_observed_video_debug")
        if isinstance(observed_videos, dict):
            self._write_observed_video_windows(
                observed_videos,
                out_dir,
                prefix,
            )
            logging.info(
                "wrote observed RGB conditioning videos: infer_idx=%d env_step=%s session=%s dir=%s",
                infer_idx,
                "unknown" if env_step is None else env_step,
                sid[:12],
                out_dir,
            )

        clean_latents = getattr(action_head, "_last_clean_video_cond", None)
        if clean_latents is not None:
            clean_frames = self._decode_latent_video(clean_latents)
            self._write_decoded_video_set(
                clean_frames, out_dir, f"{prefix}_clean_x"
            )
            logging.info(
                "wrote conditioning clean_x video: infer_idx=%d env_step=%s session=%s dir=%s",
                infer_idx, "unknown" if env_step is None else env_step, sid[:12], out_dir,
            )

        # ``y`` is [B, P, mask_channels + latent_channels, F, H, W].
        # Decode only the latent tail when its channel count matches the
        # clean VAE latent channel count.
        y_latents = getattr(action_head, "_last_y_video_cond", None)
        if y_latents is not None and clean_latents is not None:
            c_lat = clean_latents.shape[2]
            if y_latents.shape[2] >= c_lat:
                y_tail = y_latents[:, :, -c_lat:]
                y_frames = self._decode_latent_video(y_tail)
                self._write_decoded_video_set(
                    y_frames, out_dir, f"{prefix}_y_latent"
                )
                logging.info(
                    "wrote conditioning y-latent video: infer_idx=%d env_step=%s session=%s dir=%s",
                    infer_idx, "unknown" if env_step is None else env_step, sid[:12], out_dir,
                )

        sess["condition_debug_dumped"] = True

    def _flatten_bimanual_action_pred(self, pred: np.ndarray) -> np.ndarray:
        """Flatten first 8 dims of ``[P=2, T, D]`` actions to env ``[T, 16]``.

        DROID-width multi-agent checkpoints may predict D=32 per arm while
        RoboFactory still executes only 8 controls per arm. Keep client debug
        tensors in the executable 16D layout so existing dump analyzers remain
        valid.
        """
        out = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
        T = min(pred.shape[1], self.action_horizon)
        out[:T, 0:8] = pred[0, :T, :8]
        out[:T, 8:16] = pred[1, :T, :8]
        return out

    def _uses_anchor_relative_actions(self) -> bool:
        return self._relative_action or self._relative_action_per_horizon

    def _key_is_relative(self, subkey: str) -> bool:
        if not self._uses_anchor_relative_actions():
            return False
        if self._relative_action_keys:
            candidates = {subkey}
            if subkey.startswith("action."):
                candidates.add(subkey[len("action."):])
            elif subkey.startswith("state."):
                candidates.add(subkey[len("state."):])
            else:
                candidates.add(f"action.{subkey}")
                candidates.add(f"state.{subkey}")
            return bool(candidates & self._relative_action_keys)
        return "gripper" not in subkey.lower()

    def _add_reference_state_for_relative_keys(
        self,
        out: np.ndarray,
        qpos: np.ndarray,
    ) -> None:
        """Match DreamZero sim_policy: relative joint outputs become
        absolute action targets by adding the latest observed qpos.
        """
        if self._key_is_relative("panda0_joint_pos"):
            out[:, 0:7] += qpos[0:7]
        if self._key_is_relative("panda1_joint_pos"):
            out[:, 8:15] += qpos[8:15]

    def _resolved_gripper_convention(self) -> str:
        """Return the simulator gripper command convention for diagnostics.

        RoboTwin stores absolute gripper commands as ``0.0=close`` and
        ``1.0=open``. RoboFactory's ManiSkill controller uses ``-1.0=close``
        and ``+1.0=open``. The model's denormalized gripper output already
        follows the dataset statistics; this convention only controls
        diagnostic binarization/overrides and log thresholds.
        """
        convention = str(getattr(self, "gripper_convention", "auto") or "auto")
        convention = convention.strip().lower()
        if convention in ("robotwin", "robofactory"):
            return convention

        metadata_tag = self._metadata_tag()
        if metadata_tag == "robofactory":
            return "robofactory"
        return "robotwin"

    def _gripper_open_target(self) -> float:
        # Both currently supported simulator conventions use +1/open.
        return 1.0

    def _gripper_close_target(self) -> float:
        explicit = getattr(self, "gripper_close_value", None)
        if explicit is not None:
            return float(explicit)
        if self._resolved_gripper_convention() == "robofactory":
            return -1.0
        return 0.0

    def _gripper_log_close_threshold(self) -> float:
        return 0.5 * (self._gripper_open_target() + self._gripper_close_target())

    @staticmethod
    def _format_threshold_for_log(value: float) -> str:
        text = f"{value:.3f}".rstrip("0").rstrip(".")
        return text.replace("-", "neg").replace(".", "p")

    def _binarize_gripper_targets(self, out: np.ndarray) -> None:
        if self.gripper_binarize_threshold is None:
            return
        threshold = float(self.gripper_binarize_threshold)
        close_target = self._gripper_close_target()
        open_target = self._gripper_open_target()
        out[:, 7] = np.where(out[:, 7] >= threshold, open_target, close_target)
        out[:, 15] = np.where(out[:, 15] >= threshold, open_target, close_target)

    def _apply_gripper_override(self, sess: dict, out: np.ndarray) -> None:
        if self.gripper_override == "none":
            return
        infer_idx = int(sess.get("infer_idx", 0))
        should_close = self.gripper_override == "close-all"
        if self.gripper_override == "close-after-infer":
            should_close = infer_idx >= self.gripper_close_after_infer
        if not should_close:
            return
        out[:, 7] = self._gripper_close_target()
        out[:, 15] = self._gripper_close_target()
        self._last_action_debug["action_physical_after_override"] = out.copy()

    def _apply_gripper_force_open(self, sess: dict, out: np.ndarray) -> None:
        if self.gripper_force_open_until_infer <= 0:
            return
        infer_idx = int(sess.get("infer_idx", 0))
        if infer_idx >= self.gripper_force_open_until_infer:
            return
        out[:, 7] = self._gripper_open_target()
        out[:, 15] = self._gripper_open_target()
        self._last_action_debug["action_physical_after_force_open"] = out.copy()

    def _log_action_summary(self, sess: dict, sid: str, action: np.ndarray) -> None:
        """Emit compact gripper diagnostics for closed-loop eval logs."""
        infer_idx = int(sess.get("infer_idx", 0))
        prompt = str(sess.get("prompt", ""))
        close_threshold = self._gripper_log_close_threshold()
        close_threshold_label = self._format_threshold_for_log(close_threshold)

        def _gripper_summary(values: np.ndarray) -> str:
            values = np.asarray(values, dtype=np.float32).reshape(-1)
            close_count = int(np.sum(values < close_threshold))
            first = np.array2string(
                values[: min(8, values.shape[0])],
                precision=3,
                separator=",",
            )
            return (
                f"min={float(values.min()):.3f} max={float(values.max()):.3f} "
                f"mean={float(values.mean()):.3f} close_lt_{close_threshold_label}="
                f"{close_count}/{values.shape[0]} "
                f"first={first}"
            )

        debug = self._last_action_debug
        extra_parts = []
        for label, key in (
            ("norm_raw", "action_norm_raw"),
            ("physical_pre_binarize", "action_physical_pre_binarize"),
            ("physical_after_force_open", "action_physical_after_force_open"),
            ("physical_after_override", "action_physical_after_override"),
        ):
            value = debug.get(key)
            if value is None:
                continue
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 16:
                extra_parts.append(
                    f"{label}_left_gripper[{_gripper_summary(arr[:, 7])}] "
                    f"{label}_right_gripper[{_gripper_summary(arr[:, 15])}]"
                )

        logging.info(
            "Action summary session=%s infer_idx=%d prompt=%r "
            "representation=%s gripper_binarize_threshold=%s "
            "gripper_convention=%s gripper_close_target=%s "
            "gripper_force_open_until_infer=%s gripper_override=%s "
            "left_gripper[%s] right_gripper[%s]%s",
            sid[:12],
            infer_idx,
            prompt,
            self.action_representation,
            self.gripper_binarize_threshold,
            self._resolved_gripper_convention(),
            self._gripper_close_target(),
            self.gripper_force_open_until_infer,
            self.gripper_override,
            _gripper_summary(action[:, 7]),
            _gripper_summary(action[:, 15]),
            (" " + " ".join(extra_parts)) if extra_parts else "",
        )

    def _denorm_action(self, outputs, qpos: np.ndarray) -> np.ndarray:
        """Take model output ``action_pred [B=1, P=2, T_a, D_per_arm>=8]``
        (normalized to [-1, 1] via q99) and denormalize back to
        physical units. If the checkpoint was trained with DreamZero's
        ``relative_action`` path, joint slices are denormalized as
        target-current offsets and then converted back to absolute qpos
        targets using the latest RoboTwin qpos. Concatenates the two arms
        into the 16-dim flat layout the client expects:
        ``[panda0_joint(0:7), panda0_gripper(7:8), panda1_joint(8:15),
        panda1_gripper(15:16)]``.
        """
        data = outputs.data if hasattr(outputs, "data") else outputs
        if "action_pred" not in data:
            raise KeyError(
                f"Expected action_pred in model output, got {list(data.keys())}"
            )
        pred = data["action_pred"]
        try:
            import torch
        except ModuleNotFoundError:
            torch = None
        if torch is not None and isinstance(pred, torch.Tensor):
            pred = pred.detach().float().cpu().numpy()
        raw_pred = np.asarray(pred, dtype=np.float32)         # [B, P, T_a, D]
        if raw_pred.ndim != 4:
            raise ValueError(
                f"action_pred must be 4-D [B, P, T_a, D]; got {raw_pred.shape}"
            )
        # The diffusion head samples in q01/q99-normalized action space, but
        # samples are unconstrained. Clip before inverse normalization so eval
        # never sends commands outside the training/controller range.
        pred = np.clip(raw_pred, -1.0, 1.0)
        B, P, T_a, D_per_arm = pred.shape
        if P != 2 or D_per_arm < 8:
            raise ValueError(
                f"Expected P=2 arms with D>=8 per arm; got P={P} D={D_per_arm}"
            )
        self._last_action_debug = {
            "action_norm_raw": self._flatten_bimanual_action_pred(raw_pred[0]),
            "action_norm_clipped": self._flatten_bimanual_action_pred(pred[0]),
        }
        if D_per_arm > 8:
            self._last_action_debug["action_norm_raw_model_full"] = raw_pred[
                0
            ].transpose(1, 0, 2).reshape(raw_pred.shape[2], P * D_per_arm)
            self._last_action_debug["action_norm_clipped_model_full"] = pred[
                0
            ].transpose(1, 0, 2).reshape(pred.shape[2], P * D_per_arm)

        out = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
        metadata_tag = self._metadata_tag()
        emb_meta = {}
        if metadata_tag is not None:
            emb_meta = self._metadata[metadata_tag].get("statistics", {}).get("action", {})

        def _stats_for_key(key: str, width: int) -> tuple[np.ndarray, np.ndarray]:
            candidates = [key]
            if key.startswith("action."):
                candidates.append(key[len("action."):])
            else:
                candidates.append(f"action.{key}")

            stats = None
            matched_key = None
            for candidate in candidates:
                stats = emb_meta.get(candidate)
                if stats is not None:
                    matched_key = candidate
                    break
            if stats is None:
                available = ", ".join(sorted(str(k) for k in emb_meta.keys()))
                raise KeyError(
                    f"Missing action normalization stats for {key!r}. "
                    f"Tried {candidates}; available action stats: [{available}]"
                )

            try:
                q01 = np.asarray(stats["q01"], dtype=np.float32).reshape(-1)
                q99 = np.asarray(stats["q99"], dtype=np.float32).reshape(-1)
            except KeyError as exc:
                raise KeyError(
                    f"Action stats for {matched_key!r} must contain q01 and q99; "
                    f"found keys {sorted(stats.keys())}"
                ) from exc

            if q01.shape != (width,) or q99.shape != (width,):
                raise ValueError(
                    f"Action stats for {matched_key!r} have incompatible shape: "
                    f"q01={q01.shape}, q99={q99.shape}, expected ({width},)"
                )
            if not (np.isfinite(q01).all() and np.isfinite(q99).all()):
                raise ValueError(f"Action stats for {matched_key!r} contain non-finite values")
            return q01, q99

        def _denorm(slice_pred: np.ndarray, key: str, lo: int, hi: int) -> None:
            q01, q99 = _stats_for_key(key, hi - lo)
            span = q99 - q01
            zero_span = span == 0
            safe_span = span.copy()
            safe_span[zero_span] = 1.0
            T = min(slice_pred.shape[0], self.action_horizon)
            denormed = (slice_pred[:T] + 1.0) / 2.0 * safe_span + q01
            if np.any(zero_span):
                denormed[:, zero_span] = q01[zero_span]
            out[:T, lo:hi] = denormed

        p0 = pred[0, 0]                                       # [T_a, 8]
        p1 = pred[0, 1]                                       # [T_a, 8]
        _denorm(p0[:, :7],  "action.panda0_joint_pos",    0, 7)
        _denorm(p0[:, 7:8], "action.panda0_gripper_pos",  7, 8)
        _denorm(p1[:, :7],  "action.panda1_joint_pos",    8, 15)
        _denorm(p1[:, 7:8], "action.panda1_gripper_pos", 15, 16)
        self._add_reference_state_for_relative_keys(out, qpos)
        self._last_action_debug["action_physical_pre_binarize"] = out.copy()
        self._binarize_gripper_targets(out)
        self._last_action_debug["action_physical_final"] = out.copy()
        return out


class BimanualWebsocketServer:
    def __init__(
        self,
        policy: BimanualPolicy,
        host: str = "0.0.0.0",
        port: int = 5001,
    ):
        self._policy = policy
        self._host = host
        self._port = port
        self._request_lock: asyncio.Lock | None = None
        self._cfg = BimanualServerConfig(
            num_frames=policy.num_frames,
            action_horizon=policy.action_horizon,
            image_resolution=(policy.image_h, policy.image_w),
            action_dim=policy.action_dim,
            return_action_debug=policy.return_action_debug,
            action_representation=policy.action_representation,
            gripper_convention=policy._resolved_gripper_convention(),
            gripper_close_value=policy._gripper_close_target(),
            gripper_open_value=policy._gripper_open_target(),
            shared_global_wrist_window_mode=policy.shared_global_wrist_window_mode,
            reset_causal_state_each_infer=policy.reset_causal_state_each_infer,
            video_pred_rollout_mode=policy.video_pred_rollout_mode,
        )
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        self._request_lock = asyncio.Lock()
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            logging.info(
                "Bimanual policy server listening on ws://%s:%d",
                self._host, self._port,
            )
            await server.serve_forever()

    async def _handler(self, websocket):
        logging.info("Connection from %s opened", websocket.remote_address)
        pack, unpack = _make_packer()
        await websocket.send(pack(dataclasses.asdict(self._cfg)))

        while True:
            try:
                obs = unpack(await websocket.recv())
                endpoint = obs.pop("endpoint", "infer")
                assert self._request_lock is not None
                async with self._request_lock:
                    if endpoint == "reset":
                        reply: Any = self._policy.reset(obs)
                    else:
                        reply = self._policy.infer(obs)
                await websocket.send(pack(reply))
            except websockets.ConnectionClosed:
                logging.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                tb = traceback.format_exc()
                logging.error("Inference error:\n%s", tb)
                await websocket.send(tb)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error.",
                )
                raise


class DistributedPolicyProxy:
    """Rank-0 proxy that keeps non-websocket ranks in lockstep.

    The WAN action head already implements two-way inference parallelism
    internally. The websocket server must therefore make every rank enter
    ``policy.reset`` / ``policy.infer`` in the same order, while only rank 0
    reads client messages and sends replies.
    """

    def __init__(
        self,
        policy: BimanualPolicy,
        *,
        rank: int,
        world_size: int,
        dist_module: Any | None = None,
    ):
        self._policy = policy
        self._rank = int(rank)
        self._world_size = int(world_size)
        if dist_module is None:
            import torch.distributed as dist_module
        self._dist = dist_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    @property
    def is_leader(self) -> bool:
        return self._rank == 0

    def _broadcast_command(self, command: dict[str, Any] | None) -> dict[str, Any]:
        obj_list = [command if self.is_leader else None]
        self._dist.broadcast_object_list(obj_list, src=0)
        received = obj_list[0]
        if not isinstance(received, dict):
            raise RuntimeError(f"invalid distributed policy command: {received!r}")
        return received

    def _execute_command(self, command: dict[str, Any]) -> Any:
        endpoint = command.get("endpoint")
        payload = command.get("payload") or {}
        if endpoint == "reset":
            return self._policy.reset(payload)
        if endpoint == "infer":
            return self._policy.infer(payload)
        if endpoint == "shutdown":
            return None
        raise RuntimeError(f"unknown distributed policy endpoint: {endpoint!r}")

    def reset(self, info: dict) -> str:
        command = self._broadcast_command({"endpoint": "reset", "payload": info})
        return self._execute_command(command)

    def infer(self, obs: dict) -> dict:
        command = self._broadcast_command({"endpoint": "infer", "payload": obs})
        return self._execute_command(command)

    def worker_loop(self) -> None:
        if self.is_leader:
            raise RuntimeError("worker_loop must not run on distributed rank 0")
        logging.info(
            "Distributed policy worker rank %d/%d waiting for websocket commands",
            self._rank,
            self._world_size,
        )
        while True:
            command = self._broadcast_command(None)
            if command.get("endpoint") == "shutdown":
                logging.info("Distributed policy worker rank %d shutting down", self._rank)
                return
            self._execute_command(command)

    def shutdown_workers(self) -> None:
        if self._world_size <= 1 or not self.is_leader:
            return
        self._broadcast_command({"endpoint": "shutdown", "payload": {}})


def main():
    def _optional_float_env(name: str) -> float | None:
        value = os.environ.get(name, "").strip()
        return float(value) if value else None

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        required=True,
        help="LoRA training output dir (contains checkpoint-<step>/ and "
             "experiment_cfg/)",
    )
    parser.add_argument("--ckpt-setting", default="checkpoint-10")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument(
        "--inference-parallel-size",
        type=int,
        default=int(os.environ.get("DREAMZERO_INFERENCE_PARALLEL_SIZE", "0") or 0),
        help="Serving-time inference parallel size. Use 0 to infer from "
             "torchrun WORLD_SIZE. Values 1 and 2 are supported; 2 runs the "
             "WAN action head's CFG branches across two ranks while rank 0 "
             "owns the websocket server.",
    )
    # Match the LeRobot v2 raw mp4 dimensions (see
    # scripts/data/robofactory_to_lerobot_v2.py); the transform chain
    # checks input resolution exactly and the in-chain Resize downsizes
    # to the model target afterwards.
    parser.add_argument("--image-h", type=int, default=240)
    parser.add_argument("--image-w", type=int, default=320)
    parser.add_argument(
        "--model-image-h",
        type=int,
        default=None,
        help="Optional model-side resize height. Leave unset to use the "
             "checkpoint's training config.",
    )
    parser.add_argument(
        "--model-image-w",
        type=int,
        default=None,
        help="Optional model-side resize width. Must be set with "
             "--model-image-h. Leave unset to use the checkpoint's training config.",
    )
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--action-horizon", type=int, default=24)
    parser.add_argument(
        "--save-video-pred",
        action="store_true",
        help="VAE-decode the model's predicted video at each infer() call "
             "and write one mp4 per agent. Adds ~5-15s per inference; use "
             "only for diagnostics, not when chasing closed-loop speed.",
    )
    parser.add_argument(
        "--video-pred-dir",
        default=None,
        help="Output dir for predicted-video mp4s; default "
             "{ckpt_dir}/video_pred/session_{sid}.",
    )
    parser.add_argument(
        "--video-pred-rollout-mode",
        default=os.environ.get("DREAMZERO_VIDEO_PRED_ROLLOUT_MODE", "action"),
        choices=_VIDEO_PRED_ROLLOUT_MODES,
        help="Which model rollout supplies saved predicted-video latents. "
             "'action' uses the same rollout that produced the control action; "
             "'noncausal' keeps control action unchanged but runs a second "
             "noncausal flowmatch pass only for diagnostic video artifacts.",
    )
    parser.add_argument(
        "--return-action-debug",
        action="store_true",
        help="Include raw and clipped normalized action chunks in infer replies.",
    )
    parser.add_argument(
        "--prompt-override",
        default=None,
        help="Optional prompt sent to the model regardless of the client prompt. "
             "If unset, DREAMZERO_PROMPT_OVERRIDE is honored when present.",
    )
    parser.add_argument(
        "--gripper-binarize-threshold",
        type=float,
        default=None,
        help="Optional physical gripper threshold in [-1, 1]. When set, "
             "targets below the threshold become the convention-specific "
             "close command and targets at/above it become the open command. "
             "If unset, DREAMZERO_GRIPPER_BINARIZE_THRESHOLD is honored when present.",
    )
    parser.add_argument(
        "--gripper-convention",
        default=os.environ.get("DREAMZERO_GRIPPER_CONVENTION", "auto"),
        choices=("auto", "robotwin", "robofactory"),
        help="Simulator gripper convention for diagnostic binarize/override. "
             "auto selects robotwin for robotwin metadata and robofactory for "
             "robofactory metadata.",
    )
    parser.add_argument(
        "--gripper-override",
        default=os.environ.get("DREAMZERO_GRIPPER_OVERRIDE", "none"),
        choices=("none", "close-after-infer", "close-all"),
        help="Diagnostic-only gripper override. Defaults to no override.",
    )
    parser.add_argument(
        "--gripper-close-after-infer",
        type=int,
        default=int(os.environ.get("DREAMZERO_GRIPPER_CLOSE_AFTER_INFER", "0") or 0),
        help="With --gripper-override=close-after-infer, force both grippers "
             "closed starting at this infer index.",
    )
    parser.add_argument(
        "--gripper-close-value",
        type=float,
        default=_optional_float_env("DREAMZERO_GRIPPER_CLOSE_VALUE"),
        help="Physical gripper target used by diagnostic close overrides. "
             "Defaults to 0.0 for RoboTwin and -1.0 for RoboFactory.",
    )
    parser.add_argument(
        "--gripper-force-open-until-infer",
        type=int,
        default=int(os.environ.get("DREAMZERO_GRIPPER_FORCE_OPEN_UNTIL_INFER", "0") or 0),
        help="Force both gripper targets open for infer indices below this "
             "1-based boundary. For example, 6 keeps infer 1-5 open and lets "
             "infer 6 close if the model predicts close.",
    )
    parser.add_argument(
        "--shared-global-wrist-window-mode",
        default=os.environ.get(
            "DREAMZERO_SHARED_GLOBAL_WRIST_WINDOW_MODE",
            "history-current-first",
        ),
        choices=_SHARED_GLOBAL_WRIST_WINDOW_MODES,
        help="Eval diagnostic for shared-global checkpoints. history-current-first "
             "keeps the current wrist frame at index 0 and appends rolling "
             "history; repeat-current matches the historical server behavior; "
             "history-chronological keeps wrist history unchanged.",
    )
    parser.add_argument(
        "--reset-causal-state-each-infer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reset the WAN causal video/KV state before every infer() call. "
             "Useful for closed-loop replanning where each request supplies the "
             "current observation window. If unset, "
             "DREAMZERO_RESET_CAUSAL_STATE_EACH_INFER is honored.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )
    dist_ctx = _init_distributed_serving(args.inference_parallel_size)
    logging.info(
        "Distributed serving context: rank=%d local_rank=%d world_size=%d "
        "inference_parallel_size=%d device=%s leader=%s",
        dist_ctx.rank,
        dist_ctx.local_rank,
        dist_ctx.world_size,
        dist_ctx.inference_parallel_size,
        dist_ctx.device,
        dist_ctx.is_leader,
    )

    policy = BimanualPolicy(
        ckpt_dir=args.ckpt_dir,
        ckpt_setting=args.ckpt_setting,
        image_h=args.image_h,
        image_w=args.image_w,
        model_image_h=args.model_image_h,
        model_image_w=args.model_image_w,
        num_frames=args.num_frames,
        action_horizon=args.action_horizon,
        save_video_pred=args.save_video_pred,
        video_pred_dir=args.video_pred_dir,
        video_pred_rollout_mode=args.video_pred_rollout_mode,
        return_action_debug=args.return_action_debug,
        prompt_override=args.prompt_override,
        gripper_binarize_threshold=args.gripper_binarize_threshold,
        gripper_override=args.gripper_override,
        gripper_close_after_infer=args.gripper_close_after_infer,
        gripper_close_value=args.gripper_close_value,
        gripper_force_open_until_infer=args.gripper_force_open_until_infer,
        gripper_convention=args.gripper_convention,
        shared_global_wrist_window_mode=args.shared_global_wrist_window_mode,
        reset_causal_state_each_infer=args.reset_causal_state_each_infer,
        device=dist_ctx.device,
        device_mesh=dist_ctx.device_mesh,
    )
    proxy: DistributedPolicyProxy | None = None
    serving_policy: Any = policy
    if dist_ctx.enabled:
        import torch.distributed as dist

        dist.barrier()
        proxy = DistributedPolicyProxy(
            policy,
            rank=dist_ctx.rank,
            world_size=dist_ctx.world_size,
        )
        if not dist_ctx.is_leader:
            proxy.worker_loop()
            return
        serving_policy = proxy

    server = BimanualWebsocketServer(serving_policy, host=args.host, port=args.port)
    try:
        server.serve_forever()
    finally:
        if proxy is not None:
            try:
                proxy.shutdown_workers()
            except Exception:
                logging.exception("Failed to broadcast distributed worker shutdown")


if __name__ == "__main__":
    main()
