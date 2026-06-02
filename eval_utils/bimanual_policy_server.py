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
        num_frames: int = 33,
        action_horizon: int = 24,
        action_dim: int = 16,
        save_video_pred: bool = False,
        video_pred_dir: str | None = None,
        return_action_debug: bool = False,
        prompt_override: str | None = None,
        gripper_binarize_threshold: float | None = None,
    ):
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_setting = ckpt_setting
        self.image_h = image_h
        self.image_w = image_w
        self.num_frames = num_frames
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.save_video_pred = save_video_pred
        self.video_pred_dir = Path(video_pred_dir) if video_pred_dir else None
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
        self._last_action_debug: dict[str, np.ndarray] = {}
        self._relative_action = False
        self._relative_action_per_horizon = False
        self._relative_action_keys: set[str] = set()
        self.action_representation = "robotwin_delta"

        self._sessions: dict[str, dict] = {}
        self._load()

    def _effective_prompt(self, prompt: str | None) -> str:
        if self.prompt_override:
            return self.prompt_override
        return prompt or ""

    def _load(self) -> None:
        import torch
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        from safetensors.torch import load_file

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
        dropped_mismatched: dict[str, tuple] = {}

        def _filter_shape_mismatches(sd: dict) -> dict:
            kept = {}
            for k, v in sd.items():
                ref = model_state.get(k)
                if ref is not None and tuple(ref.shape) != tuple(v.shape):
                    dropped_mismatched[k] = (tuple(v.shape), tuple(ref.shape))
                    continue
                kept[k] = v
            return kept

        pretrained_index = pretrained_dir / "model.safetensors.index.json"
        if pretrained_index.is_file():
            with open(pretrained_index) as f:
                p_index = json.load(f)
            for shard_file in sorted(set(p_index["weight_map"].values())):
                shard_sd = load_file(str(pretrained_dir / shard_file))
                shard_sd = _filter_shape_mismatches(shard_sd)
                model.load_state_dict(shard_sd, strict=False)
                del shard_sd
                gc.collect()
        else:
            pretrained_safe = pretrained_dir / "model.safetensors"
            if not pretrained_safe.is_file():
                raise FileNotFoundError(
                    f"No model.safetensors[.index.json] under {pretrained_dir}"
                )
            sd = _filter_shape_mismatches(load_file(str(pretrained_safe)))
            model.load_state_dict(sd, strict=False)
        if dropped_mismatched:
            logging.info(
                "Step 2/4: dropped %d shape-mismatched tensor(s) "
                "(expected for multi-agent deltas vs DROID).",
                len(dropped_mismatched),
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
                shard_state_dict = _filter_shape_mismatches(shard_state_dict)
                model.load_state_dict(shard_state_dict, strict=False)
                del shard_state_dict
                gc.collect()
        elif weight_path.is_file():
            state_dict = load_file(str(weight_path))
            state_dict = _filter_shape_mismatches(state_dict)
            model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(
                f"No model.safetensors[.index.json] under "
                f"{self.ckpt_dir / self.ckpt_setting}"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = model.to(device=device, dtype=self._dtype)
        model.eval()
        self._model = model
        logging.info("VLA loaded onto %s in %s", device, self._dtype)

        try:
            transforms = instantiate(self._cfg.transforms)
            self._transform = transforms["robofactory"]
            # The transform pipeline needs normalization stats + modality
            # metadata before it can be applied. Mirrors sim_policy.py:365.
            from groot.vla.data.schema.lerobot import DatasetMetadata

            if "robofactory" in self._metadata:
                metadata = DatasetMetadata.model_validate(self._metadata["robofactory"])
                # If the action head specifies a target video resolution, propagate it.
                ah_cfg = getattr(getattr(self._model, "action_head", None), "config", None)
                if ah_cfg is not None:
                    target_h = getattr(ah_cfg, "target_video_height", None)
                    target_w = getattr(ah_cfg, "target_video_width", None)
                    if target_h is not None and target_w is not None and metadata.modalities.video:
                        for key in metadata.modalities.video.keys():
                            metadata.modalities.video[key].resolution = (int(target_w), int(target_h))
                self._transform.set_metadata(metadata)
                logging.info("Bimanual transform ready (metadata bound)")
            else:
                logging.warning(
                    "metadata.json lacks 'robofactory' key — transform will fail "
                    "on first infer(). Found keys: %s",
                    list(self._metadata.keys()),
                )
        except Exception:
            self._transform = None
            logging.exception(
                "transforms.robofactory failed to instantiate — server will "
                "still start, but infer() will raise."
            )

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
        return "reset successful"

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
        if self._uses_shared_global():
            # Shared-global checkpoints train ``video_global`` as the
            # current scene observation repeated across the conditioning
            # window. Keep inference identical for all three camera
            # streams instead of feeding a rolling past window into the
            # clean conditioning path.
            global_video = np.repeat(head[None], self.num_frames, axis=0)
            agent0_video = np.repeat(lft[None], self.num_frames, axis=0)
            agent1_video = np.repeat(rgt[None], self.num_frames, axis=0)
        else:
            global_video = np.stack([h for (h, _, _) in history], axis=0)
            agent0_video = np.stack([l for (_, l, _) in history], axis=0)
            agent1_video = np.stack([r for (_, _, r) in history], axis=0)

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
            # in eval mode -- but the multi-agent inference path needs
            # them (at least for shape). Flip every sub-transform's
            # ``training`` flag (the ComposedModalityTransform attribute
            # alone does NOT propagate to children) for this call: our
            # placeholder action zeros flow through harmlessly because
            # the model ignores their values and starts denoising from
            # noise. Restore after.
            prev_training = getattr(self._transform, "training", False)
            self._transform.train()
            try:
                normalized_inputs = self._transform.apply(batch)
            finally:
                if prev_training:
                    self._transform.train()
                else:
                    self._transform.eval()

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

        if self.save_video_pred:
            try:
                self._dump_video_pred(sess, sid)
                self._dump_conditioning_pred(sess, sid)
            except Exception:
                logging.exception("save_video_pred failed; continuing without")
        sess["infer_idx"] = sess.get("infer_idx", 0) + 1

        flat_action = self._denorm_action(outputs, qpos)
        self._log_action_summary(sess, sid, flat_action)
        reply = {"action_chunk": flat_action.astype(np.float32)}
        if self.return_action_debug:
            reply.update(
                {
                    key: value.astype(np.float32)
                    for key, value in self._last_action_debug.items()
                }
            )
        return reply

    def _dump_video_pred(self, sess: dict, sid: str) -> None:
        """VAE-decode the action_head's last denoised video latents and
        write one mp4 per agent. Called from infer() when save_video_pred
        is on. Adds ~5-15s per call (heavy VAE decode); diagnostic only.
        """
        import torch
        import av

        action_head = self._model.action_head
        latents = getattr(action_head, "_last_video_pred", None)
        if latents is None:
            logging.warning("action_head._last_video_pred missing; "
                            "make sure _get_action_multi_agent stashes it")
            return
        # latents: [B=1, P, C_lat, F_lat, H_lat, W_lat] in self._dtype
        B, P, C_lat, F_lat, H_lat, W_lat = latents.shape
        # VAE.decode expects [B, C, T, H, W]; fold P into batch.
        lat_bp = latents.reshape(B * P, C_lat, F_lat, H_lat, W_lat)
        with torch.inference_mode():
            frames = action_head.vae.decode(
                lat_bp.to(self._device, dtype=self._dtype),
                tiled=action_head.tiled,
                tile_size=(action_head.tile_size_height,
                           action_head.tile_size_width),
                tile_stride=(action_head.tile_stride_height,
                             action_head.tile_stride_width),
            )                                                # [B*P, C, T, H, W]
        frames = frames.float()
        frames = ((frames + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        frames = frames.cpu().numpy()                        # [B*P, C, T, H, W]
        # -> [P, T, H, W, C]; B is always 1 at inference.
        frames = frames.transpose(0, 2, 3, 4, 1).reshape(
            B, P, -1, frames.shape[3], frames.shape[4], 3
        )[0]

        out_dir = self.video_pred_dir or (self.ckpt_dir / "video_pred")
        out_dir = Path(out_dir) / f"session_{sid[:12]}"
        out_dir.mkdir(parents=True, exist_ok=True)
        step = sess.get("infer_idx", 0)
        T, H, W = frames.shape[1], frames.shape[2], frames.shape[3]
        for p in range(P):
            out_path = out_dir / f"step{step:04d}_agent{p}.mp4"
            with av.open(str(out_path), mode="w") as container:
                stream = container.add_stream("h264", rate=20)
                stream.width = W
                stream.height = H
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "23"}
                for t in range(T):
                    img = frames[p, t]                       # [H, W, 3] uint8
                    av_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                    for packet in stream.encode(av_frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
        logging.info("wrote predicted video: step=%d session=%s dir=%s",
                     step, sid[:12], out_dir)

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

    def _write_decoded_video_set(self, frames, out_dir: Path, prefix: str) -> None:
        import av

        out_dir.mkdir(parents=True, exist_ok=True)
        P, T, H, W = frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3]
        for p in range(P):
            out_path = out_dir / f"{prefix}_agent{p}.mp4"
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

    def _dump_conditioning_pred(self, sess: dict, sid: str) -> None:
        """Decode the I2V conditioning latents once for debugging.

        This checks whether the VAE encode/decode path for the current
        observation is sane. If these files are structured while
        ``stepXXXX_agent*.mp4`` is noise, the denoising/video-conditioning
        path is the problem, not the VAE diagnostic decode path.
        """
        step = sess.get("infer_idx", 0)
        if step != 0 or sess.get("condition_debug_dumped", False):
            return

        action_head = self._model.action_head
        out_dir = self.video_pred_dir or (self.ckpt_dir / "video_pred")
        out_dir = Path(out_dir) / f"session_{sid[:12]}" / "conditioning"

        clean_latents = getattr(action_head, "_last_clean_video_cond", None)
        if clean_latents is not None:
            clean_frames = self._decode_latent_video(clean_latents)
            self._write_decoded_video_set(
                clean_frames, out_dir, f"step{step:04d}_clean_x"
            )
            logging.info(
                "wrote conditioning clean_x video: step=%d session=%s dir=%s",
                step, sid[:12], out_dir,
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
                    y_frames, out_dir, f"step{step:04d}_y_latent"
                )
                logging.info(
                    "wrote conditioning y-latent video: step=%d session=%s dir=%s",
                    step, sid[:12], out_dir,
                )

        sess["condition_debug_dumped"] = True

    def _flatten_bimanual_action_pred(self, pred: np.ndarray) -> np.ndarray:
        """Flatten ``[P=2, T, D=8]`` per-arm action to client ``[T, 16]``."""
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
            return subkey in self._relative_action_keys
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

    def _action_stats_for_key(self, key: str, width: int) -> tuple[np.ndarray, np.ndarray]:
        emb_meta = (
            self._metadata.get("robofactory", {})
            .get("statistics", {})
            .get("action", {})
        )
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

    def _binarize_gripper_targets(self, out: np.ndarray) -> None:
        if self.gripper_binarize_threshold is None:
            return
        threshold = float(self.gripper_binarize_threshold)
        for dim, key in (
            (7, "action.panda0_gripper_pos"),
            (15, "action.panda1_gripper_pos"),
        ):
            q01, q99 = self._action_stats_for_key(key, 1)
            close_value = float(q01[0])
            open_value = float(q99[0])
            out[:, dim] = np.where(out[:, dim] >= threshold, open_value, close_value)

    def _log_action_summary(self, sess: dict, sid: str, action: np.ndarray) -> None:
        """Emit compact gripper diagnostics for closed-loop eval logs."""
        infer_idx = int(sess.get("infer_idx", 0))
        prompt = str(sess.get("prompt", ""))
        close_threshold = (
            float(self.gripper_binarize_threshold)
            if self.gripper_binarize_threshold is not None
            else 0.0
        )

        def _gripper_summary(values: np.ndarray) -> str:
            values = np.asarray(values, dtype=np.float32).reshape(-1)
            first = np.array2string(
                values[: min(8, values.shape[0])],
                precision=3,
                separator=",",
            )
            close_count = int(np.sum(values < close_threshold))
            return (
                f"min={float(values.min()):.3f} max={float(values.max()):.3f} "
                f"mean={float(values.mean()):.3f} "
                f"close_lt_{close_threshold:.3f}={close_count}/{values.shape[0]} "
                f"first={first}"
            )

        logging.info(
            "Action summary session=%s infer_idx=%d prompt=%r "
            "representation=%s gripper_binarize_threshold=%s "
            "left_gripper[%s] right_gripper[%s]",
            sid[:12],
            infer_idx,
            prompt,
            self.action_representation,
            self.gripper_binarize_threshold,
            _gripper_summary(action[:, 7]),
            _gripper_summary(action[:, 15]),
        )

    def _denorm_action(self, outputs, qpos: np.ndarray) -> np.ndarray:
        """Take model output ``action_pred [B=1, P=2, T_a, D_per_arm=8]``
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
        if P != 2 or D_per_arm != 8:
            raise ValueError(
                f"Expected P=2 arms with D=8 per arm; got P={P} D={D_per_arm}"
            )
        self._last_action_debug = {
            "action_norm_raw": self._flatten_bimanual_action_pred(raw_pred[0]),
            "action_norm_clipped": self._flatten_bimanual_action_pred(pred[0]),
        }

        out = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
        def _stats_for_key(key: str, width: int) -> tuple[np.ndarray, np.ndarray]:
            return self._action_stats_for_key(key, width)

        def _denorm(slice_pred: np.ndarray, key: str, lo: int, hi: int) -> None:
            q01, q99 = _stats_for_key(key, hi - lo)
            span = q99 - q01
            span[span == 0] = 1.0
            T = min(slice_pred.shape[0], self.action_horizon)
            out[:T, lo:hi] = (slice_pred[:T] + 1.0) / 2.0 * span + q01

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
        self._cfg = BimanualServerConfig(
            num_frames=policy.num_frames,
            action_horizon=policy.action_horizon,
            image_resolution=(policy.image_h, policy.image_w),
            action_dim=policy.action_dim,
            return_action_debug=policy.return_action_debug,
            action_representation=policy.action_representation,
        )
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
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


def main():
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
    # Match the LeRobot v2 raw mp4 dimensions (see
    # scripts/data/robofactory_to_lerobot_v2.py); the transform chain
    # checks input resolution exactly and the in-chain Resize downsizes
    # to the model target afterwards.
    parser.add_argument("--image-h", type=int, default=240)
    parser.add_argument("--image-w", type=int, default=320)
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
        help="Optional gripper threshold in [-1, 1]. When set, "
             "gripper targets below the threshold become the metadata q01 "
             "(close) and targets at/above it become metadata q99 (open), "
             "preserving both 0/1 and -1/+1 gripper conventions. If unset, "
             "DREAMZERO_GRIPPER_BINARIZE_THRESHOLD is honored when present.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    policy = BimanualPolicy(
        ckpt_dir=args.ckpt_dir,
        ckpt_setting=args.ckpt_setting,
        image_h=args.image_h,
        image_w=args.image_w,
        num_frames=args.num_frames,
        action_horizon=args.action_horizon,
        save_video_pred=args.save_video_pred,
        video_pred_dir=args.video_pred_dir,
        return_action_debug=args.return_action_debug,
        prompt_override=args.prompt_override,
        gripper_binarize_threshold=args.gripper_binarize_threshold,
    )
    server = BimanualWebsocketServer(policy, host=args.host, port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
