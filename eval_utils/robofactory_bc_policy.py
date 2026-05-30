"""Lightweight non-grid RoboFactory BC policy utilities.

This module is intentionally independent of the DreamZero VLA stack. It
uses the same websocket observation/action protocol as
``eval_utils.bimanual_policy_server`` but fuses the three RoboFactory
camera views as separate streams instead of stitching them into a grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch import nn


CAMERA_KEYS = ("global", "agent0", "agent1")
GRIPPER_DIMS = (7, 15)


def resize_rgb(frame: np.ndarray, image_size: int) -> np.ndarray:
    """RGB uint8 ``[H, W, 3]`` -> RGB uint8 ``[3, S, S]``."""
    frame = np.asarray(frame, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"expected RGB frame [H,W,3], got {frame.shape}")
    out = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return out.transpose(2, 0, 1).copy()


def images_to_tensor(frames_chw: np.ndarray) -> torch.Tensor:
    """Uint8 ``[V, 3, S, S]`` -> float tensor in ``[-1, 1]``."""
    x = torch.as_tensor(frames_chw, dtype=torch.float32)
    return x.div(127.5).sub(1.0)


class MultiViewBCPolicyNet(nn.Module):
    """Small BC network with separate view encodings.

    Images are encoded per view with a shared CNN, a learned view embedding
    is added, and the resulting view features are concatenated. This keeps
    camera identity explicit without any spatial grid composition.
    """

    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 16,
        horizon: int = 24,
        num_views: int = 3,
        image_size: int = 96,
        hidden_dim: int = 512,
        use_images: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_views = num_views
        self.image_size = image_size
        self.use_images = use_images

        view_dim = 128
        if use_images:
            self.view_encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
                nn.GroupNorm(4, 32),
                nn.SiLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, 64),
                nn.SiLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, 128),
                nn.SiLU(),
                nn.Conv2d(128, view_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, view_dim),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.view_embedding = nn.Parameter(torch.zeros(num_views, view_dim))
            nn.init.normal_(self.view_embedding, mean=0.0, std=0.02)
            visual_dim = num_views * view_dim
        else:
            self.view_encoder = None
            self.view_embedding = None
            visual_dim = 0

        in_dim = state_dim + 1 + visual_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, horizon * action_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        progress: torch.Tensor,
        images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return normalized action chunk ``[B, horizon, action_dim]``."""
        feats = [state, progress.reshape(progress.shape[0], 1)]
        if self.use_images:
            if images is None:
                raise ValueError("images are required when use_images=True")
            if images.ndim != 5:
                raise ValueError(f"expected images [B,V,3,S,S], got {tuple(images.shape)}")
            b, v, c, h, w = images.shape
            if v != self.num_views or c != 3:
                raise ValueError(f"expected {self.num_views} RGB views, got {tuple(images.shape)}")
            view = self.view_encoder(images.reshape(b * v, c, h, w))
            view = view.reshape(b, v, -1)
            view = view + self.view_embedding.unsqueeze(0).to(dtype=view.dtype)
            feats.append(view.reshape(b, -1))
        out = self.mlp(torch.cat(feats, dim=1))
        return out.reshape(-1, self.horizon, self.action_dim)


def _to_tensor(value, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


class RoboFactoryBCPolicy:
    """Runtime wrapper used by the BC websocket server."""

    def __init__(
        self,
        ckpt_path: str | Path,
        device: str | None = None,
        progress_steps: int | None = None,
    ) -> None:
        self.ckpt_path = Path(ckpt_path)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        self.image_size = int(cfg["image_size"])
        self.horizon = int(cfg["horizon"])
        self.action_dim = int(cfg["action_dim"])
        self.max_episode_steps = int(cfg.get("max_episode_steps", 300))
        self.progress_steps = int(progress_steps or cfg.get("progress_steps", self.max_episode_steps))
        self.replan_every = int(cfg.get("replan_every", 8))
        self.use_images = bool(cfg.get("use_images", True))

        self.model = MultiViewBCPolicyNet(
            state_dim=int(cfg["state_dim"]),
            action_dim=self.action_dim,
            horizon=self.horizon,
            num_views=int(cfg["num_views"]),
            image_size=self.image_size,
            hidden_dim=int(cfg["hidden_dim"]),
            use_images=self.use_images,
        )
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

        stats = ckpt["stats"]
        self.state_mean = _to_tensor(stats["state_mean"], self.device)
        self.state_std = _to_tensor(stats["state_std"], self.device)
        self.action_mean = _to_tensor(stats["action_mean"], self.device)
        self.action_std = _to_tensor(stats["action_std"], self.device)
        self.action_min = np.asarray(stats["action_min"], dtype=np.float32)
        self.action_max = np.asarray(stats["action_max"], dtype=np.float32)
        self.sessions: dict[str, dict] = {}

    def reset(self, info: dict) -> str:
        sid = info.get("session_id", "")
        self.sessions[sid] = {"infer_idx": 0}
        return "reset successful"

    def _session_step(self, obs: dict, sess: dict) -> int:
        if "step" in obs:
            return int(obs["step"])
        return int(sess.get("infer_idx", 0)) * self.replan_every

    def infer(self, obs: dict) -> dict:
        sid = obs.get("session_id", "")
        sess = self.sessions.setdefault(sid, {"infer_idx": 0})
        step = self._session_step(obs, sess)
        progress = np.float32(np.clip(step / max(self.progress_steps - 1, 1), 0.0, 1.0))

        qpos = np.asarray(obs["qpos"], dtype=np.float32).reshape(-1)
        if qpos.shape != (16,):
            raise ValueError(f"expected qpos shape (16,), got {qpos.shape}")
        state = ((_to_tensor(qpos, self.device) - self.state_mean) / self.state_std).unsqueeze(0)
        prog = _to_tensor([progress], self.device)

        images_t = None
        if self.use_images:
            frames = np.stack(
                [
                    resize_rgb(obs["head_rgb"], self.image_size),
                    resize_rgb(obs["left_rgb"], self.image_size),
                    resize_rgb(obs["right_rgb"], self.image_size),
                ],
                axis=0,
            )
            images_t = images_to_tensor(frames).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            pred_norm = self.model(state, prog, images_t)[0]
            pred = pred_norm * self.action_std.reshape(1, -1) + self.action_mean.reshape(1, -1)
        action = pred.detach().float().cpu().numpy()
        action = np.clip(action, self.action_min[None], self.action_max[None])
        action[:, GRIPPER_DIMS] = np.clip(action[:, GRIPPER_DIMS], -1.0, 1.0)
        sess["infer_idx"] = int(sess.get("infer_idx", 0)) + 1
        return {"action_chunk": action.astype(np.float32)}


def iter_episode_files(data_root: Path) -> Iterable[Path]:
    return sorted((data_root / "data").glob("chunk-*/episode_*.parquet"))
