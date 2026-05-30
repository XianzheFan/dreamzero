"""Train a lightweight non-grid RoboFactory bimanual BC policy.

The policy consumes the three RoboFactory cameras as separate views
(``global``, ``agent0``, ``agent1``) plus current qpos and elapsed rollout
progress, then predicts a 24-step absolute action chunk. This is meant as
a fast closed-loop baseline for LiftBarrier while the heavier DreamZero
multi-agent visual-conditioning path is being fixed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_utils.robofactory_bc_policy import (
    CAMERA_KEYS,
    GRIPPER_DIMS,
    MultiViewBCPolicyNet,
)


VIDEO_DIRS = {
    "global": "observation.images.global",
    "agent0": "observation.images.agent0",
    "agent1": "observation.images.agent1",
}


@dataclass
class TrainConfig:
    data_root: str
    output_dir: str
    image_size: int = 96
    horizon: int = 24
    state_dim: int = 16
    action_dim: int = 16
    num_views: int = 3
    hidden_dim: int = 512
    use_images: bool = True
    batch_size: int = 128
    epochs: int = 80
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.15
    seed: int = 42
    gripper_weight: float = 6.0
    first_action_weight: float = 2.0
    gripper_bce_weight: float = 0.5
    max_episode_steps: int = 300
    progress_steps: int = 0
    replan_every: int = 8


def _episode_index(path: Path) -> int:
    return int(path.stem.split("_")[1])


def _video_path(data_root: Path, episode_index: int, camera: str) -> Path:
    chunk_dir = f"chunk-{episode_index // 1000:03d}"
    return (
        data_root
        / "videos"
        / chunk_dir
        / VIDEO_DIRS[camera]
        / f"episode_{episode_index:06d}.mp4"
    )


def _read_video(path: Path, image_size: int, expected_len: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"failed to open video {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
        frames.append(rgb.transpose(2, 0, 1).copy())
    cap.release()
    if not frames:
        raise RuntimeError(f"video had no frames: {path}")
    arr = np.stack(frames, axis=0).astype(np.uint8)
    if arr.shape[0] < expected_len:
        pad = np.repeat(arr[-1:], expected_len - arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    return arr[:expected_len]


class RoboFactoryBCDemos(Dataset):
    def __init__(
        self,
        data_root: Path,
        episode_indices: list[int],
        image_size: int,
        horizon: int,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        use_images: bool,
    ) -> None:
        self.data_root = data_root
        self.image_size = image_size
        self.horizon = horizon
        self.state_mean = state_mean.astype(np.float32)
        self.state_std = state_std.astype(np.float32)
        self.action_mean = action_mean.astype(np.float32)
        self.action_std = action_std.astype(np.float32)
        self.use_images = use_images

        self.episodes: dict[int, dict] = {}
        self.samples: list[tuple[int, int]] = []
        files = {
            _episode_index(p): p
            for p in sorted((data_root / "data").glob("chunk-*/episode_*.parquet"))
        }
        for ep_idx in episode_indices:
            df = pd.read_parquet(files[ep_idx], columns=["action", "observation.state"])
            action = np.stack(df["action"].to_numpy()).astype(np.float32)
            state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
            entry = {"action": action, "state": state, "length": len(action)}
            if use_images:
                views = [
                    _read_video(_video_path(data_root, ep_idx, cam), image_size, len(action))
                    for cam in CAMERA_KEYS
                ]
                entry["images"] = np.stack(views, axis=1)  # [T, V, 3, S, S]
            self.episodes[ep_idx] = entry
            for t in range(len(action)):
                self.samples.append((ep_idx, t))

    def __len__(self) -> int:
        return len(self.samples)

    def _chunk(self, action: np.ndarray, t: int) -> np.ndarray:
        chunk = action[t : t + self.horizon]
        if len(chunk) < self.horizon:
            pad = np.repeat(action[-1:], self.horizon - len(chunk), axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        return chunk.astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx, t = self.samples[idx]
        ep = self.episodes[ep_idx]
        state = (ep["state"][t] - self.state_mean) / self.state_std
        action = self._chunk(ep["action"], t)
        action_norm = (action - self.action_mean) / self.action_std
        progress = np.float32(t / max(ep["length"] - 1, 1))
        out = {
            "state": torch.from_numpy(state.astype(np.float32)),
            "action": torch.from_numpy(action_norm.astype(np.float32)),
            "action_raw": torch.from_numpy(action.astype(np.float32)),
            "progress": torch.tensor(progress, dtype=torch.float32),
        }
        if self.use_images:
            images = ep["images"][t].astype(np.float32) / 127.5 - 1.0
            out["images"] = torch.from_numpy(images)
        return out


def _load_arrays(data_root: Path) -> tuple[list[int], np.ndarray, np.ndarray, dict[int, int]]:
    episode_indices: list[int] = []
    episode_lengths: dict[int, int] = {}
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for p in sorted((data_root / "data").glob("chunk-*/episode_*.parquet")):
        ep_idx = _episode_index(p)
        episode_indices.append(ep_idx)
        df = pd.read_parquet(p, columns=["action", "observation.state"])
        action = np.stack(df["action"].to_numpy()).astype(np.float32)
        episode_lengths[ep_idx] = int(len(action))
        actions.append(action)
        states.append(np.stack(df["observation.state"].to_numpy()).astype(np.float32))
    if not episode_indices:
        raise FileNotFoundError(f"no LeRobot parquet episodes under {data_root}")
    return episode_indices, np.concatenate(states, axis=0), np.concatenate(actions, axis=0), episode_lengths


def _split_episodes(indices: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    shuffled = list(indices)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    return train, val


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train(cfg: TrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    data_root = Path(cfg.data_root)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    episode_indices, all_states, all_actions, episode_lengths = _load_arrays(data_root)
    progress_steps = cfg.progress_steps or cfg.max_episode_steps
    state_mean = all_states.mean(axis=0)
    state_std = all_states.std(axis=0) + 1e-6
    action_mean = all_actions.mean(axis=0)
    action_std = all_actions.std(axis=0) + 1e-6
    action_min = all_actions.min(axis=0)
    action_max = all_actions.max(axis=0)

    train_eps, val_eps = _split_episodes(episode_indices, cfg.val_fraction, cfg.seed)
    print(f"episodes: train={len(train_eps)} val={len(val_eps)} frames={len(all_actions)}")
    print(f"progress_steps={progress_steps}")
    print(f"output: {output_dir}")

    train_ds = RoboFactoryBCDemos(
        data_root,
        train_eps,
        cfg.image_size,
        cfg.horizon,
        state_mean,
        state_std,
        action_mean,
        action_std,
        cfg.use_images,
    )
    val_ds = RoboFactoryBCDemos(
        data_root,
        val_eps,
        cfg.image_size,
        cfg.horizon,
        state_mean,
        state_std,
        action_mean,
        action_std,
        cfg.use_images,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiViewBCPolicyNet(
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        horizon=cfg.horizon,
        num_views=cfg.num_views,
        image_size=cfg.image_size,
        hidden_dim=cfg.hidden_dim,
        use_images=cfg.use_images,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    action_mean_t = torch.as_tensor(action_mean, dtype=torch.float32, device=device)
    action_std_t = torch.as_tensor(action_std, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_path = output_dir / "best.pt"
    latest_path = output_dir / "latest.pt"

    def run_epoch(loader: DataLoader, train_mode: bool) -> dict[str, float]:
        model.train(train_mode)
        totals = {"loss": 0.0, "mae": 0.0, "grip_acc": 0.0, "n": 0.0}
        for batch in loader:
            batch = _move(batch, device)
            pred = model(
                batch["state"],
                batch["progress"],
                batch.get("images"),
            )
            target = batch["action"]
            raw_target = batch["action_raw"]
            loss = F.smooth_l1_loss(pred, target)
            loss = loss + cfg.first_action_weight * F.smooth_l1_loss(
                pred[:, 0], target[:, 0]
            )
            loss = loss + cfg.gripper_weight * F.smooth_l1_loss(
                pred[:, :, GRIPPER_DIMS], target[:, :, GRIPPER_DIMS]
            )
            pred_raw = pred * action_std_t.reshape(1, 1, -1) + action_mean_t.reshape(1, 1, -1)
            close_target = (raw_target[:, :, GRIPPER_DIMS] < 0).float()
            close_logit = -4.0 * pred_raw[:, :, GRIPPER_DIMS]
            loss = loss + cfg.gripper_bce_weight * F.binary_cross_entropy_with_logits(
                close_logit, close_target
            )
            if train_mode:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            with torch.no_grad():
                raw_pred = pred_raw
                mae = (raw_pred - raw_target).abs().mean()
                grip_acc = (
                    (raw_pred[:, :, GRIPPER_DIMS] < 0)
                    == (raw_target[:, :, GRIPPER_DIMS] < 0)
                ).float().mean()
            bsz = float(target.shape[0])
            totals["loss"] += float(loss.detach().cpu()) * bsz
            totals["mae"] += float(mae.detach().cpu()) * bsz
            totals["grip_acc"] += float(grip_acc.detach().cpu()) * bsz
            totals["n"] += bsz
        n = max(totals.pop("n"), 1.0)
        return {k: v / n for k, v in totals.items()}

    def save(path: Path, epoch: int, val: dict[str, float]) -> None:
        save_config = asdict(cfg)
        save_config["progress_steps"] = int(progress_steps)
        torch.save(
            {
                "model": model.state_dict(),
                "config": {
                    **save_config,
                    "state_dim": cfg.state_dim,
                    "action_dim": cfg.action_dim,
                    "num_views": cfg.num_views,
                },
                "stats": {
                    "state_mean": state_mean.astype(np.float32),
                    "state_std": state_std.astype(np.float32),
                    "action_mean": action_mean.astype(np.float32),
                    "action_std": action_std.astype(np.float32),
                    "action_min": action_min.astype(np.float32),
                    "action_max": action_max.astype(np.float32),
                },
                "epoch": epoch,
                "val": val,
            },
            path,
        )

    log_path = output_dir / "train_log.jsonl"
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(train_loader, train_mode=True)
        with torch.no_grad():
            val_metrics = run_epoch(val_loader, train_mode=False)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_mae={val_metrics['mae']:.4f} "
            f"val_grip_acc={val_metrics['grip_acc']:.3f}",
            flush=True,
        )
        save(latest_path, epoch, val_metrics)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save(best_path, epoch, val_metrics)
    print(f"best checkpoint: {best_path}")
    return best_path


def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--image-size", type=int, default=96)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gripper-weight", type=float, default=6.0)
    ap.add_argument("--first-action-weight", type=float, default=2.0)
    ap.add_argument("--gripper-bce-weight", type=float, default=0.5)
    ap.add_argument("--max-episode-steps", type=int, default=300)
    ap.add_argument("--progress-steps", type=int, default=0)
    ap.add_argument("--replan-every", type=int, default=8)
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()
    return TrainConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        seed=args.seed,
        gripper_weight=args.gripper_weight,
        first_action_weight=args.first_action_weight,
        gripper_bce_weight=args.gripper_bce_weight,
        max_episode_steps=args.max_episode_steps,
        progress_steps=args.progress_steps,
        replan_every=args.replan_every,
        use_images=not args.no_images,
    )


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
