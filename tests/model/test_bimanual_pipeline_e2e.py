"""End-to-end smoke: BimanualDreamTransform -> collate -> WANPolicyHead.

Wires PR 9e's data-side fixes (per-agent V-tiling in ``_prepare_video``
+ P-preserving ``_apply_vlm_processing``) to PR 9d's
``_forward_multi_agent`` and confirms a real raw-video sample makes
it through the full pipeline:

  [T, V, H, W, C] raw video + [T_a, D] state/action
    --(BimanualDreamTransform)-->
  per-sample dict with [P, T, 2H, 2W, C] images and [P, ...] state/action
    --(trainer-style np.stack collate)-->
  batch with [B, P, ...] everything
    --(WANPolicyHead._forward_multi_agent)-->
  finite scalar loss

Heavy encoders (T5 / CLIP / VAE) are stubbed; the real bits exercised
end-to-end are the bimanual transform output contract and the
multi-agent diffusion forward.
"""

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("ATTENTION_BACKEND", "torch")

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(scope="module")
def cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the e2e bimanual pipeline smoke")


def _load_modules():
    try:
        from groot.vla.model.dreamzero.transform.bimanual_cotrain import (
            BimanualDreamTransform,
        )
        from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import (
            WANPolicyHead,
        )
    except Exception as e:  # pragma: no cover
        pytest.skip(f"deps not installed: {e}")
    return BimanualDreamTransform, WANPolicyHead


def _make_yam_raw_sample(T: int = 4, V: int = 3, H: int = 16, W: int = 16,
                         max_state_dim: int = 16, max_action_dim: int = 8):
    """A single raw-data dict mimicking what the loader hands to
    ``BimanualDreamTransform``: video [T, V, H, W, C] (C-last), plus
    the state/action grid the parent transform expects.

    The dim budgets default to ``2x`` the tiny test model's per-agent
    widths so a half-half agent split lands exactly on the model's
    ``max_state_dim`` / ``action_dim`` per agent.
    """
    video = np.random.randint(0, 256, (T, V, H, W, 3), dtype=np.uint8)
    T_s, T_a = T - 1, T - 1
    state = np.random.randn(T_s, max_state_dim).astype(np.float32)
    action = np.clip(np.random.randn(T_a, max_action_dim), -1.0, 1.0).astype(np.float32)
    return {"video": video, "state": state, "action": action}


def _bimanual_per_sample(transform, raw):
    """Walk a raw sample through just the multi-agent pieces of the
    transform that we own: ``_prepare_video`` + the new
    ``_apply_vlm_processing`` for images, and ``_split_dense`` for
    state / action. The rest of DreamTransform (tokenizer, padding,
    embodiment lookup) is replaced by a hand-built dict so we don't
    have to instantiate the full pydantic-validated parent.
    """
    tiled = transform._prepare_video({"video": raw["video"]})  # [P, T, C, 2H, 2W]
    vlm = transform._apply_vlm_processing(
        {"images": tiled, "language": "fold the blanket"}
    )
    state = raw["state"]
    action = raw["action"]
    state_split = transform._split_dense(state, transform.agent_state_dims)
    action_split = transform._split_dense(action, transform.agent_action_dims)
    action_mask = np.ones_like(action, dtype=bool)
    action_mask_split = transform._split_dense(action_mask, transform.agent_action_dims)
    return {
        "images": vlm["images"],            # [P, T, 2H, 2W, C], uint8
        "state": state_split,               # [P, T_s, 7]
        "action": action_split,             # [P, T_a, 7]
        "action_mask": action_mask_split,   # [P, T_a, 7]
        "embodiment_id": np.int64(0),
        "has_real_action": np.ones((), dtype=bool),
        "text": np.zeros(8, dtype=np.int64),
        "text_attention_mask": np.ones(8, dtype=np.int64),
        "num_agents": np.int64(transform.num_agents),
    }


def _collate(samples: list) -> dict:
    """Mimic the trainer's per-key ``np.stack`` collation."""
    out = {}
    for key in samples[0]:
        out[key] = np.stack([s[key] for s in samples], axis=0)
    return out


def _to_torch(batch: dict, device: str):
    """Convert numpy collated batch into a BatchFeature of torch tensors."""
    from transformers.feature_extraction_utils import BatchFeature
    t = {}
    for k, v in batch.items():
        if v.dtype == np.bool_:
            t[k] = torch.from_numpy(v).to(device)
        elif np.issubdtype(v.dtype, np.integer):
            t[k] = torch.from_numpy(v).to(device).long()
        elif v.ndim == 0:
            t[k] = torch.tensor(v.item(), device=device)
        else:
            t[k] = torch.from_numpy(v).to(device)
    return BatchFeature(data=t)


def _make_lite_head(num_agents: int, device: str):
    """Bare WANPolicyHead with a real tiny CausalWanModel + stubbed
    encoders (same pattern as test_wan_action_head_multi_agent)."""
    from torchvision.transforms import v2
    from groot.vla.model.dreamzero.modules.flow_match_scheduler import (
        FlowMatchScheduler,
    )
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
        CausalWanModel,
    )

    _, Head = _load_modules()
    head = Head.__new__(Head)
    torch.nn.Module.__init__(head)
    head._device = device
    head.num_frame_per_block = 1
    head.tiled = False
    head.tile_size_height = 1
    head.tile_size_width = 1
    head.tile_stride_height = 1
    head.tile_stride_width = 1
    head.normalize_video = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    head.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
    head.scheduler.set_timesteps(1000, training=True)
    head._noise_logged = True
    head.config = types.SimpleNamespace(
        decouple_video_action_noise=False,
        use_high_noise_emphasis=False,
        target_video_height=None,
        target_video_width=None,
        video_noise_beta_alpha=3.0,
        video_noise_beta_beta=1.0,
        high_noise_beta_alpha=3.0,
    )

    model = CausalWanModel(
        model_type="t2v",
        patch_size=(1, 2, 2),
        frame_seqlen=4,
        text_len=8,
        in_dim=8,
        dim=96,
        ffn_dim=192,
        freq_dim=32,
        text_dim=32,
        out_dim=8,
        num_heads=4,
        num_layers=2,
        num_frame_per_block=1,
        action_dim=4,
        num_registers=2,
        max_state_dim=8,
        max_num_embodiments=1,
        hidden_size=64,
        num_action_per_block=1,
        num_state_per_block=1,
        concat_first_frame_latent=False,
        num_agents=num_agents,
        agent_dim=4,
        simplex_pool_size=2,
    ).to(device).eval()
    model.init_weights()
    model = model.to(dtype=torch.bfloat16)
    head.model = model
    head.set_frozen_modules_to_eval_mode = types.MethodType(
        lambda self: None, head
    )
    return head


def test_e2e_yam_bimanual_through_forward(cuda_available):
    """One raw video sample -> per-agent tiles -> collate -> forward."""
    BimanualDreamTransform, _ = _load_modules()
    device = "cuda"
    torch.manual_seed(0)
    np.random.seed(0)

    # Per-agent widths match the tiny model: state width 8, action 4.
    transform = BimanualDreamTransform.__new__(BimanualDreamTransform)
    transform.__dict__["agent_video_views"] = [[0, 1], [0, 2]]
    transform.__dict__["agent_state_dims"] = [(0, 8), (8, 16)]
    transform.__dict__["agent_action_dims"] = [(0, 4), (4, 8)]

    # Two raw samples -> trainer collate -> head forward.
    T_frames = 4  # raw video timesteps == VAE F_lat in this tiny config
    samples = [
        _bimanual_per_sample(
            transform,
            _make_yam_raw_sample(T=T_frames, V=3, H=16, W=16,
                                 max_state_dim=16, max_action_dim=8),
        )
        for _ in range(2)
    ]

    # Per-sample shape sanity: every multi-agent key carries P=2.
    s0 = samples[0]
    assert s0["images"].shape == (2, T_frames, 32, 32, 3)
    assert s0["state"].shape == (2, T_frames - 1, 8)
    assert s0["action"].shape == (2, T_frames - 1, 4)

    batch = _collate(samples)
    assert batch["images"].shape == (2, 2, T_frames, 32, 32, 3)
    assert batch["state"].shape == (2, 2, T_frames - 1, 8)
    assert batch["action"].shape == (2, 2, T_frames - 1, 4)

    action_input = _to_torch(batch, device=device)

    head = _make_lite_head(num_agents=2, device=device)

    # Stub VAE / T5 so the head doesn't try to load pretrained weights.
    F_lat = T_frames
    h_lat, w_lat = 4, 4
    text_dim = head.model.text_dim
    text_len = head.model.text_len
    in_dim = head.model.in_dim

    def _stub_encode_prompt(self, input_ids, attention_mask):
        b = input_ids.shape[0]
        return torch.randn(b, text_len, text_dim, dtype=torch.bfloat16, device=device)

    def _stub_encode_video(self, video, tiled=False, tile_size=None, tile_stride=None):
        b = video.shape[0]
        return torch.randn(b, in_dim, F_lat, h_lat, w_lat, dtype=torch.bfloat16, device=device)

    head.encode_prompt = types.MethodType(_stub_encode_prompt, head)
    head.encode_video = types.MethodType(_stub_encode_video, head)

    out = head.forward(type(action_input)(data={}), action_input)
    assert torch.isfinite(out["loss"]).item()
    assert torch.isfinite(out["dynamics_loss"]).item()
    assert torch.isfinite(out["action_loss"]).item()
    # Joint denoising contract: with a real action mask the action loss
    # is non-trivial.
    assert out["action_loss"].item() > 0.0
