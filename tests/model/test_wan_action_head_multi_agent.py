"""End-to-end smoke for ``WANPolicyHead._forward_multi_agent``.

The full WANPolicyHead constructor loads pretrained T5 / CLIP / VAE
weights and is far too heavy for a unit test, so we build a bare
instance via ``__new__`` and stub the encoders. The actual diffusion
model is the real :class:`CausalWanModel` (tiny config) so we exercise
the full ``_forward_train_multi_agent`` code path inside the action
head and verify:

  * inputs with a P axis on state / action / images are routed through
    :meth:`_forward_multi_agent`;
  * the multi-agent body runs end-to-end and produces a finite loss
    dict with the expected keys + shapes;
  * the joint-denoising contract is honoured -- the dict carries both
    a non-trivial ``dynamics_loss`` and ``action_loss``.

This test requires CUDA (the simplex-RoPE branch + bf16 autocast inside
the model both assume a GPU).
"""

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("ATTENTION_BACKEND", "torch")

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(scope="module")
def cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the multi-agent WANPolicyHead smoke")


def _load_head_cls():
    try:
        from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import (
            WANPolicyHead,
        )
    except Exception as e:  # pragma: no cover
        pytest.skip(f"deps not installed: {e}")
    return WANPolicyHead


def _make_tiny_model(num_agents: int, device: str, dtype=torch.bfloat16):
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
        CausalWanModel,
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
    model = model.to(dtype=dtype)
    return model


def _make_head(num_agents: int, device: str):
    """Build a bare WANPolicyHead with just the slots ``_forward_multi_agent``
    actually reads, plus a real tiny ``CausalWanModel`` so the full
    multi-agent forward runs.
    """
    from torchvision.transforms import v2
    from groot.vla.model.dreamzero.modules.flow_match_scheduler import (
        FlowMatchScheduler,
    )

    Cls = _load_head_cls()
    head = Cls.__new__(Cls)
    # nn.Module base init -- otherwise attribute set / .to() blows up.
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

    head.model = _make_tiny_model(num_agents=num_agents, device=device)
    # ``self.dtype`` is a property over self.parameters(); registering
    # ``model`` as a submodule above makes its dtype available there.

    # Stub config: just what ``_forward_multi_agent`` reads off it.
    head.config = types.SimpleNamespace(
        decouple_video_action_noise=False,
        use_high_noise_emphasis=False,
        target_video_height=None,
        target_video_width=None,
        video_noise_beta_alpha=3.0,
        video_noise_beta_beta=1.0,
        high_noise_beta_alpha=3.0,
    )

    # Stub the heavy encoders so the head never touches VAE / T5 / CLIP.
    def _set_frozen_modules_to_eval_mode(self):
        return None

    head.set_frozen_modules_to_eval_mode = types.MethodType(
        _set_frozen_modules_to_eval_mode, head
    )

    return head


def _make_batch(B: int, P: int, T: int, F_lat: int, H: int, W: int, device: str):
    """Build a multi-agent ``action_input`` BatchFeature."""
    from transformers.feature_extraction_utils import BatchFeature

    images = torch.randint(
        0, 256, (B, P, T, H, W, 3), dtype=torch.uint8, device=device
    )
    # state / action shapes follow the model's per-block contract
    # (num_state_per_block=1, num_action_per_block=1, F_lat=2 -> T_s = F_lat-1).
    T_s = F_lat - 1
    T_a = F_lat - 1
    state = torch.randn(B, P, T_s, 8, device=device)  # max_state_dim=8
    action = torch.randn(B, P, T_a, 4, device=device).clamp(-1.0, 1.0)
    action_mask = torch.ones_like(action, dtype=torch.bool)
    embodiment_id = torch.zeros(B, dtype=torch.long, device=device)
    has_real_action = torch.ones(B, dtype=torch.bool, device=device)
    text_attention_mask = torch.ones(B, 8, dtype=torch.long, device=device)
    return BatchFeature(data={
        "images": images,
        "state": state,
        "action": action,
        "action_mask": action_mask,
        "embodiment_id": embodiment_id,
        "has_real_action": has_real_action,
        "text": torch.zeros(B, 8, dtype=torch.long, device=device),
        "text_attention_mask": text_attention_mask,
    })


def test_forward_multi_agent_runs_and_emits_loss_dict(cuda_available):
    """The full multi-agent forward must run end-to-end on a small
    config and produce a finite scalar loss + per-component losses."""
    torch.manual_seed(0)
    device = "cuda"
    B, P, T, F_lat, H, W = 1, 2, 4, 2, 4, 4

    head = _make_head(num_agents=P, device=device)

    # Stub encode_prompt / encode_video so we sidestep T5 / VAE entirely.
    text_dim = head.model.text_dim
    text_len = head.model.text_len
    in_dim = head.model.in_dim

    def _stub_encode_prompt(self, input_ids, attention_mask):
        b = input_ids.shape[0]
        return torch.randn(b, text_len, text_dim, dtype=torch.bfloat16, device=device)

    def _stub_encode_video(self, video, tiled=False, tile_size=None, tile_stride=None):
        # videos in: [B*P, C=3, T, H, W]; latents out: [B*P, in_dim, F_lat, H, W].
        b = video.shape[0]
        return torch.randn(
            b, in_dim, F_lat, H, W, dtype=torch.bfloat16, device=device
        )

    head.encode_prompt = types.MethodType(_stub_encode_prompt, head)
    head.encode_video = types.MethodType(_stub_encode_video, head)

    action_input = _make_batch(B=B, P=P, T=T, F_lat=F_lat, H=H, W=W, device=device)
    backbone_output = type(action_input)(data={})

    out = head.forward(backbone_output, action_input)

    assert "loss" in out and "dynamics_loss" in out and "action_loss" in out
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"]).item()
    assert torch.isfinite(out["dynamics_loss"]).item()
    assert torch.isfinite(out["action_loss"]).item()
    # The joint-denoising contract: with a non-zero action mask and
    # has_real_action=1, the action loss must be > 0.
    assert out["action_loss"].item() > 0.0


def test_forward_multi_agent_backward_runs(cuda_available):
    """The combined loss must be differentiable wrt model parameters
    (smoke check that no detach / no_grad escapes the inner forward)."""
    torch.manual_seed(0)
    device = "cuda"
    B, P, F_lat, H, W = 1, 2, 2, 4, 4

    head = _make_head(num_agents=P, device=device)
    head.model.train()

    text_dim = head.model.text_dim
    text_len = head.model.text_len
    in_dim = head.model.in_dim

    def _stub_encode_prompt(self, input_ids, attention_mask):
        b = input_ids.shape[0]
        return torch.randn(b, text_len, text_dim, dtype=torch.bfloat16, device=device)

    def _stub_encode_video(self, video, tiled=False, tile_size=None, tile_stride=None):
        b = video.shape[0]
        return torch.randn(
            b, in_dim, F_lat, H, W, dtype=torch.bfloat16, device=device
        )

    head.encode_prompt = types.MethodType(_stub_encode_prompt, head)
    head.encode_video = types.MethodType(_stub_encode_video, head)

    action_input = _make_batch(B=B, P=P, T=4, F_lat=F_lat, H=H, W=W, device=device)
    backbone_output = type(action_input)(data={})

    out = head.forward(backbone_output, action_input)
    out["loss"].backward()

    # At least one parameter in the model should have received a gradient.
    grads = [p.grad for p in head.model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(torch.isfinite(g).all().item() for g in grads)


def test_gripper_clean_action_loss_masks_and_close_weights():
    Cls = _load_head_cls()
    head = Cls.__new__(Cls)
    torch.nn.Module.__init__(head)
    head.config = types.SimpleNamespace(
        gripper_clean_action_loss_weight=2.0,
        gripper_clean_close_action_loss_weight=4.0,
        gripper_close_threshold=0.0,
        gripper_action_dims=[1],
    )

    clean_action_pred = torch.zeros(1, 1, 2, 3)
    actions = torch.tensor([[[[0.25, -0.80, 0.50], [0.75, 0.70, -0.25]]]])
    action_mask = torch.ones_like(actions, dtype=torch.bool)
    has_real_action = torch.ones(1, dtype=torch.bool)

    loss = head._compute_gripper_clean_action_loss(
        clean_action_pred=clean_action_pred,
        actions=actions,
        action_mask=action_mask,
        has_real_action=has_real_action,
    )

    expected = ((0.80 ** 2) * 4.0 + (0.70 ** 2)) / 2.0 * 2.0
    assert torch.isclose(loss, torch.tensor(expected, dtype=loss.dtype))


def test_clean_sample_reconstruction_matches_flow_scheduler_target():
    Cls = _load_head_cls()
    head = Cls.__new__(Cls)
    torch.nn.Module.__init__(head)

    actions = torch.tensor([[[[0.25, -0.80], [0.75, 0.70]]]], dtype=torch.float32)
    noise = torch.tensor([[[[0.90, 0.10], [-0.40, 0.30]]]], dtype=torch.float32)
    sigma = torch.tensor([[[[0.25], [0.75]]]], dtype=torch.float32)
    noisy = (1.0 - sigma) * actions + sigma * noise
    training_target = noise - noisy

    clean = head._reconstruct_clean_sample_from_flow_target(
        noisy_sample=noisy,
        model_output=training_target,
        sigma=sigma,
    )

    torch.testing.assert_close(clean, actions)


def test_clean_action_loss_mask_respects_max_sigma():
    Cls = _load_head_cls()
    head = Cls.__new__(Cls)
    torch.nn.Module.__init__(head)
    head.config = types.SimpleNamespace(gripper_clean_max_sigma=0.8)

    action_mask = torch.ones(1, 1, 4, 1, dtype=torch.bool)
    sigma = torch.tensor([[[[0.0], [0.5], [0.9], [1.0]]]], dtype=torch.float32)

    mask = head._clean_action_loss_mask(action_mask, sigma)

    assert mask.tolist() == [[[[True], [True], [False], [False]]]]


def test_gripper_clean_action_loss_ignores_fake_or_masked_actions():
    Cls = _load_head_cls()
    head = Cls.__new__(Cls)
    torch.nn.Module.__init__(head)
    head.config = types.SimpleNamespace(
        gripper_clean_action_loss_weight=2.0,
        gripper_clean_close_action_loss_weight=4.0,
        gripper_close_threshold=0.0,
        gripper_action_dims=[1],
    )

    clean_action_pred = torch.zeros(1, 1, 2, 3)
    actions = torch.ones(1, 1, 2, 3)
    action_mask = torch.ones_like(actions, dtype=torch.bool)
    action_mask[..., 1] = False

    masked_loss = head._compute_gripper_clean_action_loss(
        clean_action_pred=clean_action_pred,
        actions=actions,
        action_mask=action_mask,
        has_real_action=torch.ones(1, dtype=torch.bool),
    )
    fake_loss = head._compute_gripper_clean_action_loss(
        clean_action_pred=clean_action_pred,
        actions=actions,
        action_mask=torch.ones_like(actions, dtype=torch.bool),
        has_real_action=torch.zeros(1, dtype=torch.bool),
    )

    assert masked_loss.item() == 0.0
    assert fake_loss.item() == 0.0
