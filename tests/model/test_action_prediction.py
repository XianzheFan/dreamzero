"""Per-agent action prediction tests (PR 5b).

PR 5b adds per-agent action+state **register tokens** to the multi-agent
sequence and decodes the action portion via ``self.action_decoder`` to
produce ``action_noise_pred: [B, P, T_a, action_dim]`` -- the
joint-denoising output of DreamZero extended per-agent.

We verify:

* Forward returns a per-agent ``action_noise_pred`` of the right shape.
* State is optional (action alone is enough to trigger register
  block + decode).
* Per-agent action streams produce per-agent predictions: with hubs
  enabled there is some leakage, but the diagonal (agent_i prediction
  responds to agent_i action) must dominate.
* Backward through the action prediction loss populates
  ``action_decoder``, ``action_encoder``, and ``state_encoder`` grads.
* Without ``action``, the prediction is ``None`` (no register tokens).
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("ATTENTION_BACKEND", "torch")

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(scope="module")
def cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for multi-agent action prediction smoke")


def _make_model(num_agents=2, device="cuda"):
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
        num_hub_tokens=4,
    ).to(device).eval()
    model.init_weights()
    # Re-init the action_decoder so it doesn't sit at the zero point of
    # the small-init head; otherwise everything is exactly 0.
    for m in model.action_decoder.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.05)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    return model.to(dtype=torch.bfloat16)


def _make_inputs(
    B=1, P=2, T_a=4, D_a=4, T_s=1, D_s=8, F_lat=2, H=4, W=4, device="cuda",
    with_state=True,
):
    seq_len = P * F_lat * (H // 2) * (W // 2)
    x = torch.randn(B, P, 8, F_lat, H, W, device=device, dtype=torch.bfloat16)
    timestep = torch.randint(0, 1000, (B, F_lat), device=device).float()
    timestep_action = torch.randint(0, 1000, (B, T_a), device=device).float()
    context = torch.randn(B, 8, 32, device=device, dtype=torch.bfloat16)
    action = torch.randn(B, P, T_a, D_a, device=device, dtype=torch.bfloat16)
    out = dict(
        x=x,
        timestep=timestep,
        timestep_action=timestep_action,
        context=context,
        seq_len=seq_len,
        action=action,
    )
    if with_state:
        out["state"] = torch.randn(B, P, T_s, D_s, device=device, dtype=torch.bfloat16)
    return out


def test_action_pred_shape_with_state(cuda_available):
    torch.manual_seed(0)
    model = _make_model(num_agents=2)
    inputs = _make_inputs(with_state=True)

    with torch.no_grad():
        video, action_pred = model(**inputs)

    B, P, T_a, D_a = inputs["action"].shape
    assert action_pred.shape == (B, P, T_a, D_a)
    assert torch.isfinite(action_pred).all()


def test_action_pred_shape_without_state(cuda_available):
    """State is optional; the register block still gets built from
    action features alone."""
    torch.manual_seed(0)
    model = _make_model(num_agents=2)
    inputs = _make_inputs(with_state=False)

    with torch.no_grad():
        video, action_pred = model(**inputs)

    B, P, T_a, D_a = inputs["action"].shape
    assert action_pred.shape == (B, P, T_a, D_a)


def test_no_action_returns_none(cuda_available):
    """Without ``action`` no register block is built and the model
    returns ``action_noise_pred=None``."""
    torch.manual_seed(0)
    model = _make_model(num_agents=2)
    inputs = _make_inputs(with_state=False)
    inputs.pop("action")
    inputs.pop("timestep_action")

    with torch.no_grad():
        video, action_pred = model(**inputs)

    assert action_pred is None
    assert torch.isfinite(video).all()


def test_action_pred_responds_to_per_agent_action(cuda_available):
    """If we change agent 0's action stream, agent 0's predicted action
    must change more than agent 1's. With hubs enabled there is some
    cross-agent leakage but the diagonal must still dominate."""
    torch.manual_seed(0)
    model = _make_model(num_agents=2)
    inputs = _make_inputs(with_state=False)

    with torch.no_grad():
        _, pred_a = model(**inputs)
        # Perturb only agent 0's action.
        action_b = inputs["action"].clone()
        action_b[:, 0] = action_b[:, 0] + torch.randn_like(action_b[:, 0])
        inputs_b = dict(inputs)
        inputs_b["action"] = action_b
        _, pred_b = model(**inputs_b)

    diff_a0 = (pred_a[:, 0] - pred_b[:, 0]).abs().mean().float().item()
    diff_a1 = (pred_a[:, 1] - pred_b[:, 1]).abs().mean().float().item()
    assert diff_a0 > diff_a1, (
        f"Agent-0 action perturbation should move agent-0 prediction more "
        f"than agent-1's; got d(a0)={diff_a0:.3e}, d(a1)={diff_a1:.3e}"
    )


def test_action_pred_backward_populates_decoder_and_encoders(cuda_available):
    """End-to-end gradient flow: loss on the action prediction touches
    action_encoder, state_encoder, AND action_decoder."""
    torch.manual_seed(0)
    model = _make_model(num_agents=2).train()
    inputs = _make_inputs(with_state=True)

    video, action_pred = model(**inputs)
    # Joint loss to ensure both heads pull on autograd.
    loss = action_pred.float().pow(2).mean() + video.float().pow(2).mean()
    loss.backward()

    def _any_grad(module):
        any_param = False
        any_nonzero = False
        for p in module.parameters():
            if p.grad is not None:
                any_param = True
                if p.grad.abs().sum().item() > 0:
                    any_nonzero = True
        return any_param, any_nonzero

    for name, module in (
        ("action_decoder", model.action_decoder),
        ("action_encoder", model.action_encoder),
        ("state_encoder", model.state_encoder),
    ):
        has_grad, has_nonzero = _any_grad(module)
        assert has_grad, f"{name} parameters have no .grad after backward"
        assert has_nonzero, f"{name} grads are all zero -- module is detached"
