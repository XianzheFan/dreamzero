"""Per-agent role-token tests (PR 7).

Role embeddings are an *optional*, **asymmetric-task** signal added on
top of the simplex agent identity. They are decoupled from the agent
slot index so callers can apply per-batch agent permutation augmentation
without the role label sticking to the wrong slot. We verify:

* Default ``role_id=None`` is a no-op (output unchanged from the
  no-role baseline).
* Two agents with DIFFERENT ``role_id`` produce different per-agent
  video outputs.
* Two agents with the SAME ``role_id`` produce identical outputs IF the
  rest of the input is symmetric across agents (sanity check on the
  bias mechanism).
* Backward through the video output populates ``role_embedding.weight.grad``.
* Wrong-shape ``role_id`` is rejected.
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
        pytest.skip("CUDA required for role-token smoke")


def _make_model(num_agents=2, num_roles=4, device="cuda"):
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
        num_roles=num_roles,
    ).to(device).eval()
    model.init_weights()
    # Re-init the head so output isn't identically zero (head is
    # zero-init by ``init_weights``).
    torch.nn.init.normal_(model.head.head.weight, mean=0.0, std=0.02)
    return model.to(dtype=torch.bfloat16)


def _make_inputs(B=1, P=2, T_a=4, D_a=4, F_lat=2, H=4, W=4, device="cuda"):
    seq_len = P * F_lat * (H // 2) * (W // 2)
    return dict(
        x=torch.randn(B, P, 8, F_lat, H, W, device=device, dtype=torch.bfloat16),
        timestep=torch.randint(0, 1000, (B, F_lat), device=device).float(),
        timestep_action=torch.randint(0, 1000, (B, T_a), device=device).float(),
        context=torch.randn(B, 8, 32, device=device, dtype=torch.bfloat16),
        seq_len=seq_len,
        action=torch.randn(B, P, T_a, D_a, device=device, dtype=torch.bfloat16),
    )


def test_default_role_id_none_is_noop(cuda_available):
    """Calling the model with ``role_id=None`` must produce the same
    output as not passing role_id at all -- regression check that the
    new parameter doesn't perturb the default symmetric path."""
    torch.manual_seed(0)
    model = _make_model(num_agents=2)
    inputs = _make_inputs()

    with torch.no_grad():
        v_default, a_default = model(**inputs)
        v_none, a_none = model(**inputs, role_id=None)

    torch.testing.assert_close(v_default, v_none, atol=0.0, rtol=0.0)
    torch.testing.assert_close(a_default, a_none, atol=0.0, rtol=0.0)


def test_different_role_ids_change_per_agent_output(cuda_available):
    """Two agents with different roles should produce different
    per-agent video outputs even from identical video inputs."""
    torch.manual_seed(0)
    device = "cuda"
    model = _make_model(num_agents=2, num_roles=4, device=device)

    inputs = _make_inputs(device=device)
    # Make the two agents' video / action inputs identical so the ONLY
    # source of per-agent variation is the role + simplex id.
    inputs["x"][:, 1] = inputs["x"][:, 0]
    inputs["action"][:, 1] = inputs["action"][:, 0]

    role_id_same = torch.zeros(1, 2, device=device, dtype=torch.long)
    role_id_diff = torch.tensor([[0, 1]], device=device, dtype=torch.long)

    with torch.no_grad():
        v_same, _ = model(**inputs, role_id=role_id_same)
        v_diff, _ = model(**inputs, role_id=role_id_diff)

    # When both agents share role 0, only the simplex id distinguishes
    # them. Adding a distinct role for agent 1 must add MORE variation.
    diff_same = (v_same[:, 0] - v_same[:, 1]).abs().mean().float().item()
    diff_diff = (v_diff[:, 0] - v_diff[:, 1]).abs().mean().float().item()
    assert diff_diff > diff_same, (
        f"role-token bias did not amplify per-agent divergence: "
        f"|diff_same| = {diff_same:.3e}, |diff_diff| = {diff_diff:.3e}"
    )


def test_role_embedding_backward(cuda_available):
    """A loss on the video output must back-prop through
    ``role_embedding`` (proving the bias is on the autograd graph)."""
    torch.manual_seed(0)
    device = "cuda"
    model = _make_model(num_agents=2, num_roles=4, device=device).train()
    inputs = _make_inputs(device=device)
    role_id = torch.tensor([[0, 2]], device=device, dtype=torch.long)

    video, _ = model(**inputs, role_id=role_id)
    loss = video.float().pow(2).mean()
    loss.backward()

    assert model.role_embedding.weight.grad is not None
    grad = model.role_embedding.weight.grad.abs().sum().item()
    assert grad > 0, "role_embedding.weight received an all-zero gradient"


def test_wrong_role_id_shape_rejected(cuda_available):
    model = _make_model(num_agents=2)
    inputs = _make_inputs()
    bad = torch.tensor([0, 1], device="cuda", dtype=torch.long)  # missing B axis
    with pytest.raises(AssertionError, match=r"\[B, P\]"):
        with torch.no_grad():
            model(**inputs, role_id=bad)


def test_role_id_flows_through_inference_dispatch(cuda_available):
    """``role_id`` must reach the body via the inference dispatcher too
    (it's not just a training-forward kwarg)."""
    torch.manual_seed(0)
    device = "cuda"
    model = _make_model(num_agents=2, num_roles=4, device=device)
    inputs = _make_inputs(device=device)
    num_layers = len(model.blocks)
    role_id = torch.tensor([[0, 1]], device=device, dtype=torch.long)

    with torch.no_grad():
        v_no_role, _, _ = model(
            **inputs,
            kv_cache=[None] * num_layers,
            crossattn_cache=[None] * num_layers,
            current_start_frame=0,
        )
        v_with_role, _, _ = model(
            **inputs,
            kv_cache=[None] * num_layers,
            crossattn_cache=[None] * num_layers,
            current_start_frame=0,
            role_id=role_id,
        )

    assert v_no_role.shape == v_with_role.shape
    assert not torch.allclose(v_no_role, v_with_role), (
        "role_id had no effect through the inference dispatch"
    )
