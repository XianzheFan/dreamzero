"""Trainer batch P-axis handling smoke.

The dreamzero ``collate`` helper in
``groot.vla.model.dreamzero.transform.dreamzero_cotrain`` builds
batches by ``np.stack``-ing per-sample numpy arrays / by stacking
torch tensors along a new leading axis. So when each per-sample item
already carries a P axis (as produced by ``BimanualDreamTransform``)
the stack naturally yields ``[B, P, ...]`` -- no collator change
required.

This test just confirms the contract: a list of two per-sample dicts,
each with a P=2 leading axis on ``state``/``action``/``images``,
collates into a single dict with ``[B=2, P=2, ...]``.
"""

import numpy as np
import pytest
import torch


def test_collate_preserves_p_axis():
    """The trainer's ``np.stack`` collation must produce [B, P, ...] when
    each sample already has a leading [P, ...] axis."""
    # Fake two samples, each with a leading P=2 axis.
    samples = []
    for sample_i in range(2):
        state  = np.zeros((2, 1, 7),  dtype=np.float32)  # [P, T_s, dim]
        action = np.zeros((2, 24, 7), dtype=np.float32)  # [P, T_a, dim]
        images = np.zeros((2, 2, 33, 3, 176, 320), dtype=np.uint8)
        # Distinguish samples
        state[:] = sample_i + 1.0
        action[:] = sample_i + 2.0
        samples.append({
            "state": state,
            "action": action,
            "images": images,
        })

    # Default trainer collation: per-key np.stack.
    batch = {k: np.stack([s[k] for s in samples], axis=0) for k in samples[0]}

    assert batch["state"].shape  == (2, 2, 1, 7),  batch["state"].shape
    assert batch["action"].shape == (2, 2, 24, 7), batch["action"].shape
    assert batch["images"].shape == (2, 2, 2, 33, 3, 176, 320), batch["images"].shape

    # Per-sample values land in the right batch slot.
    np.testing.assert_array_equal(
        batch["state"][0], np.full((2, 1, 7), 1.0, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        batch["state"][1], np.full((2, 1, 7), 2.0, dtype=np.float32)
    )


def test_collate_preserves_p_axis_torch():
    """Same check but with torch.stack -- some collators wrap np tensors
    in torch.from_numpy before stacking."""
    samples = []
    for sample_i in range(3):
        samples.append(torch.zeros(2, 1, 7) + float(sample_i))
    stacked = torch.stack(samples, dim=0)
    assert stacked.shape == (3, 2, 1, 7)
    for i in range(3):
        torch.testing.assert_close(stacked[i], torch.full((2, 1, 7), float(i)))
