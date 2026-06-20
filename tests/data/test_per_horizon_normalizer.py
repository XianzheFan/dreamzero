import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_local_normalizer():
    """Load state_action.py without importing the full transform package."""
    transform_dir = _REPO_ROOT / "groot/vla/data/transform"
    old_transform_pkg = sys.modules.get("groot.vla.data.transform")

    transform_pkg = types.ModuleType("groot.vla.data.transform")
    transform_pkg.__path__ = [str(transform_dir)]
    sys.modules["groot.vla.data.transform"] = transform_pkg

    try:
        base_spec = importlib.util.spec_from_file_location(
            "groot.vla.data.transform.base",
            transform_dir / "base.py",
        )
        base_module = importlib.util.module_from_spec(base_spec)
        sys.modules["groot.vla.data.transform.base"] = base_module
        assert base_spec.loader is not None
        base_spec.loader.exec_module(base_module)

        state_action_spec = importlib.util.spec_from_file_location(
            "local_state_action_for_test",
            transform_dir / "state_action.py",
        )
        state_action_module = importlib.util.module_from_spec(state_action_spec)
        sys.modules["local_state_action_for_test"] = state_action_module
        assert state_action_spec.loader is not None
        state_action_spec.loader.exec_module(state_action_module)
        return state_action_module.Normalizer
    finally:
        if old_transform_pkg is None:
            sys.modules.pop("groot.vla.data.transform", None)
        else:
            sys.modules["groot.vla.data.transform"] = old_transform_pkg
        sys.modules.pop("groot.vla.data.transform.base", None)
        sys.modules.pop("local_state_action_for_test", None)


Normalizer = _load_local_normalizer()


def test_q99_per_horizon_stats_normalize_flattened_chunks():
    statistics = {
        "q01": [[-1.0, 0.0], [-2.0, 0.0], [-3.0, 0.0]],
        "q99": [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]],
    }
    x = torch.tensor(
        [
            [-1.0, 1.0],
            [0.0, 2.0],
            [3.0, 6.0],
            [1.0, 2.0],
            [-2.0, 0.0],
            [0.0, 3.0],
        ],
        dtype=torch.float32,
    )

    normalizer = Normalizer("q99", statistics)
    normalized = normalizer.forward(x)

    expected = torch.tensor(
        [
            [-1.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(normalized, expected)
    assert normalized.shape == x.shape
    assert torch.allclose(normalizer.inverse(normalized), x)


def test_mean_std_per_horizon_stats_do_not_require_q99_keys():
    statistics = {
        "mean": [[0.0, 10.0], [1.0, 20.0]],
        "std": [[1.0, 2.0], [2.0, 4.0]],
    }
    x = torch.tensor(
        [
            [1.0, 12.0],
            [5.0, 28.0],
            [-1.0, 8.0],
            [-3.0, 12.0],
        ],
        dtype=torch.float32,
    )

    normalizer = Normalizer("mean_std", statistics)
    normalized = normalizer.forward(x)

    expected = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [-1.0, -1.0],
            [-2.0, -2.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(normalized, expected)
    assert torch.allclose(normalizer.inverse(normalized), x)


def test_per_horizon_stats_reject_unaligned_flattened_sequence():
    normalizer = Normalizer(
        "q99",
        {
            "q01": [[-1.0], [-2.0], [-3.0]],
            "q99": [[1.0], [2.0], [3.0]],
        },
    )

    with pytest.raises(ValueError, match="Cannot align sequence length"):
        normalizer.forward(torch.zeros(4, 1))
