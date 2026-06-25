import json
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


JOINT_DIMS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_compare_module():
    module_name = "compare_liftbarrier_model_data_action_magnitude_for_test"
    module_path = REPO_ROOT / "scripts/eval/compare_liftbarrier_model_data_action_magnitude.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_dataset_stats(root: Path) -> None:
    root.mkdir(parents=True)
    payload = {
        "episodes_read": 500,
        "one_step_target_current_abs": {"p50": 0.01, "p95": 0.05},
        "horizon_target_current_abs": {"p50": 0.02, "p95": 0.20},
        "horizon_target_current_abs_by_offset": [
            {"offset": 0, "p50": 0.01, "p95": 0.05},
            {"offset": 1, "p50": 0.02, "p95": 0.10},
            {"offset": 2, "p50": 0.03, "p95": 0.30},
        ],
        "relative_stats_present": False,
    }
    (root / "liftbarrier_action_magnitude.json").write_text(json.dumps(payload))


def _write_action_dump(root: Path, variant: str, *, magnitude: float) -> None:
    dump_dir = root / variant / "action_dump"
    dump_dir.mkdir(parents=True)
    pred_chunk = np.zeros((2, 3, 16), dtype=np.float32)
    obs_qpos = np.zeros((2, 16), dtype=np.float32)
    pred_chunk[:, :, JOINT_DIMS] = magnitude
    exec_action = np.zeros((5, 16), dtype=np.float32)
    exec_action[:, JOINT_DIMS] = np.linspace(0.0, magnitude, 5, dtype=np.float32)[:, None]
    np.savez(
        dump_dir / "episode_1000.npz",
        pred_chunk=pred_chunk,
        obs_qpos=obs_qpos,
        exec_action=exec_action,
    )


def test_compare_model_data_action_magnitude_groups_variants_and_ratios(tmp_path):
    compare = _load_compare_module()
    stats_root = tmp_path / "stats"
    eval_root = tmp_path / "eval"
    _write_dataset_stats(stats_root)
    _write_action_dump(
        eval_root,
        "causal_flowmatch_resetcache_vpred_action_rp24_jscale_1p0_clip_0p35_smooth",
        magnitude=0.10,
    )
    _write_action_dump(
        eval_root,
        "causal_flowmatch_resetcache_vpred_action_rp12_jscale_1p0_clip_0p35_smooth",
        magnitude=0.05,
    )

    payload = compare.compare_model_data_action_magnitude(
        eval_root=eval_root,
        data_stats_root=stats_root,
    )

    assert payload["dataset"]["episodes_read"] == 500
    assert [variant["replan"] for variant in payload["variants"]] == [12, 24]
    rp12, rp24 = payload["variants"]
    assert rp12["episodes"] == 1
    assert rp24["episodes"] == 1
    assert rp12["pred_chunk_target_current_abs"]["p95"] == pytest.approx(0.05)
    assert rp24["pred_chunk_target_current_abs"]["p95"] == pytest.approx(0.10)
    assert rp12["comparison_to_dataset"]["chunk_p95_over_data_horizon_p95"] == pytest.approx(0.25)
    assert rp24["comparison_to_dataset"]["last_offset_p95_over_data_last_offset_p95"] == pytest.approx(1 / 3)

    text = compare.format_summary(payload)
    assert "dataset_horizon_p95=0.200000" in text
    assert "chunk_p95_over_data" in text
    assert "rp12_jscale_clip_smooth" in text
    assert "rp24_jscale_clip_smooth" in text


def test_compare_model_data_action_magnitude_requires_action_dumps(tmp_path):
    compare = _load_compare_module()
    stats_root = tmp_path / "stats"
    eval_root = tmp_path / "eval"
    _write_dataset_stats(stats_root)
    eval_root.mkdir()

    with pytest.raises(FileNotFoundError, match="no action_dump episode npz"):
        compare.compare_model_data_action_magnitude(
            eval_root=eval_root,
            data_stats_root=stats_root,
        )
