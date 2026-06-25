import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_COLUMNS = np.asarray(
    [
        "step",
        "success_margin",
        "left_tcp_to_grasp_target",
        "right_tcp_to_grasp_target",
        "left_grasping",
        "right_grasping",
        "cmd_left_gripper",
        "cmd_right_gripper",
    ],
    dtype="<U32",
)


def _load_trace_module():
    module_name = "analyze_liftbarrier_trace_diagnostics_for_test"
    module_path = REPO_ROOT / "scripts/eval/analyze_liftbarrier_trace_diagnostics.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_trace_dump(
    root: Path,
    variant: str,
    seed: int,
    *,
    left_dist: list[float],
    right_dist: list[float],
    left_close_step: int,
    right_close_step: int,
    left_grasp_step: int | None = None,
    right_grasp_step: int | None = None,
) -> None:
    dump_dir = root / variant / "action_dump"
    dump_dir.mkdir(parents=True, exist_ok=True)
    steps = np.arange(len(left_dist), dtype=np.float32)
    trace = np.zeros((len(steps), len(TRACE_COLUMNS)), dtype=np.float32)
    trace[:, 0] = steps
    trace[:, 1] = -0.1
    trace[:, 2] = np.asarray(left_dist, dtype=np.float32)
    trace[:, 3] = np.asarray(right_dist, dtype=np.float32)
    trace[:, 4] = 0.0
    trace[:, 5] = 0.0
    trace[:, 6] = 1.0
    trace[:, 7] = 1.0
    trace[steps >= left_close_step, 6] = -1.0
    trace[steps >= right_close_step, 7] = -1.0
    if left_grasp_step is not None:
        trace[steps >= left_grasp_step, 4] = 1.0
    if right_grasp_step is not None:
        trace[steps >= right_grasp_step, 5] = 1.0
    np.savez(
        dump_dir / f"episode_{seed}.npz",
        seed=seed,
        success=False,
        action_representation=np.asarray("absolute_qpos", dtype="<U32"),
        env_trace=trace,
        env_trace_columns=TRACE_COLUMNS,
    )


def test_trace_diagnostics_reports_plateau_and_close_distance(tmp_path):
    trace_diag = _load_trace_module()
    eval_root = tmp_path / "eval"
    left = [0.20, 0.12, 0.08, 0.071, 0.070, 0.070, 0.070]
    right = [0.19, 0.13, 0.09, 0.083, 0.082, 0.082, 0.082]
    _write_trace_dump(
        eval_root,
        "causal_flowmatch_resetcache_vpred_action_rp12_jscale_1p0_clip_0p35_smooth",
        1000,
        left_dist=left,
        right_dist=right,
        left_close_step=4,
        right_close_step=4,
    )

    result = trace_diag.analyze_root(
        eval_root,
        step_marks=(0, 2, 4, 6),
        contact_threshold=0.05,
        close_threshold=0.0,
        decisive_threshold=-0.5,
        tail_window=2,
    )

    variant = result["variants"][0]
    assert variant["replan"] == 12
    assert variant["success_count"] == 0
    assert variant["left"]["min"]["p50"] == pytest.approx(0.07)
    assert variant["right"]["min"]["p50"] == pytest.approx(0.082)
    assert variant["left"]["final"]["p50"] == pytest.approx(0.07)
    assert variant["left"]["target_dist_before_decisive_close"]["p50"] == pytest.approx(0.071)
    assert variant["right"]["target_dist_before_decisive_close"]["p50"] == pytest.approx(0.083)
    assert variant["left"]["decisive_close_before_contact_count"] == 1
    assert variant["right"]["decisive_close_before_contact_count"] == 1
    assert variant["left"]["grasp_count"] == 0
    assert variant["left"]["curve"]["4"]["p50"] == pytest.approx(0.07)


def test_trace_summary_contains_curve_and_grasp_counts(tmp_path):
    trace_diag = _load_trace_module()
    eval_root = tmp_path / "eval"
    _write_trace_dump(
        eval_root,
        "causal_flowmatch_resetcache_vpred_action_rp24_jscale_1p0_clip_0p35_smooth",
        1000,
        left_dist=[0.09, 0.04, 0.03],
        right_dist=[0.10, 0.08, 0.06],
        left_close_step=1,
        right_close_step=1,
        left_grasp_step=2,
    )

    result = trace_diag.analyze_root(
        eval_root,
        step_marks=(0, 1, 2),
        contact_threshold=0.05,
        close_threshold=0.0,
        decisive_threshold=-0.5,
        tail_window=1,
    )
    out = tmp_path / "summary.txt"
    trace_diag.write_summary(result, out)
    text = out.read_text()

    assert "curve_step\tleft_p50\tright_p50" in text
    assert "rp24" in text
    assert "\t1\t0\n" in text or text.endswith("\t1\t0\n")


def test_trace_diagnostics_requires_action_dumps(tmp_path):
    trace_diag = _load_trace_module()
    with pytest.raises(FileNotFoundError, match="no action_dump episode npz"):
        trace_diag.analyze_root(
            tmp_path,
            step_marks=(0,),
            contact_threshold=0.05,
            close_threshold=0.0,
            decisive_threshold=-0.5,
            tail_window=1,
        )
