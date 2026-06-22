import json
from pathlib import Path

from scripts.eval.analyze_robofactory_checkpoint_trend import (
    analyze_rows,
    load_rows,
    main,
)


def _write_grid(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")


def test_analyze_rows_reports_improving_and_worsening_metrics():
    rows = [
        {
            "checkpoint_step": 2000,
            "success_rate": 0.0,
            "action_pred_vs_future_mae_rgb_mean": 30.0,
            "mean_pred_chunk_joint_step_jerk": 0.02,
        },
        {
            "checkpoint_step": 4000,
            "success_rate": 0.5,
            "action_pred_vs_future_mae_rgb_mean": 20.0,
            "mean_pred_chunk_joint_step_jerk": 0.03,
        },
        {
            "checkpoint_step": 6000,
            "success_rate": 1.0,
            "action_pred_vs_future_mae_rgb_mean": 10.0,
            "mean_pred_chunk_joint_step_jerk": 0.04,
        },
    ]

    trend = analyze_rows(rows)
    metrics = trend["metrics"]

    assert trend["checkpoint_steps"] == [2000, 4000, 6000]
    assert metrics["success_rate"]["status"] == "improving"
    assert metrics["success_rate"]["improvement_slope_per_2k"] == 0.5
    assert metrics["action_pred_vs_future_mae_rgb_mean"]["status"] == "improving"
    assert metrics["action_pred_vs_future_mae_rgb_mean"]["improvement_slope_per_2k"] == 10.0
    assert metrics["mean_pred_chunk_joint_step_jerk"]["status"] == "worsening"
    assert metrics["mean_pred_chunk_joint_step_jerk"]["improvement_slope_per_2k"] < 0.0


def test_analyze_rows_marks_two_points_insufficient():
    rows = [
        {"checkpoint_step": 2000, "success_rate": 0.0},
        {"checkpoint_step": 4000, "success_rate": 1.0},
    ]

    trend = analyze_rows(rows)

    assert trend["metrics"]["success_rate"]["point_count"] == 2
    assert trend["metrics"]["success_rate"]["status"] == "insufficient"
    assert trend["metrics"]["success_rate"]["improvement_delta"] == 1.0


def test_load_rows_reads_checkpoint_grid_json(tmp_path):
    path = tmp_path / "checkpoint_grid.json"
    _write_grid(
        path,
        [
            {"checkpoint_step": 4000, "success_rate": 0.5},
            {"checkpoint_step": 2000, "success_rate": 0.0},
        ],
    )

    rows = load_rows([path])

    assert [row["checkpoint_step"] for row in rows] == [2000, 4000]


def test_main_prints_trend_table_and_json(tmp_path, capsys):
    path = tmp_path / "checkpoint_grid.json"
    out = tmp_path / "trend.json"
    _write_grid(
        path,
        [
            {"checkpoint_step": 2000, "success_rate": 0.0},
            {"checkpoint_step": 4000, "success_rate": 0.5},
            {"checkpoint_step": 6000, "success_rate": 1.0},
        ],
    )

    status = main([str(path), "--json-out", str(out)])

    assert status == 0
    stdout = capsys.readouterr().out
    assert "metric\tdir\tn\tfirst_step" in stdout
    assert "success_rate\thigher\t3\t2000\t0\t6000\t1" in stdout
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["metrics"]["success_rate"]["status"] == "improving"
