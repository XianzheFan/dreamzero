import numpy as np

from eval_utils.offline_eval_bimanual import (
    collect_dim_pairs,
    collect_horizon_joint_errors,
    collect_phase_joint_errors,
    gripper_open_close_metrics,
    summarize_gripper_open_close,
)


def test_collect_dim_pairs_keeps_only_valid_gripper_entries():
    pred = np.zeros((1, 2, 3, 8), dtype=np.float32)
    gt = np.zeros((1, 2, 3, 8), dtype=np.float32)
    valid = np.ones((1, 2, 3, 8), dtype=bool)

    pred[..., 7] = [[[-0.8, 0.2, 0.6], [-0.4, -0.1, 0.9]]]
    gt[..., 7] = [[[-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]]]
    valid[0, 1, 0, 7] = False

    pred_flat, gt_flat = collect_dim_pairs([(pred, gt, valid)], [7])

    np.testing.assert_allclose(pred_flat, [-0.8, 0.2, 0.6, -0.1, 0.9])
    np.testing.assert_allclose(gt_flat, [-1.0, 1.0, -1.0, -1.0, 1.0])


def test_gripper_open_close_metrics_reports_confusion_counts():
    pred = np.array([-0.8, 0.2, 0.6, -0.4, -0.1, 0.9], dtype=np.float32)
    gt = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32)

    metrics = gripper_open_close_metrics(pred, gt, threshold=0.0)

    assert metrics["n"] == 6
    assert metrics["gt_close_pred_close"] == 2
    assert metrics["gt_close_pred_open"] == 1
    assert metrics["gt_open_pred_close"] == 1
    assert metrics["gt_open_pred_open"] == 2
    assert metrics["accuracy"] == 4 / 6
    assert metrics["balanced_accuracy"] == 2 / 3
    assert metrics["close_recall"] == 2 / 3
    assert metrics["open_recall"] == 2 / 3
    assert metrics["close_precision"] == 2 / 3
    assert metrics["open_precision"] == 2 / 3
    assert metrics["gt_open_rate"] == 0.5
    assert metrics["pred_open_rate"] == 0.5


def test_summarize_gripper_open_close_prints_semantic_summary(capsys):
    pred = np.array([-0.8, 0.2, 0.6, -0.4], dtype=np.float32)
    gt = np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float32)

    metrics = summarize_gripper_open_close(pred, gt, threshold=0.0)
    captured = capsys.readouterr().out

    assert metrics["accuracy"] == 0.5
    assert "Gripper open/close @ threshold 0.000" in captured
    assert "gt close: pred close=1 pred open=1" in captured
    assert "gt open : pred close=1 pred open=1" in captured
    assert "open rate: gt= 50.0% pred= 50.0%" in captured


def test_collect_phase_joint_errors_uses_gt_first_close_windows():
    pred = np.zeros((1, 1, 5, 8), dtype=np.float32)
    gt = np.zeros((1, 1, 5, 8), dtype=np.float32)
    valid = np.ones((1, 1, 5, 8), dtype=bool)

    # Joint dim 0 errors by horizon step: 0.0, 0.1, 0.2, 0.3, 0.4.
    pred[0, 0, :, 0] = np.arange(5, dtype=np.float32) * 0.1
    gt[0, 0, :, 7] = [1.0, 1.0, -1.0, -1.0, -1.0]

    phase = collect_phase_joint_errors([(pred, gt, valid)], threshold=0.0, before=1, after=2)

    assert np.isclose(phase["gt_open"].mean(), (0.0 + 0.1) / 14)
    assert np.isclose(phase["gt_close"].mean(), (0.2 + 0.3 + 0.4) / 21)
    assert np.isclose(phase["pre_first_close[-1,-1]"].mean(), 0.1 / 7)
    assert np.isclose(phase["at_first_close[0]"].mean(), 0.2 / 7)
    assert np.isclose(phase["post_first_close[0,+2]"].mean(), (0.2 + 0.3 + 0.4) / 21)
    assert np.isclose(phase["near_first_close[-1,+2]"].mean(), (0.1 + 0.2 + 0.3 + 0.4) / 28)


def test_collect_horizon_joint_errors_reports_per_offset_means():
    pred = np.zeros((1, 2, 3, 8), dtype=np.float32)
    gt = np.zeros((1, 2, 3, 8), dtype=np.float32)
    valid = np.ones((1, 2, 3, 8), dtype=bool)
    pred[:, :, 0, 0] = 0.1
    pred[:, :, 1, 0] = 0.2
    pred[:, :, 2, 0] = 0.3

    rows = collect_horizon_joint_errors([(pred, gt, valid)])

    assert [row[:2] for row in rows] == [(0, 14), (1, 14), (2, 14)]
    np.testing.assert_allclose(
        [row[2] for row in rows],
        [0.1 / 7, 0.2 / 7, 0.3 / 7],
        rtol=1e-6,
    )
