import pytest

from eval_utils.gripper_convention import (
    gripper_values_mismatch_message,
    resolve_gripper_values_from_server_meta,
)


def _meta(left_open=1.0, left_close=-1.0, right_open=1.0, right_close=-1.0):
    return {
        "gripper_action_values": {
            "left": {"open": left_open, "close": left_close},
            "right": {"open": right_open, "close": right_close},
        }
    }


def test_resolve_gripper_values_uses_fallback_without_server_metadata():
    assert resolve_gripper_values_from_server_meta({}, 1.0, -1.0) == (1.0, -1.0)


def test_resolve_gripper_values_accepts_signed_metadata():
    assert resolve_gripper_values_from_server_meta(_meta(), 0.9, -0.9) == (1.0, -1.0)


def test_resolve_gripper_values_rejects_left_right_mismatch():
    with pytest.raises(ValueError, match="left/right open gripper values disagree"):
        resolve_gripper_values_from_server_meta(_meta(right_open=0.5), 1.0, -1.0)


def test_resolve_gripper_values_rejects_reversed_open_close_order():
    with pytest.raises(ValueError, match="close gripper command to be lower"):
        resolve_gripper_values_from_server_meta(_meta(left_open=-1.0, left_close=1.0), 1.0, -1.0)


def test_mismatch_message_is_none_when_client_matches_metadata():
    assert gripper_values_mismatch_message(_meta(), 1.0, -1.0) is None


def test_mismatch_message_reports_client_metadata_difference():
    message = gripper_values_mismatch_message(_meta(left_open=0.8, right_open=0.8), 1.0, -1.0)

    assert message is not None
    assert "client_open=1" in message
    assert "metadata_open=0.8" in message
