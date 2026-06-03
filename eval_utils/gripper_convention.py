from __future__ import annotations

from typing import Any

import numpy as np


def resolve_gripper_values_from_server_meta(
    meta: dict[str, Any],
    fallback_open: float,
    fallback_close: float,
) -> tuple[float, float]:
    """Return symmetric client gripper values inferred from server metadata.

    The policy server reports per-arm q01/q99-derived values where q01 is the
    close command and q99 is the open command. Eval sends one scalar convention
    to both arms, so require both arms to agree closely before using metadata.
    """
    values = meta.get("gripper_action_values")
    if not isinstance(values, dict):
        return fallback_open, fallback_close

    pairs = []
    for side in ("left", "right"):
        side_values = values.get(side)
        if not isinstance(side_values, dict):
            raise ValueError(f"server meta missing {side} gripper values: {values}")
        pairs.append((float(side_values["open"]), float(side_values["close"])))

    open_values = np.asarray([p[0] for p in pairs], dtype=np.float32)
    close_values = np.asarray([p[1] for p in pairs], dtype=np.float32)
    if not (np.isfinite(open_values).all() and np.isfinite(close_values).all()):
        raise ValueError(f"non-finite server gripper values: {values}")
    if float(np.max(np.abs(open_values - open_values[0]))) > 1e-3:
        raise ValueError(f"left/right open gripper values disagree: {values}")
    if float(np.max(np.abs(close_values - close_values[0]))) > 1e-3:
        raise ValueError(f"left/right close gripper values disagree: {values}")
    if not close_values[0] < open_values[0]:
        raise ValueError(
            "expected close gripper command to be lower than open command, "
            f"got {values}"
        )
    return float(open_values[0]), float(close_values[0])


def gripper_values_mismatch_message(
    meta: dict[str, Any],
    client_open: float,
    client_close: float,
) -> str | None:
    values = meta.get("gripper_action_values")
    if not isinstance(values, dict):
        return None
    meta_open, meta_close = resolve_gripper_values_from_server_meta(
        meta,
        fallback_open=client_open,
        fallback_close=client_close,
    )
    if abs(meta_open - client_open) <= 1e-3 and abs(meta_close - client_close) <= 1e-3:
        return None
    return (
        "client gripper binarize values differ from checkpoint metadata: "
        f"client_open={client_open:.6g} client_close={client_close:.6g} "
        f"metadata_open={meta_open:.6g} metadata_close={meta_close:.6g}. "
        "Pass --client-gripper-values-from-server-metadata to use metadata."
    )
