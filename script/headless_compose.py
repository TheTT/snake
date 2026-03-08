"""Frame composition helpers for free-space multi-panel output."""

from __future__ import annotations

import numpy as np


def compose_fsm_quad(
    frame_main: np.ndarray,
    frame_left: np.ndarray,
    frame_top: np.ndarray,
    frame_persp: np.ndarray,
    out_h: int,
    out_w: int,
    split_x_ratio: float,
    split_y_ratio: float,
) -> np.ndarray:
    """Compose 4 views into one frame with configurable split positions.

    All input frames are expected to be rendered at their target panel sizes,
    so no post-render image scaling is performed.
    """
    split_x = int(round(out_w * split_x_ratio))
    split_y = int(round(out_h * split_y_ratio))
    split_x = max(1, min(out_w - 1, split_x))
    split_y = max(1, min(out_h - 1, split_y))

    w_left = split_x
    h_top = split_y

    out = np.zeros((out_h, out_w, 3), dtype=frame_main.dtype)
    out[:h_top, :w_left] = frame_main
    out[:h_top, split_x:] = frame_left
    out[split_y:, :w_left] = frame_top
    out[split_y:, split_x:] = frame_persp
    return out
