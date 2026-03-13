from __future__ import annotations

from typing import Sequence
from shapefit_types import Axis

import math
import numpy as np


def _rot_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c]
    ], dtype=np.float64)


def _rot_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ], dtype=np.float64)


def _rot_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


class fk:
    def __init__(self, lengths_m: Sequence[float], joint_axes: Sequence[Axis], init_angles: Sequence[float]) -> None:
        ...

    def setval(self, j: int, angle: float) -> None:
        ...

    def geti(self, idx: int) -> np.ndarray:
        ...

    def getf(self, pos: float) -> np.ndarray:
        ...

    def _sync(self) -> None:
        ...

    def _upd(self, idx: int) -> None:
        ...


__all__ = ["fk"]
