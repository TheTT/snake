from __future__ import annotations

from typing import Sequence
from enum import IntEnum

import math
import numpy as np


class Axis(IntEnum):
    X = 0
    Y = 1
    Z = 2


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


class FK:
    def __init__(
        self, jn: int,
        *,
        init_angles: np.ndarray,
        seg_len: Sequence[float],
        joint_axes: Sequence[Axis],
        joint_signs: Sequence[float],
    ) -> None:
        self.jn = int(jn)
        # store arrays with leading sequence dimension
        self.l = np.asarray(seg_len, dtype=np.float64)
        self.M = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], repeats=self.jn + 1, axis=0)
        self.Dlt = np.zeros((max(0, self.jn), 3), dtype=np.float64)
        self.v = np.asarray(init_angles, dtype=np.float64).copy()
        self.ja = np.asarray(joint_axes, dtype=object)
        self.js = np.asarray(joint_signs, dtype=np.float64)
        self.p = np.zeros((self.jn + 2, 3), dtype=np.float64)

        self.p[1] = np.array([-float(self.l[0]), 0.0, 0.0], dtype=np.float64)
        self.d = int(1)

    def setval(self, i: int, angle: float) -> None:
        if i < 0 or i >= self.jn:
            raise ValueError(f"Joint index out of range: {i}")
        if abs(self.v[i] - angle) < 1e-12:
            return
        self.d = min(self.d, i + 1)
        self.v[i] = angle

    def geti(self, i: int) -> np.ndarray:
        if i < 0 or i >= self.jn + 2:
            raise ValueError(f"Joint index out of range: {i}")
        while i > self.d:
            self._sync()
        return self.p[i]

    def getf(self, pos: float) -> np.ndarray:
        if pos < 0.0 or pos > self.jn + 1:
            raise ValueError(f"Position out of range: {pos}")
        i, j = math.floor(pos), math.ceil(pos)
        if i == j:
            return self.geti(i)
        pi = self.geti(i)
        pj = self.geti(j)
        t = pos - i
        return (1.0 - t) * pi + t * pj
    
    def getallp(self) -> np.ndarray:
        while self.d <= self.jn:
            self._sync()
        return self.p

    def _sync(self) -> None:
        d = self.d
        rot_fn = {
            Axis.X: _rot_x,
            Axis.Y: _rot_y,
            Axis.Z: _rot_z,
        }[self.ja[d - 1]]
        rot = rot_fn(self.js[d - 1] * float(self.v[d - 1]))
        # matrix multiply previous frame by joint rotation to get new frame
        self.M[d] = self.M[d - 1] @ rot
        # update position of point d (endpoint of previous segment)
        seg_len = self.l[d]
        self.p[d + 1] = self.p[d] + (self.M[d] @ np.array([-seg_len, 0.0, 0.0], dtype=np.float64))
        self.d += 1


__all__ = ["FK"]
