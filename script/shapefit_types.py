from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from enum import IntEnum

import numpy as np

Vec3 = tuple[float, float, float]
CurveFn = Callable[[float, float], Vec3]
TwistFn = Callable[[float, float], float]


@dataclass
class FitState:
    joint_yz: np.ndarray
    joint_x: np.ndarray  # no init 90deg
    last_t: float | None = None
    last_align_plane_n: np.ndarray | None = None


@dataclass
class FitParam:
    fplist: np.ndarray
    twist_no_base: np.ndarray
BackendFn = Callable[[np.ndarray, FitParam], np.ndarray]


class Axis(IntEnum):
    X = 0
    Y = 1
    Z = 2


def create_initial_state(num_yz_joints: int, num_x_joints: int) -> FitState:
    return FitState(
        joint_yz=np.zeros(num_yz_joints, dtype=np.float64),
        joint_x=np.zeros(num_x_joints, dtype=np.float64),
        last_t=None,
        last_align_plane_n=None,
    )
