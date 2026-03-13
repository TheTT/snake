from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from enum import IntEnum
from typing import Sequence

import numpy as np

Vec3 = tuple[float, float, float]
CurveFn = Callable[[float, float], Vec3]
TwistFn = Callable[[float, float], float]


@dataclass
class FitState:
    joint_tar: np.ndarray
    last_align_plane_n: np.ndarray | None = None


class Axis(IntEnum):
    X = 0
    Y = 1
    Z = 2


@dataclass
class FitParam:
    fplist: np.ndarray
    twist_no_base: np.ndarray
    joint_axes: Sequence[Axis]
    joint_signs: Sequence[float]
BackendFn = Callable[[np.ndarray, FitParam], np.ndarray]


def create_initial_state(num_joints: int) -> FitState:
    return FitState(
        joint_tar=np.zeros(num_joints, dtype=np.float64),
        last_align_plane_n=None,
    )
