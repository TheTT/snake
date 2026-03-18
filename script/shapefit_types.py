from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any, Union, Sequence
from enum import IntEnum

import numpy as np

CurveFn = Callable[[float, float], np.ndarray]
TwistFn = Callable[[float, float], float]


@dataclass
class FitState:
    joint_tar: np.ndarray
    last_UP: np.ndarray


class Axis(IntEnum):
    X = 0
    Y = 1
    Z = 2


@dataclass
class FitParam:
    jn: int
    fplist: np.ndarray
    hint_i: np.ndarray
    twist: np.ndarray
    joint_axes: Sequence[Axis]
    joint_signs: Sequence[float]
    seglen: Sequence[float]
    hint_rad: int


@dataclass
class DebugInfo:
    fk_points: np.ndarray
    expected_points: np.ndarray
    hint_i: np.ndarray


BackendFn = Callable[[np.ndarray, FitParam], Union[np.ndarray, tuple[np.ndarray, Any]]]


def create_initial_state(num_joints: int) -> FitState:
    return FitState(
        joint_tar=np.zeros(num_joints, dtype=np.float64),
        last_UP=np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
