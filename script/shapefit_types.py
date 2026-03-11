from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from enum import Enum

import numpy as np

Vec3 = tuple[float, float, float]
CurveFn = Callable[[float, float], Vec3]
TwistFn = Callable[[float, float], float]


class Backend(Enum):
    JACOB = "jacob"
    LINEAR = "linear"
    ANNEAL = "anneal"
    ACF = "acf"
    STEP = "step"


@dataclass
class FitConfig:
    startup_ramp_sec: float = 1.0
    max_iter: int = 3
    damping: float = 1e-2
    finite_diff_eps: float = 1e-4
    integration_samples: int = 5
    lowpass_tau_sec: float = 0.05
    temporal_reg_weight: float = 1e-2
    backend: Backend = Backend.ACF


@dataclass
class FitState:
    yz_and_head: np.ndarray
    twist_filtered: np.ndarray
    last_t: float | None = None


def create_initial_state(num_yz_joints: int, num_x_joints: int) -> FitState:
    return FitState(
        yz_and_head=np.zeros(num_yz_joints + 6, dtype=np.float64),
        twist_filtered=np.zeros(num_x_joints, dtype=np.float64),
        last_t=None,
    )
