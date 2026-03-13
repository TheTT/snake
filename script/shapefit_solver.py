from __future__ import annotations

import math
from typing import Callable, Sequence
from shapefit_types import Axis

import numpy as np

from shapefit_types import TwistFn, CurveFn, BackendFn, FitState, FitParam


def compute_twist_filtered(
    *,
    g_fn: TwistFn,
    t: float,
    x_joint_indices_0b: Sequence[int],
    lengths_m: Sequence[float],
    base_twist_rad: float,
    state: FitState,
) -> np.ndarray:
    """Compute per-x-joint twist, apply startup ramp and lowpass, update state.

    Returns twist values with base_twist removed (for FK/geometry use).
    """
    ...
    # return twist_no_base


def solve_shape_for_time(
    state: FitState,
    *,
    f_fn: CurveFn,
    g_fn: TwistFn,
    t: float,
    n_joints: int,
    joint_axes: Sequence[Axis],
    joint_signs: Sequence[float],
    x_joint_indices_0b: Sequence[int],
    lengths_m: Sequence[float],
    base_twist_rad: float,
    backend_fn: BackendFn,
) -> None:
    twist_no_base = compute_twist_filtered(
        g_fn=g_fn,
        t=t,
        x_joint_indices_0b=x_joint_indices_0b,
        lengths_m=lengths_m,
        base_twist_rad=base_twist_rad,
        state=state,
    )

    fplist = np.zeros((n_joints, 3), dtype=np.float64)  # Transformed sampling points
    backend_param = FitParam(
        fplist=fplist,
        twist_no_base=twist_no_base,
        joint_axes=joint_axes,
        joint_signs=joint_signs,
    )
    v = backend_fn(
        state.joint_tar,
        backend_param,
    )
    state.joint_tar[:] = v[:]
