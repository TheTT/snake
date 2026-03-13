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
    *,
    f_fn: CurveFn,
    g_fn: TwistFn,
    t: float,
    joint_axes: Sequence[Axis],
    joint_signs: Sequence[float],
    x_joint_indices_0b: Sequence[int],
    yz_joint_indices_0b: Sequence[int],
    lengths_m: Sequence[float],
    base_twist_rad: float,
    state: FitState,
    backend_fn: BackendFn,
) -> np.ndarray:
    n_joints = len(joint_axes)

    twist_no_base = compute_twist_filtered(
        g_fn=g_fn,
        t=t,
        x_joint_indices_0b=x_joint_indices_0b,
        lengths_m=lengths_m,
        base_twist_rad=base_twist_rad,
        state=state,
    )

    fplist = np.zeros((n_joints, 3), dtype=np.float64)  # 变换后的采样点
    backend_param = FitParam(
        fplist=fplist,
        twist_no_base=twist_no_base,
    )
    v = backend_fn(
        state.joint_yz,
        backend_param
    )

    v = np.asarray(v, dtype=np.float64).copy()
    n_yz = len(yz_joint_indices_0b)
    if v.size >= n_yz:
        state.joint_yz[:n_yz] = v[:n_yz]

    yz_vars = state.joint_yz[:n_yz].copy()
    joint_angles = assemble_joint(
        x_joint_indices_0b,
        yz_joint_indices_0b,
        joint_signs,
        state.joint_x,
        yz_vars,
        n_joints,
    )

    return joint_angles
