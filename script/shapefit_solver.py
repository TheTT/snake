from __future__ import annotations

import math
from typing import Callable, Sequence
from shapefit_types import Axis

import numpy as np

from shapefit_types import TwistFn, CurveFn, BackendFn, FitState, FitParam


def compute_twist(
    g_fn: TwistFn,
    t: float,
    *,
    x_joint_indices_0b: Sequence[int],
    seglen: Sequence[float],
    state: FitState,
) -> np.ndarray:
    """Compute per-x-joint twist. Returns twist values without base_twist (for FK/geometry use).
    """
    ...


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
    seglen: Sequence[float],
    backend_fn: BackendFn,
) -> None:
    twist_no_base = compute_twist(
        g_fn=g_fn,
        t=t,
        x_joint_indices_0b=x_joint_indices_0b,
        seglen=seglen,
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
