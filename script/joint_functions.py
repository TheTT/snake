"""Unified joint command interface built on curve fitting.

This module exposes:
- JOINT_FUNCTIONS: actuator name -> function(t)
- get_theoretical_local_matrices: cached local orientation chain
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence

import numpy as np
from shapefit_types import Axis

from gait_function import f, g
from shape_fit import DebugInfo, create_initial_state, solve_shape_for_time
import anneal_backend

N_JOINTS = 18

# Manual polyline segment lengths (meters). Must have length N_JOINTS + 1.
POLYLINE_SEGMENT_LENGTHS_M = [
    0.083,
    0.049, 0.0805, 0.076,
    0.049, 0.0805, 0.076,
    0.049, 0.0805, 0.076,
    0.049, 0.0805, 0.076,
    0.049, 0.0805, 0.076,
    0.049, 0.0805,
    0.113
]

BODY_LENGTH_M = float(sum(float(x) for x in POLYLINE_SEGMENT_LENGTHS_M))

JOINT_AXES = [
    Axis.Z, Axis.X, Axis.Y,
    Axis.Z, Axis.X, Axis.Y,
    Axis.Z, Axis.X, Axis.Y,
    Axis.Z, Axis.X, Axis.Y,
    Axis.Z, Axis.X, Axis.Y,
    Axis.Z, Axis.X, Axis.Y,
]

JOINT_SIGNS = [
    +1, -1, -1,
    +1, -1, -1,
    +1, -1, -1,
    +1, -1, -1,
    +1, -1, -1,
    +1, -1, -1,
]

TWIST_X_BASE_ANGLE_RAD = math.pi / 2.0

Mat3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]

if len(JOINT_AXES) != N_JOINTS:
    raise ValueError(f"JOINT_AXES length must be {N_JOINTS}, got {len(JOINT_AXES)}")
if len(JOINT_SIGNS) != N_JOINTS:
    raise ValueError(f"JOINT_SIGNS length must be {N_JOINTS}, got {len(JOINT_SIGNS)}")
if len(POLYLINE_SEGMENT_LENGTHS_M) != N_JOINTS + 1:
    raise ValueError(
        "POLYLINE_SEGMENT_LENGTHS_M length must be "
        f"{N_JOINTS + 1}, got {len(POLYLINE_SEGMENT_LENGTHS_M)}"
    )

X_JOINT_INDICES_0B = [i for i, a in enumerate(JOINT_AXES) if a == Axis.X]

_FIT_STATE = create_initial_state(
    num_joints=N_JOINTS,
)

_LATEST_JOINT_ANGLES: List[float] = [0.0 for _ in range(N_JOINTS)]
_LAST_REFRESH_T: float | None = None
_LAST_DEBUG_INFO: DebugInfo | None = None


def _refresh_theoretical_state(t: float) -> None:
    global _LAST_REFRESH_T, _LAST_DEBUG_INFO

    if _LAST_REFRESH_T == t:
        return

    _LAST_DEBUG_INFO = solve_shape_for_time(
        _FIT_STATE,
        f_fn=f,
        g_fn=g,
        t=float(t),
        n_joints=N_JOINTS,
        joint_axes=JOINT_AXES,
        joint_signs=JOINT_SIGNS,
        x_joint_indices_0b=X_JOINT_INDICES_0B,
        seglen=POLYLINE_SEGMENT_LENGTHS_M,
        totlen=BODY_LENGTH_M,
        backend_fn=lambda v0, p: anneal_backend.anneal_backend(v0, p, window_size=3),
    )

    for i in range(N_JOINTS):
        _LATEST_JOINT_ANGLES[i] = (float(_FIT_STATE.joint_tar[i]) + TWIST_X_BASE_ANGLE_RAD) if (JOINT_AXES[i] == Axis.X) else (float(_FIT_STATE.joint_tar[i]))

    _LAST_REFRESH_T = float(t)


def get_debug_info(t: float) -> DebugInfo | None:
    _refresh_theoretical_state(float(t))
    return _LAST_DEBUG_INFO


def _make_joint_function(joint_index: int) -> Callable[[float], float]:
    def _joint_fn(t: float) -> float:
        _refresh_theoretical_state(float(t))
        return _LATEST_JOINT_ANGLES[joint_index]

    return _joint_fn


JOINT_FUNCTIONS: Dict[str, Callable[[float], float]] = {
    f"joint_{i + 1}_pos": _make_joint_function(i) for i in range(N_JOINTS)
}
