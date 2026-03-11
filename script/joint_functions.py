"""Unified joint command interface built on curve fitting.

This module exposes:
- JOINT_FUNCTIONS: actuator name -> function(t)
- get_theoretical_local_matrices: cached local orientation chain
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List

import numpy as np

from gait_function import f, g
from shape_fit import FitConfig, create_initial_state, solve_shape_for_time

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

JOINT_AXES = [
    "z", "x", "y",
    "z", "x", "y",
    "z", "x", "y",
    "z", "x", "y",
    "z", "x", "y",
    "z", "x", "z",
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

X_JOINT_INDICES_0B = [i for i, a in enumerate(JOINT_AXES) if a == "x"]
YZ_JOINT_INDICES_0B = [i for i, a in enumerate(JOINT_AXES) if a in ("y", "z")]

_FIT_CFG = FitConfig(
    startup_ramp_sec=1.0,
    max_iter=3,
    damping=1e-2,
    finite_diff_eps=1e-4,
    integration_samples=5,
    lowpass_tau_sec=0.05,
)
_FIT_STATE = create_initial_state(
    num_yz_joints=len(YZ_JOINT_INDICES_0B),
    num_x_joints=len(X_JOINT_INDICES_0B),
)

_LATEST_JOINT_ANGLES: List[float] = [0.0 for _ in range(N_JOINTS)]
THEORETICAL_LOCAL_MATRICES: List[Mat3] = [
    (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    for _ in range(N_JOINTS + 1)
]
_LAST_REFRESH_T: float | None = None


def _np_to_mat3(r: np.ndarray) -> Mat3:
    return (
        (float(r[0, 0]), float(r[0, 1]), float(r[0, 2])),
        (float(r[1, 0]), float(r[1, 1]), float(r[1, 2])),
        (float(r[2, 0]), float(r[2, 1]), float(r[2, 2])),
    )


def _refresh_theoretical_state(t: float) -> None:
    global _LAST_REFRESH_T

    if _LAST_REFRESH_T == t:
        return

    joint_angles, _points, frames, _head = solve_shape_for_time(
        f_fn=f,
        g_fn=g,
        t=float(t),
        joint_axes=JOINT_AXES,
        joint_signs=JOINT_SIGNS,
        x_joint_indices_0b=X_JOINT_INDICES_0B,
        yz_joint_indices_0b=YZ_JOINT_INDICES_0B,
        lengths_m=POLYLINE_SEGMENT_LENGTHS_M,
        base_twist_rad=TWIST_X_BASE_ANGLE_RAD,
        state=_FIT_STATE,
        cfg=_FIT_CFG,
    )

    for i in range(N_JOINTS):
        _LATEST_JOINT_ANGLES[i] = float(joint_angles[i])

    THEORETICAL_LOCAL_MATRICES[0] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    for i in range(1, N_JOINTS + 1):
        THEORETICAL_LOCAL_MATRICES[i] = _np_to_mat3(frames[i - 1])

    _LAST_REFRESH_T = float(t)


def _make_joint_function(joint_index: int) -> Callable[[float], float]:
    def _joint_fn(t: float) -> float:
        _refresh_theoretical_state(float(t))
        return _LATEST_JOINT_ANGLES[joint_index]

    return _joint_fn


JOINT_FUNCTIONS: Dict[str, Callable[[float], float]] = {
    f"joint_{i + 1}_pos": _make_joint_function(i) for i in range(N_JOINTS)
}


def get_theoretical_local_matrices(_gait_fn: object = None, t: float | None = None) -> List[Mat3]:
    """Return cached theoretical local coordinate matrices.

    `_gait_fn` is ignored and kept only for backward compatibility.
    """
    if t is not None:
        _refresh_theoretical_state(float(t))
    return THEORETICAL_LOCAL_MATRICES
