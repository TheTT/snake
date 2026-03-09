"""Unified gait interface for snake.xml actuators.

This module exposes:
- JOINT_FUNCTIONS: actuator name -> function(t) generated for all joints
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, TypedDict

from gait_function import zro

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

# Component bitmask: bit0=x, bit1=y, bit2=z.
MASK_X = 1
MASK_Y = 2
MASK_Z = 4

# Component index in vec3f.
IDX_X = 0
IDX_Y = 1
IDX_Z = 2

# Per-joint axis definition for joint_1..joint_18.
# Edit this list directly when tuning each segment by hand.
# Allowed values: "x", "y", "z".
JOINT_AXES = [
    "z", "x", "y",
    "z", "x", "y",
    "z", "x", "y",
    "z", "x", "y",
    "z", "x", "y",
    "z", "x", "z",
]

# Per-joint direction sign for joint_1..joint_18.
# Use +1 for normal direction and -1 for reversed direction.
JOINT_SIGNS = [
    +1, -1, -1,
    +1, -1, -1,
    +1, -1, -1,
    +1, -1, -1,
    +1, -1, -1,
    +1, -1, -1,
]

AXIS_TO_COMPONENT = {
    "x": (IDX_X, MASK_X),
    "y": (IDX_Y, MASK_Y),
    "z": (IDX_Z, MASK_Z),
}

STARTUP_RAMP_SEC = 1.0
TWIST_X_BASE_ANGLE_RAD = math.pi / 2.0

Vec3f = tuple[float, float, float]
Mat3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
GaitFn = Callable[[float, float, int], Vec3f]


class JointConfig(TypedDict):
    c: float
    axis_name: str
    component_index: int
    component_mask: int
    sign: float
    base_offset: float


def _identity3() -> Mat3:
    return (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


def _matmul3(a: Mat3, b: Mat3) -> Mat3:
    return (
        (
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ),
        (
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ),
        (
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ),
    )


def _rot_axis(axis_name: str, angle: float) -> Mat3:
    ca = math.cos(angle)
    sa = math.sin(angle)

    if axis_name == "x":
        return (
            (1.0, 0.0, 0.0),
            (0.0, ca, -sa),
            (0.0, sa, ca),
        )
    if axis_name == "y":
        return (
            (ca, 0.0, sa),
            (0.0, 1.0, 0.0),
            (-sa, 0.0, ca),
        )
    if axis_name == "z":
        return (
            (ca, -sa, 0.0),
            (sa, ca, 0.0),
            (0.0, 0.0, 1.0),
        )

    raise ValueError(f"Invalid axis '{axis_name}'; expected x/y/z")


def _smoothstep01(x: float) -> float:
    """C1-smooth interpolation from 0 to 1 for x in [0, 1]."""
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)

def _joint_angle_at_time(
    c: float,
    component_index: int,
    component_mask: int,
    sign: float,
    base_offset: float,
    gait_fn: GaitFn,
    t: float,
) -> float:
    if t <= 0.0:
        return base_offset

    if t < STARTUP_RAMP_SEC:
        # Smoothly morph from base offset to base offset + gait(c, t=0).
        alpha = _smoothstep01(t / STARTUP_RAMP_SEC)
        target0 = gait_fn(c, 0.0, component_mask)[component_index]
        return base_offset + sign * alpha * target0

    # After ramp, run gait from local time 0 to ensure continuity at handoff.
    return base_offset + sign * gait_fn(c, t - STARTUP_RAMP_SEC, component_mask)[component_index]


def _make_joint_function(c: float, component_index: int, component_mask: int, sign: float, base_offset: float, gait_fn: GaitFn, joint_index: int) -> Callable[[float], float]:
    def _joint_fn(t: float) -> float:
        _refresh_theoretical_state(t, gait_fn)
        return _LATEST_JOINT_ANGLES[joint_index]

    return _joint_fn


# Theoretical state cache (head frame fixed as identity).
# Index convention:
# - THEORETICAL_LOCAL_MATRICES[0] = head frame = I
# - THEORETICAL_LOCAL_MATRICES[i] = segment i frame for i in [1..N_JOINTS]
THEORETICAL_LOCAL_MATRICES: List[Mat3] = [_identity3() for _ in range(N_JOINTS + 1)]
_LATEST_JOINT_ANGLES: List[float] = [0.0 for _ in range(N_JOINTS)]
_LAST_REFRESH_T: float | None = None
_LAST_REFRESH_GAIT_ID: int | None = None


def _refresh_theoretical_state(t: float, gait_fn: GaitFn) -> None:
    """Refresh cached joint angles and theoretical local frames once per (t, gait_fn)."""
    global _LAST_REFRESH_T, _LAST_REFRESH_GAIT_ID

    gait_id = id(gait_fn)
    if _LAST_REFRESH_T == t and _LAST_REFRESH_GAIT_ID == gait_id:
        return

    for idx, cfg in enumerate(_JOINT_CONFIGS):
        _LATEST_JOINT_ANGLES[idx] = _joint_angle_at_time(
            c=cfg["c"],
            component_index=cfg["component_index"],
            component_mask=cfg["component_mask"],
            sign=cfg["sign"],
            base_offset=cfg["base_offset"],
            gait_fn=gait_fn,
            t=t,
        )

    current = _identity3()
    THEORETICAL_LOCAL_MATRICES[0] = current
    for i, cfg in enumerate(_JOINT_CONFIGS, start=1):
        rot_i = _rot_axis(cfg["axis_name"], _LATEST_JOINT_ANGLES[i - 1])
        current = _matmul3(current, rot_i)
        THEORETICAL_LOCAL_MATRICES[i] = current

    _LAST_REFRESH_T = t
    _LAST_REFRESH_GAIT_ID = gait_id


# Programmatically generate map actuator_name -> function(t).
JOINT_FUNCTIONS: Dict[str, Callable[[float], float]] = {}
_JOINT_CONFIGS: List[JointConfig] = []

if len(JOINT_AXES) != N_JOINTS:
    raise ValueError(f"JOINT_AXES length must be {N_JOINTS}, got {len(JOINT_AXES)}")
if len(JOINT_SIGNS) != N_JOINTS:
    raise ValueError(f"JOINT_SIGNS length must be {N_JOINTS}, got {len(JOINT_SIGNS)}")

for i in range(1, N_JOINTS + 1):
    # Normalized body coordinate: c = (i - 1) / (N - 1), with N=18.
    c = (i - 1) / (N_JOINTS - 1)
    axis_name = JOINT_AXES[i - 1].lower()
    if axis_name not in AXIS_TO_COMPONENT:
        raise ValueError(f"Invalid axis '{JOINT_AXES[i - 1]}' at joint_{i}; expected x/y/z")

    component_index, component_mask = AXIS_TO_COMPONENT[axis_name]
    sign = float(JOINT_SIGNS[i - 1])
    if sign not in (-1.0, 1.0):
        raise ValueError(f"Invalid sign '{JOINT_SIGNS[i - 1]}' at joint_{i}; expected +1 or -1")

    base_offset = TWIST_X_BASE_ANGLE_RAD if axis_name == "x" else 0.0
    _JOINT_CONFIGS.append(
        {
            "c": c,
            "axis_name": axis_name,
            "component_index": component_index,
            "component_mask": component_mask,
            "sign": sign,
            "base_offset": base_offset,
        }
    )

    actuator_name = f"joint_{i}_pos"
    JOINT_FUNCTIONS[actuator_name] = _make_joint_function(
        c,
        component_index,
        component_mask,
        sign,
        base_offset,
        zro,
        i - 1,
    )


def get_theoretical_local_matrices(t: float | None = None, gait_fn: GaitFn = zro) -> List[Mat3]:
    """Return cached theoretical local coordinate matrices.

    If `t` is provided, cache is refreshed at that time before returning.
    """
    if t is not None:
        _refresh_theoretical_state(t, gait_fn)
    return THEORETICAL_LOCAL_MATRICES
