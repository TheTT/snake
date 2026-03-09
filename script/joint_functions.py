"""Unified gait interface for snake.xml actuators.

This module exposes:
- JOINT_FUNCTIONS: actuator name -> function(t) generated for all joints
"""

from __future__ import annotations

from typing import Callable, Dict

from gait_function import zro

N_JOINTS = 18

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


def _smoothstep01(x: float) -> float:
    """C1-smooth interpolation from 0 to 1 for x in [0, 1]."""
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)

def _make_joint_function(c: float, component_index: int, component_mask: int, sign: float, gait_fn: Callable[[float, float, int], tuple]) -> Callable[[float], float]:
    def _joint_fn(t: float) -> float:
        if t <= 0.0:
            return 0.0

        if t < STARTUP_RAMP_SEC:
            # Smoothly morph from zero to the static pose gait_fn(c, t=0).
            alpha = _smoothstep01(t / STARTUP_RAMP_SEC)
            target0 = gait_fn(c, 0.0, component_mask)[component_index]
            return sign * alpha * target0

        # After ramp, run gait from local time 0 to ensure continuity at handoff.
        return sign * gait_fn(c, t - STARTUP_RAMP_SEC, component_mask)[component_index]

    return _joint_fn


# Programmatically generate map actuator_name -> function(t).
JOINT_FUNCTIONS: Dict[str, Callable[[float], float]] = {}

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

    actuator_name = f"joint_{i}_pos"
    JOINT_FUNCTIONS[actuator_name] = _make_joint_function(c, component_index, component_mask, sign, zro)
