"""Unified gait interface for snake.xml actuators.

This module exposes:
- JOINT_FUNCTIONS: actuator name -> function(t) generated for all joints
"""

from __future__ import annotations

from typing import Callable, Dict

from gait_function import f

N_JOINTS = 18

# Component bitmask: bit0=x, bit1=y, bit2=z.
MASK_X = 1
MASK_Y = 2
MASK_Z = 4

# Component index in vec3f.
IDX_X = 0
IDX_Y = 1
IDX_Z = 2

# Axis cycle for joint_1..joint_18: z, -x, -y, z, -x, -y, ...
# Tuple layout: (component_index, sign, single_component_mask)
AXIS_CYCLE = (
    (IDX_Z, MASK_Z),
    (IDX_X, MASK_X),
    (IDX_Y, MASK_Y),
)

STARTUP_RAMP_SEC = 1.0


def _smoothstep01(x: float) -> float:
    """C1-smooth interpolation from 0 to 1 for x in [0, 1]."""
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)

def _make_joint_function(c: float, component_index: int, component_mask: int) -> Callable[[float], float]:
    def _joint_fn(t: float) -> float:
        if t <= 0.0:
            return 0.0

        if t < STARTUP_RAMP_SEC:
            # Smoothly morph from zero to the static pose f(c, t=0).
            alpha = _smoothstep01(t / STARTUP_RAMP_SEC)
            target0 = f(c, 0.0, component_mask)[component_index]
            return alpha * target0

        # After ramp, run gait from local time 0 to ensure continuity at handoff.
        return f(c, t - STARTUP_RAMP_SEC, component_mask)[component_index]

    return _joint_fn


# Programmatically generate map actuator_name -> function(t).
JOINT_FUNCTIONS: Dict[str, Callable[[float], float]] = {}

for i in range(1, N_JOINTS + 1):
    # Normalized body coordinate: c = (i - 1) / (N - 1), with N=18.
    c = (i - 1) / (N_JOINTS - 1)
    component_index, component_mask = AXIS_CYCLE[(i - 1) % len(AXIS_CYCLE)]
    actuator_name = f"joint_{i}_pos"
    JOINT_FUNCTIONS[actuator_name] = _make_joint_function(c, component_index, component_mask)
