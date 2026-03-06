"""Unified gait interface for snake.xml actuators.

This module exposes:
- JOINT_FUNCTIONS: actuator name -> function(t) generated for all joints
"""

from __future__ import annotations

from typing import Callable, Dict

from gait_function import MASK_X, MASK_Y, MASK_Z, f

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
    (IDX_Z, 1.0, MASK_Z),
    (IDX_X, -1.0, MASK_X),
    (IDX_Y, -1.0, MASK_Y),
)

def _make_joint_function(c: float, component_index: int, sign: float, component_mask: int) -> Callable[[float], float]:
    def _joint_fn(t: float) -> float:
        # Apply axis mapping and sign correction for negative axes (-x, -y).
        return sign * f(c, t, component_mask)[component_index]

    return _joint_fn


# Programmatically generate map actuator_name -> function(t).
JOINT_FUNCTIONS: Dict[str, Callable[[float], float]] = {}

for i in range(1, N_JOINTS + 1):
    # Normalized body coordinate: c = (i - 1) / (N - 1), with N=18.
    c = (i - 1) / (N_JOINTS - 1)
    component_index, sign, component_mask = AXIS_CYCLE[(i - 1) % len(AXIS_CYCLE)]
    actuator_name = f"joint_{i}_pos"
    JOINT_FUNCTIONS[actuator_name] = _make_joint_function(c, component_index, sign, component_mask)
