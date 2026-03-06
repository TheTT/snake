"""Per-joint control functions for snake.xml actuators.

Each function must accept exactly one argument: time t (seconds),
and return a target joint angle in radians.

Edit these functions to design your gait.
"""

from __future__ import annotations

import math
from typing import Callable, Dict


# Global gait knobs (optional).
AMP = 0.6
OMEGA = 2.5
PHASE = math.pi / 3.0
BIAS = 0.0


def _wave(t: float, idx: int, *, amp: float = AMP, omega: float = OMEGA, phase: float = PHASE, bias: float = BIAS) -> float:
    """Default traveling wave used by all joints unless customized."""
    return bias + amp * math.sin(omega * t - idx * phase)


def joint_z_1(t: float) -> float:
    return _wave(t, 1)


def joint_z_4(t: float) -> float:
    return _wave(t, 4)


def joint_z_7(t: float) -> float:
    return _wave(t, 7)


def joint_z_10(t: float) -> float:
    return _wave(t, 10)


def joint_z_13(t: float) -> float:
    return _wave(t, 13)


def joint_z_16(t: float) -> float:
    return _wave(t, 16)


def joint_z_18(t: float) -> float:
    return _wave(t, 18)


def joint_x_2(t: float) -> float:
    return _wave(t, 2)


def joint_x_5(t: float) -> float:
    return _wave(t, 5)


def joint_x_8(t: float) -> float:
    return _wave(t, 8)


def joint_x_11(t: float) -> float:
    return _wave(t, 11)


def joint_x_14(t: float) -> float:
    return _wave(t, 14)


def joint_x_17(t: float) -> float:
    return _wave(t, 17)


def joint_y_3(t: float) -> float:
    return _wave(t, 3)


def joint_y_6(t: float) -> float:
    return _wave(t, 6)


def joint_y_9(t: float) -> float:
    return _wave(t, 9)


def joint_y_12(t: float) -> float:
    return _wave(t, 12)


def joint_y_15(t: float) -> float:
    return _wave(t, 15)


# Map actuator name in snake.xml -> function(t).
JOINT_FUNCTIONS: Dict[str, Callable[[float], float]] = {
    "joint_2_pos": joint_x_2,
    "joint_5_pos": joint_x_5,
    "joint_8_pos": joint_x_8,
    "joint_11_pos": joint_x_11,
    "joint_14_pos": joint_x_14,
    "joint_17_pos": joint_x_17,
    "joint_3_pos": joint_y_3,
    "joint_6_pos": joint_y_6,
    "joint_9_pos": joint_y_9,
    "joint_12_pos": joint_y_12,
    "joint_15_pos": joint_y_15,
    "joint_1_pos": joint_z_1,
    "joint_4_pos": joint_z_4,
    "joint_7_pos": joint_z_7,
    "joint_10_pos": joint_z_10,
    "joint_13_pos": joint_z_13,
    "joint_16_pos": joint_z_16,
    "joint_18_pos": joint_z_18,
}
