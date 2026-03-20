"""Unified joint command interface built on curve fitting.

This module exposes:
- JOINT_FUNCTIONS: actuator name -> function(t)
- get_theoretical_local_matrices: cached local orientation chain
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Dict
from enum import IntEnum
from functools import lru_cache

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

class Axis(IntEnum):
    X = 0
    Y = 1
    Z = 2

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

X_JOINT_INDICES_0B = [i for i, a in enumerate(JOINT_AXES) if a == Axis.X]


GAIT_PARAMS_PATH = Path(__file__).resolve().parent / "gait" / "sw.json"


@lru_cache(maxsize=1)
def _load_gait_params() -> dict:
    with GAIT_PARAMS_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected JSON object in {GAIT_PARAMS_PATH}, got {type(cfg).__name__}")
    return cfg


def _cfg_float(cfg: dict, key: str, default: float = 0.0) -> float:
    val = cfg.get(key, default)
    if val is None:
        return float(default)
    return float(val)


# Center arc-length of each joint, using the midpoint of the adjacent polyline spans.
_JOINT_CENTER_LS_M = tuple(
    sum(POLYLINE_SEGMENT_LENGTHS_M[: i + 1]) for i in range(N_JOINTS)
)

# Local integration length for each joint.
# Use the average of the two adjacent polyline segments as a midpoint-rule approximation.
_JOINT_SPANS_M = tuple(
    0.5 * (POLYLINE_SEGMENT_LENGTHS_M[i] + POLYLINE_SEGMENT_LENGTHS_M[i + 1])
    for i in range(N_JOINTS)
)


def _shape_functions(l: float, t: float, cfg: dict) -> tuple[float, float, float]:
    """
    Returns:
        kappa_a, kappa_b, tau

    Notes:
        wave_number = number of full cycles over BODY_LENGTH_M
        period      = wave period in seconds
        rollv       = a0 in the paper
        a1          = twisting gradient in the paper
    """
    ka = _cfg_float(cfg, "ka", 0.03)
    kb = _cfg_float(cfg, "kb", 0.09)
    gain = _cfg_float(cfg, "gain", 1.0)

    wave_number = _cfg_float(cfg, "wave_number", 1.0)
    period = _cfg_float(cfg, "period", 5.0)
    phi0 = _cfg_float(cfg, "phi_offset", 0.0)
    rollv = _cfg_float(cfg, "rollv", 0.0)
    a1 = _cfg_float(cfg, "a1", 0.0)

    # cycles per meter
    ktheta = wave_number / BODY_LENGTH_M

    # cycles per second -> rad/s factor is absorbed by 2*pi in phase
    omega_t = 1.0 / period if period != 0.0 else 0.0

    psi = 2.0 * math.pi * (ktheta * l - omega_t * t)

    denom = (
        ka * ka * (math.sin(psi) ** 2)
        + kb * kb * (math.cos(psi) ** 2)
        + ktheta * ktheta
    ) ** 1.5

    if denom == 0.0:
        kappa_f = 0.0
    else:
        kappa_f = (ka * kb * ktheta * ktheta) / denom

    # complete-frame phase
    phase = psi + phi0 + (a1 * l + rollv) * t

    kappa_a = gain * kappa_f * math.sin(phase)
    kappa_b = gain * kappa_f * math.cos(phase)

    # torsion density; the joint output integrates this over one span
    tau = a1 * t

    return kappa_a, kappa_b, tau


def _make_joint_function(joint_index: int) -> Callable[[float], float]:
    if not (0 <= joint_index < N_JOINTS):
        raise IndexError(f"joint_index out of range: {joint_index}")

    cfg = _load_gait_params()

    l_i = _JOINT_CENTER_LS_M[joint_index]
    span_i = _JOINT_SPANS_M[joint_index]
    axis = JOINT_AXES[joint_index]
    sign = JOINT_SIGNS[joint_index]

    def joint_fn(t: float) -> float:
        kappa_a, kappa_b, tau = _shape_functions(l_i, float(t), cfg)

        # Keep this mapping consistent with your mechanism:
        #   Axis.Z -> dorsal
        #   Axis.Y -> lateral
        #   Axis.X -> twist
        if axis == Axis.Z:
            raw_angle = span_i * kappa_b
        elif axis == Axis.Y:
            raw_angle = span_i * kappa_a
        elif axis == Axis.X:
            raw_angle = span_i * tau - TWIST_X_BASE_ANGLE_RAD
        else:
            raise ValueError(f"Unknown axis: {axis}")

        return sign * raw_angle

    return joint_fn


JOINT_FUNCTIONS: Dict[str, Callable[[float], float]] = {
    f"joint_{i + 1}_pos": _make_joint_function(i) for i in range(N_JOINTS)
}