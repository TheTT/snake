"""Unified joint command interface built on curve fitting.

This module exposes:
- JOINT_FUNCTIONS: actuator name -> function(t)
- get_theoretical_local_matrices: cached local orientation chain
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Dict, Tuple
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
    v = cfg.get(key, default)
    if v is None:
        return float(default)
    return float(v)


def _joint_center_l(joint_index: int) -> float:
    # Keep the same convention as your previous polyline setup:
    # joint i is placed at the cumulative length up to segment i.
    return float(sum(POLYLINE_SEGMENT_LENGTHS_M[: joint_index + 1]))


def _joint_span(joint_index: int) -> float:
    # Midpoint-rule window around each joint.
    # This approximates the paper's segment-wise integral over [l_i - L0/2, l_i + L0/2].
    return 0.5 * float(
        POLYLINE_SEGMENT_LENGTHS_M[joint_index] + POLYLINE_SEGMENT_LENGTHS_M[joint_index + 1]
    )


def _simpson_integrate(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 9,
) -> float:
    """
    Simpson integration on [a, b].
    n must be odd and >= 3. If even, it will be bumped by 1.
    """
    if n < 3:
        n = 3
    if n % 2 == 0:
        n += 1

    h = (b - a) / (n - 1)
    s = f(a) + f(b)
    for i in range(1, n - 1):
        x = a + i * h
        s += (4.0 if (i % 2 == 1) else 2.0) * f(x)
    return s * h / 3.0


def _shape_functions(l: float, t: float, cfg: dict) -> Tuple[float, float, float]:
    """
    Returns:
        kappa_a, kappa_b, tau

    Paper-consistent structure:
      kappa_a^C = kappa_F * sin((a1*l + a0)*t + phi_F^B)
      kappa_b^C = kappa_F * cos((a1*l + a0)*t + phi_F^B)
      tau^C     = a1 * t

    Here wave_number is interpreted as total cycles over BODY_LENGTH_M.
    """
    ka = _cfg_float(cfg, "ka", 0.03)
    kb = _cfg_float(cfg, "kb", 0.09)
    gain = _cfg_float(cfg, "gain", 1.0)

    wave_number = _cfg_float(cfg, "wave_number", 1.0)
    period = _cfg_float(cfg, "period", 5.0)
    phi_offset = _cfg_float(cfg, "phi_offset", 0.0)

    rollv = _cfg_float(cfg, "rollv", 0.0)  # a0
    a1 = _cfg_float(cfg, "a1", 0.0)

    # wave_number counts cycles over the whole body
    ktheta = wave_number / BODY_LENGTH_M
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

    phase = psi + phi_offset + (a1 * l + rollv) * t

    kappa_a = gain * kappa_f * math.sin(phase)
    kappa_b = gain * kappa_f * math.cos(phase)
    tau = a1 * t

    return kappa_a, kappa_b, tau


def _make_joint_function(joint_index: int) -> Callable[[float], float]:
    if not (0 <= joint_index < N_JOINTS):
        raise IndexError(f"joint_index out of range: {joint_index}")

    l_i = _joint_center_l(joint_index)
    span_i = _joint_span(joint_index)
    axis = JOINT_AXES[joint_index]
    sign = JOINT_SIGNS[joint_index]

    def joint_fn(t: float) -> float:
        cfg = _load_gait_params()
        t = float(t)

        # Twist joints are exact: tau^C = a1 * t, constant in l.
        if axis == Axis.X:
            a1 = _cfg_float(cfg, "a1", 0.0)
            raw_angle = span_i * (a1 * t)
            return TWIST_X_BASE_ANGLE_RAD + sign * raw_angle

        # For bending joints, integrate the rotated curvature over the local span.
        a = l_i - 0.5 * span_i
        b = l_i + 0.5 * span_i

        if axis == Axis.Z:
            # Dorsal joint in your convention -> κ_b
            integrand = lambda l: _shape_functions(l, t, cfg)[1]
        elif axis == Axis.Y:
            # Lateral joint in your convention -> κ_a
            integrand = lambda l: _shape_functions(l, t, cfg)[0]
        else:
            raise ValueError(f"Unknown axis: {axis}")

        raw_angle = _simpson_integrate(integrand, a, b, n=int(cfg.get("integration_steps", 9)))
        return sign * raw_angle

    return joint_fn


JOINT_FUNCTIONS: Dict[str, Callable[[float], float]] = {
    f"joint_{i + 1}_pos": _make_joint_function(i) for i in range(N_JOINTS)
}