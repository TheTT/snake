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
    return float(default if v is None else v)


def _joint_center_l(joint_index: int) -> float:
    return float(sum(POLYLINE_SEGMENT_LENGTHS_M[: joint_index + 1]))


def _joint_span(joint_index: int) -> float:
    return 0.5 * float(
        POLYLINE_SEGMENT_LENGTHS_M[joint_index] + POLYLINE_SEGMENT_LENGTHS_M[joint_index + 1]
    )


def _simpson_integrate(f: Callable[[float], float], a: float, b: float, n: int = 9) -> float:
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


MODULE_COUNT = N_JOINTS // 3
MODULE_LENGTH_M = BODY_LENGTH_M / MODULE_COUNT
TWIST_EPS = 1e-12


def _shape_functions(l: float, t: float, cfg: dict) -> Tuple[float, float, float]:
    """
    Engineering-friendly version:
      - ka, kb directly set the two principal bending amplitudes
      - twist rotates these two components in the normal plane
      - tau = a1 * t
    """
    ka = _cfg_float(cfg, "ka", 0.03)
    kb = _cfg_float(cfg, "kb", 0.09)
    gain = _cfg_float(cfg, "gain", 1.0)

    wave_number = _cfg_float(cfg, "wave_number", 1.0)
    period = _cfg_float(cfg, "period", 5.0)
    phi_offset = _cfg_float(cfg, "phi_offset", 0.0)

    rollv = _cfg_float(cfg, "rollv", 0.0)  # a0
    a1 = _cfg_float(cfg, "a1", 0.0)

    # wave_number = number of cycles over the whole body length
    ktheta = wave_number / BODY_LENGTH_M
    omega_t = 1.0 / period if period != 0.0 else 0.0

    # Base sidewinding phase
    psi = 2.0 * math.pi * (ktheta * l - omega_t * t) + phi_offset

    # Principal bending components in the bellows frame
    # Use a flat ellipse directly: one axis gets ka, the other gets kb.
    # You can swap sin/cos if your frame convention feels 90° shifted.
    kappa_a_b = gain * ka * math.sin(psi)
    kappa_b_b = gain * kb * math.cos(psi)

    # Twist/rolling rotation about the tangent
    phi_b_c = (a1 * l + rollv) * t
    c = math.cos(phi_b_c)
    s = math.sin(phi_b_c)

    # Rotate from bellows frame B to complete frame C
    kappa_a_c = c * kappa_a_b + s * kappa_b_b
    kappa_b_c = -s * kappa_a_b + c * kappa_b_b

    # torsion
    tau_c = a1 * t

    return kappa_a_c, kappa_b_c, tau_c


def _shape_functions_piecewise(
    l: float,
    t: float,
    cfg: dict,
    *,
    l_ref: float,
) -> Tuple[float, float, float]:
    """
    Paper-consistent piecewise approximation for twisting sidewinding.

    For a1 != 0:
      - freeze phi_B^C = (a1*l + a0)*t at the joint center l_ref
      - use omega_t = 0 (paper's twisting experiments)
    For a1 == 0:
      - this function should not be used; keep rolling mode path unchanged.
    """
    ka = _cfg_float(cfg, "ka", 0.03)
    kb = _cfg_float(cfg, "kb", 0.09)
    gain = _cfg_float(cfg, "gain", 1.0)

    wave_number = _cfg_float(cfg, "wave_number", 1.0)
    phi_offset = _cfg_float(cfg, "phi_offset", 0.0)

    a0 = _cfg_float(cfg, "rollv", 0.0)
    a1 = _cfg_float(cfg, "a1", 0.0)

    # Twisting sidewinding in the paper uses omega_t = 0 in the turning experiments.
    omega_t = 0.0 if abs(a1) > TWIST_EPS else _cfg_float(cfg, "omega_t", 0.0)

    ktheta = wave_number / BODY_LENGTH_M
    psi = 2.0 * math.pi * (ktheta * l - omega_t * t) + phi_offset

    # Base bending in bellows frame
    kappa_a_b = gain * ka * math.sin(psi)
    kappa_b_b = gain * kb * math.cos(psi)

    # Freeze the twist phase at the joint center
    phi_b_c_ref = (a1 * l_ref + a0) * t
    c = math.cos(phi_b_c_ref)
    s = math.sin(phi_b_c_ref)

    kappa_a_c = c * kappa_a_b + s * kappa_b_b
    kappa_b_c = -s * kappa_a_b + c * kappa_b_b

    tau_c = a1 * t
    return kappa_a_c, kappa_b_c, tau_c


def _make_joint_function(joint_index: int) -> Callable[[float], float]:
    if not (0 <= joint_index < N_JOINTS):
        raise IndexError(f"joint_index out of range: {joint_index}")

    cfg = _load_gait_params()
    l_i = _joint_center_l(joint_index)
    span_i = _joint_span(joint_index)
    axis = JOINT_AXES[joint_index]
    sign = JOINT_SIGNS[joint_index]

    a1 = _cfg_float(cfg, "a1", 0.0)
    turning_mode = abs(a1) > TWIST_EPS

    def joint_fn(t: float) -> float:
        t = float(t)

        # Twist joint
        if axis == Axis.X:
            if not turning_mode:
                # keep rolling mode unchanged
                return TWIST_X_BASE_ANGLE_RAD
            raw = MODULE_LENGTH_M * a1 * t
            return TWIST_X_BASE_ANGLE_RAD + sign * raw

        # Rolling mode: keep your original behavior unchanged
        if not turning_mode:
            a = l_i - 0.5 * span_i
            b = l_i + 0.5 * span_i

            if axis == Axis.Y:
                integrand = lambda l: _shape_functions(l, t, cfg)[0]
            elif axis == Axis.Z:
                integrand = lambda l: _shape_functions(l, t, cfg)[1]
            else:
                raise ValueError(f"Unknown axis: {axis}")

            raw = _simpson_integrate(
                integrand,
                a,
                b,
                n=int(cfg.get("integration_steps", 9)),
            )
            return sign * raw

        # Twisting mode: paper-consistent piecewise approximation
        a = l_i - 0.5 * MODULE_LENGTH_M
        b = l_i + 0.5 * MODULE_LENGTH_M

        if axis == Axis.Y:
            integrand = lambda l: _shape_functions_piecewise(l, t, cfg, l_ref=l_i)[0]
        elif axis == Axis.Z:
            integrand = lambda l: _shape_functions_piecewise(l, t, cfg, l_ref=l_i)[1]
        else:
            raise ValueError(f"Unknown axis: {axis}")

        raw = _simpson_integrate(
            integrand,
            a,
            b,
            n=int(cfg.get("integration_steps", 9)),
        )
        return sign * raw

    return joint_fn


JOINT_FUNCTIONS: Dict[str, Callable[[float], float]] = {
    f"joint_{i + 1}_pos": _make_joint_function(i) for i in range(N_JOINTS)
}
