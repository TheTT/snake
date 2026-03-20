"""Unified joint command interface built on curve fitting.

This module exposes:
- JOINT_FUNCTIONS: actuator name -> function(t)
- get_theoretical_local_matrices: cached local orientation chain
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Dict, List
from enum import IntEnum


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


# 关节中心弧长：第 i 个关节位于前 i+1 段长度之后
_JOINT_CENTER_LS_M = tuple(
    sum(POLYLINE_SEGMENT_LENGTHS_M[: i + 1]) for i in range(N_JOINTS)
)

# 每个关节对应的局部积分长度，取相邻两段的平均，作为中点积分近似
_JOINT_SPANS_M = tuple(
    0.5 * (POLYLINE_SEGMENT_LENGTHS_M[i] + POLYLINE_SEGMENT_LENGTHS_M[i + 1])
    for i in range(N_JOINTS)
)


_SW_JSON_PATH = Path(__file__).resolve().parent / "gait" / "sw.json"
_GAIT_DEFAULTS = {
    "ka": 0.02,
    "kb": 0.08,
    "wave_number": 1.0 / (2.0 * math.pi),
    "period": 1.0,
    "phi_offset": 0.0,
    "rollv": 0.0,
    "a1": 0.0,
}
_GAIT_PARAMS = dict(_GAIT_DEFAULTS)
try:
    raw = json.loads(_SW_JSON_PATH.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        for k in _GAIT_DEFAULTS.keys():
            if k in raw and raw[k] is not None:
                _GAIT_PARAMS[k] = float(raw[k])
        # Backward compatibility for old config keys.
        if "wave_number" not in raw and ("ktheta" in raw or "k_theta" in raw):
            if "ktheta" in raw and raw["ktheta"] is not None:
                _GAIT_PARAMS["wave_number"] = float(raw["ktheta"])
            elif "k_theta" in raw and raw["k_theta"] is not None:
                _GAIT_PARAMS["wave_number"] = float(raw["k_theta"])
        if "period" not in raw and ("omega_t" in raw or "omega" in raw):
            omega_t = 0.0
            if "omega_t" in raw and raw["omega_t"] is not None:
                omega_t = float(raw["omega_t"])
            elif "omega" in raw and raw["omega"] is not None:
                omega_t = float(raw["omega"])
            _GAIT_PARAMS["period"] = (1.0 / omega_t) if omega_t > 1e-12 else _GAIT_DEFAULTS["period"]
except Exception:
    # Keep defaults if file is missing or malformed.
    pass


def _cfg_float(cfg: dict, *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return float(cfg[k])
    return float(default)


def _shape_functions(l: float, t: float, cfg: dict) -> tuple[float, float, float]:
    """
    Returns:
        kappa_a, kappa_b, tau
    """
    ka = _cfg_float(cfg, "ka", "k_a", default=0.02)
    kb = _cfg_float(cfg, "kb", "k_b", default=0.08)
    wave_number = _cfg_float(cfg, "wave_number", "k", default=1.0 / (2.0 * math.pi))
    period = _cfg_float(cfg, "period", "T", default=1.0)
    omega_t = (1.0 / period) if period > 1e-12 else 0.0
    phi0 = _cfg_float(cfg, "phi_offset", "phi0", default=0.0)

    # rolling coefficient (paper's a0), and twisting gradient (paper's a1)
    rollv = _cfg_float(cfg, "rollv", "a0", default=0.0)
    a1 = _cfg_float(cfg, "a1", "twistv", default=0.0)

    # Elliptical helix parameter
    psi = 2.0 * math.pi * (wave_number * l - omega_t * t)

    denom = (
        ka * ka * (math.sin(psi) ** 2)
        + kb * kb * (math.cos(psi) ** 2)
        + wave_number * wave_number
    ) ** 1.5

    if denom == 0.0:
        kappa_f = 0.0
    else:
        kappa_f = (ka * kb * wave_number * wave_number) / denom

    # complete-frame phase: base sidewinding phase + rolling/twisting term
    phase = psi + phi0 + (a1 * l + rollv) * t

    kappa_a = kappa_f * math.sin(phase)
    kappa_b = kappa_f * math.cos(phase)
    tau = a1 * t

    return kappa_a, kappa_b, tau


def _make_joint_function(joint_index: int) -> Callable[[float], float]:
    if not (0 <= joint_index < N_JOINTS):
        raise IndexError(f"joint_index out of range: {joint_index}")

    l_i = _JOINT_CENTER_LS_M[joint_index]
    span_i = _JOINT_SPANS_M[joint_index]
    axis = JOINT_AXES[joint_index]
    sign = JOINT_SIGNS[joint_index]
    cfg = _GAIT_PARAMS

    def joint_fn(t: float) -> float:
        kappa_a, kappa_b, tau = _shape_functions(l_i, float(t), cfg)

        # Z -> dorsal bending, Y -> lateral bending, X -> twist
        if axis == Axis.Z:
            angle = span_i * kappa_b
        elif axis == Axis.Y:
            angle = span_i * kappa_a
        elif axis == Axis.X:
            angle = span_i * tau
        else:
            raise ValueError(f"Unknown axis: {axis}")

        cmd = sign * angle
        if axis == Axis.X:
            cmd += TWIST_X_BASE_ANGLE_RAD
        return cmd

    return joint_fn


JOINT_FUNCTIONS: Dict[str, Callable[[float], float]] = {
    f"joint_{i + 1}_pos": _make_joint_function(i) for i in range(N_JOINTS)
}