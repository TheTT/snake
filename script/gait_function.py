"""Singleton gait function used by joint controller mapping."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

# Component bitmask: bit0=x, bit1=y, bit2=z.
MASK_X = 1
MASK_Y = 2
MASK_Z = 4

Vec3f = tuple[float, float, float]

SCRIPT_DIR = Path(__file__).resolve().parent
F_CONFIG_PATH = SCRIPT_DIR / "gait" / "f.json"


def _load_f_config() -> dict[str, float]:
    """Load f parameters once at module import time."""
    with F_CONFIG_PATH.open("r", encoding="utf-8") as fobj:
        raw = json.load(fobj)

    required = ["ah", "av", "k", "freq"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise KeyError(f"Missing keys in {F_CONFIG_PATH}: {missing}")

    return {
        "ah": float(raw["ah"]),
        "av": float(raw["av"]),
        "k": float(raw["k"]),
        "freq": float(raw["freq"]),
    }


F_CFG = _load_f_config()


def _build_f() -> Callable[[float, float, int], Vec3f]:
    """Build singleton gait function with internal static-like constants."""
    # f-specific constants loaded once from script/gait/f.json.
    ah = F_CFG["ah"]
    av = F_CFG["av"]
    k = F_CFG["k"]
    freq = F_CFG["freq"]

    def _f(c: float, t: float, mask: int = 7) -> Vec3f:
        """Return (x, y, z) in radians, computing only components enabled by mask (0..7)."""
        base = 2.0 * math.pi * (k * c - freq * t)

        x = 0.0
        y = 0.0
        z = 0.0

        if mask & MASK_X:
            x = 0.0
        if mask & MASK_Y:
            y = av * math.cos(base)
        if mask & MASK_Z:
            z = ah * math.sin(base)

        return x, y, z

    return _f


foo = _build_f()


def zro(c: float, t: float, mask: int = 7) -> Vec3f:
    """Zero gait: every component is always zero.

    This control function ignores `c`, `t`, and `mask` and returns
    a (0.0, 0.0, 0.0) tuple so every joint angle remains zero.
    """
    return 0.0, 0.0, 0.0
