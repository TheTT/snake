"""Singleton gait function used by joint controller mapping."""

from __future__ import annotations

import math
from typing import Callable

# Component bitmask: bit0=x, bit1=y, bit2=z.
MASK_X = 1
MASK_Y = 2
MASK_Z = 4

Vec3f = tuple[float, float, float]


def _build_f() -> Callable[[float, float, int], Vec3f]:
    """Build singleton gait function with internal static-like constants."""
    # f-specific constants (kept internal to this module).
    ah = 0.6
    av = 0.6
    k = 1.0
    freq = 0.4

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


f = _build_f()
