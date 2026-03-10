"""Curve inputs for the new shape-fit controller.

Exports:
- f(t, s): evolving 3D target curve
- g(t, s): twist field along body
"""

from __future__ import annotations

import math
from typing import Tuple

Vec3 = Tuple[float, float, float]


# Sample flattened helix parameters.
BODY_LENGTH_M = 1.2
HELIX_TURNS = 1.6
HELIX_ANGULAR_SPEED = 0.8  # rad/s
HELIX_RADIUS_Y = 0.08
HELIX_RADIUS_Z = 0.03


def f(t: float, s: float) -> Vec3:
    """Sample evolving target curve: a flattened helix in 3D.

    Args:
        t: time in seconds, t >= 0.
        s: normalized arc parameter in [0, 1].
    """
    ss = max(0.0, min(1.0, float(s)))
    phase = 2.0 * math.pi * HELIX_TURNS * ss - HELIX_ANGULAR_SPEED * float(t)

    x = BODY_LENGTH_M * ss
    y = HELIX_RADIUS_Y * math.cos(phase)
    z = HELIX_RADIUS_Z * math.sin(phase)
    return x, y, z


def g(t: float, s: float) -> float:
    """Sample twist field: always zero."""
    _ = t
    _ = s
    return 0.0
