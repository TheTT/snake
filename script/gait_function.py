"""Curve inputs for the new shape-fit controller.

Exports:
- f(t, s): evolving 3D target curve
- g(t, s): twist field along body
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Tuple

Vec3 = Tuple[float, float, float]


BODY_LENGTH_M = 1.2

# Params are loaded from script/gait/f.json.
_F_JSON_PATH = Path(__file__).resolve().parent / "gait" / "f.json"
_DEFAULTS = {
    "ah": 0.15,
    "av": 0.30,
    "k": 2.0,
    "freq": 0.05,
}


def _load_f_params() -> dict[str, float]:
    params = dict(_DEFAULTS)
    try:
        data = json.loads(_F_JSON_PATH.read_text(encoding="utf-8"))
        for k in ("ah", "av", "k", "freq"):
            if k in data:
                params[k] = float(data[k])
    except (OSError, ValueError, TypeError):
        # Keep defaults when config is missing or malformed.
        pass
    return params


_F_PARAMS = _load_f_params()
HELIX_RADIUS_Z = float(_F_PARAMS["ah"])
HELIX_RADIUS_Y = float(_F_PARAMS["av"])
HELIX_TURNS = float(_F_PARAMS["k"])
HELIX_FREQ_HZ = float(_F_PARAMS["freq"])


def f(t: float, s: float) -> Vec3:
    """Sample evolving target curve: a flattened helix in 3D.

    Args:
        t: time in seconds, t >= 0.
        s: normalized arc parameter in [0, 1].
    """
    ss = max(0.0, min(1.0, float(s)))
    phase = 2.0 * math.pi * (HELIX_TURNS * ss - HELIX_FREQ_HZ * float(t))

    x = BODY_LENGTH_M * ss
    y = HELIX_RADIUS_Y * math.cos(phase)
    z = HELIX_RADIUS_Z * math.sin(phase)
    return x, y, z


def g(t: float, s: float) -> float:
    """Sample twist field: always zero."""
    _ = t
    _ = s
    return 0.0
