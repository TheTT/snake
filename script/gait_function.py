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
import numpy as np


# Load parameters from script/gait/f.json (ah,av,k,freq)
_F_JSON_PATH = Path(__file__).resolve().parent / "gait" / "f.json"
_F_PARAMS = {"ah": 0.15, "av": 0.30, "k": 2.0, "freq": 0.05}
try:
    raw = json.loads(_F_JSON_PATH.read_text(encoding="utf-8"))
    for k in ("ah", "av", "k", "freq"):
        if k in raw:
            _F_PARAMS[k] = float(raw[k])
except Exception:
    # keep defaults on error
    pass


# def f(t: float, s: float) -> np.ndarray:
#     """Sample evolving target curve: a flattened helix in 3D.

#     Args:
#         t: time in seconds, t >= 0.
#         s: normalized arc parameter in [0, 1].
#     """
#     ss = max(0.0, min(1.0, float(s)))

#     # spatial wavenumber k applies to normalized s
#     k = float(_F_PARAMS["k"])
#     freq = float(_F_PARAMS["freq"])  # temporal frequency (Hz)
#     ah = float(_F_PARAMS["ah"])  # z amplitude
#     av = float(_F_PARAMS["av"])  # y amplitude

#     phase = 2.0 * math.pi * (k * ss - freq * float(t))

#     x = ss
#     y = av * math.cos(phase)
#     z = ah * math.sin(phase)
#     return np.array([x, y, z], dtype=np.float64)

def f(t: float, s: float) -> np.ndarray:
    ah = float(_F_PARAMS["ah"])  # z amplitude
    av = float(_F_PARAMS["av"])  # y amplitude
    freq = float(_F_PARAMS["freq"])  # temporal frequency (Hz)

    phase = 2.0 * math.pi * freq * float(t)

    x = - s
    y = av * s * s * math.cos(phase)
    if t < 5.0:
        y *= t / 5.0
    z = 0
    return np.array([x, y, z], dtype=np.float64)


def g(t: float, s: float) -> float:
    """Sample twist field: always zero."""
    return math.pi * min(1.0, t / 5.0)
