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

import joint_functions as jf


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


def f(t: float, s: float) -> Vec3:
    """Sample evolving target curve: a flattened helix in 3D.

    Args:
        t: time in seconds, t >= 0.
        s: normalized arc parameter in [0, 1].
    """
    ss = max(0.0, min(1.0, float(s)))
    body_len = jf.BODY_LENGTH_M

    # spatial wavenumber k applies to normalized s
    k = float(_F_PARAMS["k"])
    freq = float(_F_PARAMS["freq"])  # temporal frequency (Hz)
    ah = float(_F_PARAMS["ah"])  # z amplitude
    av = float(_F_PARAMS["av"])  # y amplitude

    phase = 2.0 * math.pi * (k * ss - freq * float(t))

    x = body_len * ss
    y = av * math.cos(phase)
    z = ah * math.sin(phase)
    return x, y, z

# def f(t: float, s: float) -> Vec3:
#     ss = max(0.0, min(1.0, float(s)))
#     _ensure_body_length()
#     body_len = BODY_LENGTH_M if BODY_LENGTH_M is not None else _DEFAULT_BODY_LENGTH_M

#     ah = float(_F_PARAMS["ah"])  # z amplitude
#     av = float(_F_PARAMS["av"])  # y amplitude
#     freq = float(_F_PARAMS["freq"])  # temporal frequency (Hz)

#     phase = 2.0 * math.pi * freq * float(t)

#     x = body_len * ss
#     y = av * s * s * math.cos(phase)
#     z = 0
#     return x, y, z


def g(t: float, s: float) -> float:
    """Sample twist field: always zero."""
    _ = t
    _ = s
    return 0.0
