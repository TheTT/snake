from __future__ import annotations

from typing import Sequence

import numpy as np

from shapefit_geometry import smoothstep01
from shapefit_types import CurveFn, TwistFn


def integrate_avg_g(g_fn: TwistFn, t: float, s_lo: float, s_hi: float, samples: int) -> float:
    s_lo_f = float(s_lo)
    s_hi_f = float(s_hi)
    if s_hi_f - s_lo_f < 1e-8:
        return float(g_fn(t, 0.5 * (s_lo_f + s_hi_f)))

    n = max(3, int(samples))
    ss = np.linspace(s_lo_f, s_hi_f, n, dtype=np.float64)
    vals = np.array([float(g_fn(t, float(s))) for s in ss], dtype=np.float64)
    vals[~np.isfinite(vals)] = 0.0
    ds = np.diff(ss)
    area = float(0.5 * np.sum((vals[:-1] + vals[1:]) * ds))
    return area / (s_hi_f - s_lo_f)

