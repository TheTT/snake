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


def line_curve(lengths_m: Sequence[float], s: float) -> np.ndarray:
    total = float(np.sum(np.asarray(lengths_m, dtype=np.float64)))
    return np.array((total * s, 0.0, 0.0), dtype=np.float64)


def target_point(f_fn: CurveFn, lengths_m: Sequence[float], t: float, s: float, startup_ramp_sec: float) -> np.ndarray:
    if t <= 0.0:
        return line_curve(lengths_m, s)

    if t < startup_ramp_sec:
        c1 = smoothstep01(t / startup_ramp_sec)
        line = line_curve(lengths_m, s)
        target0 = np.asarray(f_fn(0.0, s), dtype=np.float64)
        return (1.0 - c1) * line + c1 * target0

    return np.asarray(f_fn(t - startup_ramp_sec, s), dtype=np.float64)


def build_target_points(
    f_fn: CurveFn,
    lengths_m: Sequence[float],
    t: float,
    seg_mid_s: np.ndarray,
    startup_ramp_sec: float,
) -> np.ndarray:
    return np.array(
        [target_point(f_fn, lengths_m, t, float(s), startup_ramp_sec) for s in seg_mid_s],
        dtype=np.float64,
    )
