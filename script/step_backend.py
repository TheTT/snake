from __future__ import annotations

import math
import numpy as np
from typing import Callable

from shapefit_types import FitParam
from fwd_kine import FK


def _point_segment_dist_sq(p, a, b):
    ab = b - a
    t = np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12)
    t = max(0.0, min(1.0, t))
    q = a + t * ab
    d = p - q
    return float(np.dot(d, d))


def _dist_to_polyline(p, curve_pts, hint_i, radius):
    """
    curve_pts: Nx3 polyline
    hint_i: approximate index (uses locality)
    """
    n = len(curve_pts)
    if n < 2:
        d = p - curve_pts[0] if n == 1 else p
        return float(np.dot(d, d)), 0

    start = max(0, hint_i - int(radius))
    end = min(n - 2, hint_i + int(radius))

    best = 1e30
    best_i = start

    for i in range(start, end + 1):
        d = _point_segment_dist_sq(
            p,
            curve_pts[i],
            curve_pts[i + 1],
        )
        if d < best:
            best = d
            best_i = i

    return best, int(best_i)


def _optimize_single_joint(
    seg_i: int,
    initval: float,
    *,
    fk: FK,
    dist_fn: Callable[[np.ndarray], float],
) -> float:
    """Anneal v[seg_i] from initval to minimize distance from curve to p[seg_i + 1.5]."""
    ...


def step_backend(
    v0: np.ndarray,
    param: FitParam,
) -> np.ndarray:
    ...
