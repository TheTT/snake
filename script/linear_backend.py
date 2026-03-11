from __future__ import annotations

from typing import Callable

import numpy as np

from shapefit_types import FitConfig

ResidualFn = Callable[[np.ndarray], np.ndarray]


def solve_with_linear_placeholder(
    *,
    residual_fn: ResidualFn,
    v0: np.ndarray,
    cfg: FitConfig,
    precomp: dict | None = None,
) -> np.ndarray:
    """Placeholder linear solver backend with same interface as other backends.

    Returns `v0` copy for now; replace with a linear-programming or
    least-squares closed-form solver implementation later.
    """
    _ = residual_fn
    _ = cfg
    _ = precomp
    return np.asarray(v0, dtype=np.float64).copy()
