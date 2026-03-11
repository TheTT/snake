from __future__ import annotations

from typing import Callable

import numpy as np

from shapefit_types import FitConfig

ResidualFn = Callable[[np.ndarray], np.ndarray]


def solve_with_anneal_placeholder(
    *,
    residual_fn: ResidualFn,
    v0: np.ndarray,
    cfg: FitConfig,
) -> np.ndarray:
    """Placeholder annealing backend with the same interface as jacob backend.

    This function intentionally does not implement simulated annealing yet.
    It returns the initial state unchanged so call sites can be wired now.
    """
    _ = residual_fn
    _ = cfg
    return np.asarray(v0, dtype=np.float64).copy()
