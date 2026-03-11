from __future__ import annotations

from typing import Callable

import numpy as np

from shapefit_types import FitConfig

ResidualFn = Callable[[np.ndarray], np.ndarray]


def forward_difference_jacobian(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    v: np.ndarray,
    r0: np.ndarray,
    eps: float,
) -> np.ndarray:
    nvar = int(v.size)
    jac = np.zeros((r0.size, nvar), dtype=np.float64)

    for i in range(nvar):
        vp = v.copy()
        vp[i] += eps
        rp = residual_fn(vp)
        jac[:, i] = (rp - r0) / eps

    return jac


def solve_with_jacob_least_squares(
    *,
    residual_fn: ResidualFn,
    v0: np.ndarray,
    cfg: FitConfig,
    precomp: dict | None = None,
) -> np.ndarray:
    """Damped Gauss-Newton backend using finite-difference Jacobian."""
    _ = precomp
    v = np.asarray(v0, dtype=np.float64).copy()

    for _ in range(cfg.max_iter):
        r0 = np.asarray(residual_fn(v), dtype=np.float64)
        eps = max(cfg.finite_diff_eps, 1e-8)
        jac = np.asarray(forward_difference_jacobian(residual_fn, v, r0, eps), dtype=np.float64)

        lhs = jac.T @ jac + cfg.damping * np.eye(v.size, dtype=np.float64)
        rhs = -1.0 * (jac.T @ r0)
        try:
            dv = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            break

        v = v + dv
        if float(np.linalg.norm(dv)) < 1e-5:
            break

    return v
