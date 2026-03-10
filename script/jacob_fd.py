from __future__ import annotations

from typing import Callable

import numpy as np


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
