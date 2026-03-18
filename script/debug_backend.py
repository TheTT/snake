from __future__ import annotations

import math
import numpy as np
from typing import Any

from shapefit_types import Axis, FitParam
from fwd_kine import FK


def debug_backend(
    v0: np.ndarray,
    param: FitParam,
) -> tuple[np.ndarray, Any]:
    """A simple debug backend that ignores the f curve.

    Sets all Y-axis joints to +25 degrees and returns the FK in meta.
    """
    fk = FK(
        jn=param.jn,
        init_angles=v0,
        seg_len=param.seglen,
        joint_axes=param.joint_axes,
        joint_signs=param.joint_signs,
    )

    v = v0.copy()

    x_i = 0
    for i, axis in enumerate(param.joint_axes):
        if axis == Axis.X:
            v[i] = float(v[i])
        elif axis == Axis.Y:
            v[i] = float(math.radians(25.0))
        else:
            v[i] = 0.0
        fk.setval(i, float(v[i]))

    # compute nearest indices if fplist exists, otherwise zeros
    allp = fk.getallp()
    n_nodes = allp.shape[0]
    nearest_idx = np.zeros(n_nodes, dtype=np.int32)
    if getattr(param, "fplist", None) is not None and len(param.fplist) > 0:
        for ni in range(n_nodes):
            p = allp[ni]
            diffs = param.fplist - p[None, :]
            d2 = np.sum(diffs * diffs, axis=1)
            nearest_idx[ni] = int(np.argmin(d2))

    meta = {
        "fk": fk,
        "nearest_idx": nearest_idx,
    }

    return v, meta
