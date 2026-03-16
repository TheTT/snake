from __future__ import annotations

import math
import numpy as np
from typing import Callable, Any

from shapefit_types import Axis, FitParam
from fwd_kine import FK


def _point_segment_dist_sq(p, a, b):
    ab = b - a
    t = np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12)
    t = max(0.0, min(1.0, t))
    q = a + t * ab
    d = p - q
    return float(np.dot(d, d))


def _dist_to_polyline(
    p: np.ndarray,
    curve_pts: np.ndarray,
    hint_i: int,
    radius: float,
) -> float:
    """
    curve_pts: Nx3 polyline
    hint_i: approximate index (uses locality)
    """
    n = len(curve_pts)
    if n < 2:
        d = p - curve_pts[0] if n == 1 else p
        return float(np.dot(d, d))

    # DEBUG
    # start = max(0, hint_i - int(radius))
    # end = min(n - 2, hint_i + int(radius))
    start = 0
    end = n - 2

    best = 1e30

    for i in range(start, end + 1):
        d = _point_segment_dist_sq(
            p,
            curve_pts[i],
            curve_pts[i + 1],
        )
        if d < best:
            best = d

    return best


# Const parameters for anneal


def _optimize_single_joint(
    seg_i: int,
    initval: float,
    *,
    fk: FK,
    dist_fn: Callable[[np.ndarray], float],
) -> float:
    """Anneal v[seg_i] from initval to minimize distance from curve to p[seg_i + 2]."""
    def dist(val: float) -> float:
        fk.setval(seg_i, val)
        return dist_fn(fk.geti(seg_i + 2))

    # tmp: compare initval, initval + 0.001, initval - 0.001
    best_dist = dist(initval)
    # olddist = best_dist
    best_val = initval
    for delta in [0.01, -0.01]:
        if delta < - math.pi/2 or delta > math.pi/2:
            continue
        val = initval + delta
        d = dist(val)
        if d < best_dist:
            best_dist = d
            best_val = val
    # best_val = initval + 0.01
    # print(olddist, "->", best_dist)

    return best_val


def step_backend(
    v0: np.ndarray,
    param: FitParam,
) -> tuple[np.ndarray, Any]:
    fk = FK(
        jn=param.jn,
        init_angles=v0,
        seg_len=param.seglen,
        joint_axes=param.joint_axes,
        joint_signs=param.joint_signs,
    )
    v = v0.copy()

    # n8 = fk.geti(8)
    # print("\noldp[8]=(",n8[0],",",n8[1],",",n8[2],")")
    # # n8h = param.fplist[param.hint_i[8]]
    # # print("hint[8]=(",n8h[0],",",n8h[1],",",n8h[2],")")
    # n8d = _dist_to_polyline(
    #     p=n8,
    #     curve_pts=param.fplist,
    #     hint_i=param.hint_i[8],
    #     radius=param.hint_rad,
    # )
    # print("oldd[8]=",n8d)

    x_i = 0
    for i, axis in enumerate(param.joint_axes):
        if axis == Axis.X:
            v[i] = param.twist[x_i]
            x_i += 1
        else:
            v[i] = _optimize_single_joint(
                seg_i=i,
                initval=v[i],
                fk=fk,
                dist_fn=lambda p: _dist_to_polyline(
                    p=p,
                    curve_pts=param.fplist,
                    hint_i=param.hint_i[i + 2],
                    radius=param.hint_rad,
                ),
            )
        fk.setval(i, v[i])

    # n8 = fk.geti(8)
    # print("newp[8]=(",n8[0],",",n8[1],",",n8[2],")")
    # # n8h = param.fplist[param.hint_i[8]]
    # # print("hint[8]=(",n8h[0],",",n8h[1],",",n8h[2],")")
    # n8d = _dist_to_polyline(
    #     p=n8,
    #     curve_pts=param.fplist,
    #     hint_i=param.hint_i[8],
    #     radius=param.hint_rad,
    # )
    # print("newd[8]=",n8d)

    # compute closest fplist index for each FK node
    allp = fk.getallp()
    n_nodes = allp.shape[0]
    nearest_idx = np.zeros(n_nodes, dtype=np.int32)
    # brute-force nearest; fplist size is small (~200)
    for ni in range(n_nodes):
        p = allp[ni]
        diffs = param.fplist - p[None, :]
        d2 = np.sum(diffs * diffs, axis=1)
        nearest_idx[ni] = int(np.argmin(d2))

    # pack fk and nearest indices into a single Any (dict) for backward compatibility
    meta = {
        "fk": fk,
        "nearest_idx": nearest_idx,
    }
    return v, meta
