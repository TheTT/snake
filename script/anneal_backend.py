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


def dist_fn_constructor(
    node_indices: list[int],
    *,
    fk: FK,
    curve_pts: np.ndarray,
    hint_is: np.ndarray,
    radius: float,
) -> Callable[[], float]:
    def obj() -> float:
        s = 0.0
        for node in node_indices:
            p = fk.geti(int(node))
            d = _dist_to_polyline(
                p=p,
                curve_pts=curve_pts,
                hint_i=hint_is[node],
                radius=radius,
            )
            s += d * d
        return s

    return obj


def _optimize_joints(
    seg_is: list[int],
    initvals: list[float],
    *,
    fk: FK,
    obj_fn: Callable[[], float],
    n_iters: int = 400,
    temp0: float = 0.1,
    step_size: float = 0.05,
) -> tuple[np.ndarray, dict]:
    assert len(seg_is) == len(initvals)

    k = len(seg_is)
    cur = np.array(initvals, dtype=float)
    best = cur.copy()

    # initialize FK to current values
    for si, val in zip(seg_is, cur):
        fk.setval(si, float(val))

    cur_obj = float(obj_fn())
    best_obj = cur_obj

    rng = np.random.default_rng()

    for it in range(n_iters):
        T = temp0 * (1.0 - it / max(1, n_iters))

        # propose a candidate by perturbing each joint (one-at-a-time)
        for idx in range(k):
            si = seg_is[idx]
            old_val = cur[idx]
            # gaussian proposal
            cand = old_val + rng.normal(0.0, step_size)

            fk.setval(si, float(cand))
            cand_obj = float(obj_fn())

            delta = cand_obj - cur_obj
            accept = False
            if delta <= 0.0:
                accept = True
            else:
                # Metropolis acceptance
                if T > 0.0:
                    prob = math.exp(-delta / T)
                    if rng.random() < prob:
                        accept = True

            if accept:
                cur[idx] = cand
                cur_obj = cand_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best = cur.copy()
            else:
                # revert FK to old value
                fk.setval(si, float(old_val))

    # ensure FK left at best values
    for si, val in zip(seg_is, best):
        fk.setval(si, float(val))

    meta = {"fk": fk, "best_obj": best_obj}
    return best, meta


def anneal_backend(
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
            obj = dist_fn_constructor(
                [i + 2],
                fk=fk,
                curve_pts=param.fplist,
                hint_is=param.hint_i,
                radius=param.hint_rad,
            )
            best_arr = _optimize_joints(
                seg_is=[i],
                initvals=[v[i]],
                fk=fk,
                obj_fn=obj,
            )
            v[i] = float(best_arr[0])
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
