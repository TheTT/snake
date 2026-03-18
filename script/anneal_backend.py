from __future__ import annotations

import math
import numpy as np
from typing import Callable, Any
from dataclasses import dataclass
import random

from shapefit_types import Axis, FitParam
from fwd_kine import FK


def _point_segment_dist_sq(p, a, b):
    ab = b - a
    t = np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12)
    t = max(0.0, min(1.0, t))
    q = a + t * ab
    d = p - q
    return float(np.dot(d, d))


def _dist_to_polyline_sq(
    p: np.ndarray,
    curve_pts: np.ndarray,
    hint_i: int,
    radius: int,
) -> float:
    """
    curve_pts: Nx3 polyline
    hint_i: approximate index (uses locality)
    """
    n = len(curve_pts)

    start = max(0, hint_i - radius)
    end = min(n - 1, hint_i + radius)
    # start = 0
    # end = n - 1

    best = 1e30

    for i in range(start, end):
        d = _point_segment_dist_sq(
            p,
            curve_pts[i],
            curve_pts[i + 1],
        )
        if d < best:
            best = d

    return best


def _cost_builder(
    ps: list[int],
    curve_pts: np.ndarray,
    hint_is: np.ndarray,
    radius: int,
):
    def cost(fk: FK) -> float:
        total = 0.0
        for pi in ps:
            p = fk.geti(pi)
            hint_i = hint_is[pi]
            total += _dist_to_polyline_sq(
                p=p,
                curve_pts=curve_pts,
                hint_i=hint_i,
                radius=radius,
            )
        return total

    return cost


# Const parameters for anneal
@dataclass
class AnnealConfig:
    max_evals: int = 20
    temp0: float = 0.03
    alpha: float = 0.9
    sigma0: float = 0.01
    min_sigma: float = 1e-4
    max_sigma: float = 0.1
    accept_low: float = 0.2
    accept_high: float = 0.5
    adapt_rate: float = 0.85
    early_stop: int = 6


def _optimize_single_joint(
    seg_i: list[int],
    initval: list[float],
    *,
    fk,
    cost: Callable[[FK], float],
    cfg: AnnealConfig,
) -> list[float]:

    def clamp(v):
        return [max(-math.pi/2, min(math.pi/2, x)) for x in v]

    def set_vals(v):
        for i, val in zip(seg_i, v):
            fk.setval(i, val)

    # init
    cur = clamp(list(initval))
    set_vals(cur)
    best = list(cur)
    best_cost = cost(fk)
    cur_cost = best_cost

    evals = 1
    T = cfg.temp0
    sigma = cfg.sigma0

    no_improve = 0
    acc_cnt = 0
    tried = 0

    while evals < cfg.max_evals:
        cand = [c + random.gauss(0, sigma) for c in cur]
        cand = clamp(cand)

        if all(abs(a - b) < 1e-12 for a, b in zip(cand, cur)):
            T *= cfg.alpha
            continue

        set_vals(cand)
        c_cost = cost(fk)
        evals += 1
        tried += 1

        delta = c_cost - cur_cost
        accept = False

        if delta <= 0:
            accept = True
        elif T > 1e-12 and random.random() < math.exp(-delta / T):
            accept = True

        if accept:
            cur = cand
            cur_cost = c_cost
            acc_cnt += 1

            if c_cost < best_cost:
                best = list(cand)
                best_cost = c_cost
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        # sigma 自适应（低频触发）
        if tried >= 5:
            rate = acc_cnt / tried
            if rate < cfg.accept_low:
                sigma = max(cfg.min_sigma, sigma * cfg.adapt_rate)
            elif rate > cfg.accept_high:
                sigma = min(cfg.max_sigma, sigma / cfg.adapt_rate)
            acc_cnt = 0
            tried = 0

        T *= cfg.alpha

        if no_improve >= cfg.early_stop:
            break

    # ---- 极简局部精修（只一轮，极低成本）----
    step = 0.005
    for i in range(len(best)):
        for s in (step, -step):
            cand = list(best)
            cand[i] = max(-math.pi/2, min(math.pi/2, cand[i] + s))
            set_vals(cand)
            c_cost = cost(fk)
            evals += 1
            if c_cost < best_cost:
                best = cand
                best_cost = c_cost
            if evals >= cfg.max_evals:
                break
        if evals >= cfg.max_evals:
            break

    set_vals(best)
    return best

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
            res = _optimize_single_joint(
                seg_i=[i],
                initval=[v[i]],
                fk=fk,
                cost=_cost_builder(
                    ps=[i + 2],
                    curve_pts=param.fplist,
                    hint_is=param.hint_i,
                    radius=param.hint_rad,
                ),
                cfg=AnnealConfig()
            )
            v[i] = res[0]
        # fk.setval(i, v[i])

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
