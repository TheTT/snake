from __future__ import annotations

import math
from collections import deque
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


def _optimize_multi(
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

    return best


def anneal_backend(
    v0: np.ndarray,
    param: FitParam,
    window_size: int = 1,
) -> np.ndarray:
    fk = FK(
        jn=param.jn,
        init_angles=v0,
        seg_len=param.seglen,
        joint_axes=param.joint_axes,
        joint_signs=param.joint_signs,
    )
    v = v0.copy()

    n_joints = len(param.joint_axes)

    win = max(1, min(int(window_size), n_joints))
    cfg = AnnealConfig()

    twist_by_joint = np.zeros(n_joints, dtype=np.float64)
    x_i = 0
    for i, axis in enumerate(param.joint_axes):
        if axis == Axis.X:
            twist_by_joint[i] = float(param.twist[x_i])
            x_i += 1

    def _apply_entering_joint(jidx: int) -> None:
        if param.joint_axes[jidx] == Axis.X:
            v[jidx] = float(twist_by_joint[jidx])
            fk.setval(jidx, float(v[jidx]))

    def _optimize_window(segs: list[int], ps: list[int]) -> None:
        if not segs:
            return

        init_vals = [float(v[k]) for k in segs]
        res = _optimize_multi(
            seg_i=segs,
            initval=init_vals,
            fk=fk,
            cost=_cost_builder(
                ps=ps,
                curve_pts=param.fplist,
                hint_is=param.hint_i,
                radius=param.hint_rad,
            ),
            cfg=cfg,
        )
        for idx, val in zip(segs, res):
            v[idx] = float(val)
            fk.setval(idx, float(v[idx]))

    win_indices = deque(range(win))
    ps_deque = deque((k + 2) for k in range(win))
    segs_deque = deque(k for k in range(win) if param.joint_axes[k] != Axis.X)

    for j in win_indices:
        _apply_entering_joint(j)
    _optimize_window(list(segs_deque), list(ps_deque))

    for right in range(win, n_joints):
        left = win_indices.popleft()
        ps_deque.popleft()
        if param.joint_axes[left] != Axis.X and segs_deque and segs_deque[0] == left:
            segs_deque.popleft()

        win_indices.append(right)
        ps_deque.append(right + 2)
        _apply_entering_joint(right)
        if param.joint_axes[right] != Axis.X:
            segs_deque.append(right)

        _optimize_window(list(segs_deque), list(ps_deque))

    # Small temporal smoothing to reduce command jitter in dynamic simulation.
    # Keep X-axis twist joints unchanged; smooth the optimized joints only.
    smooth_alpha = 0.2
    keep = 1.0 - smooth_alpha
    for i, axis in enumerate(param.joint_axes):
        if axis == Axis.X:
            continue
        v[i] = keep * float(v0[i]) + smooth_alpha * float(v[i])
        v[i] = max(-math.pi / 2, min(math.pi / 2, float(v[i])))

    return v
