# anneal_backend.py
from __future__ import annotations

import math
import random
from typing import Callable, Sequence

import numpy as np

from shapefit_geometry import assemble_joint_angles
from shapefit_types import FitConfig

ResidualFn = Callable[[np.ndarray], np.ndarray]

# Numerical eps
EPS = 1e-12


def _clamp(x: float, a: float, b: float) -> float:
    return a if x < a else (b if x > b else x)


def _point_segment_distance_sq(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Squared distance from point p to segment [a,b] (analytic)."""
    v = b - a
    vv = float(np.dot(v, v))
    if vv < EPS:
        # degenerate segment
        d = p - a
        return float(np.dot(d, d))
    t = float(np.dot(p - a, v) / vv)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    q = a + t * v
    d = p - q
    return float(np.dot(d, d))


def _point_polyline_distance_sq_local(p: np.ndarray, curve: np.ndarray, seg_i: int, radius: int = 2) -> float:
    """
    Compute squared distance from p to polyline `curve` but only checking segments
    in index range [seg_i - radius, seg_i + radius + 1] (clipped).
    curve shape: (N,3)
    seg_i refers to segment index roughly corresponding to p; we clip indices to valid range.
    """
    if curve is None or curve.ndim != 2 or curve.shape[0] < 1:
        return float(np.dot(p, p))
    n = curve.shape[0]
    best = float("inf")
    # segments are [i, i+1]
    start = max(0, seg_i - radius)
    end = min(n - 2, seg_i + radius)  # last segment index is n-2
    # if there are no segments (n < 2), fallback to nearest sample point
    if n == 1:
        d = p - curve[0]
        return float(np.dot(d, d))
    for i in range(start, end + 1):
        a = curve[i]
        b = curve[i + 1]
        dsq = _point_segment_distance_sq(p, a, b)
        if dsq < best:
            best = dsq
    return best


def solve_with_anneal_placeholder(
    *,
    residual_fn: ResidualFn,
    v0: np.ndarray,
    cfg: FitConfig,
    precomp: dict | None = None,
) -> np.ndarray:
    """
    Simulated-annealing local-window backend.

    Strategy:
      - Sweep windows of 3 consecutive yz joints (stride 1) from front to back.
      - For each window, run a small simulated-annealing loop to minimize the sum of squared
        distances between the corresponding segment midpoints and the target curve (local polyline).
      - Step size default = 1 degree (in radians). Angles clamped to [-pi/2, pi/2] cumulative per joint.
      - Uses precomp['fk_callbacks'] same contract as other backends.

    Assumptions / notes:
      - If precomp contains 'aligned_tgt' (an Nx3 array), it will be used directly.
        Otherwise, if precomp contains 'align_func', call align_func(tgt, head_translation, head_rpy, joint_axes).
        Otherwise fallback to 'tgt' unchanged.
      - Joint-index ↔ segment-index mapping: uses int(joint_idx) as the primary segment index for evaluating proximity.
    """
    v = np.asarray(v0, dtype=np.float64).copy()
    _ = residual_fn

    if precomp is None:
        return v

    fk_callbacks = precomp.get("fk_callbacks", None)
    if not isinstance(fk_callbacks, dict):
        return v

    required_fk = ("fk_init", "fk_apply_delta", "fk_get_midpoint", "fk_invalidate_from", "fk_ensure_upto")
    if not all(callable(fk_callbacks.get(k, None)) for k in required_fk):
        return v

    lengths_m = np.asarray(precomp.get("lengths_m", []), dtype=np.float64)
    joint_axes = tuple(precomp.get("joint_axes", ()))
    joint_signs = np.asarray(precomp.get("joint_signs", []), dtype=np.float64)
    joint_signs_seq = [float(x) for x in joint_signs]
    x_joint_indices_0b = tuple(int(x) for x in precomp.get("x_joint_indices_0b", ()))
    yz_joint_indices_0b = tuple(int(y) for y in precomp.get("yz_joint_indices_0b", ()))
    x_twist = np.asarray(precomp.get("twist_filtered", []), dtype=np.float64)
    tgt = np.asarray(precomp.get("tgt", []), dtype=np.float64)

    # validation
    if len(joint_axes) == 0 or lengths_m.size != len(joint_axes) + 1:
        return v
    if joint_signs.size != len(joint_axes):
        return v
    if len(yz_joint_indices_0b) == 0:
        return v
    if tgt.ndim != 2 or tgt.shape[0] == 0 or tgt.shape[1] != 3:
        return v

    n_yz = len(yz_joint_indices_0b)
    if v.size < n_yz + 6:
        return v

    # variables
    yz_vars = v[:n_yz].copy()
    head = v[n_yz: n_yz + 6].copy()
    head_translation = np.asarray(head[:3], dtype=np.float64)
    head_rpy = np.asarray(head[3:], dtype=np.float64)

    # aligned targets: prefer precomputed or provided align_func
    if "aligned_tgt" in precomp and isinstance(precomp["aligned_tgt"], np.ndarray):
        aligned_tgt = precomp["aligned_tgt"]
    elif "align_func" in precomp and callable(precomp["align_func"]):
        # align_func(tgt, head_translation, head_rpy, joint_axes) -> aligned array
        aligned_tgt = np.asarray(precomp["align_func"](tgt, head_translation, head_rpy, joint_axes), dtype=np.float64)
    else:
        # fallback: assume tgt is already in the right frame
        aligned_tgt = tgt

    # assemble initial joint angles and init FK
    joint_angles = assemble_joint_angles(
        x_joint_indices_0b,
        yz_joint_indices_0b,
        joint_signs_seq,
        x_twist,
        yz_vars,
        len(joint_axes),
    )

    fk_init = fk_callbacks["fk_init"]
    fk_apply_delta = fk_callbacks["fk_apply_delta"]
    fk_get_midpoint = fk_callbacks["fk_get_midpoint"]
    fk_invalidate_from = fk_callbacks["fk_invalidate_from"]
    fk_ensure_upto = fk_callbacks["fk_ensure_upto"]

    fk_init(
        head_translation=head_translation,
        head_rpy=head_rpy,
        joint_angles_base_x=joint_angles,
        yz_init=joint_angles,
        lengths_m=lengths_m,
        joint_axes=joint_axes,
    )

    # track applied deltas for clamping [-pi/2, pi/2]
    applied_deltas = np.zeros((len(joint_axes),), dtype=np.float64)

    # annealing parameters (tunable)
    step_deg = float(precomp.get("anneal_step_deg", 1.0))  # degrees
    step = math.radians(step_deg)  # radians step
    # iterations_per_window: base on cfg.max_iter but ensure reasonable minimum
    iterations_per_window = max(20, int(max(1, getattr(cfg, "max_iter", 20))))
    # temperature schedule
    T0 = float(precomp.get("anneal_T0", 1.0))
    Tmin = float(precomp.get("anneal_Tmin", 1e-3))

    n_pass = max(1, int(getattr(cfg, "max_iter", 1)))

    # helper: cost for a set of joint indices (window)
    def _window_cost_for_joint_indices(joint_indices_window: Sequence[int]) -> float:
        # For each joint in window, map to a primary segment index and sum squared distances
        s = 0.0
        for jj in joint_indices_window:
            seg_i = max(0, min(int(jj), aligned_tgt.shape[0] - 1))
            # evaluate midpoint at seg_i
            try:
                mid = np.asarray(fk_get_midpoint(seg_i), dtype=np.float64)
            except Exception:
                # if fk fails, return large cost
                return 1e300
            s += _point_polyline_distance_sq_local(mid, aligned_tgt, seg_i, radius=2)
        return float(s)

    # Sweep passes
    for pass_i in range(n_pass):
        # front-to-back windows over yz_joint_indices_0b with stride 1
        max_start = max(0, n_yz - 3)
        for start_idx in range(0, max_start + 1):
            # window joint indices (these are indices into yz_vars array)
            window_yz_indices = [start_idx, start_idx + 1, start_idx + 2]
            # convert to actual joint indices (0-based in whole joint list)
            window_joint_indices = [yz_joint_indices_0b[i] for i in window_yz_indices]

            # determine minimal segment index touched for fk_invalidate_from
            segs = [max(0, min(int(j), aligned_tgt.shape[0] - 1)) for j in window_joint_indices]
            min_seg = max(0, min(segs) - 2)
            max_seg = min(aligned_tgt.shape[0] - 1, max(segs) + 2)

            # ensure FK up to needed
            fk_ensure_upto(max_seg + 1)

            # current window cost
            curr_cost = _window_cost_for_joint_indices(window_joint_indices)

            # annealing loop
            for it in range(iterations_per_window):
                # temperature schedule: linear decay
                t_fraction = (it + 1) / float(iterations_per_window)
                T = max(Tmin, T0 * (1.0 - t_fraction))

                # propose random discrete perturbations for the 3 yz joints: each ∈ {-1,0,1} * step
                # map to global joint indices
                proposals = []
                for k_local, yz_idx in enumerate(window_yz_indices):
                    sign_choice = random.choice((-1, 0, 1))
                    delta = float(sign_choice) * step
                    # convert sign by joint_signs mapping so delta applies in FK coordinate if needed
                    global_jidx = int(window_joint_indices[k_local])
                    sign = float(joint_signs[global_jidx]) if (0 <= global_jidx < joint_signs.size) else 1.0
                    # delta in FK joint units
                    delta_joint = sign * delta
                    # clamp w.r.t applied_deltas so cumulative stays in [-pi/2, pi/2]
                    if 0 <= global_jidx < applied_deltas.size:
                        proposed = applied_deltas[global_jidx] + delta_joint
                        clamped = _clamp(proposed, -math.pi / 2.0, math.pi / 2.0)
                        actual_delta_joint = clamped - applied_deltas[global_jidx]
                    else:
                        actual_delta_joint = delta_joint
                    proposals.append((global_jidx, actual_delta_joint, delta))  # keep delta (signed by sign) for yz_vars update later

                # if all proposals are near zero skip
                if all(abs(p[1]) < 1e-12 for p in proposals):
                    continue

                # apply proposals sequentially (record applied order)
                applied_seq = []
                for (gidx, actual_delta_joint, unused_delta_for_yz) in proposals:
                    if abs(actual_delta_joint) < 1e-12:
                        applied_seq.append((gidx, 0.0))
                        continue
                    fk_apply_delta(int(gidx), actual_delta_joint)
                    applied_seq.append((gidx, actual_delta_joint))

                # invalidate FK from min_seg and ensure upto
                fk_invalidate_from(min_seg)
                fk_ensure_upto(max_seg + 1)

                # compute new cost
                new_cost = _window_cost_for_joint_indices(window_joint_indices)

                delta_cost = new_cost - curr_cost

                accept = False
                if delta_cost <= 0.0:
                    accept = True
                else:
                    # probabilistic acceptance
                    try:
                        prob = math.exp(-delta_cost / max(T, 1e-300))
                    except OverflowError:
                        prob = 0.0
                    if random.random() < prob:
                        accept = True

                if accept:
                    # commit: update applied_deltas and yz_vars for each applied change
                    curr_cost = new_cost
                    for (gidx, actual_delta_joint) in applied_seq:
                        if abs(actual_delta_joint) < 1e-12:
                            continue
                        if 0 <= gidx < applied_deltas.size:
                            applied_deltas[gidx] += actual_delta_joint
                        # find corresponding yz_vars index if this gidx is one of yz_joint_indices_0b
                        # note: window_joint_indices[k_local] corresponds to yz index at window_yz_indices[k_local]
                        for k_local, yz_idx in enumerate(window_yz_indices):
                            if window_joint_indices[k_local] == gidx:
                                # the delta in yz_vars should be actual_delta_joint / sign (reverse convert)
                                sign = float(joint_signs[gidx]) if (0 <= gidx < joint_signs.size) else 1.0
                                if abs(sign) > 1e-12:
                                    yz_vars[yz_idx] += actual_delta_joint / sign
                else:
                    # rollback: apply negative of applied_seq in reverse order
                    for (gidx, actual_delta_joint) in reversed(applied_seq):
                        if abs(actual_delta_joint) < 1e-12:
                            continue
                        fk_apply_delta(int(gidx), -actual_delta_joint)
                    fk_invalidate_from(min_seg)
                    fk_ensure_upto(max_seg + 1)
                    # nothing else to do

            # end annealing loop for this window

    # end passes sweep

    v[:n_yz] = yz_vars
    return v
