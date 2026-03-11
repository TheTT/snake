# anneal_backend.py
from __future__ import annotations

import math
import random
from typing import Callable, Sequence

import numpy as np

from shapefit_geometry import assemble_joint_angles
from shapefit_types import FitConfig

ResidualFn = Callable[[np.ndarray], np.ndarray]

EPS = 1e-12


def _clamp(x: float, a: float, b: float) -> float:
    return a if x < a else (b if x > b else x)


def _point_segment_distance_sq(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Squared distance from point p to segment [a,b] (analytic)."""
    v = b - a
    vv = float(np.dot(v, v))
    if vv < EPS:
        d = p - a
        return float(np.dot(d, d))
    t = float(np.dot(p - a, v) / vv)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    q = a + t * v
    d = p - q
    return float(np.dot(d, d))


def _point_polyline_distance_sq_local(
    p: np.ndarray,
    curve: np.ndarray,
    seg_i: int,
    radius: int = 2,
    min_seg_i: int = 0,
) -> tuple[float, int]:
    """Nearest squared distance from p to local polyline neighborhood, plus best segment index."""
    if curve is None or curve.ndim != 2 or curve.shape[0] < 1:
        return float(np.dot(p, p)), 0

    n = int(curve.shape[0])
    if n == 1:
        d = p - curve[0]
        return float(np.dot(d, d)), 0

    center = max(0, min(int(seg_i), n - 2))
    lo = max(int(min_seg_i), center - int(radius), 0)
    hi = min(n - 2, center + int(radius))
    if lo > hi:
        lo = max(0, min(int(min_seg_i), n - 2))
        hi = lo

    best = float("inf")
    best_i = lo
    for i in range(lo, hi + 1):
        dsq = _point_segment_distance_sq(p, curve[i], curve[i + 1])
        if dsq < best:
            best = dsq
            best_i = i
    return best, int(best_i)


def _is_finite_scalar(x: float) -> bool:
    return bool(np.isfinite(x))


def solve_with_anneal_placeholder(
    *,
    residual_fn: ResidualFn,
    v0: np.ndarray,
    cfg: FitConfig,
    precomp: dict | None = None,
) -> np.ndarray:
    """Simulated-annealing backend using nearest distance to piecewise-linear target."""
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

    # aligned targets: only use curve f-derived target (or user-provided aligned variant)
    if "aligned_tgt" in precomp and isinstance(precomp["aligned_tgt"], np.ndarray):
        aligned_tgt = np.asarray(precomp["aligned_tgt"], dtype=np.float64)
    elif "align_func" in precomp and callable(precomp["align_func"]):
        aligned_tgt = np.asarray(precomp["align_func"](tgt, head_translation, head_rpy, joint_axes), dtype=np.float64)
    else:
        aligned_tgt = np.asarray(tgt, dtype=np.float64)

    # assemble initial joint angles and init FK
    joint_angles = assemble_joint_angles(
        x_joint_indices_0b,
        yz_joint_indices_0b,
        joint_signs_seq,
        x_twist,
        yz_vars,
        len(joint_axes),
    )

    # Hard clamp yz joints to physical range before entering annealing.
    for yz_k, jidx in enumerate(yz_joint_indices_0b):
        jj = int(jidx)
        if jj < 0 or jj >= len(joint_angles):
            continue
        clamped_joint = _clamp(float(joint_angles[jj]), -math.pi / 2.0, math.pi / 2.0)
        joint_angles[jj] = clamped_joint
        sign = float(joint_signs[jj]) if 0 <= jj < joint_signs.size else 1.0
        if abs(sign) > EPS:
            yz_vars[yz_k] = clamped_joint / sign

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

    # Track current absolute joint angles; all proposals are clamped to +/-90 deg.
    current_joint_angles = np.asarray(joint_angles, dtype=np.float64).copy()

    # Annealing parameters.
    step_deg = float(precomp.get("anneal_step_deg", 0.6))
    step = max(1e-4, math.radians(step_deg))
    iterations_per_window = int(precomp.get("anneal_iterations_per_window", max(6, 2 * int(cfg.max_iter))))
    iterations_per_window = max(4, iterations_per_window)
    T0 = float(precomp.get("anneal_T0", 1.0))
    Tmin = float(precomp.get("anneal_Tmin", 1e-3))
    stagnation_limit = int(precomp.get("anneal_stagnation_limit", 6))
    local_radius = int(precomp.get("anneal_polyline_radius", 2))

    n_pass = max(1, min(3, int(getattr(cfg, "max_iter", 1))))

    # Sequential hint: nearest target segment index should vary smoothly along body.
    seg_hint_by_joint: dict[int, int] = {
        int(j): max(0, min(int(j), aligned_tgt.shape[0] - 2)) for j in yz_joint_indices_0b
    }

    def _window_cost(window_joint_indices: Sequence[int], hint_map: dict[int, int]) -> tuple[float, dict[int, int]]:
        s = 0.0
        min_seg_i = 0
        out_hints: dict[int, int] = {}
        for jj in window_joint_indices:
            seg_i = max(0, min(int(jj), aligned_tgt.shape[0] - 1))
            center = max(min_seg_i, int(hint_map.get(int(jj), seg_i)))
            mid = np.asarray(fk_get_midpoint(seg_i), dtype=np.float64)
            if not np.all(np.isfinite(mid)):
                return 1e300, out_hints
            dsq, best_seg_i = _point_polyline_distance_sq_local(
                mid,
                aligned_tgt,
                seg_i=center,
                radius=local_radius,
                min_seg_i=min_seg_i,
            )
            if not _is_finite_scalar(dsq):
                return 1e300, out_hints
            s += dsq
            min_seg_i = best_seg_i
            out_hints[int(jj)] = best_seg_i
        s = float(s)
        if not _is_finite_scalar(s):
            return 1e300, out_hints
        return s, out_hints

    for _ in range(n_pass):
        max_start = max(0, n_yz - 1)
        for start_idx in range(0, max_start + 1):
            end_idx = min(n_yz, start_idx + 3)
            window_yz_indices = list(range(start_idx, end_idx))
            if not window_yz_indices:
                continue
            window_joint_indices = [yz_joint_indices_0b[i] for i in window_yz_indices]

            segs = [max(0, min(int(j), aligned_tgt.shape[0] - 1)) for j in window_joint_indices]
            min_seg = max(0, min(segs) - 2)
            max_seg = min(aligned_tgt.shape[0] - 1, max(segs) + 2)
            fk_ensure_upto(max_seg + 1)
            curr_cost, curr_hints = _window_cost(window_joint_indices, seg_hint_by_joint)
            stagnation = 0

            for it in range(iterations_per_window):
                t_fraction = (it + 1) / float(iterations_per_window)
                T = max(Tmin, T0 * (1.0 - t_fraction))

                # Lighter proposal: perturb one joint per step.
                yz_idx = random.choice(window_yz_indices)
                gidx = int(yz_joint_indices_0b[yz_idx])
                sign = float(joint_signs[gidx]) if (0 <= gidx < joint_signs.size) else 1.0
                dyz = float(random.choice((-1, 1))) * step
                delta_joint = sign * dyz
                if 0 <= gidx < current_joint_angles.size:
                    proposed_angle = float(current_joint_angles[gidx] + delta_joint)
                    clamped_angle = _clamp(proposed_angle, -math.pi / 2.0, math.pi / 2.0)
                    actual_delta_joint = clamped_angle - float(current_joint_angles[gidx])
                else:
                    actual_delta_joint = delta_joint

                if abs(actual_delta_joint) < EPS:
                    stagnation += 1
                    if stagnation >= stagnation_limit:
                        break
                    continue

                fk_apply_delta(gidx, actual_delta_joint)
                fk_invalidate_from(max(0, min_seg))
                fk_ensure_upto(max_seg + 1)
                new_cost, new_hints = _window_cost(window_joint_indices, curr_hints)
                delta_cost = new_cost - curr_cost
                if not _is_finite_scalar(delta_cost):
                    delta_cost = 1e300

                if delta_cost <= 0.0:
                    accept = True
                else:
                    try:
                        prob = math.exp(-delta_cost / max(T, 1e-12))
                    except OverflowError:
                        prob = 0.0
                    accept = random.random() < prob

                if accept:
                    curr_cost = new_cost
                    curr_hints = new_hints
                    if 0 <= gidx < current_joint_angles.size:
                        current_joint_angles[gidx] = _clamp(
                            float(current_joint_angles[gidx] + actual_delta_joint),
                            -math.pi / 2.0,
                            math.pi / 2.0,
                        )
                    if abs(sign) > EPS:
                        yz_vars[yz_idx] = current_joint_angles[gidx] / sign
                    stagnation = 0
                else:
                    fk_apply_delta(gidx, -actual_delta_joint)
                    fk_invalidate_from(max(0, min_seg))
                    fk_ensure_upto(max_seg + 1)
                    stagnation += 1
                    if stagnation >= stagnation_limit:
                        break

            seg_hint_by_joint.update(curr_hints)

    v[:n_yz] = yz_vars
    # Head translation and rotation are not optimized in this backend.
    v[n_yz: n_yz + 6] = v0[n_yz: n_yz + 6]
    return v
