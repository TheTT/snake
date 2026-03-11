# acf_backend.py
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


def _point_polyline_distance_sq_local_by_samples(p: np.ndarray, samples_s: np.ndarray, samples_xyz: np.ndarray, s_m: float, radius_segments: int = 2) -> float:
    """
    samples_s: shape (M,) monotonic increasing arc-length parameters (meters)
    samples_xyz: shape (M,3)
    s_m: query arc-length position (meters)
    radius_segments: how many neighboring segments (on each side) to check
    Returns squared distance.
    """
    if samples_xyz is None or samples_xyz.ndim != 2 or samples_xyz.shape[0] == 0:
        d = p
        return float(np.dot(d, d))
    M = samples_xyz.shape[0]
    if M == 1:
        d = p - samples_xyz[0]
        return float(np.dot(d, d))
    # find nearest sample index by s
    idx = int(np.searchsorted(samples_s, s_m))
    # candidate segments are [i, i+1] where i ranges
    start = max(0, idx - radius_segments)
    end = min(M - 2, idx + radius_segments)
    best = float("inf")
    for i in range(start, end + 1):
        a = samples_xyz[i]
        b = samples_xyz[i + 1]
        dsq = _point_segment_distance_sq(p, a, b)
        if dsq < best:
            best = dsq
    return best


def _golden_section_minimize_scalar(func: Callable[[float], float], a: float, b: float, max_iter: int = 30) -> tuple[float, float]:
    """Return (s_best, f_best). Uses golden-section on [a,b]."""
    gr = (math.sqrt(5.0) - 1.0) / 2.0  # 0.618...
    x1 = b - gr * (b - a)
    x2 = a + gr * (b - a)
    f1 = func(x1)
    f2 = func(x2)
    for _ in range(max_iter):
        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + gr * (b - a)
            f2 = func(x2)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - gr * (b - a)
            f1 = func(x1)
        if abs(b - a) < 1e-8:
            break
    if f1 < f2:
        return x1, f1
    return x2, f2


def solve_with_acf_placeholder(
    *,
    residual_fn: ResidualFn,
    v0: np.ndarray,
    cfg: FitConfig,
    precomp: dict | None = None,
) -> np.ndarray:
    """
    acfed chain fitting backend using sliding 3-joint windows and local acfing.
    Does NOT use `tgt` as point-samples; instead requires either:
      - precomp['curve_samples_arc'] = ndarray (M,4) columns [s, x, y, z], where s is arc-length (meters)
      OR
      - precomp['curve_fn'] = callable(s) -> (3,) point, with s in [s_min, s_max].
        When using curve_fn, precomp may provide 'curve_param_range' = (s_min, s_max).
    Also uses precomp['lengths_m'] to compute robot segment midpoint arc positions.

    FK contract same as other backends: precomp['fk_callbacks'] must provide fk_init, fk_apply_delta,
    fk_get_midpoint, fk_invalidate_from, fk_ensure_upto.

    Returns updated v (same format v[:n_yz] = yz_vars).
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

    # curve sources (do NOT use precomp['tgt'])
    curve_samples_arc = precomp.get("curve_samples_arc", None)  # expected shape (M,4): s,x,y,z
    curve_fn = precomp.get("curve_fn", None)  # callable(s) -> (3,)
    curve_param_range = precomp.get("curve_param_range", None)  # optional (s_min, s_max)

    # validation
    if len(joint_axes) == 0 or lengths_m.size != len(joint_axes) + 1:
        return v
    if joint_signs.size != len(joint_axes):
        return v
    if len(yz_joint_indices_0b) == 0:
        return v
    if curve_samples_arc is None and not callable(curve_fn):
        # no valid curve representation provided (we intentionally do not use 'tgt')
        return v

    n_yz = len(yz_joint_indices_0b)
    if v.size < n_yz + 6:
        return v

    # variables
    yz_vars = v[:n_yz].copy()
    head = v[n_yz: n_yz + 6].copy()
    head_translation = np.asarray(head[:3], dtype=np.float64)
    head_rpy = np.asarray(head[3:], dtype=np.float64)

    # prepare curve samples if provided
    samples_s = None
    samples_xyz = None
    if curve_samples_arc is not None:
        arr = np.asarray(curve_samples_arc, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 4:
            samples_s = arr[:, 0].copy()
            samples_xyz = arr[:, 1:4].copy()
        else:
            # invalid format: ignore samples (will fallback to curve_fn if present)
            samples_s = None
            samples_xyz = None

    # compute segment midpoint arc positions from lengths_m
    # lengths_m has size = number_of_segments, segments indexed 0..S-1
    S = lengths_m.size
    # cumulative lengths at segment starts
    starts = np.zeros((S,), dtype=np.float64)
    for i in range(1, S):
        starts[i] = starts[i - 1] + float(lengths_m[i - 1])
    midpoints_s = starts + 0.5 * lengths_m  # arc-length position of each segment midpoint

    # choose curve param bounds if using curve_fn
    if callable(curve_fn):
        if curve_param_range is not None and isinstance(curve_param_range, (tuple, list)) and len(curve_param_range) == 2:
            s_min, s_max = float(curve_param_range[0]), float(curve_param_range[1])
        elif samples_s is not None:
            s_min, s_max = float(samples_s[0]), float(samples_s[-1])
        else:
            # fallback: use robot total length as bounds
            s_min = 0.0
            s_max = float(np.sum(lengths_m))
    else:
        s_min = s_max = 0.0  # unused

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

    # acfing parameters (tunable via precomp)
    step_deg = float(precomp.get("acf_step_deg", 1.0))
    step = math.radians(step_deg)
    iterations_per_window = int(precomp.get("acf_iters_per_window", max(20, int(getattr(cfg, "max_iter", 20)))))
    T0 = float(precomp.get("acf_T0", 1.0))
    Tmin = float(precomp.get("acf_Tmin", 1e-3))
    random_seed = precomp.get("random_seed", None)
    if random_seed is not None:
        random.seed(int(random_seed))

    # how many neighboring curve segments to check when using samples (method3)
    radius_segments = int(precomp.get("curve_search_radius_segments", 3))

    # helper: compute squared distance from segment midpoint (by seg_i) to curve (using samples or curve_fn)
    def _midpoint_to_curve_sq(seg_i: int) -> float:
        # ensure seg_i in [0, S-1]
        if seg_i < 0:
            seg_i = 0
        if seg_i >= S:
            seg_i = S - 1
        # get FK midpoint (3D)
        try:
            mid_pt = np.asarray(fk_get_midpoint(int(seg_i)), dtype=np.float64)
        except Exception:
            # if FK fails for some reason, return large cost
            return 1e300
        s_m = float(midpoints_s[int(seg_i)])
        # prefer analytical sample-based projection if samples available
        if samples_xyz is not None and samples_s is not None:
            return _point_polyline_distance_sq_local_by_samples(mid_pt, samples_s, samples_xyz, s_m, radius_segments)
        # otherwise use curve_fn via 1D local minimization around s_m
        if callable(curve_fn):
            # search window: +/- window_len (use a few segment lengths)
            avg_seg = float(np.mean(lengths_m)) if lengths_m.size > 0 else (s_max - s_min) / 10.0
            window = max(avg_seg * 3.0, 0.1)  # at least 0.1m
            a = max(s_min, s_m - window)
            b = min(s_max, s_m + window)
            if b <= a:
                # degenerate -> evaluate at s_m directly
                try:
                    p = np.asarray(curve_fn(s_m), dtype=np.float64)
                    d = mid_pt - p
                    return float(np.dot(d, d))
                except Exception:
                    return 1e300

            def f_scalar(s: float) -> float:
                try:
                    p = np.asarray(curve_fn(s), dtype=np.float64)
                except Exception:
                    return 1e300
                d = mid_pt - p
                return float(np.dot(d, d))

            s_best, f_best = _golden_section_minimize_scalar(f_scalar, a, b, max_iter=30)
            return float(f_best)
        # fallback
        d = mid_pt
        return float(np.dot(d, d))

    # helper: cost for a window of joint indices (global joint indices)
    def _window_cost_for_joint_indices(window_joint_indices: Sequence[int]) -> float:
        s = 0.0
        # For each joint in the window, map to a primary segment index and sum squared distances.
        for jj in window_joint_indices:
            seg_i = max(0, min(int(jj), S - 1))
            s += _midpoint_to_curve_sq(seg_i)
        return float(s)

    # sweep passes
    n_pass = max(1, int(getattr(cfg, "max_iter", 1)))
    for pass_i in range(n_pass):
        # sliding windows over yz joints (local window size = 3)
        if n_yz < 3:
            window_starts = [0]
        else:
            window_starts = list(range(0, n_yz - 3 + 1))
        for start_local in window_starts:
            window_yz_local = [start_local, start_local + 1, start_local + 2]
            window_joint_indices = [yz_joint_indices_0b[i] for i in window_yz_local]

            # determine seg index range influenced (for fk invalidation)
            segs = [max(0, min(int(j), S - 1)) for j in window_joint_indices]
            min_seg = max(0, min(segs) - 2)
            max_seg = min(S - 1, max(segs) + 2)
            fk_ensure_upto(max_seg + 1)

            curr_cost = _window_cost_for_joint_indices(window_joint_indices)

            # acfing loop for this window
            for it in range(iterations_per_window):
                t_frac = (it + 1) / float(iterations_per_window)
                T = max(Tmin, T0 * (1.0 - t_frac))

                # propose perturbs for the 3 local yz vars: choices -1, 0, +1 times step
                proposals = []
                for k_local, yz_local_idx in enumerate(window_yz_local):
                    sign_choice = random.choice((-1, 0, 1))
                    delta_yz = float(sign_choice) * step  # in yz_vars units
                    global_jidx = int(window_joint_indices[k_local])
                    sign = float(joint_signs[global_jidx]) if (0 <= global_jidx < joint_signs.size) else 1.0
                    delta_joint = sign * delta_yz
                    # clamp cumulative applied delta
                    if 0 <= global_jidx < applied_deltas.size:
                        proposed = applied_deltas[global_jidx] + delta_joint
                        clamped = _clamp(proposed, -math.pi / 2.0, math.pi / 2.0)
                        actual_delta_joint = clamped - applied_deltas[global_jidx]
                    else:
                        actual_delta_joint = delta_joint
                    proposals.append((global_jidx, actual_delta_joint, k_local, delta_yz, sign))

                if all(abs(p[1]) < 1e-12 for p in proposals):
                    continue

                # apply proposals sequentially and record
                applied_seq = []
                for (gidx, actual_delta_joint, k_local, delta_yz, sign) in proposals:
                    if abs(actual_delta_joint) < 1e-12:
                        applied_seq.append((gidx, 0.0))
                        continue
                    fk_apply_delta(int(gidx), actual_delta_joint)
                    applied_seq.append((gidx, actual_delta_joint))

                fk_invalidate_from(min_seg)
                fk_ensure_upto(max_seg + 1)

                new_cost = _window_cost_for_joint_indices(window_joint_indices)
                delta_cost = new_cost - curr_cost

                accept = False
                if delta_cost <= 0.0:
                    accept = True
                else:
                    try:
                        prob = math.exp(-delta_cost / max(T, 1e-300))
                    except OverflowError:
                        prob = 0.0
                    if random.random() < prob:
                        accept = True

                if accept:
                    # commit changes: update applied_deltas and yz_vars
                    curr_cost = new_cost
                    for (gidx, applied_delta) in applied_seq:
                        if abs(applied_delta) < 1e-12:
                            continue
                        if 0 <= gidx < applied_deltas.size:
                            applied_deltas[gidx] += applied_delta
                        # update yz_vars if this gidx corresponds to a local yz var
                        for (k_local, yz_local_idx) in enumerate(window_yz_local):
                            if window_joint_indices[k_local] == gidx:
                                sign = float(joint_signs[gidx]) if (0 <= gidx < joint_signs.size) else 1.0
                                if abs(sign) > 1e-12:
                                    yz_vars[yz_local_idx] += applied_delta / sign
                else:
                    # rollback: undo in reverse
                    for (gidx, applied_delta) in reversed(applied_seq):
                        if abs(applied_delta) < 1e-12:
                            continue
                        fk_apply_delta(int(gidx), -applied_delta)
                    fk_invalidate_from(min_seg)
                    fk_ensure_upto(max_seg + 1)
                    # no changes to yz_vars or applied_deltas

            # end acfing per window

    # end passes

    v[:n_yz] = yz_vars
    return v
