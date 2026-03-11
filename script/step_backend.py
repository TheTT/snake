from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

import linear_backend
from shapefit_geometry import assemble_joint_angles
from shapefit_types import FitConfig

ResidualFn = Callable[[np.ndarray], np.ndarray]


def _point_segment_dist_sq(p, a, b):
    ab = b - a
    t = np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12)
    t = max(0.0, min(1.0, t))
    q = a + t * ab
    d = p - q
    return float(np.dot(d, d))


def _dist_to_polyline(p, curve_pts, hint_i, radius):
    """
    curve_pts: Nx3 polyline
    hint_i: approximate index (uses locality)
    """
    n = len(curve_pts)
    if n < 2:
        d = p - curve_pts[0] if n == 1 else p
        return float(np.dot(d, d)), 0

    start = max(0, hint_i - int(radius))
    end = min(n - 2, hint_i + int(radius))

    best = 1e30
    best_i = start

    for i in range(start, end + 1):
        d = _point_segment_dist_sq(
            p,
            curve_pts[i],
            curve_pts[i + 1],
        )
        if d < best:
            best = d
            best_i = i

    return best, int(best_i)


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array((1.0, 0.0, 0.0), dtype=np.float64)
    return v / n


def solve_with_step_placeholder(
    *,
    residual_fn: ResidualFn,
    v0: np.ndarray,
    cfg: FitConfig,
    precomp: dict | None = None,
) -> np.ndarray:

    v = np.asarray(v0, dtype=np.float64).copy()

    if precomp is None:
        return v

    fk = precomp.get("fk_callbacks", None)
    if fk is None:
        return v

    lengths = np.asarray(precomp["lengths_m"], dtype=np.float64)

    curve_pts = np.asarray(precomp["curve_samples"], dtype=np.float64)

    joint_axes = tuple(precomp["joint_axes"])
    joint_signs = np.asarray(precomp["joint_signs"], dtype=np.float64)

    x_joint_indices = tuple(precomp["x_joint_indices_0b"])
    yz_joint_indices = tuple(precomp["yz_joint_indices_0b"])

    x_twist = np.asarray(precomp["twist_filtered"], dtype=np.float64)

    n_yz = len(yz_joint_indices)

    yz_vars = v[:n_yz].copy()

    head = v[n_yz : n_yz + 6]

    head_translation = head[:3]
    head_rpy = head[3:]

    # Keep target curve in the same world frame convention used by rendering/other backends.
    if "aligned_curve_samples" in precomp and isinstance(precomp["aligned_curve_samples"], np.ndarray):
        curve_pts = np.asarray(precomp["aligned_curve_samples"], dtype=np.float64)
    elif "align_func" in precomp and callable(precomp["align_func"]):
        curve_pts = np.asarray(precomp["align_func"](curve_pts, head_translation, head_rpy, joint_axes), dtype=np.float64)

    if curve_pts.ndim != 2 or curve_pts.shape[1] != 3 or curve_pts.shape[0] == 0:
        return v

    joint_angles = assemble_joint_angles(
        x_joint_indices,
        yz_joint_indices,
        joint_signs.tolist(),
        x_twist,
        yz_vars,
        len(joint_axes),
    )

    fk_init = fk["fk_init"]
    fk_apply_delta = fk["fk_apply_delta"]
    fk_get_midpoint = fk["fk_get_midpoint"]
    fk_invalidate_from = fk["fk_invalidate_from"]
    fk_ensure_upto = fk["fk_ensure_upto"]

    fk_init(
        head_translation=head_translation,
        head_rpy=head_rpy,
        joint_angles_base_x=joint_angles,
        yz_init=joint_angles,
        lengths_m=lengths,
        joint_axes=joint_axes,
    )

    step = 2 * cfg.finite_diff_eps

    passes = max(1, int(cfg.max_iter))

    applied = np.zeros(len(joint_axes))

    n_fk_segments = int(lengths.size)
    auto_radius = max(6, int(round(float(curve_pts.shape[0]) / max(1.0, float(n_fk_segments)))))
    poly_radius = int(precomp.get("step_polyline_radius", auto_radius))
    dir_weight = float(precomp.get("step_direction_weight", 0.5 * float(np.mean(lengths) ** 2)))
    downstream_span = int(precomp.get("step_downstream_span", max(6, n_fk_segments // 2)))
    downstream_decay = float(precomp.get("step_downstream_decay", 0.92))

    def _primary_segment_for_joint(jidx: int) -> int:
        # In this FK convention, joint j rotates the frame after segment j,
        # so the first directly affected segment is j+1.
        return max(0, min(int(jidx) + 1, n_fk_segments - 1))

    def _curve_hint_from_fk_segment(seg_i: int) -> int:
        # Map FK segment index [0, n_fk_segments-1] into curve sample index [0, n_curve-1].
        n_curve = int(curve_pts.shape[0])
        if n_curve <= 1 or n_fk_segments <= 1:
            return 0
        ratio = float(max(0, min(int(seg_i), n_fk_segments - 1))) / float(n_fk_segments - 1)
        return int(round(ratio * float(n_curve - 1)))

    def segment_cost(seg_i):
        seg_i = max(0, min(int(seg_i), n_fk_segments - 1))
        fk_ensure_upto(min(seg_i + 2, n_fk_segments))

        p = np.asarray(fk_get_midpoint(seg_i), dtype=np.float64)
        hint_i = _curve_hint_from_fk_segment(seg_i)
        dsq, best_curve_seg = _dist_to_polyline(p, curve_pts, hint_i, poly_radius)

        # Direction consistency term: discourage body tangent opposite to curve tangent.
        if seg_i < n_fk_segments - 1:
            p_next = np.asarray(fk_get_midpoint(seg_i + 1), dtype=np.float64)
            body_tan = _safe_normalize(p_next - p)
        elif seg_i > 0:
            p_prev = np.asarray(fk_get_midpoint(seg_i - 1), dtype=np.float64)
            body_tan = _safe_normalize(p - p_prev)
        else:
            body_tan = np.array((1.0, 0.0, 0.0), dtype=np.float64)

        c0 = curve_pts[best_curve_seg]
        c1 = curve_pts[min(best_curve_seg + 1, curve_pts.shape[0] - 1)]
        curve_tan = _safe_normalize(np.asarray(c1 - c0, dtype=np.float64))
        cosang = float(np.dot(body_tan, curve_tan))
        orient_penalty = 0.5 * (1.0 - max(-1.0, min(1.0, cosang)))
        return float(dsq + dir_weight * orient_penalty)

    def _window_cost(window_joint_indices: Sequence[int]) -> float:
        if len(window_joint_indices) == 0:
            return 0.0
        primary = [_primary_segment_for_joint(int(j)) for j in window_joint_indices]
        seg_start = max(0, min(primary))
        seg_end = min(n_fk_segments - 1, max(primary) + max(0, downstream_span))
        csum = 0.0
        for seg in range(seg_start, seg_end + 1):
            w = float(downstream_decay ** max(0, seg - seg_start))
            csum += w * segment_cost(seg)
        return float(csum)

    for _ in range(passes):

        for start in range(max(1, n_yz - 2)):

            window = yz_joint_indices[start : start + 3]

            base_cost = _window_cost(window)

            for local_k, j in enumerate(window):

                sign = joint_signs[j]
                # DEBUG
                axis_name = str(joint_axes[int(j)]).lower() if 0 <= int(j) < len(joint_axes) else ""
                # Experimental switch requested by user: reverse angle increment on y-axis joints.
                drive_sign = -sign if axis_name == "y" else sign

                best_delta = 0.0
                best_cost = base_cost

                for dy in (step, -step):

                    delta_joint = drive_sign * dy

                    proposed = applied[j] + delta_joint

                    if proposed > math.pi / 2:
                        continue
                    if proposed < -math.pi / 2:
                        continue

                    fk_apply_delta(j, delta_joint)

                    fk_invalidate_from(int(j) + 1)

                    c = _window_cost(window)

                    fk_apply_delta(j, -delta_joint)

                    fk_invalidate_from(int(j) + 1)

                    if c < best_cost:
                        best_cost = c
                        best_delta = delta_joint

                if best_delta != 0.0:

                    fk_apply_delta(j, best_delta)

                    fk_invalidate_from(int(j) + 1)

                    applied[j] += best_delta

                    if abs(sign) > 1e-12:
                        yz_vars[start + local_k] += best_delta / sign

    v[:n_yz] = yz_vars

    return v
