from __future__ import annotations

import math
import random
from typing import Callable

import numpy as np

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
    anneal_iters = int(precomp.get("step_anneal_iters", 16))
    anneal_step_scale = float(precomp.get("step_anneal_step_scale", 2.0))
    geom_tol = float(precomp.get("step_geom_tol", 1e-12))
    random_seed = precomp.get("random_seed", None)
    if random_seed is not None:
        random.seed(int(random_seed))

    yz_var_pos_by_joint = {int(j): idx for idx, j in enumerate(yz_joint_indices)}

    def _curve_hint_from_fk_segment(seg_i: int) -> int:
        # Map FK segment index [0, n_fk_segments-1] into curve sample index [0, n_curve-1].
        n_curve = int(curve_pts.shape[0])
        if n_curve <= 1 or n_fk_segments <= 1:
            return 0
        ratio = float(max(0, min(int(seg_i), n_fk_segments - 1))) / float(n_fk_segments - 1)
        return int(round(ratio * float(n_curve - 1)))

    def _segment_midpoint_distance_sq_to_curve(seg_i: int) -> float:
        # Pure geometric distance from midpoint to curve (without orientation penalty).
        seg_i = max(0, min(int(seg_i), n_fk_segments - 1))
        fk_ensure_upto(min(seg_i + 1, n_fk_segments))
        p = np.asarray(fk_get_midpoint(seg_i), dtype=np.float64)
        hint_i = _curve_hint_from_fk_segment(seg_i)
        dsq, _ = _dist_to_polyline(p, curve_pts, hint_i, poly_radius)
        return float(dsq)

    def _anneal_joint_delta(
        j: int,
        drive_sign: float,
        primary_seg_i: int,
        applied_base: float,
        curr_geom_base: float,
    ) -> tuple[float, float, int, int, int]:
        """One-joint local search minimizing midpoint distance of its primary segment.

        Returns (best_delta_from_base, best_geom_cost, tried, accepted, clipped).
        """
        curr_delta = 0.0
        curr_geom = float(curr_geom_base)
        best_delta = 0.0
        best_geom = float(curr_geom_base)
        n_tried = 0
        n_accepted = 0
        n_clipped = 0

        n_iter = max(2, int(anneal_iters))
        for it in range(n_iter):
            frac = (it + 1) / float(n_iter)

            # Start with larger local proposals, then shrink toward the end.
            scale = 1.0 + anneal_step_scale * (1.0 - frac)
            trial_step = step * scale
            if random.random() < 0.35:
                trial_step = step
            dsign = -1.0 if random.random() < 0.5 else 1.0
            delta_joint = float(drive_sign * dsign * trial_step)

            proposed = float(applied_base + curr_delta + delta_joint)
            if proposed > math.pi / 2 or proposed < -math.pi / 2:
                n_clipped += 1
                continue

            n_tried += 1
            fk_apply_delta(j, delta_joint)
            fk_invalidate_from(int(j) + 1)
            new_geom = _segment_midpoint_distance_sq_to_curve(primary_seg_i)

            # Only accept strictly better geometry for this one primary segment.
            accept = new_geom < (curr_geom - geom_tol)

            if accept:
                n_accepted += 1
                curr_delta += delta_joint
                curr_geom = float(new_geom)
                if curr_geom < (best_geom - geom_tol):
                    best_geom = curr_geom
                    best_delta = curr_delta
            else:
                fk_apply_delta(j, -delta_joint)
                fk_invalidate_from(int(j) + 1)

        # Restore baseline state for caller; caller will apply best_delta once.
        if abs(curr_delta) > 1e-12:
            fk_apply_delta(j, -curr_delta)
            fk_invalidate_from(int(j) + 1)

        return float(best_delta), float(best_geom), int(n_tried), int(n_accepted), int(n_clipped)

    # 第4节(索引3)前一关节是索引2，该日志只跟踪这个关节(若非x轴)的退火前后变化。
    seg4_i = max(0, min(3, n_fk_segments - 1))
    seg4_prev_joint = max(0, seg4_i - 1)

    for pass_idx in range(passes):
        # Segment-joint mapping rule:
        # segment 0 has no previous joint; segment s>=1 maps to previous joint (s-1).
        for seg_i in range(1, n_fk_segments):
            j_int = int(seg_i - 1)
            if j_int < 0 or j_int >= len(joint_axes):
                continue

            # Only optimize when previous joint is not twist x.
            axis_name = str(joint_axes[j_int]).lower()
            if axis_name == "x":
                continue

            sign = float(joint_signs[j_int])
            drive_sign = sign
            primary_seg_i = int(seg_i)
            curr_geom = _segment_midpoint_distance_sq_to_curve(primary_seg_i)

            seg4_before = None
            if j_int == seg4_prev_joint:
                seg4_before = math.sqrt(max(0.0, _segment_midpoint_distance_sq_to_curve(seg4_i)))

            best_delta, _best_geom, n_tried, n_accepted, n_clipped = _anneal_joint_delta(
                j_int,
                float(drive_sign),
                int(primary_seg_i),
                float(applied[j_int]),
                float(curr_geom),
            )

            if best_delta != 0.0:
                fk_apply_delta(j_int, best_delta)
                fk_invalidate_from(j_int + 1)
                applied[j_int] += best_delta

                if abs(sign) > 1e-12:
                    yz_pos = yz_var_pos_by_joint.get(j_int, None)
                    if yz_pos is not None:
                        yz_vars[yz_pos] += best_delta / sign

            if seg4_before is not None:
                seg4_after = math.sqrt(max(0.0, _segment_midpoint_distance_sq_to_curve(seg4_i)))
                print(
                    f"[STEP] 第4节前一关节退火(pass={pass_idx + 1}/{passes})后: "
                    f"seg={seg4_i}, joint={seg4_prev_joint}, axis={axis_name}; "
                    f"中点到曲线距离 调整前={seg4_before:.9e} m, 调整后={seg4_after:.9e} m, "
                    f"delta={seg4_after - seg4_before:+.3e} m; "
                    f"best_delta={best_delta:+.3e} rad; "
                    f"tries={n_tried}, accepted={n_accepted}, clipped={n_clipped}"
                )

    v[:n_yz] = yz_vars

    return v
