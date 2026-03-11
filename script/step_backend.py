from __future__ import annotations

import math
from typing import Callable, Sequence

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


def _dist_to_polyline(p, curve_pts, hint_i):
    """
    curve_pts: Nx3 polyline
    hint_i: approximate index (uses locality)
    """
    n = len(curve_pts)

    start = max(0, hint_i - 3)
    end = min(n - 2, hint_i + 3)

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

    seg_hint = 0

    def segment_cost(seg_i):

        fk_ensure_upto(seg_i + 1)

        p = np.asarray(fk_get_midpoint(seg_i))

        return _dist_to_polyline(p, curve_pts, seg_hint)

    for _ in range(passes):

        for start in range(max(1, n_yz - 2)):

            window = yz_joint_indices[start : start + 3]

            base_cost = 0.0

            for j in window:

                seg_i = int(j)

                base_cost += segment_cost(seg_i)

            for local_k, j in enumerate(window):

                sign = joint_signs[j]

                best_delta = 0.0
                best_cost = base_cost

                for dy in (step, -step):

                    delta_joint = sign * dy

                    proposed = applied[j] + delta_joint

                    if proposed > math.pi / 2:
                        continue
                    if proposed < -math.pi / 2:
                        continue

                    fk_apply_delta(j, delta_joint)

                    fk_invalidate_from(j)

                    c = 0.0

                    for jj in window:

                        c += segment_cost(int(jj))

                    fk_apply_delta(j, -delta_joint)

                    fk_invalidate_from(j)

                    if c < best_cost:
                        best_cost = c
                        best_delta = delta_joint

                if best_delta != 0.0:

                    fk_apply_delta(j, best_delta)

                    fk_invalidate_from(j)

                    applied[j] += best_delta

                    if abs(sign) > 1e-12:
                        yz_vars[start + local_k] += best_delta / sign

    v[:n_yz] = yz_vars

    return v
