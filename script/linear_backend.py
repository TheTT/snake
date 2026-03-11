from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np

from shapefit_geometry import _rpy_to_mat3, assemble_joint_angles
from shapefit_types import FitConfig

ResidualFn = Callable[[np.ndarray], np.ndarray]


def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.asarray(fallback, dtype=np.float64)
    return v / n


def _axis_vec(axis_name: str) -> np.ndarray:
    if axis_name == "x":
        return np.array((1.0, 0.0, 0.0), dtype=np.float64)
    if axis_name == "y":
        return np.array((0.0, 1.0, 0.0), dtype=np.float64)
    if axis_name == "z":
        return np.array((0.0, 0.0, 1.0), dtype=np.float64)
    raise ValueError(f"Unsupported axis {axis_name}")


def _rot_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    ax = _normalize(axis, np.array((1.0, 0.0, 0.0), dtype=np.float64))
    x, y, z = float(ax[0]), float(ax[1]), float(ax[2])
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    return np.array(
        (
            (t * x * x + c, t * x * y - s * z, t * x * z + s * y),
            (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
            (t * x * z - s * y, t * y * z + s * x, t * z * z + c),
        ),
        dtype=np.float64,
    )


def _align_targets(
    tgt: np.ndarray,
    head_translation: np.ndarray,
    head_rpy: np.ndarray,
    joint_axes: Sequence[str],
) -> np.ndarray:
    aligned = np.asarray(tgt, dtype=np.float64).copy()
    if aligned.shape[0] < 2:
        return aligned
    # New behavior: rotate first (about the curve start), then translate the start
    # to the head translation. This reduces issues caused by rotating about a point
    # that may move under translation.
    f0 = aligned[0].copy()

    head_rot = _rpy_to_mat3(float(head_rpy[0]), float(head_rpy[1]), float(head_rpy[2]))
    head_axis = _normalize(head_rot @ np.array((1.0, 0.0, 0.0), dtype=np.float64), np.array((1.0, 0.0, 0.0), dtype=np.float64))

    # source tangent (at curve start)
    src = aligned[1] - aligned[0]
    tangent = _normalize(src, head_axis)

    # 1) Rotate around curve start so the local tangent aligns with head axis
    c = max(-1.0, min(1.0, float(np.dot(tangent, head_axis))))
    ang = math.acos(c)
    axis = np.cross(tangent, head_axis)
    if float(np.linalg.norm(axis)) < 1e-10:
        rot1 = np.eye(3, dtype=np.float64)
    else:
        rot1 = _rot_about_axis(axis, ang)
    aligned = (rot1 @ (aligned - f0).T).T + f0

    # 2) Translate so the rotated start matches head center
    aligned = aligned + (head_translation - aligned[0])

    if aligned.shape[0] < 3:
        return aligned

    # 3) Rotate around head axis (about head_translation) to make the first joint axis
    # close to orthogonal to the local osculating plane normal.
    d0 = aligned[1] - aligned[0]
    d1 = aligned[2] - aligned[1]
    plane_n = np.cross(d0, d1)
    if float(np.linalg.norm(plane_n)) < 1e-10:
        return aligned
    plane_n = _normalize(plane_n, np.array((0.0, 0.0, 1.0), dtype=np.float64))

    first_axis_world = _normalize(head_rot @ _axis_vec(joint_axes[0]), np.array((0.0, 1.0, 0.0), dtype=np.float64))

    best_phi = 0.0
    best_cost = abs(float(np.dot(first_axis_world, plane_n)))
    for phi in np.linspace(-math.pi, math.pi, 73, dtype=np.float64):
        rp = _rot_about_axis(head_axis, float(phi))
        n_phi = rp @ plane_n
        cost = abs(float(np.dot(first_axis_world, n_phi)))
        if cost < best_cost:
            best_cost = cost
            best_phi = float(phi)

    if abs(best_phi) > 1e-8:
        rot2 = _rot_about_axis(head_axis, best_phi)
        aligned = (rot2 @ (aligned - head_translation).T).T + head_translation

    return aligned


def _segment_cost(
    fk_get_midpoint: Callable[[int], np.ndarray],
    aligned_tgt: np.ndarray,
    seg_i: int,
) -> float:
    if seg_i < 0 or seg_i >= aligned_tgt.shape[0]:
        return 0.0
    mid_i = np.asarray(fk_get_midpoint(seg_i), dtype=np.float64)
    c = float(np.sum((mid_i - aligned_tgt[seg_i]) ** 2))
    if seg_i + 1 < aligned_tgt.shape[0]:
        mid_ip1 = np.asarray(fk_get_midpoint(seg_i + 1), dtype=np.float64)
        c += float(np.sum((mid_ip1 - aligned_tgt[seg_i + 1]) ** 2))
    return c


def solve_with_linear_placeholder(
    *,
    residual_fn: ResidualFn,
    v0: np.ndarray,
    cfg: FitConfig,
    precomp: dict | None = None,
) -> np.ndarray:
    """Linear sequential projection backend using incremental FK callbacks.

    This backend keeps x-twist read-only and updates yz joints in-place by
    coordinate-descent style segment projection.
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

    yz_vars = v[:n_yz].copy()
    head = v[n_yz: n_yz + 6].copy()
    head_translation = np.asarray(head[:3], dtype=np.float64)
    head_rpy = np.asarray(head[3:], dtype=np.float64)

    aligned_tgt = _align_targets(tgt, head_translation, head_rpy, joint_axes)

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

    # Track applied deltas per joint and clamp to +/- 90 degrees
    applied_deltas = np.zeros((len(joint_axes),), dtype=np.float64)

    step = max(1e-4, 2.0 * float(cfg.finite_diff_eps))
    n_pass = max(1, int(cfg.max_iter))

    for _ in range(n_pass):
        for yz_k, joint_idx in enumerate(yz_joint_indices_0b):
            seg_i = max(0, min(int(joint_idx), aligned_tgt.shape[0] - 1))

            fk_ensure_upto(seg_i + 1)
            best_cost = _segment_cost(fk_get_midpoint, aligned_tgt, seg_i)
            best_delta_joint = 0.0

            sign = float(joint_signs[int(joint_idx)])
            cand_dyz = (step, -step)
            for dyz in cand_dyz:
                delta_joint = sign * float(dyz)
                # clamp candidate so cumulative applied stays in [-pi/2, pi/2]
                jidx = int(joint_idx)
                if 0 <= jidx < applied_deltas.size:
                    proposed = applied_deltas[jidx] + delta_joint
                    clamped = float(max(-math.pi / 2.0, min(math.pi / 2.0, proposed)))
                    actual_delta = clamped - applied_deltas[jidx]
                else:
                    actual_delta = delta_joint

                if abs(actual_delta) < 1e-12:
                    continue

                fk_apply_delta(int(joint_idx), actual_delta)
                fk_invalidate_from(seg_i)
                fk_ensure_upto(seg_i + 1)
                c = _segment_cost(fk_get_midpoint, aligned_tgt, seg_i)
                fk_apply_delta(int(joint_idx), -actual_delta)
                fk_invalidate_from(seg_i)
                if c < best_cost:
                    best_cost = c
                    best_delta_joint = actual_delta

            if abs(best_delta_joint) > 0.0:
                jidx = int(joint_idx)
                # ensure we do not exceed +/- 90 degrees when applying
                if 0 <= jidx < applied_deltas.size:
                    proposed = applied_deltas[jidx] + best_delta_joint
                    clamped = float(max(-math.pi / 2.0, min(math.pi / 2.0, proposed)))
                    actual_apply = clamped - applied_deltas[jidx]
                else:
                    actual_apply = best_delta_joint

                if abs(actual_apply) > 0.0:
                    fk_apply_delta(int(joint_idx), actual_apply)
                    fk_invalidate_from(seg_i)
                    if 0 <= jidx < applied_deltas.size:
                        applied_deltas[jidx] += actual_apply
                    if abs(sign) > 1e-12:
                        yz_vars[yz_k] += actual_apply / sign

    v[:n_yz] = yz_vars
    return v
