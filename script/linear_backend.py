from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np

from shapefit_geometry import _rpy_to_mat3, assemble_joint_angles
from shapefit_types import FitConfig

ResidualFn = Callable[[np.ndarray], np.ndarray]

EPS = 1e-12


def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < EPS:
        return np.asarray(fallback, dtype=np.float64)
    return v / n


def _axis_vec(axis_name: str) -> np.ndarray:
    # accept upper/lower case
    an = axis_name.lower()
    if an == "x":
        return np.array((1.0, 0.0, 0.0), dtype=np.float64)
    if an == "y":
        return np.array((0.0, 1.0, 0.0), dtype=np.float64)
    if an == "z":
        return np.array((0.0, 0.0, 1.0), dtype=np.float64)
    raise ValueError(f"Unsupported axis '{axis_name}'; expected 'x','y' or 'z'.")


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
    # rotate about curve start, then translate
    f0 = aligned[0].copy()

    head_rot = _rpy_to_mat3(float(head_rpy[0]), float(head_rpy[1]), float(head_rpy[2]))
    head_axis = _normalize(head_rot @ np.array((1.0, 0.0, 0.0), dtype=np.float64),
                           np.array((1.0, 0.0, 0.0), dtype=np.float64))

    # source tangent (at curve start)
    src = aligned[1] - aligned[0]
    tangent = _normalize(src, head_axis)

    # 1) Rotate around curve start so the local tangent aligns with head axis
    # compute dot and handle parallel / anti-parallel cases robustly
    c = float(np.dot(tangent, head_axis))
    c = max(-1.0, min(1.0, c))
    ang = math.acos(c)

    # if angle is tiny => no rotation
    if ang < 1e-8:
        rot1 = np.eye(3, dtype=np.float64)
    else:
        axis = np.cross(tangent, head_axis)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-10:
            # tangent and head_axis are (nearly) parallel or anti-parallel
            if c > 0.0:
                # already aligned
                rot1 = np.eye(3, dtype=np.float64)
            else:
                # opposite direction: choose arbitrary perp axis and rotate by pi
                # pick a stable perpendicular vector
                fallback = np.array((1.0, 0.0, 0.0), dtype=np.float64)
                perp = np.cross(tangent, fallback)
                if float(np.linalg.norm(perp)) < 1e-8:
                    # tangent was nearly parallel to fallback; choose Y
                    fallback = np.array((0.0, 1.0, 0.0), dtype=np.float64)
                    perp = np.cross(tangent, fallback)
                perp = _normalize(perp, np.array((0.0, 0.0, 1.0), dtype=np.float64))
                rot1 = _rot_about_axis(perp, math.pi)
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

    first_axis_world = _normalize(head_rot @ _axis_vec(joint_axes[0]),
                                  np.array((0.0, 1.0, 0.0), dtype=np.float64))

    # We want to rotate plane_n around head_axis to minimize |first_axis_world · rotated_plane_n|.
    # Solve analytically: let h=head_axis. Decompose p=plane_n into p_par + p_perp (w.r.t h).
    # If p_perp is zero (p parallel to h) => nothing to do.
    h = head_axis
    p = plane_n
    p_par_comp = float(np.dot(p, h))
    p_perp = p - p_par_comp * h
    p_perp_norm = float(np.linalg.norm(p_perp))
    if p_perp_norm < 1e-10:
        # plane normal is parallel to head axis (no meaningful rotation)
        return aligned

    # build orthonormal basis (u, v) in plane perpendicular to h, with u aligned to p_perp
    u = p_perp / p_perp_norm
    v = np.cross(h, u)  # already orthogonal to both
    # project first_axis_world into the (u,v) plane
    a = first_axis_world
    A = float(np.dot(a, u))
    B = float(np.dot(a, v))
    # we want phi such that A*cos(phi) + B*sin(phi) is minimized in absolute value.
    # choose phi = atan2(-A, B) which makes A*cos + B*sin = 0 (if solvable).
    best_phi = math.atan2(-A, B)

    # if best_phi is negligibly small, skip rotate
    if abs(best_phi) > 1e-10:
        rot2 = _rot_about_axis(h, best_phi)
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

            if abs(best_delta_joint) > 1e-12:
                jidx = int(joint_idx)
                # ensure we do not exceed +/- 90 degrees when applying
                if 0 <= jidx < applied_deltas.size:
                    proposed = applied_deltas[jidx] + best_delta_joint
                    clamped = float(max(-math.pi / 2.0, min(math.pi / 2.0, proposed)))
                    actual_apply = clamped - applied_deltas[jidx]
                else:
                    actual_apply = best_delta_joint

                if abs(actual_apply) > 1e-12:
                    fk_apply_delta(int(joint_idx), actual_apply)
                    fk_invalidate_from(seg_i)
                    if 0 <= jidx < applied_deltas.size:
                        applied_deltas[jidx] += actual_apply
                    if abs(sign) > 1e-12:
                        yz_vars[yz_k] += actual_apply / sign

    v[:n_yz] = yz_vars
    return v
