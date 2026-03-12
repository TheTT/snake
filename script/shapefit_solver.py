from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np

import step_backend
from shapefit_geometry import (
    _rot_axis,
    _rpy_to_mat3,
    assemble_joint_angles,
    forward_points_and_frames,
    segment_midpoint_s,
    smoothstep01,
    x_joint_s_intervals,
)
from shapefit_target import build_target_points, integrate_avg_g, target_point
from shapefit_types import CurveFn, FitConfig, FitState, TwistFn, Backend


def _align_targets(
    tgt: np.ndarray,
    head_translation: np.ndarray,
    head_rpy: np.ndarray,
    joint_axes: Sequence[str],
    prev_plane: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    # Inline copy of previous linear_backend._align_targets so step backend
    # can use alignment without depending on linear_backend.
    aligned = np.asarray(tgt, dtype=np.float64).copy()
    if aligned.shape[0] < 2:
        return aligned, None
    f0 = aligned[0].copy()

    head_rot = _rpy_to_mat3(float(head_rpy[0]), float(head_rpy[1]), float(head_rpy[2]))
    head_axis = np.asarray(head_rot @ np.array((1.0, 0.0, 0.0), dtype=np.float64), dtype=np.float64)
    hnorm = float(np.linalg.norm(head_axis))
    if hnorm < 1e-12:
        head_axis = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    else:
        head_axis = head_axis / hnorm
    # define h (head axis) early for use in zero-curvature projection
    h = head_axis

    src = aligned[1] - aligned[0]
    tangent_norm = float(np.linalg.norm(src))
    if tangent_norm < 1e-12:
        tangent = head_axis
    else:
        tangent = src / tangent_norm

    c = float(np.dot(tangent, head_axis))
    c = max(-1.0, min(1.0, c))
    ang = math.acos(c)

    def _rot_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
        ax = axis.copy()
        an = float(np.linalg.norm(ax))
        if an < 1e-12:
            ax = np.array((1.0, 0.0, 0.0), dtype=np.float64)
        else:
            ax = ax / an
        x, y, z = float(ax[0]), float(ax[1]), float(ax[2])
        co = math.cos(angle)
        s = math.sin(angle)
        t = 1.0 - co
        return np.array(
            (
                (t * x * x + co, t * x * y - s * z, t * x * z + s * y),
                (t * x * y + s * z, t * y * y + co, t * y * z - s * x),
                (t * x * z - s * y, t * y * z + s * x, t * z * z + co),
            ),
            dtype=np.float64,
        )

    if ang < 1e-8:
        rot1 = np.eye(3, dtype=np.float64)
    else:
        axis = np.cross(tangent, head_axis)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-10:
            if c > 0.0:
                rot1 = np.eye(3, dtype=np.float64)
            else:
                fallback = np.array((1.0, 0.0, 0.0), dtype=np.float64)
                perp = np.cross(tangent, fallback)
                if float(np.linalg.norm(perp)) < 1e-8:
                    fallback = np.array((0.0, 1.0, 0.0), dtype=np.float64)
                    perp = np.cross(tangent, fallback)
                perp_norm = float(np.linalg.norm(perp))
                if perp_norm < 1e-12:
                    perp = np.array((0.0, 0.0, 1.0), dtype=np.float64)
                else:
                    perp = perp / perp_norm
                rot1 = _rot_about_axis(perp, math.pi)
        else:
            rot1 = _rot_about_axis(axis, ang)

    aligned = (rot1 @ (aligned - f0).T).T + f0
    aligned = aligned + (head_translation - aligned[0])

    if aligned.shape[0] < 3:
        return aligned, None

    d0 = aligned[1] - aligned[0]
    d1 = aligned[2] - aligned[1]
    plane_n = np.cross(d0, d1)
    pn_norm = float(np.linalg.norm(plane_n))
    if pn_norm < 1e-10:
        # curvature nearly zero: attempt to derive plane normal from previous up
        if prev_plane is not None:
            prev = np.asarray(prev_plane, dtype=np.float64)
            prev_norm = float(np.linalg.norm(prev))
            if prev_norm >= 1e-12:
                prev_u = prev / prev_norm
                # project previous up into plane perpendicular to head axis h and normalize
                proj = prev_u - float(np.dot(prev_u, h)) * h
                proj_norm = float(np.linalg.norm(proj))
                if proj_norm >= 1e-12:
                    plane_n = proj / proj_norm
                else:
                    return aligned, None
            else:
                return aligned, None
        else:
            return aligned, None
    else:
        plane_n = plane_n / pn_norm

    first_axis_world = np.asarray(head_rot @ _rot_axis(joint_axes[0], 0.0)[:, 0], dtype=np.float64)
    # fallback
    if float(np.linalg.norm(first_axis_world)) < 1e-12:
        first_axis_world = np.array((0.0, 1.0, 0.0), dtype=np.float64)

    h = head_axis
    p = plane_n
    p_par_comp = float(np.dot(p, h))
    p_perp = p - p_par_comp * h
    p_perp_norm = float(np.linalg.norm(p_perp))
    if p_perp_norm < 1e-10:
        return aligned, None

    u = p_perp / p_perp_norm
    v = np.cross(h, u)
    a = first_axis_world
    A = float(np.dot(a, u))
    B = float(np.dot(a, v))
    best_phi = math.atan2(-A, B)

    # Prepare two candidate rotations: best_phi and best_phi + pi (opposite)
    if abs(best_phi) <= 1e-10:
        # no meaningful rotation
        return aligned, plane_n

    rot2_a = _rot_about_axis(h, best_phi)
    rot2_b = _rot_about_axis(h, best_phi + math.pi)

    aligned_a = (rot2_a @ (aligned - head_translation).T).T + head_translation
    aligned_b = (rot2_b @ (aligned - head_translation).T).T + head_translation

    # resulting rotated plane normals
    d0a = aligned_a[1] - aligned_a[0]
    d1a = aligned_a[2] - aligned_a[1]
    plane_na = np.cross(d0a, d1a)
    if float(np.linalg.norm(plane_na)) >= 1e-12:
        plane_na = plane_na / float(np.linalg.norm(plane_na))
    else:
        plane_na = plane_n

    d0b = aligned_b[1] - aligned_b[0]
    d1b = aligned_b[2] - aligned_b[1]
    plane_nb = np.cross(d0b, d1b)
    if float(np.linalg.norm(plane_nb)) >= 1e-12:
        plane_nb = plane_nb / float(np.linalg.norm(plane_nb))
    else:
        plane_nb = plane_n

    # If previous plane is provided, choose candidate closer to previous (use cross product magnitude)
    if prev_plane is not None:
        prev = np.asarray(prev_plane, dtype=np.float64)
        pn_norm = float(np.linalg.norm(prev))
        if pn_norm >= 1e-12:
            prev = prev / pn_norm
            # cross magnitude ~ sin(angle) -> smaller means more aligned (smaller angular difference)
            diff_a = float(np.linalg.norm(np.cross(prev, plane_na)))
            diff_b = float(np.linalg.norm(np.cross(prev, plane_nb)))
            if diff_a <= diff_b:
                return aligned_a, plane_na
            else:
                return aligned_b, plane_nb

    # No previous plane: choose the candidate whose rotation angle magnitude is smaller
    if abs(best_phi) <= abs(best_phi + math.pi):
        return aligned_a, plane_na
    else:
        return aligned_b, plane_nb


def _build_fk_callbacks() -> dict[str, Callable[..., Any]]:
    cache: dict[str, Any] = {
        "initialized": False,
    }

    def fk_init(
        head_translation: Sequence[float] | np.ndarray,
        head_rpy: Sequence[float] | np.ndarray,
        joint_angles_base_x: Sequence[float] | np.ndarray,
        yz_init: Sequence[float] | np.ndarray,
        lengths_m: Sequence[float] | np.ndarray,
        joint_axes: Sequence[str],
    ) -> None:
        lengths = np.asarray(lengths_m, dtype=np.float64)
        n_joints = len(joint_axes)
        if lengths.size != n_joints + 1:
            raise ValueError("lengths_m length must be n_joints + 1")

        angles = np.asarray(joint_angles_base_x, dtype=np.float64).copy()
        if angles.size != n_joints:
            raise ValueError("joint_angles_base_x length must match joint count")
        yz = np.asarray(yz_init, dtype=np.float64)
        if yz.size == n_joints:
            angles[:] = yz
        elif yz.size > n_joints:
            raise ValueError("yz_init length cannot exceed joint count")
        elif yz.size > 0:
            angles[: yz.size] = yz

        points = np.zeros((n_joints + 2, 3), dtype=np.float64)
        frames = np.zeros((n_joints + 1, 3, 3), dtype=np.float64)

        points[0] = np.asarray(head_translation, dtype=np.float64)
        head_rot = _rpy_to_mat3(float(head_rpy[0]), float(head_rpy[1]), float(head_rpy[2]))

        cache.update(
            {
                "initialized": True,
                "n_joints": n_joints,
                "joint_axes": tuple(joint_axes),
                "lengths": lengths,
                "angles": angles,
                "points": points,
                "frames": frames,
                "head_rot": head_rot,
                "valid_upto_segment": -1,
            }
        )

    def fk_ensure_upto(i: int) -> None:
        if not cache.get("initialized", False):
            raise RuntimeError("fk_init must be called before fk_ensure_upto")

        n_joints = int(cache["n_joints"])
        seg_idx = max(0, min(int(i), n_joints))
        # DEBUG
        valid_upto = int(cache.get("valid_upto_segment", -1))

        points = cache["points"]
        frames = cache["frames"]
        lengths = cache["lengths"]
        angles = cache["angles"]
        joint_axes = cache["joint_axes"]
        head_rot = cache["head_rot"]

        # DEBUG
        start_seg = 0
        for seg in range(start_seg, min(seg_idx, n_joints - 1) + 1):
            rot_before = head_rot if seg == 0 else frames[seg - 1]
            points[seg + 1] = points[seg] + rot_before @ np.array((float(lengths[seg]), 0.0, 0.0), dtype=np.float64)
            frames[seg] = rot_before @ _rot_axis(joint_axes[seg], float(angles[seg]))

        if seg_idx == n_joints:
            if n_joints > 0 and valid_upto < n_joints:
                tail_rot = frames[n_joints - 1]
                points[n_joints + 1] = points[n_joints] + tail_rot @ np.array(
                    (float(lengths[n_joints]), 0.0, 0.0), dtype=np.float64
                )
                frames[n_joints] = tail_rot

        cache["valid_upto_segment"] = seg_idx

    def fk_invalidate_from(i: int) -> None:
        if not cache.get("initialized", False):
            raise RuntimeError("fk_init must be called before fk_invalidate_from")
        seg_idx = max(0, min(int(i), int(cache["n_joints"])))
        cache["valid_upto_segment"] = min(int(cache["valid_upto_segment"]), seg_idx - 1)

    def fk_apply_delta(i: int, delta_rad: float) -> None:
        if not cache.get("initialized", False):
            raise RuntimeError("fk_init must be called before fk_apply_delta")
        j = int(i)
        n_joints = int(cache["n_joints"])
        if j < 0 or j >= n_joints:
            raise IndexError(f"joint index out of range: {j}")

        cache["angles"][j] += float(delta_rad)
        # Invalidate from j (not j+1): frames[j] depends on angles[j] and must be recomputed.
        fk_invalidate_from(j)

    def fk_get_midpoint(i: int) -> np.ndarray:
        if not cache.get("initialized", False):
            raise RuntimeError("fk_init must be called before fk_get_midpoint")

        seg_idx = int(i)
        n_segs = int(cache["n_joints"]) + 1
        if seg_idx < 0 or seg_idx >= n_segs:
            raise IndexError(f"segment index out of range: {seg_idx}")

        fk_ensure_upto(seg_idx)
        points = cache["points"]
        return 0.5 * (points[seg_idx] + points[seg_idx + 1])

    return {
        "fk_init": fk_init,
        "fk_apply_delta": fk_apply_delta,
        "fk_get_midpoint": fk_get_midpoint,
        "fk_invalidate_from": fk_invalidate_from,
        "fk_ensure_upto": fk_ensure_upto,
    }


def compute_twist_filtered(
    *,
    g_fn: TwistFn,
    t: float,
    x_joint_indices_0b: Sequence[int],
    lengths_m: Sequence[float],
    base_twist_rad: float,
    state: FitState,
    cfg: FitConfig,
) -> np.ndarray:
    """Compute per-x-joint twist, apply startup ramp and lowpass, update state.

    Returns twist values with base_twist removed (for FK/geometry use).
    """
    c1 = 1.0
    if t < cfg.startup_ramp_sec:
        c1 = smoothstep01(max(0.0, t) / cfg.startup_ramp_sec)

    intervals = x_joint_s_intervals(x_joint_indices_0b, lengths_m)
    twist_raw = np.array(
        [
            integrate_avg_g(g_fn, max(0.0, t), float(lo), float(hi), cfg.integration_samples)
            for lo, hi in intervals
        ],
        dtype=np.float64,
    )
    twist_target = base_twist_rad + c1 * twist_raw

    dt = 0.0 if state.last_t is None else max(0.0, float(t - state.last_t))
    if dt <= 0.0:
        alpha = 0.0
    else:
        alpha = math.exp(-dt / max(cfg.lowpass_tau_sec, 1e-6))
    state.twist_filtered = alpha * state.twist_filtered + (1.0 - alpha) * twist_target
    state.last_t = float(t)

    twist_no_base = np.asarray(state.twist_filtered, dtype=np.float64) - float(base_twist_rad)
    return twist_no_base


def solve_shape_for_time(
    *,
    f_fn: CurveFn,
    g_fn: TwistFn,
    t: float,
    joint_axes: Sequence[str],
    joint_signs: Sequence[float],
    x_joint_indices_0b: Sequence[int],
    yz_joint_indices_0b: Sequence[int],
    lengths_m: Sequence[float],
    base_twist_rad: float,
    state: FitState,
    cfg: FitConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_joints = len(joint_axes)

    # Compute and filter per-x-joint twist (separated function)
    twist_no_base = compute_twist_filtered(
        g_fn=g_fn,
        t=t,
        x_joint_indices_0b=x_joint_indices_0b,
        lengths_m=lengths_m,
        base_twist_rad=base_twist_rad,
        state=state,
        cfg=cfg,
    )

    seg_mid_s = segment_midpoint_s(lengths_m)
    tgt = build_target_points(f_fn, lengths_m, t, seg_mid_s, cfg.startup_ramp_sec)
    fk_callbacks = _build_fk_callbacks()

    # NOTE: residual construction removed — `step_backend` does not use it.

    # Swap this backend import to anneal_backend.solve_with_anneal_placeholder
    # without touching residual construction logic.
    def _align_closure(tgt_arr, head_translation, head_rpy, joint_axes):
        aligned_res, chosen_plane = _align_targets(
            tgt_arr, head_translation, head_rpy, joint_axes, prev_plane=state.last_align_plane_n
        )
        state.last_align_plane_n = None if chosen_plane is None else np.asarray(chosen_plane, dtype=np.float64)
        return aligned_res

    precomp = {
        "lengths_m": np.asarray(lengths_m, dtype=np.float64),
        # curve_samples: Nx3 points sampled uniformly by arc-length (meters)
        "curve_samples": None,
        # alignment helper used by step backend (or precomputed aligned_curve_samples)
        "align_func": _align_closure,
        # twist values without base offset for geometry-based solvers
        "twist_filtered": np.asarray(twist_no_base, dtype=np.float64).copy(),
        "joint_axes": tuple(joint_axes),
        "joint_signs": np.asarray(joint_signs, dtype=np.float64),
        "x_joint_indices_0b": tuple(int(i) for i in x_joint_indices_0b),
        "yz_joint_indices_0b": tuple(int(i) for i in yz_joint_indices_0b),
        "fk_callbacks": fk_callbacks,
    }

    backend_map = {
        Backend.STEP: step_backend.solve_with_step_placeholder,
    }

    backend_fn = backend_map.get(cfg.backend, step_backend.solve_with_step_placeholder)
    # Build uniform arc-length curve samples (in meters) for precomp.
    total_length = float(np.sum(lengths_m))
    n_samples = int(precomp.get("curve_samples_n", 200))

    s_m = np.linspace(0.0, total_length, max(2, int(n_samples)), dtype=np.float64)
    s_norm = s_m / max(total_length, 1e-12)
    curve_pts = np.array([target_point(f_fn, lengths_m, t, float(s), cfg.startup_ramp_sec) for s in s_norm], dtype=np.float64)
    # update precomp with concrete samples
    precomp["curve_samples"] = curve_pts
    precomp["curve_samples_arc"] = np.column_stack((s_m, curve_pts))

    v = backend_fn(
        v0=state.yz_and_head,
        cfg=cfg,
        precomp=precomp,
    )

    # Only update robot yz joint variables in state; use backend-returned head
    v = np.asarray(v, dtype=np.float64).copy()
    n_yz = len(yz_joint_indices_0b)
    if v.size >= n_yz:
        state.yz_and_head[:n_yz] = v[:n_yz]

    # Use preserved yz vars from state, but use head computed by backend (not written back to state)
    yz_vars = state.yz_and_head[:n_yz].copy()
    head = v[n_yz : n_yz + 6].copy() if v.size >= n_yz + 6 else state.yz_and_head[n_yz : n_yz + 6].copy()
    # Return joint angles that include the base twist (these are sent to the
    # robot controller). But compute FK frames/points using twist without the
    # base so the model's built-in -90deg is not applied to geometry.
    joint_angles_with_base = assemble_joint_angles(
        x_joint_indices_0b,
        yz_joint_indices_0b,
        joint_signs,
        state.twist_filtered,
        yz_vars,
        n_joints,
    )

    joint_angles_for_fk = assemble_joint_angles(
        x_joint_indices_0b,
        yz_joint_indices_0b,
        joint_signs,
        twist_no_base,
        yz_vars,
        n_joints,
    )

    points, frames = forward_points_and_frames(
        joint_angles=np.asarray(joint_angles_for_fk, dtype=np.float64),
        joint_axes=joint_axes,
        lengths_m=lengths_m,
        head_translation=np.asarray(head[:3], dtype=np.float64),
        head_rpy=np.asarray(head[3:], dtype=np.float64),
    )
    return joint_angles_with_base, points, frames, head
