from __future__ import annotations

import math
from typing import Sequence

import numpy as np


from shapefit_geometry import (
    assemble_joint_angles,
    cumulative_s,
    forward_points_and_frames,
    smoothstep01,
    x_joint_s_intervals,
)
from shapefit_target import integrate_avg_g
from shapefit_types import CurveFn, FitConfig, FitState, TwistFn


def integrate_twist(
    *,
    g_fn: TwistFn,
    t: float,
    x_joint_indices_0b: Sequence[int],
    lengths_m: Sequence[float],
    base_twist_rad: float,
    state: FitState,
    cfg: FitConfig,
) -> np.ndarray:
    """Compute per-x-joint twist and update state's low-pass filtered twist.

    Returns the updated `state.twist_filtered` (this value does NOT include any
    `base_twist_rad` offset; callers should add model offsets when assembling
    final joint angles).
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
    # Do not include the model's base twist offset here; keep the filtered
    # twist purely from `g_fn` integration. The model's static +/-90deg offset
    # should be applied when producing joint angles for the robot.
    twist_target = c1 * twist_raw

    dt = 0.0 if state.last_t is None else max(0.0, float(t - state.last_t))
    if dt <= 0.0:
        alpha = 0.0
    else:
        alpha = math.exp(-dt / max(cfg.lowpass_tau_sec, 1e-6))
    state.twist_filtered = alpha * state.twist_filtered + (1.0 - alpha) * twist_target
    state.last_t = float(t)
    return np.asarray(state.twist_filtered, dtype=np.float64).copy()


def integrate_yz_by_segments(
    *,
    f_fn: CurveFn,
    t: float,
    joint_axes: Sequence[str],
    joint_signs: Sequence[float],
    x_joint_indices_0b: Sequence[int],
    yz_joint_indices_0b: Sequence[int],
    lengths_m: Sequence[float],
    twist_no_base: np.ndarray,
    head_translation: np.ndarray,
    head_rot: np.ndarray,
    integration_samples: int = 200,
) -> np.ndarray:
    """Integrate bending on segments [head, each x joint, tail] and assign to yz joints.

    Segment partition follows normalized chain positions at:
    [0.0, s(x_joint_1), ..., s(x_joint_m), 1.0].
    For each segment, compute a discrete accumulated rotation vector from the
    curve, project it onto the segment-start local y/z axes, then distribute the
    projected y/z angle equally to y/z joints inside that segment.
    """
    yz_len = len(yz_joint_indices_0b)
    if yz_len == 0:
        return np.zeros(0, dtype=np.float64)

    n_joints = len(joint_axes)
    s_nodes = cumulative_s(lengths_m)
    yz_vars = np.zeros(yz_len, dtype=np.float64)
    yz_map = {int(idx): i for i, idx in enumerate(yz_joint_indices_0b)}

    x_sorted = sorted(int(v) for v in x_joint_indices_0b)
    x_positions = [float(s_nodes[i + 1]) for i in x_sorted]
    boundaries = [0.0, *x_positions, 1.0]

    yz_zero = np.zeros(yz_len, dtype=np.float64)
    x_only_joint_angles = assemble_joint_angles(
        x_joint_indices_0b,
        yz_joint_indices_0b,
        joint_signs,
        np.asarray(twist_no_base, dtype=np.float64),
        yz_zero,
        n_joints,
    )
    _, frames_x = forward_points_and_frames(
        joint_angles=np.asarray(x_only_joint_angles, dtype=np.float64),
        joint_axes=joint_axes,
        lengths_m=lengths_m,
        head_translation=head_translation,
        head_rpy=np.zeros(3, dtype=np.float64),
        head_rot=head_rot,
    )

    def _segment_start_frame(seg_index: int) -> np.ndarray:
        if seg_index == 0:
            return head_rot
        x_joint_idx = x_sorted[seg_index - 1]
        return np.asarray(frames_x[x_joint_idx], dtype=np.float64)

    eps = 1e-9
    base_steps = max(int(integration_samples), 8)
    for seg_i in range(len(boundaries) - 1):
        s0 = float(boundaries[seg_i])
        s1 = float(boundaries[seg_i + 1])
        if s1 <= s0 + eps:
            continue

        # joints located in this segment: (s0, s1]
        joints_in_segment = []
        for joint_idx in yz_joint_indices_0b:
            jj = int(joint_idx)
            sj = float(s_nodes[jj + 1])
            if (sj > s0 + eps) and (sj <= s1 + eps):
                joints_in_segment.append(jj)
        if not joints_in_segment:
            continue

        seg_steps = max(8, int(round(base_steps * (s1 - s0))))
        s_vals = np.linspace(s0, s1, seg_steps + 1, dtype=np.float64)
        pts = np.array([np.asarray(f_fn(float(t), float(ss)), dtype=np.float64) for ss in s_vals], dtype=np.float64)
        diffs = pts[1:] - pts[:-1]
        lens = np.linalg.norm(diffs, axis=1)
        tangents = np.zeros_like(diffs)
        valid = lens > eps
        if not np.any(valid):
            continue
        tangents[valid] = diffs[valid] / lens[valid, None]

        rot_vec = np.zeros(3, dtype=np.float64)
        for j in range(tangents.shape[0] - 1):
            d0 = tangents[j]
            d1 = tangents[j + 1]
            if (np.linalg.norm(d0) <= eps) or (np.linalg.norm(d1) <= eps):
                continue
            c = np.cross(d0, d1)
            s_norm = float(np.linalg.norm(c))
            dot = float(np.clip(np.dot(d0, d1), -1.0, 1.0))
            if s_norm <= eps:
                continue
            phi = math.atan2(s_norm, dot)
            axis = c / s_norm
            rot_vec += axis * phi

        frame0 = _segment_start_frame(seg_i)
        y_axis = frame0[:, 1]
        z_axis = frame0[:, 2]
        theta_y = float(np.dot(rot_vec, y_axis))
        theta_z = float(np.dot(rot_vec, z_axis))

        y_joints = [j for j in joints_in_segment if joint_axes[j] == "y"]
        z_joints = [j for j in joints_in_segment if joint_axes[j] == "z"]

        if y_joints:
            each_y = theta_y / float(len(y_joints))
            for j in y_joints:
                map_idx = yz_map[j]
                sign = float(joint_signs[j])
                yz_vars[map_idx] = each_y / sign if abs(sign) > eps else each_y

        if z_joints:
            each_z = theta_z / float(len(z_joints))
            for j in z_joints:
                map_idx = yz_map[j]
                sign = float(joint_signs[j])
                yz_vars[map_idx] = each_z / sign if abs(sign) > eps else each_z

    return yz_vars


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_joints = len(joint_axes)
    # DEBUG
    first_run = state.last_t is None
    # integrate and low-pass filter twist x -> updates state.twist_filtered
    # Note: `integrate_twist` returns a filtered twist that does NOT include
    # the model's static base_twist offset; we add that back only when
    # assembling final joint angles so the internal integration is unbiased.
    twist_filtered = integrate_twist(
        g_fn=g_fn,
        t=t,
        x_joint_indices_0b=x_joint_indices_0b,
        lengths_m=lengths_m,
        base_twist_rad=base_twist_rad,
        state=state,
        cfg=cfg,
    )

    # Add model base twist back for the actual joint angle outputs.
    twist_for_joints = base_twist_rad + twist_filtered

    ep = 1e-4
    p0 = np.asarray(f_fn(float(t), 0.0), dtype=np.float64)
    p1 = np.asarray(f_fn(float(t), ep), dtype=np.float64)
    p2 = np.asarray(f_fn(float(t), 2.0 * ep), dtype=np.float64)

    v = p1 - p0
    v_norm = float(np.linalg.norm(v))
    T = v / v_norm if v_norm > 1e-12 else np.array([1.0, 0.0, 0.0], dtype=np.float64)

    accel = p2 - 2.0 * p1 + p0
    accel_proj = accel - np.dot(accel, T) * T
    acc_norm = float(np.linalg.norm(accel_proj))

    if acc_norm > 1e-12:
        N = accel_proj / acc_norm
    else:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        tmp = tmp - np.dot(tmp, T) * T
        N = tmp / np.linalg.norm(tmp)

    B = np.cross(T, N)

    head_rot = np.column_stack((T, N, B))

    yz_vars = integrate_yz_by_segments(
        f_fn=f_fn,
        t=t,
        joint_axes=joint_axes,
        joint_signs=joint_signs,
        x_joint_indices_0b=x_joint_indices_0b,
        yz_joint_indices_0b=yz_joint_indices_0b,
        lengths_m=lengths_m,
        twist_no_base=twist_filtered,
        head_translation=p0,
        head_rot=head_rot,
        integration_samples=200,
    )

    # Do not store or return head; use zero head for FK unless external head is provided elsewhere.
    head = np.zeros(6, dtype=np.float64)
    # Update state with yz only
    state.yz_and_head = yz_vars
    joint_angles = assemble_joint_angles(
        x_joint_indices_0b,
        yz_joint_indices_0b,
        joint_signs,
        twist_for_joints,
        yz_vars,
        n_joints,
    )
    # DEBUG On first solve, output control angles grouped by axis
    if first_run:
        x_idx = [i for i, a in enumerate(joint_axes) if a == "x"]
        y_idx = [i for i, a in enumerate(joint_axes) if a == "y"]
        z_idx = [i for i, a in enumerate(joint_axes) if a == "z"]
        x_angles = [float(joint_angles[i]) for i in x_idx]
        y_angles = [float(joint_angles[i]) for i in y_idx]
        z_angles = [float(joint_angles[i]) for i in z_idx]
        print("initial_joint_groups:", {"x": x_angles, "y": y_angles, "z": z_angles})
    # Compute FK for geometry using twist without the model base offset.
    joint_angles_for_fk = assemble_joint_angles(
        x_joint_indices_0b,
        yz_joint_indices_0b,
        joint_signs,
        twist_filtered,
        yz_vars,
        n_joints,
    )
    points, frames = forward_points_and_frames(
        joint_angles=np.asarray(joint_angles_for_fk, dtype=np.float64),
        joint_axes=joint_axes,
        lengths_m=lengths_m,
        head_translation=p0,
        head_rpy=np.zeros(3, dtype=np.float64),
        head_rot=head_rot,
    )
    return joint_angles, points, frames
