from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from jacob_fd import forward_difference_jacobian
from shapefit_geometry import (
    assemble_joint_angles,
    forward_points_and_frames,
    segment_midpoint_s,
    smoothstep01,
    x_joint_s_intervals,
)
from shapefit_target import build_target_points, integrate_avg_g
from shapefit_types import CurveFn, FitConfig, FitState, TwistFn


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

    seg_mid_s = segment_midpoint_s(lengths_m)
    tgt = build_target_points(f_fn, lengths_m, t, seg_mid_s, cfg.startup_ramp_sec)

    def residual(v: np.ndarray) -> np.ndarray:
        yz_vars = v[: len(yz_joint_indices_0b)]
        head = v[len(yz_joint_indices_0b):]
        head_translation = np.asarray(head[:3], dtype=np.float64)
        head_rpy = np.asarray(head[3:], dtype=np.float64)

        joint_angles = assemble_joint_angles(
            x_joint_indices_0b,
            yz_joint_indices_0b,
            joint_signs,
            state.twist_filtered,
            yz_vars,
            n_joints,
        )
        points, _frames = forward_points_and_frames(
            joint_angles=np.asarray(joint_angles, dtype=np.float64),
            joint_axes=joint_axes,
            lengths_m=lengths_m,
            head_translation=head_translation,
            head_rpy=head_rpy,
        )
        mids = 0.5 * (points[:-1] + points[1:])
        return (mids - tgt).reshape(-1)

    v = state.yz_and_head.copy()
    for _ in range(cfg.max_iter):
        r0 = residual(v)
        eps = max(cfg.finite_diff_eps, 1e-8)
        jac = forward_difference_jacobian(residual, v, r0, eps)
        jac = np.asarray(jac, dtype=np.float64)
        r0 = np.asarray(r0, dtype=np.float64)

        lhs = jac.T @ jac + cfg.damping * np.eye(v.size, dtype=np.float64)
        rhs = -1.0 * (jac.T @ r0)
        try:
            dv = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            break

        v = v + dv
        if float(np.linalg.norm(dv)) < 1e-5:
            break

    state.yz_and_head = v

    yz_vars = v[: len(yz_joint_indices_0b)]
    head = v[len(yz_joint_indices_0b):]
    joint_angles = assemble_joint_angles(
        x_joint_indices_0b,
        yz_joint_indices_0b,
        joint_signs,
        state.twist_filtered,
        yz_vars,
        n_joints,
    )
    points, frames = forward_points_and_frames(
        joint_angles=np.asarray(joint_angles, dtype=np.float64),
        joint_axes=joint_axes,
        lengths_m=lengths_m,
        head_translation=np.asarray(head[:3], dtype=np.float64),
        head_rpy=np.asarray(head[3:], dtype=np.float64),
    )
    return joint_angles, points, frames, head
