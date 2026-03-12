from __future__ import annotations

import math
from typing import Sequence

import numpy as np


from shapefit_geometry import (
    assemble_joint_angles,
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

    # Skip solving for `yz` joints: only integrate x (twist) and pass empty yz.
    # State now stores only yz variables (no head). Head is kept internal and not returned.
    yz_len = len(yz_joint_indices_0b)
    if getattr(state, "yz_and_head", None) is None or state.yz_and_head.size < yz_len:
        yz_vars = np.zeros(yz_len, dtype=np.float64)
    else:
        yz_vars = np.asarray(state.yz_and_head[:yz_len], dtype=np.float64).copy()

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
    points, frames = forward_points_and_frames(
        joint_angles=np.asarray(joint_angles, dtype=np.float64),
        joint_axes=joint_axes,
        lengths_m=lengths_m,
        head_translation=np.asarray(head[:3], dtype=np.float64),
        head_rpy=np.asarray(head[3:], dtype=np.float64),
    )
    return joint_angles, points, frames
