"""Curve-to-shape fitting utilities for snake theoretical joint angles.

This module implements:
- x-twist evaluation from interval-averaged g(t, s)
- dynamic initialization blending line(s) -> f(0, s)
- damped Gauss-Newton solve for yz joints + head 6DoF
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

Vec3 = tuple[float, float, float]
CurveFn = Callable[[float, float], Vec3]
TwistFn = Callable[[float, float], float]


@dataclass
class FitConfig:
    startup_ramp_sec: float = 1.0
    max_iter: int = 3
    damping: float = 1e-2
    finite_diff_eps: float = 1e-4
    integration_samples: int = 5
    lowpass_tau_sec: float = 0.05
    temporal_reg_weight: float = 1e-2


@dataclass
class FitState:
    yz_and_head: np.ndarray
    twist_filtered: np.ndarray
    last_t: float | None = None


def smoothstep01(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _rot_x(a: float) -> np.ndarray:
    ca, sa = math.cos(a), math.sin(a)
    return np.array(((1.0, 0.0, 0.0), (0.0, ca, -sa), (0.0, sa, ca)), dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    ca, sa = math.cos(a), math.sin(a)
    return np.array(((ca, 0.0, sa), (0.0, 1.0, 0.0), (-sa, 0.0, ca)), dtype=np.float64)


def _rot_z(a: float) -> np.ndarray:
    ca, sa = math.cos(a), math.sin(a)
    return np.array(((ca, -sa, 0.0), (sa, ca, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)


def _rot_axis(axis_name: str, angle: float) -> np.ndarray:
    if axis_name == "x":
        return _rot_x(angle)
    if axis_name == "y":
        return _rot_y(angle)
    if axis_name == "z":
        return _rot_z(angle)
    raise ValueError(f"Unsupported axis {axis_name}")


def _rpy_to_mat3(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def cumulative_s(lengths_m: Sequence[float]) -> np.ndarray:
    lens = np.asarray(lengths_m, dtype=np.float64)
    if lens.ndim != 1 or lens.size == 0:
        raise ValueError("lengths_m must be a non-empty 1D sequence")
    total = float(np.sum(lens))
    if total <= 0.0:
        raise ValueError("sum(lengths_m) must be > 0")
    out = np.zeros(lens.size + 1, dtype=np.float64)
    out[1:] = np.cumsum(lens)
    return out / total


def x_joint_s_intervals(x_joint_indices_0b: Sequence[int], lengths_m: Sequence[float]) -> np.ndarray:
    s_nodes = cumulative_s(lengths_m)
    x_sorted = sorted(int(v) for v in x_joint_indices_0b)
    if not x_sorted:
        return np.zeros((0, 2), dtype=np.float64)

    # Joint i (0-based) is at node s_nodes[i + 1] in the chain used by FK.
    s_x = [float(s_nodes[i + 1]) for i in x_sorted]

    bounds = [0.0]
    for i in range(len(s_x) - 1):
        bounds.append(0.5 * (s_x[i] + s_x[i + 1]))
    bounds.append(1.0)

    intervals = []
    for i in range(len(s_x)):
        intervals.append((bounds[i], bounds[i + 1]))
    return np.asarray(intervals, dtype=np.float64)


def _integrate_avg_g(g_fn: TwistFn, t: float, s_lo: float, s_hi: float, samples: int) -> float:
    s_lo_f = float(s_lo)
    s_hi_f = float(s_hi)
    if s_hi_f - s_lo_f < 1e-8:
        return float(g_fn(t, 0.5 * (s_lo_f + s_hi_f)))

    n = max(3, int(samples))
    ss = np.linspace(s_lo_f, s_hi_f, n, dtype=np.float64)
    vals = np.array([float(g_fn(t, float(s))) for s in ss], dtype=np.float64)
    vals[~np.isfinite(vals)] = 0.0
    area = float(np.trapezoid(vals, ss))
    return area / (s_hi_f - s_lo_f)


def _segment_midpoint_s(lengths_m: Sequence[float]) -> np.ndarray:
    s_nodes = cumulative_s(lengths_m)
    return 0.5 * (s_nodes[:-1] + s_nodes[1:])


def _line_curve(lengths_m: Sequence[float], s: float) -> np.ndarray:
    total = float(np.sum(np.asarray(lengths_m, dtype=np.float64)))
    return np.array((total * s, 0.0, 0.0), dtype=np.float64)


def _target_point(f_fn: CurveFn, lengths_m: Sequence[float], t: float, s: float, startup_ramp_sec: float) -> np.ndarray:
    if t <= 0.0:
        return _line_curve(lengths_m, s)

    if t < startup_ramp_sec:
        c1 = smoothstep01(t / startup_ramp_sec)
        line = _line_curve(lengths_m, s)
        target0 = np.asarray(f_fn(0.0, s), dtype=np.float64)
        return (1.0 - c1) * line + c1 * target0

    return np.asarray(f_fn(t - startup_ramp_sec, s), dtype=np.float64)


def _assemble_joint_angles(
    x_joint_indices_0b: Sequence[int],
    yz_joint_indices_0b: Sequence[int],
    joint_signs: Sequence[float],
    twist_angles: np.ndarray,
    yz_vars: np.ndarray,
    n_joints: int,
) -> np.ndarray:
    out = np.zeros(n_joints, dtype=np.float64)
    x_map = {int(idx): i for i, idx in enumerate(x_joint_indices_0b)}
    yz_map = {int(idx): i for i, idx in enumerate(yz_joint_indices_0b)}

    for j in range(n_joints):
        sign = float(joint_signs[j])
        if j in x_map:
            out[j] = float(twist_angles[x_map[j]])
        elif j in yz_map:
            out[j] = sign * float(yz_vars[yz_map[j]])
        else:
            out[j] = 0.0
    return out


def forward_points_and_frames(
    joint_angles: Sequence[float],
    joint_axes: Sequence[str],
    lengths_m: Sequence[float],
    head_translation: Sequence[float],
    head_rpy: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    n_joints = len(joint_axes)
    if len(joint_angles) != n_joints:
        raise ValueError("joint_angles length must match joint_axes")
    if len(lengths_m) != n_joints + 1:
        raise ValueError("lengths_m length must be n_joints + 1")

    points = np.zeros((n_joints + 2, 3), dtype=np.float64)
    frames = np.zeros((n_joints + 1, 3, 3), dtype=np.float64)

    pos = np.asarray(head_translation, dtype=np.float64).copy()
    rot = _rpy_to_mat3(float(head_rpy[0]), float(head_rpy[1]), float(head_rpy[2]))
    points[0] = pos

    for i in range(n_joints):
        pos = pos + rot @ np.array((float(lengths_m[i]), 0.0, 0.0), dtype=np.float64)
        points[i + 1] = pos
        joint_rot = _rot_axis(joint_axes[i], float(joint_angles[i]))
        rot = rot @ joint_rot
        frames[i] = rot

    pos = pos + rot @ np.array((float(lengths_m[n_joints]), 0.0, 0.0), dtype=np.float64)
    points[n_joints + 1] = pos
    frames[n_joints] = rot

    return points, frames


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
            _integrate_avg_g(g_fn, max(0.0, t), float(lo), float(hi), cfg.integration_samples)
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

    seg_mid_s = _segment_midpoint_s(lengths_m)

    def residual(v: np.ndarray) -> np.ndarray:
        yz_vars = v[: len(yz_joint_indices_0b)]
        head = v[len(yz_joint_indices_0b):]
        head_translation = head[:3]
        head_rpy = head[3:]

        joint_angles = _assemble_joint_angles(
            x_joint_indices_0b,
            yz_joint_indices_0b,
            joint_signs,
            state.twist_filtered,
            yz_vars,
            n_joints,
        )
        points, _frames = forward_points_and_frames(
            joint_angles=joint_angles,
            joint_axes=joint_axes,
            lengths_m=lengths_m,
            head_translation=head_translation,
            head_rpy=head_rpy,
        )
        mids = 0.5 * (points[:-1] + points[1:])

        tgt = np.array(
            [_target_point(f_fn, lengths_m, t, float(s), cfg.startup_ramp_sec) for s in seg_mid_s],
            dtype=np.float64,
        )
        return (mids - tgt).reshape(-1)

    v = state.yz_and_head.copy()
    for _ in range(cfg.max_iter):
        r0 = residual(v)
        nvar = v.size
        jac = np.zeros((r0.size, nvar), dtype=np.float64)
        eps = max(cfg.finite_diff_eps, 1e-8)

        # forward finite difference (one residual eval per var) for speed
        for i in range(nvar):
            vp = v.copy()
            vp[i] += eps
            rp = residual(vp)
            jac[:, i] = (rp - r0) / eps

        lhs = jac.T @ jac + cfg.damping * np.eye(nvar, dtype=np.float64)
        rhs = -(jac.T @ r0)
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
    joint_angles = _assemble_joint_angles(
        x_joint_indices_0b,
        yz_joint_indices_0b,
        joint_signs,
        state.twist_filtered,
        yz_vars,
        n_joints,
    )
    points, frames = forward_points_and_frames(
        joint_angles=joint_angles,
        joint_axes=joint_axes,
        lengths_m=lengths_m,
        head_translation=head[:3],
        head_rpy=head[3:],
    )
    return joint_angles, points, frames, head


def create_initial_state(num_yz_joints: int, num_x_joints: int) -> FitState:
    return FitState(
        yz_and_head=np.zeros(num_yz_joints + 6, dtype=np.float64),
        twist_filtered=np.zeros(num_x_joints, dtype=np.float64),
        last_t=None,
    )
