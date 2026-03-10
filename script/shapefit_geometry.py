from __future__ import annotations

import math
from typing import Sequence

import numpy as np


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


def segment_midpoint_s(lengths_m: Sequence[float]) -> np.ndarray:
    s_nodes = cumulative_s(lengths_m)
    return 0.5 * (s_nodes[:-1] + s_nodes[1:])


def assemble_joint_angles(
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
    joint_angles: Sequence[float] | np.ndarray,
    joint_axes: Sequence[str],
    lengths_m: Sequence[float],
    head_translation: Sequence[float] | np.ndarray,
    head_rpy: Sequence[float] | np.ndarray,
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
