"""Joint-order and polyline helpers for headless rendering."""

from __future__ import annotations

from typing import List, Sequence

import mujoco
import numpy as np


def fsm_joint_ids_in_order(model: mujoco.MjModel) -> List[int]:
    """Return joint ids in joint_1_pos..joint_N_pos actuator order."""
    ids: List[int] = []
    i = 1
    while True:
        actuator_name = f"joint_{i}_pos"
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if act_id < 0:
            break

        joint_id = int(model.actuator_trnid[act_id, 0])
        if joint_id >= 0:
            ids.append(joint_id)
        i += 1

    return ids


def dim_scene_model_geoms(scene: mujoco.MjvScene, alpha: float) -> None:
    """Dim current scene geoms by setting their alpha for this view only."""
    a = float(alpha)
    for i in range(scene.ngeom):
        scene.geoms[i].rgba[3] = a


def append_joint_polyline(
    scene: mujoco.MjvScene,
    points: Sequence[np.ndarray],
    segment_colors: Sequence[np.ndarray],
    radius: float = 0.008,
) -> None:
    """Append capsule segments connecting consecutive 3D points."""
    if len(points) < 2:
        return
    if not segment_colors:
        return

    for i in range(len(points) - 1):
        if scene.ngeom >= scene.maxgeom:
            break

        p0 = np.asarray(points[i], dtype=np.float64)
        p1 = np.asarray(points[i + 1], dtype=np.float64)
        dist = float(np.linalg.norm(p1 - p0))
        if dist < 1e-9:
            continue

        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).ravel(),
            segment_colors[i % len(segment_colors)],
        )

        p0_arr = np.asarray(p0, dtype=np.float64).reshape(3)
        p1_arr = np.asarray(p1, dtype=np.float64).reshape(3)

        try:
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                float(radius),
                p0_arr,
                p1_arr,
            )
        except TypeError as exc:
            raise TypeError(
                "mjv_connector failed in append_joint_polyline with "
                f"radius={float(radius)}, p0={p0_arr.tolist()}, p1={p1_arr.tolist()}"
            ) from exc
        scene.ngeom += 1


def append_joint_axis_markers(
    scene: mujoco.MjvScene,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_ids: Sequence[int],
    *,
    allowed_axis_indices: Sequence[int] = (1, 2),
    length_m: float = 0.12,
    radius: float = 0.002,
    rgba: np.ndarray | None = None,
) -> None:
    """Append capsule markers centered at each selected joint anchor along joint axis.

    Axis selection is done in joint-local coordinates via model.jnt_axis.
    """
    if rgba is None:
        rgba = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64)

    half_len = 0.5 * float(length_m)
    allowed = set(int(v) for v in allowed_axis_indices)

    for jid in joint_ids:
        if scene.ngeom >= scene.maxgeom:
            break
        if jid < 0 or jid >= model.njnt:
            continue

        axis_local = np.asarray(model.jnt_axis[jid], dtype=np.float64)
        axis_local_norm = float(np.linalg.norm(axis_local))
        if axis_local_norm < 1e-9:
            continue
        dominant_axis = int(np.argmax(np.abs(axis_local)))
        if dominant_axis not in allowed:
            continue

        anchor = np.asarray(data.xanchor[jid], dtype=np.float64)
        axis_world = np.asarray(data.xaxis[jid], dtype=np.float64)
        axis_world_norm = float(np.linalg.norm(axis_world))
        if axis_world_norm < 1e-9:
            continue
        axis_world = axis_world / axis_world_norm

        p0 = anchor - half_len * axis_world
        p1 = anchor + half_len * axis_world

        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).ravel(),
            rgba,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            float(radius),
            p0.reshape(3),
            p1.reshape(3),
        )
        scene.ngeom += 1


def _rotate_vector_about_axis(vec: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate vector around unit axis using Rodrigues' formula."""
    k = _safe_unit(np.asarray(axis, dtype=np.float64))
    v = np.asarray(vec, dtype=np.float64)
    ca = float(np.cos(angle_rad))
    sa = float(np.sin(angle_rad))
    return v * ca + np.cross(k, v) * sa + k * float(np.dot(k, v)) * (1.0 - ca)


def _joint_body_axes_world(model: mujoco.MjModel, data: mujoco.MjData, joint_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x,y,z) body axes in world frame for the body owning this joint."""
    body_id = int(model.jnt_bodyid[joint_id])
    if body_id < 0 or body_id >= model.nbody:
        x_axis = _safe_unit(np.asarray(data.xaxis[joint_id], dtype=np.float64))
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(x_axis, ref))) > 0.95:
            ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        y_axis = _safe_unit(np.cross(x_axis, ref))
        z_axis = _safe_unit(np.cross(x_axis, y_axis))
        return x_axis, y_axis, z_axis

    rot = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    x_from_col = _safe_unit(rot[:, 0])
    x_from_row = _safe_unit(rot[0, :])
    joint_x = _safe_unit(np.asarray(data.xaxis[joint_id], dtype=np.float64))

    if float(np.dot(x_from_col, joint_x)) >= float(np.dot(x_from_row, joint_x)):
        return x_from_col, _safe_unit(rot[:, 1]), _safe_unit(rot[:, 2])
    return x_from_row, _safe_unit(rot[1, :]), _safe_unit(rot[2, :])


def append_twist_axis_markers(
    scene: mujoco.MjvScene,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_ids: Sequence[int],
    *,
    twist_length_m: float = 0.1,
    twist_radius: float = 0.003,
    twist_base_angle_rad: float = np.pi / 2.0,
    twist_rgba: np.ndarray | None = None,
) -> None:
    """Append two green twist markers (local y/z) for x-axis joints.

    Each marker starts at the joint midpoint (xanchor) and points along local y or z.
    The local frame is compensated by -twist_base_angle_rad around local x to cancel
    the initial fixed x-joint twist offset.
    """
    if twist_rgba is None:
        twist_rgba = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)

    twist_length = float(twist_length_m)

    for joint_id in joint_ids:
        if scene.ngeom >= scene.maxgeom:
            break
        if joint_id < 0 or joint_id >= model.njnt:
            continue

        axis_local = np.asarray(model.jnt_axis[joint_id], dtype=np.float64)
        if float(np.linalg.norm(axis_local)) < 1e-9:
            continue
        if int(np.argmax(np.abs(axis_local))) != 0:
            continue

        x_world, y_world, z_world = _joint_body_axes_world(model, data, joint_id)
        y_twist = _safe_unit(_rotate_vector_about_axis(y_world, x_world, -twist_base_angle_rad))
        z_twist = _safe_unit(_rotate_vector_about_axis(z_world, x_world, -twist_base_angle_rad))
        anchor = np.asarray(data.xanchor[joint_id], dtype=np.float64)

        for direction in (y_twist, z_twist):
            if scene.ngeom >= scene.maxgeom:
                break

            p0 = anchor
            p1 = anchor + twist_length * direction
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.eye(3, dtype=np.float64).ravel(),
                twist_rgba,
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                float(twist_radius),
                p0.reshape(3),
                p1.reshape(3),
            )
            scene.ngeom += 1


def _safe_unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return vec / norm


def build_section_polyline_points(
    joint_axis_points: Sequence[np.ndarray],
    segment_lengths: Sequence[float],
) -> List[np.ndarray]:
    """Build polyline points from manual segment lengths."""
    anchors = [np.asarray(p, dtype=np.float64).copy() for p in joint_axis_points]
    n_anchors = len(anchors)
    if n_anchors == 0:
        return []
    if len(segment_lengths) != n_anchors + 1:
        raise ValueError(
            f"segment_lengths must have length N+1 ({n_anchors + 1}), got {len(segment_lengths)}"
        )

    def dir_for(i: int) -> np.ndarray:
        if i == 0:
            vec = anchors[1] - anchors[0] if n_anchors > 1 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        elif i >= n_anchors:
            vec = (
                anchors[-1] - anchors[-2]
                if n_anchors > 1
                else np.array([1.0, 0.0, 0.0], dtype=np.float64)
            )
        else:
            vec = anchors[i] - anchors[i - 1]
        return _safe_unit(vec)

    points: List[np.ndarray] = []
    d0 = dir_for(0)
    p0 = anchors[0] - float(segment_lengths[0]) * d0
    points.append(p0)

    for i in range(0, n_anchors):
        di = dir_for(i)
        pi_prev = points[-1]
        pi = pi_prev + float(segment_lengths[i]) * di
        points.append(pi)

    d_n = dir_for(n_anchors)
    p_last = points[-1] + float(segment_lengths[n_anchors]) * d_n
    points.append(p_last)

    return points


def polyline_segment_lengths(points: Sequence[np.ndarray]) -> List[float]:
    """Return individual segment lengths for consecutive-point polyline."""
    if len(points) < 2:
        return []

    out: List[float] = []
    for i in range(len(points) - 1):
        p0 = np.asarray(points[i], dtype=np.float64)
        p1 = np.asarray(points[i + 1], dtype=np.float64)
        out.append(float(np.linalg.norm(p1 - p0)))
    return out
