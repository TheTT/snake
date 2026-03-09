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
