"""Camera math helpers for headless rendering."""

from __future__ import annotations

import numpy as np
import mujoco


def tracking_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    """Track the middle body so the snake remains centered in frame."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    body_count = model.nbody
    mid_body_id = max(1, body_count // 2)
    cam.trackbodyid = mid_body_id
    return cam


def safe_normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def camera_direction_to_az_el(direction: np.ndarray) -> tuple[float, float]:
    """Convert a world direction vector to MuJoCo camera azimuth/elevation."""
    d = safe_normalize(direction)
    azimuth = float(np.degrees(np.arctan2(d[1], d[0])))
    elevation = float(np.degrees(np.arcsin(np.clip(d[2], -1.0, 1.0))))
    return azimuth, elevation


def principal_axis_and_com(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    prev_axis: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate whole-body center of mass and dominant body-distribution axis."""
    masses = np.asarray(model.body_mass, dtype=np.float64)
    com_positions = np.asarray(data.xipos, dtype=np.float64)

    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        center = np.mean(com_positions, axis=0)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return center, axis

    center = np.sum(com_positions * masses[:, None], axis=0) / total_mass
    rel = com_positions - center
    weighted = rel * np.sqrt(masses[:, None])
    cov = weighted.T @ weighted

    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    axis = safe_normalize(axis)

    if np.linalg.norm(axis) < 1e-9:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    if prev_axis is not None and float(np.dot(axis, prev_axis)) < 0.0:
        axis = -axis

    return center, axis


def orthogonal_basis(front: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build right-handed orthonormal basis (front, left, top) from front axis."""
    front_n = safe_normalize(front)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(front_n, ref))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    left = safe_normalize(np.cross(ref, front_n))
    top = safe_normalize(np.cross(front_n, left))
    return front_n, left, top


def apply_global_fov_scale(model: mujoco.MjModel, fov_scale: float) -> tuple[float, float]:
    """Scale MuJoCo global perspective FOV once and return (base, scaled)."""
    base_fovy = float(model.vis.global_.fovy)
    scaled_fovy = float(np.clip(base_fovy * float(fov_scale), 5.0, 170.0))
    model.vis.global_.fovy = scaled_fovy
    return base_fovy, scaled_fovy


def vfov_to_hfov(vfov_deg: float, aspect_w_over_h: float) -> float:
    """Convert vertical FOV (deg) to horizontal FOV (deg)."""
    a = max(float(aspect_w_over_h), 1e-9)
    v = np.radians(float(vfov_deg))
    h = 2.0 * np.arctan(np.tan(v * 0.5) * a)
    return float(np.degrees(h))


def hfov_to_vfov(hfov_deg: float, aspect_w_over_h: float) -> float:
    """Convert horizontal FOV (deg) to vertical FOV (deg)."""
    a = max(float(aspect_w_over_h), 1e-9)
    h = np.radians(float(hfov_deg))
    v = 2.0 * np.arctan(np.tan(h * 0.5) / a)
    return float(np.degrees(v))


def compute_panel_fovy_for_equal_long_side(panel_w: int, panel_h: int, target_long_side_fov_deg: float) -> float:
    """Compute panel vertical FOV so each panel has the same long-side FOV."""
    w = max(int(panel_w), 1)
    h = max(int(panel_h), 1)
    aspect = float(w) / float(h)

    if w >= h:
        vfov = hfov_to_vfov(target_long_side_fov_deg, aspect)
    else:
        vfov = float(target_long_side_fov_deg)

    return float(np.clip(vfov, 5.0, 170.0))


def set_model_fovy(model: mujoco.MjModel, fovy: float) -> None:
    """Set model global perspective FOV (degrees)."""
    model.vis.global_.fovy = float(np.clip(fovy, 5.0, 170.0))
