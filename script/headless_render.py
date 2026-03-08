"""Headless MuJoCo renderer for snake.xml.

Control is isolated in `joint_functions.py`.
All runtime parameters are loaded from script/headless.json.
"""

from __future__ import annotations

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List

import imageio
import mujoco
import numpy as np

from joint_functions import JOINT_FUNCTIONS


RESOLUTIONS = {
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2160p": (3840, 2160),
}

SCRIPT_DIR = Path(__file__).resolve().parent
HEADLESS_CONFIG_PATH = SCRIPT_DIR / "headless.json"
RES_DIR = (SCRIPT_DIR / "../res").resolve()


def _print_progress(current: int, total: int, *, width: int = 32) -> None:
    """Render an in-place terminal progress bar."""
    if total <= 0:
        return

    current = max(0, min(current, total))
    ratio = current / total
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    msg = f"\rRendering video: [{bar}] {current}/{total} ({ratio * 100:5.1f}%)"
    print(msg, end="", file=sys.stdout, flush=True)


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path)


def parse_args() -> argparse.Namespace:
    """Parse runtime args not stored in config file."""
    parser = argparse.ArgumentParser(description="Headless render for snake.xml")
    parser.add_argument(
        "--output",
        default="snake_headless.mp4",
        help="Output filename only (saved under ../res/)",
    )
    return parser.parse_args()


def _load_config() -> Dict[str, Any]:
    """Load headless render config from script/headless.json."""
    with HEADLESS_CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = json.load(f)

    required = [
        "xml",
        "duration",
        "settle",
        "fps",
        "resolution",
        "camera_distance",
        "camera_azimuth",
        "camera_elevation",
        "sim_timestep",
        "video_speed",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing keys in {HEADLESS_CONFIG_PATH}: {missing}")

    if cfg["resolution"] not in RESOLUTIONS:
        raise ValueError(f"Invalid resolution: {cfg['resolution']}")

    if float(cfg["sim_timestep"]) <= 0.0:
        raise ValueError("sim_timestep must be > 0")
    if float(cfg["video_speed"]) <= 0.0:
        raise ValueError("video_speed must be > 0")
    if int(cfg["fps"]) <= 0:
        raise ValueError("fps must be > 0")

    # Optional switch: run the snake in free space with no gravity/contact/external forces.
    cfg.setdefault("free_space_mode", False)

    return cfg


def _apply_free_space_mode(model: mujoco.MjModel, enabled: bool) -> None:
    """Switch environment to a free-space approximation with no external field/contact."""
    if not enabled:
        return

    model.opt.gravity[:] = 0.0
    model.opt.wind[:] = 0.0
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    # Hide and disable the floor geom (visual + contact) when in free-space mode.
    try:
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, b"floor")
    except Exception:
        # mj_name2id may raise if name not found; try with str name for older bindings.
        try:
            floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        except Exception:
            floor_id = -1

    if floor_id is not None and floor_id >= 0 and floor_id < model.ngeom:
        # Make invisible by setting alpha to 0 and disable contact
        try:
            model.geom_rgba[floor_id, 3] = 0.0
        except Exception:
            pass
        try:
            model.geom_contype[floor_id] = 0
            model.geom_conaffinity[floor_id] = 0
        except Exception:
            pass


def _clear_external_forces(data: mujoco.MjData) -> None:
    """Ensure no user-applied external wrench/force leaks into dynamics."""
    data.xfrc_applied[:] = 0.0
    data.qfrc_applied[:] = 0.0


def _actuator_names(model: mujoco.MjModel) -> List[str]:
    """Return actuator names via stable MuJoCo API across versions."""
    names: List[str] = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        names.append(name if name is not None else f"actuator_{i}")
    return names


def _build_actuator_function_table(model: mujoco.MjModel) -> Dict[int, Callable[[float], float]]:
    """Create mapping actuator index -> function(t) using actuator names."""
    actuator_names = _actuator_names(model)
    table: Dict[int, Callable[[float], float]] = {}

    for i, name in enumerate(actuator_names):
        if name in JOINT_FUNCTIONS:
            table[i] = JOINT_FUNCTIONS[name]

    return table


def _apply_time_only_controls(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    table: Dict[int, Callable[[float], float]],
    t: float,
) -> None:
    """Apply control at time t, clipping by actuator ctrl range if limited."""
    for i in range(model.nu):
        func = table.get(i)
        if func is None:
            data.ctrl[i] = 0.0
            continue

        u = float(func(t))

        # Respect ctrlrange for limited actuators.
        if model.actuator_ctrllimited[i]:
            lo, hi = model.actuator_ctrlrange[i]
            u = float(np.clip(u, lo, hi))

        data.ctrl[i] = u


def _tracking_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    """Track the middle body so the snake remains centered in frame."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    body_count = model.nbody
    mid_body_id = max(1, body_count // 2)

    cam.trackbodyid = mid_body_id
    return cam


def _safe_normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _camera_direction_to_az_el(direction: np.ndarray) -> tuple[float, float]:
    """Convert a world direction vector to MuJoCo camera azimuth/elevation."""
    d = _safe_normalize(direction)
    azimuth = float(np.degrees(np.arctan2(d[1], d[0])))
    elevation = float(np.degrees(np.arcsin(np.clip(d[2], -1.0, 1.0))))
    return azimuth, elevation


def _principal_axis_and_com(
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
    axis = _safe_normalize(axis)

    if np.linalg.norm(axis) < 1e-9:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # Keep sign continuity so the main view does not flip 180 degrees frame-to-frame.
    if prev_axis is not None and float(np.dot(axis, prev_axis)) < 0.0:
        axis = -axis

    return center, axis


def _orthogonal_basis(front: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build right-handed orthonormal basis (front, left, top) from front axis."""
    front_n = _safe_normalize(front)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(front_n, ref))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    left = _safe_normalize(np.cross(ref, front_n))
    top = _safe_normalize(np.cross(front_n, left))
    return front_n, left, top


def _set_projection_mode(renderer: mujoco.Renderer, *, orthographic: bool) -> None:
    """Best-effort toggle for projection type across MuJoCo python versions."""
    if not orthographic:
        return

    scene = renderer.scene
    cameras = getattr(scene, "camera", None)
    if cameras is None:
        return

    # Newer bindings expose two GL cameras (stereo). Set both if available.
    for cam in cameras:
        if hasattr(cam, "orthographic"):
            cam.orthographic = 1


def render_headless(cfg: Dict[str, Any]) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")

    xml_path = _resolve_path(SCRIPT_DIR, str(cfg["xml"]))
    args = parse_args()
    output_name = Path(str(args.output)).name
    output_path = RES_DIR / output_name

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = float(cfg["sim_timestep"])
    free_space_mode = bool(cfg.get("free_space_mode", False))
    _apply_free_space_mode(model, free_space_mode)

    # Start from zero generalized velocity so initial linear/angular momentum is zero.
    data.qvel[:] = 0.0
    _clear_external_forces(data)
    mujoco.mj_forward(model, data)

    width, height = RESOLUTIONS[str(cfg["resolution"])]
    settle_steps = int(float(cfg["settle"]) / model.opt.timestep)
    motion_steps = int(float(cfg["duration"]) / model.opt.timestep)
    total_steps = settle_steps + motion_steps
    base_fps = int(cfg["fps"])
    speed = float(cfg["video_speed"])
    # Keep output fps fixed; increase simulation interval per frame to speed up video.
    render_every = max(1, int(round(speed / (base_fps * model.opt.timestep))))
    total_frames = 0 if motion_steps <= 0 else ((motion_steps - 1) // render_every) + 1
    output_fps = base_fps

    table = _build_actuator_function_table(model)
    actuator_names = _actuator_names(model)

    mapped = [actuator_names[i] for i in sorted(table.keys())]
    unmapped = [name for name in actuator_names if name not in JOINT_FUNCTIONS]

    print(f"Model loaded: {xml_path}")
    print(f"timestep={model.opt.timestep:.6f}s, nu={model.nu}, nbody={model.nbody}")
    print(f"free_space_mode={free_space_mode}")
    print(f"video_fps={output_fps} (fixed), speed={speed}, render_every={render_every}")
    print(f"Mapped actuators ({len(mapped)}): {mapped}")
    if unmapped:
        print(f"Unmapped actuators will be zeroed ({len(unmapped)}): {unmapped}")

    if total_frames > 0:
        _print_progress(0, total_frames)

    cam = _tracking_camera(model)
    cam.distance = float(cfg["camera_distance"])
    cam.azimuth = float(cfg["camera_azimuth"])
    cam.elevation = float(cfg["camera_elevation"])

    view_width = width
    view_height = height
    if free_space_mode:
        view_width = width // 2
        view_height = height // 2

    persp_cam = _tracking_camera(model)
    persp_cam.distance = float(cfg["camera_distance"])
    persp_cam.azimuth = float(cfg["camera_azimuth"])
    persp_cam.elevation = float(cfg["camera_elevation"])

    dyn_main_cam = mujoco.MjvCamera()
    dyn_left_cam = mujoco.MjvCamera()
    dyn_top_cam = mujoco.MjvCamera()
    for c in (dyn_main_cam, dyn_left_cam, dyn_top_cam):
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.fixedcamid = -1
        c.trackbodyid = -1
        c.distance = float(cfg["camera_distance"])

    prev_axis: np.ndarray | None = None

    with mujoco.Renderer(model, width=view_width, height=view_height) as renderer, imageio.get_writer(str(output_path), fps=output_fps) as writer:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        written_frames = 0

        for step in range(total_steps):
            if free_space_mode:
                _clear_external_forces(data)

            if step >= settle_steps:
                t = (step - settle_steps) * model.opt.timestep
                _apply_time_only_controls(model, data, table, t)

            mujoco.mj_step(model, data)

            if step >= settle_steps and (step - settle_steps) % render_every == 0:
                if free_space_mode:
                    com, principal_axis = _principal_axis_and_com(model, data, prev_axis)
                    prev_axis = principal_axis.copy()
                    front, left, top = _orthogonal_basis(principal_axis)

                    for c in (dyn_main_cam, dyn_left_cam, dyn_top_cam):
                        c.lookat[:] = com

                    dyn_main_cam.azimuth, dyn_main_cam.elevation = _camera_direction_to_az_el(front)
                    dyn_left_cam.azimuth, dyn_left_cam.elevation = _camera_direction_to_az_el(left)
                    dyn_top_cam.azimuth, dyn_top_cam.elevation = _camera_direction_to_az_el(top)

                    renderer.update_scene(data, camera=dyn_main_cam)
                    _set_projection_mode(renderer, orthographic=True)
                    frame_main = renderer.render()

                    renderer.update_scene(data, camera=dyn_left_cam)
                    _set_projection_mode(renderer, orthographic=True)
                    frame_left = renderer.render()

                    renderer.update_scene(data, camera=dyn_top_cam)
                    _set_projection_mode(renderer, orthographic=True)
                    frame_top = renderer.render()

                    renderer.update_scene(data, camera=persp_cam)
                    frame_persp = renderer.render()

                    top_row = np.concatenate([frame_main, frame_left], axis=1)
                    bottom_row = np.concatenate([frame_top, frame_persp], axis=1)
                    frame = np.concatenate([top_row, bottom_row], axis=0)
                else:
                    renderer.update_scene(data, camera=cam)
                    frame = renderer.render()

                writer.append_data(frame)
                written_frames += 1
                _print_progress(written_frames, total_frames)

    if total_frames > 0:
        print(file=sys.stdout, flush=True)

    print(f"Video saved: {output_path}")


if __name__ == "__main__":
    render_headless(_load_config())
