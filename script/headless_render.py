"""Headless MuJoCo renderer for drive/snake.xml.

Control is isolated in `joint_functions.py`.
Each actuator is driven by a user-editable function f(t) that receives only time.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable, Dict, List

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless render for snake.xml using time-only joint functions")
    default_xml = Path(__file__).resolve().parents[1] / "snake.xml"

    parser.add_argument("--xml", type=Path, default=default_xml, help="Path to MuJoCo XML model")
    parser.add_argument("--output", type=Path, default=Path("snake_headless.mp4"), help="Output video path")
    parser.add_argument("--duration", type=float, default=20.0, help="Motion duration in seconds")
    parser.add_argument("--settle", type=float, default=2.0, help="Pre-roll settle time in seconds")
    parser.add_argument("--fps", type=int, default=60, help="Output video FPS")
    parser.add_argument("--resolution", choices=RESOLUTIONS.keys(), default="720p", help="Output video resolution")
    parser.add_argument("--camera-distance", type=float, default=1.6, help="Tracking camera distance")
    parser.add_argument("--camera-azimuth", type=float, default=180.0, help="Tracking camera azimuth")
    parser.add_argument("--camera-elevation", type=float, default=-25.0, help="Tracking camera elevation")
    return parser.parse_args()


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


def render_headless(args: argparse.Namespace) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)

    width, height = RESOLUTIONS[args.resolution]
    settle_steps = int(args.settle / model.opt.timestep)
    motion_steps = int(args.duration / model.opt.timestep)
    total_steps = settle_steps + motion_steps
    render_every = max(1, int(round(1.0 / (args.fps * model.opt.timestep))))

    table = _build_actuator_function_table(model)
    actuator_names = _actuator_names(model)

    mapped = [actuator_names[i] for i in sorted(table.keys())]
    unmapped = [name for name in actuator_names if name not in JOINT_FUNCTIONS]

    print(f"Model loaded: {args.xml}")
    print(f"timestep={model.opt.timestep:.6f}s, nu={model.nu}, nbody={model.nbody}")
    print(f"Mapped actuators ({len(mapped)}): {mapped}")
    if unmapped:
        print(f"Unmapped actuators will be zeroed ({len(unmapped)}): {unmapped}")

    cam = _tracking_camera(model)
    cam.distance = args.camera_distance
    cam.azimuth = args.camera_azimuth
    cam.elevation = args.camera_elevation

    with mujoco.Renderer(model, width=width, height=height) as renderer, imageio.get_writer(str(args.output), fps=args.fps) as writer:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1

        for step in range(total_steps):
            if step >= settle_steps:
                t = (step - settle_steps) * model.opt.timestep
                _apply_time_only_controls(model, data, table, t)

            mujoco.mj_step(model, data)

            if step >= settle_steps and (step - settle_steps) % render_every == 0:
                renderer.update_scene(data, camera=cam)
                frame = renderer.render()
                writer.append_data(frame)

    print(f"Video saved: {args.output}")


if __name__ == "__main__":
    render_headless(parse_args())
