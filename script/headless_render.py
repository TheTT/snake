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

    return cfg


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


def render_headless(cfg: Dict[str, Any]) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")

    xml_path = _resolve_path(SCRIPT_DIR, str(cfg["xml"]))
    args = parse_args()
    output_name = Path(str(args.output)).name
    output_path = RES_DIR / output_name

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = float(cfg["sim_timestep"])

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

    with mujoco.Renderer(model, width=width, height=height) as renderer, imageio.get_writer(str(output_path), fps=output_fps) as writer:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        written_frames = 0

        for step in range(total_steps):
            if step >= settle_steps:
                t = (step - settle_steps) * model.opt.timestep
                _apply_time_only_controls(model, data, table, t)

            mujoco.mj_step(model, data)

            if step >= settle_steps and (step - settle_steps) % render_every == 0:
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
