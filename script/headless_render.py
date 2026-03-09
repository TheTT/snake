"""Headless MuJoCo renderer for snake.xml.

Control is isolated in `joint_functions.py`.
All runtime parameters are loaded from script/headless.json.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import imageio
import mujoco
import numpy as np

from headless_camera import (
    apply_global_fov_scale,
    camera_direction_to_az_el,
    compute_panel_fovy_for_equal_long_side,
    orthogonal_basis,
    principal_axis_and_com,
    set_model_fovy,
    tracking_camera,
    vfov_to_hfov,
)
from headless_common import RESOLUTIONS, RES_DIR, SCRIPT_DIR, print_progress, resolve_path
from headless_compose import compose_fsm_quad
from headless_config import load_config
from headless_control import (
    actuator_names,
    apply_free_space_mode,
    apply_time_only_controls,
    build_actuator_function_table,
    clear_external_forces,
)
from joint_functions import JOINT_FUNCTIONS


def _fsm_joint_ids_in_order(model: mujoco.MjModel) -> List[int]:
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


def _hide_scene_model_geoms(scene: mujoco.MjvScene) -> None:
    """Hide current scene geoms by setting alpha=0 for this view only."""
    for i in range(scene.ngeom):
        scene.geoms[i].rgba[3] = 0.0


def _append_joint_polyline(
    scene: mujoco.MjvScene,
    points: Sequence[np.ndarray],
    rgba: np.ndarray,
    radius: float = 0.008,
) -> None:
    """Append capsule segments connecting consecutive 3D points."""
    if len(points) < 2:
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
            rgba,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            radius,
            p0[0],
            p0[1],
            p0[2],
            p1[0],
            p1[1],
            p1[2],
        )
        scene.ngeom += 1


def parse_args() -> argparse.Namespace:
    """Parse runtime args not stored in config file."""
    parser = argparse.ArgumentParser(description="Headless render for snake.xml")
    parser.add_argument(
        "--output",
        default="snake_headless.mp4",
        help="Output filename only (saved under ../res/)",
    )
    parser.add_argument(
        "-s",
        "--fsm",
        action="store_true",
        help="Enable free-space mode (no gravity/contact/external forces)",
    )
    return parser.parse_args()


def render_headless(cfg: Dict[str, Any]) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")

    xml_path = resolve_path(SCRIPT_DIR, str(cfg["xml"]))
    args = parse_args()
    output_name = Path(str(args.output)).name
    output_path = RES_DIR / output_name

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = float(cfg["sim_timestep"])
    free_space_mode = bool(args.fsm)
    view_fov_scale = float(cfg["view_fov_scale"])
    base_fovy, scaled_fovy = apply_global_fov_scale(model, view_fov_scale)
    apply_free_space_mode(model, free_space_mode)

    # Start from zero generalized velocity so initial linear/angular momentum is zero.
    data.qvel[:] = 0.0
    clear_external_forces(data)
    mujoco.mj_forward(model, data)

    width, height = RESOLUTIONS[str(cfg["resolution"])]
    settle_steps = int(float(cfg["settle"]) / model.opt.timestep)
    motion_steps = int(float(cfg["duration"]) / model.opt.timestep)
    total_steps = settle_steps + motion_steps
    base_fps = int(cfg["fps"])
    speed = float(cfg["video_speed"])
    fsm_split_x_ratio = float(cfg["fsm_split_x_ratio"])
    fsm_split_y_ratio = float(cfg["fsm_split_y_ratio"])
    # Keep output fps fixed; increase simulation interval per frame to speed up video.
    render_every = max(1, int(round(speed / (base_fps * model.opt.timestep))))
    total_frames = 0 if motion_steps <= 0 else ((motion_steps - 1) // render_every) + 1
    output_fps = base_fps

    table = build_actuator_function_table(model, JOINT_FUNCTIONS)
    names = actuator_names(model)

    mapped = [names[i] for i in sorted(table.keys())]
    unmapped = [name for name in names if name not in JOINT_FUNCTIONS]

    print(f"Model loaded: {xml_path}")
    print(f"timestep={model.opt.timestep:.6f}s, nu={model.nu}, nbody={model.nbody}")
    print(f"free_space_mode={free_space_mode}")
    print(f"view_fov_scale={view_fov_scale}, fovy={base_fovy:.2f}->{scaled_fovy:.2f}")
    if free_space_mode:
        print(f"fsm_split_x_ratio={fsm_split_x_ratio}, fsm_split_y_ratio={fsm_split_y_ratio}")
    print(f"video_fps={output_fps} (fixed), speed={speed}, render_every={render_every}")
    print(f"Mapped actuators ({len(mapped)}): {mapped}")
    if unmapped:
        print(f"Unmapped actuators will be zeroed ({len(unmapped)}): {unmapped}")

    if total_frames > 0:
        print_progress(0, total_frames)

    cam = tracking_camera(model)
    cam.distance = float(cfg["camera_distance"])
    cam.azimuth = float(cfg["camera_azimuth"])
    cam.elevation = float(cfg["camera_elevation"])

    view_width = width
    view_height = height

    split_x = int(round(width * fsm_split_x_ratio))
    split_y = int(round(height * fsm_split_y_ratio))
    split_x = max(1, min(width - 1, split_x))
    split_y = max(1, min(height - 1, split_y))
    w_left = split_x
    w_right = width - split_x
    h_top = split_y
    h_bottom = height - split_y

    full_aspect = float(width) / float(max(height, 1))
    if width >= height:
        target_long_side_fov = vfov_to_hfov(scaled_fovy, full_aspect)
    else:
        target_long_side_fov = scaled_fovy

    fovy_main = compute_panel_fovy_for_equal_long_side(w_left, h_top, target_long_side_fov)
    fovy_left = compute_panel_fovy_for_equal_long_side(w_right, h_top, target_long_side_fov)
    fovy_top = compute_panel_fovy_for_equal_long_side(w_left, h_bottom, target_long_side_fov)
    fovy_persp = compute_panel_fovy_for_equal_long_side(w_right, h_bottom, target_long_side_fov)

    if free_space_mode:
        print(
            "fsm_panel_sizes="
            f"TL({w_left}x{h_top}) TR({w_right}x{h_top}) "
            f"BL({w_left}x{h_bottom}) BR({w_right}x{h_bottom})"
        )
        print(
            "fsm_panel_fovy="
            f"TL({fovy_main:.2f}) TR({fovy_left:.2f}) "
            f"BL({fovy_top:.2f}) BR({fovy_persp:.2f})"
        )
        print(f"fsm_target_long_side_fov={target_long_side_fov:.2f}")

    persp_cam = tracking_camera(model)
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

    if free_space_mode:
        fsm_joint_ids = _fsm_joint_ids_in_order(model)
        poly_rgba = np.array([0.1, 1.0, 0.1, 1.0], dtype=np.float32)

        with (
            mujoco.Renderer(model, width=w_left, height=h_top) as renderer_main,
            mujoco.Renderer(model, width=w_right, height=h_top) as renderer_left,
            mujoco.Renderer(model, width=w_left, height=h_bottom) as renderer_top,
            mujoco.Renderer(model, width=w_right, height=h_bottom) as renderer_persp,
            imageio.get_writer(str(output_path), fps=output_fps) as writer,
        ):
            for r in (renderer_main, renderer_left, renderer_top, renderer_persp):
                r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1

            written_frames = 0

            for step in range(total_steps):
                clear_external_forces(data)

                if step >= settle_steps:
                    t = (step - settle_steps) * model.opt.timestep
                    apply_time_only_controls(model, data, table, t)

                mujoco.mj_step(model, data)

                if step >= settle_steps and (step - settle_steps) % render_every == 0:
                    poly_points = [data.xanchor[jid].copy() for jid in fsm_joint_ids]

                    com, principal_axis = principal_axis_and_com(model, data, prev_axis)
                    prev_axis = principal_axis.copy()
                    front, left, top = orthogonal_basis(principal_axis)

                    for c in (dyn_main_cam, dyn_left_cam, dyn_top_cam):
                        c.lookat[:] = com

                    dyn_main_cam.azimuth, dyn_main_cam.elevation = camera_direction_to_az_el(front)
                    dyn_left_cam.azimuth, dyn_left_cam.elevation = camera_direction_to_az_el(left)
                    dyn_top_cam.azimuth, dyn_top_cam.elevation = camera_direction_to_az_el(top)

                    set_model_fovy(model, fovy_main)
                    renderer_main.update_scene(data, camera=dyn_main_cam)
                    _append_joint_polyline(renderer_main.scene, poly_points, poly_rgba)
                    frame_main = renderer_main.render()

                    set_model_fovy(model, fovy_left)
                    renderer_left.update_scene(data, camera=dyn_left_cam)
                    _append_joint_polyline(renderer_left.scene, poly_points, poly_rgba)
                    frame_left = renderer_left.render()

                    set_model_fovy(model, fovy_top)
                    renderer_top.update_scene(data, camera=dyn_top_cam)
                    _append_joint_polyline(renderer_top.scene, poly_points, poly_rgba)
                    frame_top = renderer_top.render()

                    set_model_fovy(model, fovy_persp)
                    renderer_persp.update_scene(data, camera=persp_cam)
                    _hide_scene_model_geoms(renderer_persp.scene)
                    _append_joint_polyline(renderer_persp.scene, poly_points, poly_rgba)
                    frame_persp = renderer_persp.render()

                    frame = compose_fsm_quad(
                        frame_main,
                        frame_left,
                        frame_top,
                        frame_persp,
                        height,
                        width,
                        fsm_split_x_ratio,
                        fsm_split_y_ratio,
                    )

                    writer.append_data(frame)
                    written_frames += 1
                    print_progress(written_frames, total_frames)
    else:
        with mujoco.Renderer(model, width=view_width, height=view_height) as renderer, imageio.get_writer(str(output_path), fps=output_fps) as writer:
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
            set_model_fovy(model, scaled_fovy)
            written_frames = 0

            for step in range(total_steps):
                if step >= settle_steps:
                    t = (step - settle_steps) * model.opt.timestep
                    apply_time_only_controls(model, data, table, t)

                mujoco.mj_step(model, data)

                if step >= settle_steps and (step - settle_steps) % render_every == 0:
                    renderer.update_scene(data, camera=cam)
                    frame = renderer.render()

                    writer.append_data(frame)
                    written_frames += 1
                    print_progress(written_frames, total_frames)

    if total_frames > 0:
        print(file=sys.stdout, flush=True)

    print(f"Video saved: {output_path}")


if __name__ == "__main__":
    render_headless(load_config())
