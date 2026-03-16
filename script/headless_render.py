"""Headless MuJoCo renderer for snake.xml.

Control is isolated in `joint_functions.py`.
All runtime parameters are loaded from script/headless.json.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Must be set before importing mujoco so the GL backend is selected correctly.
# This script is headless-oriented: never use GLFW by default.
# - if MUJOCO_GL is empty or glfw -> force egl
# - if user explicitly sets osmesa/egl -> respect it
# _mujoco_gl = os.environ.get("MUJOCO_GL", "").strip().lower()
# if _mujoco_gl in ("", "glfw"):
#     os.environ["MUJOCO_GL"] = "egl"

# _selected_gl = os.environ.get("MUJOCO_GL", "egl").strip().lower() or "egl"
# if _selected_gl in ("egl", "osmesa"):
#     os.environ["PYOPENGL_PLATFORM"] = _selected_gl
# else:
#     os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np

from headless_camera import (
    apply_global_fov_scale,
    compute_panel_fovy_for_equal_long_side,
    tracking_camera,
    vfov_to_hfov,
)
from headless_common import RESOLUTIONS, RES_DIR, SCRIPT_DIR, print_progress, resolve_path
from headless_config import load_config
from headless_control import (
    actuator_names,
    apply_free_space_mode,
    build_actuator_function_table,
    clear_external_forces,
    pin_base_free_joint,
)
from headless_joint_utils import (
    fsm_joint_ids_in_order,
    build_section_polyline_points,
    polyline_segment_lengths,
)
from headless_plot import plot_joint_angles_mod3
from headless_render_passes import run_fk_render_loop, run_free_space_render_loop, run_standard_render_loop
from gait_function import f as midline_f
from joint_functions import JOINT_FUNCTIONS, POLYLINE_SEGMENT_LENGTHS_M, get_debug_info


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
    parser.add_argument(
        "--fk",
        action="store_true",
        help="Enable FK debug mode (2x2: MuJoCo, FK nodes, target nodes, blank)",
    )
    return parser.parse_args()


def render_headless(cfg: Dict[str, Any]) -> None:
    xml_path = resolve_path(SCRIPT_DIR, str(cfg["xml"]))
    args = parse_args()
    output_name = Path(str(args.output)).name
    output_path = RES_DIR / output_name

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = float(cfg["sim_timestep"])
    free_space_mode = bool(args.fsm)
    fk_mode = bool(args.fk)
    if free_space_mode and fk_mode:
        raise ValueError("--fsm and --fk cannot be enabled together")
    view_fov_scale = float(cfg["view_fov_scale"])
    base_fovy, scaled_fovy = apply_global_fov_scale(model, view_fov_scale)
    apply_free_space_mode(model, free_space_mode or fk_mode)

    # Start from zero generalized velocity so initial linear/angular momentum is zero.
    data.qvel[:] = 0.0
    clear_external_forces(data)
    if fk_mode:
        pin_base_free_joint(model, data)
    mujoco.mj_forward(model, data)

    # --- Sync simulation qpos to commanded initial angles (t=0) to avoid jump ---
    # For each actuator in JOINT_FUNCTIONS, find mapped joint qpos address and set
    # it to the command value at t=0. Then run forward kinematics again.
    for act_name, fn in JOINT_FUNCTIONS.items():
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        if act_id < 0:
            continue
        try:
            joint_id = int(model.actuator_trnid[act_id, 0])
        except Exception:
            continue
        if joint_id < 0:
            continue
        qpos_addr = int(model.jnt_qposadr[joint_id])
        try:
            desired = float(fn(0.0))
        except Exception:
            continue
        # Only set if qpos address is valid
        if 0 <= qpos_addr < model.nq:
            data.qpos[qpos_addr] = desired
    mujoco.mj_forward(model, data)

    width, height = RESOLUTIONS[str(cfg["resolution"])]
    settle_steps = int(float(cfg["settle"]) / model.opt.timestep)
    motion_steps = int(float(cfg["duration"]) / model.opt.timestep)
    total_steps = settle_steps + motion_steps
    base_fps = int(cfg["fps"])
    speed = float(cfg["video_speed"])
    fsm_split_x_ratio = float(cfg["fsm"]["split_x_ratio"])
    fsm_split_y_ratio = float(cfg["fsm"]["split_y_ratio"])
    fsm_show_f_curve_overlay = bool(cfg["fsm"]["show_f_curve_overlay"])
    fsm_show_local_axes = bool(cfg["fsm"]["show_local_axes"])
    fsm_show_shell_overlay = bool(cfg["fsm"]["show_shell_overlay"])
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
    print(f"fk_mode={fk_mode}")
    print(
        "gl_backend="
        f"{os.environ.get('MUJOCO_GL', '<unset>')}, "
        f"DISPLAY={os.environ.get('DISPLAY', '<unset>')}"
    )
    print(f"view_fov_scale={view_fov_scale}, fovy={base_fovy:.2f}->{scaled_fovy:.2f}")
    if free_space_mode:
        print(
            f"fsm_split_x_ratio={fsm_split_x_ratio}, fsm_split_y_ratio={fsm_split_y_ratio}"
        )
        print(
            f"fsm_show_f_curve_overlay={fsm_show_f_curve_overlay}, "
            f"fsm_show_local_axes={fsm_show_local_axes}, "
            f"fsm_show_shell_overlay={fsm_show_shell_overlay}"
        )
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

    segment_colors = [
    ]
    base_alpha = 0.5 if fsm_show_f_curve_overlay else 1.0
    segment_colors = [
        np.array([0.1, 0.3, 1.0, base_alpha], dtype=np.float32),  # blue
        np.array([1.0, 1.0, 1.0, base_alpha], dtype=np.float32),  # white
        np.array([1.0, 0.9, 0.1, base_alpha], dtype=np.float32),  # yellow
    ]

    # Record actual post-step joint qpos values in joint_1..joint_N order.
    plot_joint_ids = fsm_joint_ids_in_order(model)
    plot_qpos_addrs = [int(model.jnt_qposadr[jid]) for jid in plot_joint_ids]
    sample_times: List[float] = []
    sample_angles: List[List[float]] = [[] for _ in plot_qpos_addrs]

    if free_space_mode:
        fsm_joint_ids = fsm_joint_ids_in_order(model)

        xml_joint_axis_points = [data.xanchor[jid].copy() for jid in fsm_joint_ids]
        xml_poly_points = build_section_polyline_points(
            xml_joint_axis_points,
            POLYLINE_SEGMENT_LENGTHS_M,
        )
        xml_segment_lengths = polyline_segment_lengths(xml_poly_points)
        xml_poly_len = float(sum(xml_segment_lengths))
        xml_segment_mm = [round(v * 1000.0, 3) for v in xml_segment_lengths]
        print(f"fsm_segment_lengths_count={len(xml_segment_lengths)}")
        print(f"fsm_segment_lengths_mm={xml_segment_mm}")
        print(f"fsm_polyline_length_from_xml={xml_poly_len:.6f} m")
        run_free_space_render_loop(
            model=model,
            data=data,
            output_path=output_path,
            output_fps=output_fps,
            total_steps=total_steps,
            settle_steps=settle_steps,
            render_every=render_every,
            total_frames=total_frames,
            table=table,
            fsm_joint_ids=fsm_joint_ids,
            polyline_segment_lengths_m=POLYLINE_SEGMENT_LENGTHS_M,
            segment_colors=segment_colors,
            plot_qpos_addrs=plot_qpos_addrs,
            sample_times=sample_times,
            sample_angles=sample_angles,
            w_left=w_left,
            w_right=w_right,
            h_top=h_top,
            h_bottom=h_bottom,
            width=width,
            height=height,
            fsm_split_x_ratio=fsm_split_x_ratio,
            fsm_split_y_ratio=fsm_split_y_ratio,
            fovy_main=fovy_main,
            fovy_left=fovy_left,
            fovy_top=fovy_top,
            fovy_persp=fovy_persp,
            persp_cam=persp_cam,
            dyn_main_cam=dyn_main_cam,
            dyn_left_cam=dyn_left_cam,
            dyn_top_cam=dyn_top_cam,
            show_f_curve_overlay=fsm_show_f_curve_overlay,
            f_curve_fn=midline_f,
            show_local_axes=fsm_show_local_axes,
            show_shell_overlay=fsm_show_shell_overlay,
        )
    elif fk_mode:
        fk_joint_ids = fsm_joint_ids_in_order(model)
        run_fk_render_loop(
            model=model,
            data=data,
            output_path=output_path,
            output_fps=output_fps,
            total_steps=total_steps,
            settle_steps=settle_steps,
            render_every=render_every,
            total_frames=total_frames,
            table=table,
            plot_qpos_addrs=plot_qpos_addrs,
            sample_times=sample_times,
            sample_angles=sample_angles,
            width=width,
            height=height,
            camera_distance=float(cfg["camera_distance"]),
            scaled_fovy=scaled_fovy,
            joint_ids_in_order=fk_joint_ids,
            debug_info_fn=get_debug_info,
        )
    else:
        run_standard_render_loop(
            model=model,
            data=data,
            output_path=output_path,
            output_fps=output_fps,
            total_steps=total_steps,
            settle_steps=settle_steps,
            render_every=render_every,
            total_frames=total_frames,
            table=table,
            cam=cam,
            scaled_fovy=scaled_fovy,
            plot_qpos_addrs=plot_qpos_addrs,
            sample_times=sample_times,
            sample_angles=sample_angles,
            view_width=view_width,
            view_height=view_height,
        )

    if total_frames > 0:
        print(file=sys.stdout, flush=True)

    plot_path = RES_DIR / f"{Path(output_name).stem}_jagl.png"
    plot_joint_angles_mod3(sample_times, sample_angles, plot_path, segment_colors)

    print(f"Video saved: {output_path}")


if __name__ == "__main__":
    render_headless(load_config())
