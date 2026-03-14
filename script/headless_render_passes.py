"""Render loop implementations for headless rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Sequence

import imageio
import mujoco
import numpy as np
from shapefit_types import DebugInfo

from headless_camera import camera_direction_to_az_el, orthogonal_basis, principal_axis_and_com, set_model_fovy
from headless_common import print_progress
from headless_compose import compose_fsm_quad
from headless_control import apply_time_only_controls, clear_external_forces, pin_base_free_joint
from headless_joint_utils import (
    append_pair_connectors,
    append_joint_axis_markers,
    append_point_markers,
    append_joint_polyline,
    append_twist_axis_markers,
    build_section_polyline_points,
    dim_scene_model_geoms,
)


ControlTable = Dict[int, Callable[[float], float]]


def _record_joint_samples(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    t: float,
    plot_qpos_addrs: Sequence[int],
    sample_times: List[float],
    sample_angles: List[List[float]],
) -> None:
    sample_times.append(t)
    for idx, qaddr in enumerate(plot_qpos_addrs):
        if 0 <= qaddr < model.nq:
            sample_angles[idx].append(float(data.qpos[qaddr]))


def run_free_space_render_loop(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    output_path: Path,
    output_fps: int,
    total_steps: int,
    settle_steps: int,
    render_every: int,
    total_frames: int,
    table: ControlTable,
    fsm_joint_ids: Sequence[int],
    polyline_segment_lengths_m: Sequence[float],
    segment_colors: Sequence[np.ndarray],
    plot_qpos_addrs: Sequence[int],
    sample_times: List[float],
    sample_angles: List[List[float]],
    w_left: int,
    w_right: int,
    h_top: int,
    h_bottom: int,
    width: int,
    height: int,
    fsm_split_x_ratio: float,
    fsm_split_y_ratio: float,
    fovy_main: float,
    fovy_left: float,
    fovy_top: float,
    fovy_persp: float,
    persp_cam: mujoco.MjvCamera,
    dyn_main_cam: mujoco.MjvCamera,
    dyn_left_cam: mujoco.MjvCamera,
    dyn_top_cam: mujoco.MjvCamera,
    show_f_curve_overlay: bool = False,
    f_curve_fn: Callable[[float, float], tuple[float, float, float]] | None = None,
    show_local_axes: bool = False,
    show_shell_overlay: bool = False,
) -> None:
    prev_axis: np.ndarray | None = None

    with (
        mujoco.Renderer(model, width=w_left, height=h_top) as renderer_main,
        mujoco.Renderer(model, width=w_right, height=h_top) as renderer_left,
        mujoco.Renderer(model, width=w_left, height=h_bottom) as renderer_top,
        mujoco.Renderer(model, width=w_right, height=h_bottom) as renderer_persp,
        imageio.get_writer(str(output_path), fps=output_fps) as writer,
    ):
        for renderer in (renderer_main, renderer_left, renderer_top, renderer_persp):
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1

        written_frames = 0

        for step in range(total_steps):
            clear_external_forces(data)

            if step >= settle_steps:
                t = (step - settle_steps) * model.opt.timestep
            else:
                t = 0.0
            apply_time_only_controls(model, data, table, t)

            mujoco.mj_step(model, data)

            if step >= settle_steps and (step - settle_steps) % render_every == 0:
                _record_joint_samples(model, data, t, plot_qpos_addrs, sample_times, sample_angles)

                joint_axis_points = [data.xanchor[jid].copy() for jid in fsm_joint_ids]
                poly_points = build_section_polyline_points(joint_axis_points, polyline_segment_lengths_m)

                com, principal_axis = principal_axis_and_com(model, data, prev_axis)
                prev_axis = principal_axis.copy()
                front, left, top = orthogonal_basis(principal_axis)

                for cam in (dyn_main_cam, dyn_left_cam, dyn_top_cam):
                    cam.lookat[:] = com

                dyn_main_cam.azimuth, dyn_main_cam.elevation = camera_direction_to_az_el(front)
                dyn_left_cam.azimuth, dyn_left_cam.elevation = camera_direction_to_az_el(left)
                dyn_top_cam.azimuth, dyn_top_cam.elevation = camera_direction_to_az_el(top)

                set_model_fovy(model, fovy_main)
                renderer_main.update_scene(data, camera=dyn_main_cam)
                frame_main = renderer_main.render()

                set_model_fovy(model, fovy_left)
                renderer_left.update_scene(data, camera=dyn_left_cam)
                frame_left = renderer_left.render()

                set_model_fovy(model, fovy_top)
                renderer_top.update_scene(data, camera=dyn_top_cam)
                frame_top = renderer_top.render()

                set_model_fovy(model, fovy_persp)
                renderer_persp.update_scene(data, camera=persp_cam)
                if show_shell_overlay:
                    dim_scene_model_geoms(renderer_persp.scene, 0.05)
                else:
                    # Hide model shell in FSM BR panel when overlay is disabled.
                    dim_scene_model_geoms(renderer_persp.scene, 0.0)
                append_joint_polyline(renderer_persp.scene, poly_points, segment_colors)
                if show_local_axes:
                    append_joint_axis_markers(
                        renderer_persp.scene,
                        model,
                        data,
                        fsm_joint_ids,
                        allowed_axis_indices=(1, 2),  # y/z-axis joints
                        length_m=0.16,
                        radius=0.007,
                        rgba=np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64),
                    )
                    append_twist_axis_markers(
                        renderer_persp.scene,
                        model,
                        data,
                        fsm_joint_ids,
                        twist_length_m=0.1,
                        twist_radius=0.007,
                        twist_base_angle_rad=np.pi / 2.0,
                        twist_rgba=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
                    )
                if show_f_curve_overlay and f_curve_fn is not None:
                    s_samples = np.linspace(0.0, 1.0, 20, dtype=np.float64)
                    f_points = [
                        np.asarray(f_curve_fn(float(t), float(sv)), dtype=np.float64)
                        for sv in s_samples
                    ]
                    # Transform the purple curve to visually align with the midline:
                    # 1) Rotate around the curve start to align the initial tangent with the
                    #    first segment joint direction
                    # 2) Translate the rotated start point to the head anchor
                    f0 = f_points[0].copy()
                    # head anchor use the first joint's xanchor
                    head_anchor = np.asarray(joint_axis_points[0], dtype=np.float64)

                    # Target direction: vector from the first joint to the second joint (if available)
                    if len(joint_axis_points) > 1:
                        tgt_dir = np.asarray(joint_axis_points[1], dtype=np.float64) - np.asarray(joint_axis_points[0], dtype=np.float64)
                    else:
                        tgt_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    tgt_norm = float(np.linalg.norm(tgt_dir))
                    if tgt_norm < 1e-9:
                        tgt_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    else:
                        tgt_dir = tgt_dir / tgt_norm

                    # Source direction: the first tangent of f
                    if len(f_points) > 1:
                        src_dir = f_points[1] - f_points[0]
                    else:
                        src_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    src_norm = float(np.linalg.norm(src_dir))
                    if src_norm < 1e-9:
                        src_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    else:
                        src_dir = src_dir / src_norm

                    cross = np.cross(src_dir, tgt_dir)
                    cross_norm = float(np.linalg.norm(cross))
                    if cross_norm < 1e-9:
                        angle = 0.0
                        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    else:
                        axis = cross / cross_norm
                        angle = float(np.arccos(np.clip(float(np.dot(src_dir, tgt_dir)), -1.0, 1.0)))

                    def _rodrigues_rotate(v: np.ndarray, k: np.ndarray, theta: float) -> np.ndarray:
                        ca = float(np.cos(theta))
                        sa = float(np.sin(theta))
                        return v * ca + np.cross(k, v) * sa + k * float(np.dot(k, v)) * (1.0 - ca)

                    rotated = []
                    for p in f_points:
                        v = p - f0
                        v_rot = _rodrigues_rotate(v, axis, angle)
                        rotated.append(v_rot + f0)

                    # Translate to head_anchor
                    delta = head_anchor - rotated[0]
                    f_points_trans = [p + delta for p in rotated]

                    append_joint_polyline(
                        renderer_persp.scene,
                        f_points_trans,
                        [np.array([0.75, 0.2, 0.95, 1.0], dtype=np.float32)],
                        radius=0.004,
                    )
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


def run_standard_render_loop(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    output_path: Path,
    output_fps: int,
    total_steps: int,
    settle_steps: int,
    render_every: int,
    total_frames: int,
    table: ControlTable,
    cam: mujoco.MjvCamera,
    scaled_fovy: float,
    plot_qpos_addrs: Sequence[int],
    sample_times: List[float],
    sample_angles: List[List[float]],
    view_width: int,
    view_height: int,
) -> None:
    with (
        mujoco.Renderer(model, width=view_width, height=view_height) as renderer,
        imageio.get_writer(str(output_path), fps=output_fps) as writer,
    ):
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        set_model_fovy(model, scaled_fovy)
        written_frames = 0

        for step in range(total_steps):
            if step >= settle_steps:
                t = (step - settle_steps) * model.opt.timestep
            else:
                t = 0.0
            apply_time_only_controls(model, data, table, t)

            mujoco.mj_step(model, data)

            if step >= settle_steps and (step - settle_steps) % render_every == 0:
                _record_joint_samples(model, data, t, plot_qpos_addrs, sample_times, sample_angles)

                renderer.update_scene(data, camera=cam)
                frame = renderer.render()

                writer.append_data(frame)
                written_frames += 1
                print_progress(written_frames, total_frames)


def run_fk_render_loop(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    output_path: Path,
    output_fps: int,
    total_steps: int,
    settle_steps: int,
    render_every: int,
    total_frames: int,
    table: ControlTable,
    plot_qpos_addrs: Sequence[int],
    sample_times: List[float],
    sample_angles: List[List[float]],
    width: int,
    height: int,
    camera_distance: float,
    scaled_fovy: float,
    joint_ids_in_order: Sequence[int],
    debug_info_fn: Callable[[float], DebugInfo | None],
) -> None:
    split_x_ratio = 0.5
    split_y_ratio = 0.5

    w_left = width // 2
    w_right = width - w_left
    h_top = height // 2
    h_bottom = height - h_top

    orbit_cam = mujoco.MjvCamera()
    orbit_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    orbit_cam.fixedcamid = -1
    orbit_cam.trackbodyid = -1
    orbit_cam.distance = float(camera_distance)
    orbit_cam.elevation = 0.0

    with (
        mujoco.Renderer(model, width=w_left, height=h_top) as renderer_mj,
        mujoco.Renderer(model, width=w_right, height=h_top) as renderer_fk,
        mujoco.Renderer(model, width=w_left, height=h_bottom) as renderer_target,
        mujoco.Renderer(model, width=w_right, height=h_bottom) as renderer_overlay,
        imageio.get_writer(str(output_path), fps=output_fps) as writer,
    ):
        for renderer in (renderer_mj, renderer_fk, renderer_target, renderer_overlay):
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1

        written_frames = 0

        for step in range(total_steps):
            clear_external_forces(data)

            if step >= settle_steps:
                t = (step - settle_steps) * model.opt.timestep
            else:
                t = 0.0

            apply_time_only_controls(model, data, table, t)
            pin_base_free_joint(model, data)
            mujoco.mj_step(model, data)
            pin_base_free_joint(model, data)
            mujoco.mj_forward(model, data)

            if step >= settle_steps and (step - settle_steps) % render_every == 0:
                _record_joint_samples(model, data, t, plot_qpos_addrs, sample_times, sample_angles)

                center = np.zeros(3, dtype=np.float64)
                if joint_ids_in_order:
                    mid_jid = joint_ids_in_order[len(joint_ids_in_order) // 2]
                    center = np.asarray(data.xanchor[mid_jid], dtype=np.float64)
                orbit_cam.lookat[:] = center
                orbit_cam.azimuth = 90.0 - 360.0 * (float(t) / 10.0)
                orbit_cam.elevation = 0.0

                set_model_fovy(model, scaled_fovy)
                renderer_mj.update_scene(data, camera=orbit_cam)
                frame_mj = renderer_mj.render()

                debug_info = debug_info_fn(float(t))

                renderer_fk.update_scene(data, camera=orbit_cam)
                dim_scene_model_geoms(renderer_fk.scene, 0.0)
                if debug_info is not None:
                    append_point_markers(
                        renderer_fk.scene,
                        [row for row in np.asarray(debug_info.fk_points, dtype=np.float64)],
                        radius=0.012,
                        rgba=np.array([0.1, 0.8, 1.0, 1.0], dtype=np.float64),
                    )
                frame_fk = renderer_fk.render()

                renderer_target.update_scene(data, camera=orbit_cam)
                dim_scene_model_geoms(renderer_target.scene, 0.0)
                if debug_info is not None:
                    append_point_markers(
                        renderer_target.scene,
                        [row for row in np.asarray(debug_info.expected_points, dtype=np.float64)],
                        radius=0.012,
                        rgba=np.array([1.0, 0.55, 0.1, 1.0], dtype=np.float64),
                    )
                frame_target = renderer_target.render()

                renderer_overlay.update_scene(data, camera=orbit_cam)
                dim_scene_model_geoms(renderer_overlay.scene, 0.0)
                if debug_info is not None:
                    append_pair_connectors(
                        renderer_overlay.scene,
                        [row for row in np.asarray(debug_info.expected_points, dtype=np.float64)],
                        [row for row in np.asarray(debug_info.fk_points, dtype=np.float64)],
                        radius=0.006,
                        colors=[
                            np.array([0.1, 0.3, 1.0, 1.0], dtype=np.float64),
                            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
                            np.array([1.0, 0.9, 0.1, 1.0], dtype=np.float64),
                        ],
                    )
                frame_overlay = renderer_overlay.render()

                frame = compose_fsm_quad(
                    frame_mj,
                    frame_fk,
                    frame_target,
                    frame_overlay,
                    height,
                    width,
                    split_x_ratio,
                    split_y_ratio,
                )
                writer.append_data(frame)
                written_frames += 1
                print_progress(written_frames, total_frames)
