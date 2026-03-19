"""Render loop implementations for headless rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Sequence

import imageio
import mujoco
import numpy as np

from headless_camera import camera_direction_to_az_el, orthogonal_basis, principal_axis_and_com, set_model_fovy
from headless_common import print_progress
from headless_compose import compose_fsm_quad
from headless_control import apply_time_only_controls, clear_external_forces
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

