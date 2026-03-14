from __future__ import annotations

from pathlib import Path

import imageio
import mujoco
import numpy as np

from common import compose_lr, print_progress
from control import clear_external_forces, fix_base_pose, set_controls
from fwd_kine import FK
from spec import JOINT_AXES, JOINT_SIGNS, N_JOINTS, SEGMENT_LENGTHS_M


def append_fk_points_as_spheres(
    scene: mujoco.MjvScene,
    points: np.ndarray,
    radius: float,
) -> None:
    n = points.shape[0]
    for i in range(n):
        if scene.ngeom >= scene.maxgeom:
            break

        p = np.asarray(points[i], dtype=np.float64)
        geom = scene.geoms[scene.ngeom]

        if i == 0:
            rgba = np.array([1.0, 0.1, 0.1, 1.0], dtype=np.float64)
        elif i == n - 1:
            rgba = np.array([0.1, 0.8, 1.0, 1.0], dtype=np.float64)
        else:
            rgba = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([float(radius), 0.0, 0.0], dtype=np.float64),
            p,
            np.eye(3, dtype=np.float64).ravel(),
            rgba,
        )
        scene.ngeom += 1


def run_compare_render_loop(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    output_path: Path,
    fps: int,
    width_half: int,
    height: int,
    render_frames: int,
    sim_steps_per_frame: int,
    actuator_target: dict[int, float],
    free_qadr: int,
    free_dadr: int,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    joint_ids: list[int],
    target_angles: np.ndarray,
    fk_offset: np.ndarray,
    camera_distance: float,
    camera_elevation_deg: float,
    fk_point_radius: float,
) -> None:
    cam_left = mujoco.MjvCamera()
    cam_right = mujoco.MjvCamera()
    for cam in (cam_left, cam_right):
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        cam.trackbodyid = -1
        cam.distance = float(camera_distance)
        cam.elevation = float(camera_elevation_deg)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        mujoco.Renderer(model, width=width_half, height=height) as renderer_left,
        mujoco.Renderer(model, width=width_half, height=height) as renderer_right,
        imageio.get_writer(str(output_path), fps=fps) as writer,
    ):
        renderer_left.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        renderer_right.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1

        for frame_idx in range(render_frames):
            for _ in range(sim_steps_per_frame):
                set_controls(model, data, actuator_target)
                clear_external_forces(data)
                fix_base_pose(data, free_qadr, free_dadr, base_pos, base_quat)
                mujoco.mj_step(model, data)
                fix_base_pose(data, free_qadr, free_dadr, base_pos, base_quat)
                mujoco.mj_forward(model, data)

            azimuth = 360.0 * (float(frame_idx) / float(render_frames))

            center_joint_id = joint_ids[N_JOINTS // 2]
            center_pos = np.asarray(data.xanchor[center_joint_id], dtype=np.float64)
            cam_left.lookat[:] = center_pos
            cam_left.azimuth = azimuth

            renderer_left.update_scene(data, camera=cam_left)
            frame_left = renderer_left.render()

            fk = FK(
                jn=N_JOINTS,
                init_angles=target_angles,
                seg_len=SEGMENT_LENGTHS_M,
                joint_axes=JOINT_AXES,
                joint_signs=JOINT_SIGNS,
            )
            fk_points = fk.getallp().copy() + fk_offset[None, :]
            fk_center = fk_points[(N_JOINTS // 2) + 1]

            cam_right.lookat[:] = fk_center
            cam_right.azimuth = azimuth

            renderer_right.update_scene(data, camera=cam_right)
            for i in range(renderer_right.scene.ngeom):
                renderer_right.scene.geoms[i].rgba[3] = 0.0
            append_fk_points_as_spheres(
                renderer_right.scene,
                fk_points,
                radius=float(fk_point_radius),
            )
            frame_right = renderer_right.render()

            writer.append_data(compose_lr(frame_left, frame_right))
            print_progress(frame_idx + 1, render_frames)
