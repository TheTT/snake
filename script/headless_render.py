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
from joint_functions import JOINT_FUNCTIONS, POLYLINE_SEGMENT_LENGTHS_M

HEAD_TO_FIRST_AXIS_M = 0.083
TAIL_TO_LAST_AXIS_M = 0.113


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


def _dim_scene_model_geoms(scene: mujoco.MjvScene, alpha: float) -> None:
    """Dim current scene geoms by setting their alpha for this view only."""
    a = float(alpha)
    for i in range(scene.ngeom):
        scene.geoms[i].rgba[3] = a


def _append_joint_polyline(
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
                "mjv_connector failed in _append_joint_polyline with "
                f"radius={float(radius)}, p0={p0_arr.tolist()}, p1={p1_arr.tolist()}"
            ) from exc
        scene.ngeom += 1


def _safe_unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return vec / norm


def _build_section_polyline_points(
    joint_axis_points: Sequence[np.ndarray],
    segment_lengths: Sequence[float],
) -> List[np.ndarray]:
    """Build polyline points from manual segment lengths.

    Construction:
    - segment_lengths has length N+1 for N joints
    - directions di for segment i are approximated by normalize(A[min(i+1,N-1)] - A[max(i,0)])
    - start at head point p0 = A0 - L0 * d0, then p_{k+1} = p_k + L_k * d_k
    """
    anchors = [np.asarray(p, dtype=np.float64).copy() for p in joint_axis_points]
    N = len(anchors)
    if N == 0:
        return []
    if len(segment_lengths) != N + 1:
        raise ValueError(f"segment_lengths must have length N+1 ({N+1}), got {len(segment_lengths)}")

    def dir_for(i: int) -> np.ndarray:
        # direction for segment i: use vector between anchors[min(i, N-1)] and anchors[max(i-1, 0)]
        if i == 0:
            vec = anchors[1] - anchors[0] if N > 1 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        elif i >= N:
            vec = anchors[-1] - anchors[-2] if N > 1 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            vec = anchors[i] - anchors[i - 1]
        return _safe_unit(vec)

    points: List[np.ndarray] = []
    d0 = dir_for(0)
    p0 = anchors[0] - float(segment_lengths[0]) * d0
    points.append(p0)

    for i in range(0, N):
        di = dir_for(i)
        pi_prev = points[-1]
        pi = pi_prev + float(segment_lengths[i]) * di
        points.append(pi)

    # last segment (index N) uses direction dir_for(N)
    dN = dir_for(N)
    p_last = points[-1] + float(segment_lengths[N]) * dN
    points.append(p_last)

    return points


def _polyline_segment_lengths(points: Sequence[np.ndarray]) -> List[float]:
    """Return individual segment lengths for consecutive-point polyline."""
    if len(points) < 2:
        return []

    out: List[float] = []
    for i in range(len(points) - 1):
        p0 = np.asarray(points[i], dtype=np.float64)
        p1 = np.asarray(points[i + 1], dtype=np.float64)
        out.append(float(np.linalg.norm(p1 - p0)))
    return out


def _polyline_length(points: Sequence[np.ndarray]) -> float:
    """Return total length of consecutive-point polyline."""
    if len(points) < 2:
        return 0.0

    length = 0.0
    for i in range(len(points) - 1):
        p0 = np.asarray(points[i], dtype=np.float64)
        p1 = np.asarray(points[i + 1], dtype=np.float64)
        length += float(np.linalg.norm(p1 - p0))
    return length


def _plot_joint_angles_mod3(
    times: Sequence[float],
    joint_angles: Sequence[Sequence[float]],
    output_path: Path,
    segment_colors: Sequence[np.ndarray],
) -> None:
    """Plot joint angle trajectories with colors by joint-index mod 3.

    - black background, white axes/ticks/text
    - y values folded into [0, 2pi)
    """
    if not times or not joint_angles:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skip angle plot: matplotlib import failed ({exc})")
        return

    times_arr = np.asarray(times, dtype=np.float64)
    two_pi = float(2.0 * np.pi)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    for idx, angles in enumerate(joint_angles):
        if not angles:
            continue
        y = np.mod(np.asarray(angles, dtype=np.float64), two_pi)
        c_rgba = np.asarray(segment_colors[idx % len(segment_colors)], dtype=np.float64)
        c_rgb = tuple(np.clip(c_rgba[:3], 0.0, 1.0).tolist())
        ax.plot(times_arr, y, color=c_rgb, linewidth=1.0, alpha=0.95)

    ax.set_xlim(float(times_arr[0]), float(times_arr[-1]))
    ax.set_ylim(0.0, two_pi)
    ax.set_xlabel("time (s)", color="white")
    ax.set_ylabel("angle mod 2pi (rad)", color="white")
    ax.set_title("Joint Angles Colored by Joint Index mod 3", color="white")

    for spine in ax.spines.values():
        spine.set_color("white")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    ax.grid(color="#333333", linewidth=0.6, alpha=0.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Angle plot saved: {output_path}")


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
    segment_colors = [
        np.array([0.1, 0.3, 1.0, 1.0], dtype=np.float32),  # blue
        np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),  # white
        np.array([1.0, 0.9, 0.1, 1.0], dtype=np.float32),  # yellow
    ]

    # Record actual post-step joint qpos values in joint_1..joint_N order.
    plot_joint_ids = _fsm_joint_ids_in_order(model)
    plot_qpos_addrs = [int(model.jnt_qposadr[jid]) for jid in plot_joint_ids]
    sample_times: List[float] = []
    sample_angles: List[List[float]] = [[] for _ in plot_qpos_addrs]

    if free_space_mode:
        fsm_joint_ids = _fsm_joint_ids_in_order(model)

        xml_joint_axis_points = [data.xanchor[jid].copy() for jid in fsm_joint_ids]
        xml_poly_points = _build_section_polyline_points(
            xml_joint_axis_points,
            POLYLINE_SEGMENT_LENGTHS_M,
        )
        xml_segment_lengths = _polyline_segment_lengths(xml_poly_points)
        xml_poly_len = float(sum(xml_segment_lengths))
        xml_segment_mm = [round(v * 1000.0, 3) for v in xml_segment_lengths]
        print(f"fsm_segment_lengths_count={len(xml_segment_lengths)}")
        print(f"fsm_segment_lengths_mm={xml_segment_mm}")
        print(f"fsm_polyline_length_from_xml={xml_poly_len:.6f} m")

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
                else:
                    # Hold commanded initial pose during settle to avoid drift/jump.
                    t = 0.0
                apply_time_only_controls(model, data, table, t)

                mujoco.mj_step(model, data)

                if step >= settle_steps and (step - settle_steps) % render_every == 0:
                    sample_times.append(t)
                    for idx, qaddr in enumerate(plot_qpos_addrs):
                        if 0 <= qaddr < model.nq:
                            sample_angles[idx].append(float(data.qpos[qaddr]))

                    joint_axis_points = [data.xanchor[jid].copy() for jid in fsm_joint_ids]
                    poly_points = _build_section_polyline_points(
                        joint_axis_points,
                        POLYLINE_SEGMENT_LENGTHS_M,
                    )

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
                    frame_main = renderer_main.render()

                    set_model_fovy(model, fovy_left)
                    renderer_left.update_scene(data, camera=dyn_left_cam)
                    frame_left = renderer_left.render()

                    set_model_fovy(model, fovy_top)
                    renderer_top.update_scene(data, camera=dyn_top_cam)
                    frame_top = renderer_top.render()

                    set_model_fovy(model, fovy_persp)
                    renderer_persp.update_scene(data, camera=persp_cam)
                    _dim_scene_model_geoms(renderer_persp.scene, 0.05)
                    _append_joint_polyline(renderer_persp.scene, poly_points, segment_colors)
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
                else:
                    # Hold commanded initial pose during settle to avoid drift/jump.
                    t = 0.0
                apply_time_only_controls(model, data, table, t)

                mujoco.mj_step(model, data)

                if step >= settle_steps and (step - settle_steps) % render_every == 0:
                    sample_times.append(t)
                    for idx, qaddr in enumerate(plot_qpos_addrs):
                        if 0 <= qaddr < model.nq:
                            sample_angles[idx].append(float(data.qpos[qaddr]))

                    renderer.update_scene(data, camera=cam)
                    frame = renderer.render()

                    writer.append_data(frame)
                    written_frames += 1
                    print_progress(written_frames, total_frames)

    if total_frames > 0:
        print(file=sys.stdout, flush=True)

    plot_path = RES_DIR / f"{Path(output_name).stem}_jagl.png"
    _plot_joint_angles_mod3(sample_times, sample_angles, plot_path, segment_colors)

    print(f"Video saved: {output_path}")


if __name__ == "__main__":
    render_headless(load_config())
