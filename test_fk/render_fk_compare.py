from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import mujoco
import numpy as np

from common import resolve_path
from config import load_config, parse_resolution
from control import (
    build_actuator_table,
    clear_external_forces,
    find_free_joint,
    fix_base_pose,
    joint_ids_in_order,
    set_controls,
    set_no_gravity,
)
from render_passes import run_compare_render_loop
from spec import N_JOINTS


def _target_angles_array(cfg: Dict[str, Any]) -> np.ndarray:
    arr = np.asarray(cfg["target_angles_rad"], dtype=np.float64)
    if arr.shape != (N_JOINTS,):
        raise ValueError(f"target_angles_rad shape must be ({N_JOINTS},)")
    return arr


def render_compare(cfg: Dict[str, Any], cfg_path: Path) -> None:
    xml_path = resolve_path(cfg_path.parent, str(cfg["xml"]))
    output_path = resolve_path(cfg_path.parent, str(cfg["output"]))

    target_angles = _target_angles_array(cfg)
    width_total, height = parse_resolution(cfg["resolution"])
    if width_total % 2 != 0:
        width_total -= 1
    width_half = width_total // 2

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = float(cfg["sim_timestep"])
    set_no_gravity(model)

    _, free_qadr, free_dadr = find_free_joint(model)

    base_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)  # yaw 180deg, faces -x
    data.qpos[free_qadr:free_qadr + 3] = 0.0
    data.qpos[free_qadr + 3:free_qadr + 7] = base_quat
    data.qvel[free_dadr:free_dadr + 6] = 0.0

    joint_ids = joint_ids_in_order(model)
    if len(joint_ids) != N_JOINTS:
        raise ValueError(f"Expected {N_JOINTS} ordered joints, got {len(joint_ids)}")
    qaddrs = [int(model.jnt_qposadr[jid]) for jid in joint_ids]

    mujoco.mj_forward(model, data)

    # Shift base so the first joint anchor is exactly at world origin (head reference point fixed at origin).
    anchor0 = np.asarray(data.xanchor[joint_ids[0]], dtype=np.float64).copy()
    base_pos = -anchor0
    fix_base_pose(data, free_qadr, free_dadr, base_pos, base_quat)
    mujoco.mj_forward(model, data)

    actuator_target = build_actuator_table(model, N_JOINTS)
    start_angles = np.array([float(data.qpos[qa]) for qa in qaddrs], dtype=np.float64)

    ramp_steps = max(1, int(round(float(cfg["ramp_duration_s"]) / model.opt.timestep)))
    hold_steps = max(0, int(round(float(cfg["hold_duration_s"]) / model.opt.timestep)))
    fps = int(cfg["fps"])
    render_frames = max(1, int(round(float(cfg["render_duration_s"]) * fps)))
    sim_steps_per_frame = max(1, int(round((1.0 / fps) / model.opt.timestep)))

    # Smoothly apply target angles then hold before rendering starts.
    for s in range(ramp_steps + hold_steps):
        if s < ramp_steps:
            alpha = float(s + 1) / float(ramp_steps)
            cmd = (1.0 - alpha) * start_angles + alpha * target_angles
        else:
            cmd = target_angles

        for i in range(N_JOINTS):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint_{i + 1}_pos")
            if aid >= 0:
                actuator_target[int(aid)] = float(cmd[i])

        set_controls(model, data, actuator_target)
        clear_external_forces(data)
        fix_base_pose(data, free_qadr, free_dadr, base_pos, base_quat)
        mujoco.mj_step(model, data)
        fix_base_pose(data, free_qadr, free_dadr, base_pos, base_quat)
        mujoco.mj_forward(model, data)

    fovy = float(np.clip(cfg.get("camera_fovy_deg", 45.0), 5.0, 170.0))
    model.vis.global_.fovy = fovy

    fk_offset = np.asarray(cfg["fk_separate_offset"], dtype=np.float64)
    if fk_offset.shape != (3,):
        raise ValueError("fk_separate_offset must be [x, y, z]")

    run_compare_render_loop(
        model=model,
        data=data,
        output_path=output_path,
        fps=fps,
        width_half=width_half,
        height=height,
        render_frames=render_frames,
        sim_steps_per_frame=sim_steps_per_frame,
        actuator_target=actuator_target,
        free_qadr=free_qadr,
        free_dadr=free_dadr,
        base_pos=base_pos,
        base_quat=base_quat,
        joint_ids=joint_ids,
        target_angles=target_angles,
        fk_offset=fk_offset,
        camera_distance=float(cfg["camera_distance"]),
        camera_elevation_deg=float(cfg["camera_elevation_deg"]),
        fk_point_radius=float(cfg.get("fk_point_radius", 0.018)),
    )

    print("\nDone.")
    print(f"Output video: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FK vs MuJoCo side-by-side renderer (standalone test_fk)")
    parser.add_argument(
        "--config",
        default=str((Path(__file__).resolve().parent / "test_fk_config.json")),
        help="Path to config json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    render_compare(cfg, cfg_path)


if __name__ == "__main__":
    main()
