from __future__ import annotations

from typing import Dict, List

import mujoco
import numpy as np


def joint_ids_in_order(model: mujoco.MjModel) -> List[int]:
    ids: List[int] = []
    i = 1
    while True:
        act_name = f"joint_{i}_pos"
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        if act_id < 0:
            break
        jid = int(model.actuator_trnid[act_id, 0])
        if jid >= 0:
            ids.append(jid)
        i += 1
    return ids


def build_actuator_table(model: mujoco.MjModel, n_joints: int) -> Dict[int, float]:
    table: Dict[int, float] = {}
    for i in range(n_joints):
        name = f"joint_{i + 1}_pos"
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            table[int(aid)] = 0.0
    return table


def set_no_gravity(model: mujoco.MjModel) -> None:
    model.opt.gravity[:] = 0.0
    model.opt.wind[:] = 0.0
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)


def find_free_joint(model: mujoco.MjModel) -> tuple[int, int, int]:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_free_joint")
    if jid < 0:
        raise ValueError("Cannot find base_free_joint")
    qadr = int(model.jnt_qposadr[jid])
    dadr = int(model.jnt_dofadr[jid])
    return jid, qadr, dadr


def fix_base_pose(
    data: mujoco.MjData,
    free_qadr: int,
    free_dadr: int,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
) -> None:
    data.qpos[free_qadr:free_qadr + 3] = base_pos
    data.qpos[free_qadr + 3:free_qadr + 7] = base_quat
    data.qvel[free_dadr:free_dadr + 6] = 0.0


def clear_external_forces(data: mujoco.MjData) -> None:
    data.xfrc_applied[:] = 0.0
    data.qfrc_applied[:] = 0.0


def set_controls(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_target_by_id: Dict[int, float],
) -> None:
    for aid in range(model.nu):
        u = float(actuator_target_by_id.get(aid, 0.0))
        if bool(model.actuator_ctrllimited[aid]):
            lo, hi = model.actuator_ctrlrange[aid]
            u = float(np.clip(u, lo, hi))
        data.ctrl[aid] = u
