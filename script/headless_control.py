"""Control and dynamics helpers for headless rendering."""

from __future__ import annotations

from typing import Callable, Dict, List

import mujoco
import numpy as np


def apply_free_space_mode(model: mujoco.MjModel, enabled: bool) -> None:
    """Switch environment to a free-space approximation with no external field/contact."""
    if not enabled:
        return

    model.opt.gravity[:] = 0.0
    model.opt.wind[:] = 0.0
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)

    try:
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, b"floor")
    except Exception:
        try:
            floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        except Exception:
            floor_id = -1

    if floor_id is not None and 0 <= floor_id < model.ngeom:
        try:
            model.geom_rgba[floor_id, 3] = 0.0
        except Exception:
            pass
        try:
            model.geom_contype[floor_id] = 0
            model.geom_conaffinity[floor_id] = 0
        except Exception:
            pass


def clear_external_forces(data: mujoco.MjData) -> None:
    """Ensure no user-applied external wrench/force leaks into dynamics."""
    data.xfrc_applied[:] = 0.0
    data.qfrc_applied[:] = 0.0


def actuator_names(model: mujoco.MjModel) -> List[str]:
    """Return actuator names via stable MuJoCo API across versions."""
    names: List[str] = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        names.append(name if name is not None else f"actuator_{i}")
    return names


def build_actuator_function_table(
    model: mujoco.MjModel,
    function_map: Dict[str, Callable[[float], float]],
) -> Dict[int, Callable[[float], float]]:
    """Create mapping actuator index -> function(t) using actuator names."""
    names = actuator_names(model)
    table: Dict[int, Callable[[float], float]] = {}

    for i, name in enumerate(names):
        if name in function_map:
            table[i] = function_map[name]

    return table


def apply_time_only_controls(
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
        if model.actuator_ctrllimited[i]:
            lo, hi = model.actuator_ctrlrange[i]
            u = float(np.clip(u, lo, hi))

        data.ctrl[i] = u
