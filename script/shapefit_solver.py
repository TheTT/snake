from __future__ import annotations

import math
from typing import Callable, Sequence
from shapefit_types import Axis

import numpy as np
from fwd_kine import FK

from shapefit_types import TwistFn, CurveFn, BackendFn, FitState, FitParam, DebugInfo


def compute_twist(
    g_fn: TwistFn,
    t: float,
    *,
    x_joint_indices_0b: Sequence[int],
    seglen: Sequence[float],
) -> np.ndarray:
    """Compute per-x-joint twist. Returns twist values without base_twist (for FK/geometry use).
    """
    n_x = len(x_joint_indices_0b)
    total_len = float(sum(float(s) for s in seglen))

    # joint positions in normalized arc-length [0, 1]
    cum = np.cumsum(np.asarray(seglen, dtype=np.float64)) / total_len
    x_pos = np.array([float(cum[int(i)]) for i in x_joint_indices_0b], dtype=np.float64)

    boundaries = np.empty(n_x + 1, dtype=np.float64)
    boundaries[0] = 0.0
    boundaries[1:-1] = 0.5 * (x_pos[:-1] + x_pos[1:])
    boundaries[-1] = 1.0

    # Integrate g_fn on each boundary interval with trapezoidal rule.
    twist = np.zeros(n_x, dtype=np.float64)
    n_int_samples = 16
    t_float = float(t)
    for i in range(n_x):
        a = float(boundaries[i])
        b = float(boundaries[i + 1])
        grid = np.linspace(a, b, n_int_samples + 1, dtype=np.float64)
        vals = np.array([float(g_fn(t_float, float(s))) for s in grid], dtype=np.float64)
        twist[i] = float(0.5 * np.sum((vals[1:] + vals[:-1]) * (grid[1:] - grid[:-1])))

    return twist


_SAMPLE_NUMBER = 200
# bigger hint radius for better convergence, but slower
_HINT_RADIUS = 50


def get_fsample(
    f_fn: CurveFn,
    t: float,
    n_samples: int,
) -> np.ndarray:
    """Sample points on the curve."""
    return np.array([f_fn(float(t), float(s / (n_samples - 1))) for s in range(n_samples)], dtype=np.float64)


def get_ftrans(
    fplist: np.ndarray,
    totlen: float,
    first_joint_axis: Axis,
    frame_x_axis: np.ndarray,
    last_UP: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute transform to map curve into a robot-following frame.
    fplist: fplist[0]=[0,0,0] ensured
    Returns (ftrans, new_UP).
    """
    # get scale to match the length
    diffs = np.diff(fplist, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    scale = totlen / seg_lengths.sum()

    x_axis = np.asarray(frame_x_axis, dtype=np.float64)
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm > 1e-9:
        x_axis = x_axis / x_norm
    else:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # project last_UP onto plane orthogonal to x_axis
    proj = last_UP - (float(np.dot(last_UP, x_axis))) * x_axis
    proj_norm = float(np.linalg.norm(proj))
    if proj_norm > 1e-9:
        binorm_proj = proj / proj_norm
    else:
        if abs(x_axis[0]) < 0.9:
            tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        proj = tmp - (float(np.dot(tmp, x_axis))) * x_axis
        proj /= float(np.linalg.norm(proj))
        binorm_proj = proj

    # choose sign closest to last_UP if provided
    cur_UP = binorm_proj
    if float(np.dot(last_UP, cur_UP)) < 0.0:
        cur_UP = -cur_UP

    if first_joint_axis == Axis.Y:
        y_axis = cur_UP
        z_axis = np.cross(x_axis, y_axis)
    else:  # first_joint_axis == Axis.Z
        z_axis = cur_UP
        y_axis = np.cross(z_axis, x_axis)

    ftrans = float(scale) * np.row_stack([x_axis, y_axis, z_axis])
    new_UP = cur_UP
    # print("x(",x_axis[0],",",x_axis[1],",",x_axis[2],")")
    # print("y(",y_axis[0],",",y_axis[1],",",y_axis[2],")")
    # print("z(",z_axis[0],",",z_axis[1],",",z_axis[2],")")
    return ftrans, new_UP


def get_hint(
    fplist: np.ndarray,
    jn: int,
    seglen: Sequence[float],
    state: FitState,
) -> np.ndarray:
    """Get hint indices for closest curve point to head, each joint and tail.
    Uses locality to speed up the search.
    """
    hint_i = np.zeros(jn + 2, dtype=np.int32)
    hint_i[0] = 0
    i = 0
    d = 0.0
    for j in range(1, len(fplist)):
        d += float(np.linalg.norm(fplist[j] - fplist[j - 1]))
        while i < jn + 1 and d > seglen[i]:
            d -= seglen[i]
            i += 1
            hint_i[i] = j
        if i >= jn + 1:
            break
    hint_i[-1] = len(fplist) - 1

    return hint_i


def solve_shape_for_time(
    state: FitState,
    *,
    f_fn: CurveFn,
    g_fn: TwistFn,
    t: float,
    n_joints: int,
    joint_axes: Sequence[Axis],
    joint_signs: Sequence[float],
    x_joint_indices_0b: Sequence[int],
    seglen: Sequence[float],
    totlen: float,
    backend_fn: BackendFn,
) -> DebugInfo:
    twist_no_base = compute_twist(
        g_fn=g_fn,
        t=t,
        x_joint_indices_0b=x_joint_indices_0b,
        seglen=seglen,
    )

    # Use previous solved shape as robot-following frame, so f curve's own
    # rotation/evolution is preserved and not canceled by per-frame re-alignment.
    prev_fk = FK(
        jn=n_joints,
        init_angles=state.joint_tar,
        seg_len=seglen,
        joint_axes=joint_axes,
        joint_signs=joint_signs,
    )
    prev_fk_points = prev_fk.getallp().copy()
    frame_x_axis = prev_fk_points[-1] - prev_fk_points[0]
    if float(np.linalg.norm(frame_x_axis)) <= 1e-9:
        frame_x_axis = np.array([-1.0, 0.0, 0.0], dtype=np.float64)

    fplist = get_fsample(f_fn, t, _SAMPLE_NUMBER)
    # ftrans is a 3*3 transform
    ftrans, state.last_UP = get_ftrans(
        fplist,
        totlen,
        joint_axes[0],
        frame_x_axis,
        state.last_UP,
    )
    fplist = (ftrans @ fplist.T).T
    # Hint indices for closest curve point
    hint_i = get_hint(
        fplist=fplist,
        jn=n_joints,
        seglen=seglen,
        state=state,
    )

    # print()
    # for k in (6, 12):
    #     hi = int(hint_i[k])
    #     print(k, hi, fplist[hi].tolist(), flush=True)

    backend_param = FitParam(
        jn=n_joints,
        fplist=fplist,
        hint_i=hint_i,
        twist=twist_no_base,
        joint_axes=joint_axes,
        joint_signs=joint_signs,
        seglen=seglen,
        hint_rad=_HINT_RADIUS,
    )
    backend_ret = backend_fn(
        state.joint_tar,
        backend_param,
    )

    # backend_fn may return:
    # - just angles (np.ndarray)
    # - (angles, fk_obj)
    # - (angles, meta) where meta is Any packing fk and nearest_idx (dict or tuple)
    if isinstance(backend_ret, tuple) and len(backend_ret) >= 2 and isinstance(backend_ret[0], np.ndarray):
        v = backend_ret[0]
        meta = backend_ret[1]
        fk = None
        node_hint_idx = None
        if isinstance(meta, dict):
            fk = meta.get("fk", None)
            node_hint_idx = meta.get("nearest_idx", None)
        elif isinstance(meta, (tuple, list)):
            if len(meta) >= 1:
                fk = meta[0]
            if len(meta) >= 2:
                node_hint_idx = meta[1]
        else:
            # backward compat: meta may itself be an FK object
            fk = meta
    else:
        v = backend_ret
        fk = None
        node_hint_idx = None

    state.joint_tar[:] = v[:]

    if fk is None:
        fk = FK(
            jn=n_joints,
            init_angles=state.joint_tar,
            seg_len=seglen,
            joint_axes=joint_axes,
            joint_signs=joint_signs,
        )
    fk_points = fk.getallp().copy()
    # prefer node_hint_idx returned from backend if available
    if node_hint_idx is not None:
        hint_for_debug = np.asarray(node_hint_idx, dtype=np.int32)
    else:
        hint_for_debug = hint_i.copy()

    expected_points = fplist[hint_for_debug].copy()
    return DebugInfo(
        fk_points=fk_points,
        expected_points=expected_points,
        hint_i=hint_for_debug,
    )
