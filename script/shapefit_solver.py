from __future__ import annotations

import math
from typing import Callable, Sequence
from shapefit_types import Axis

import numpy as np

from shapefit_types import TwistFn, CurveFn, BackendFn, FitState, FitParam


def compute_twist(
    g_fn: TwistFn,
    t: float,
    *,
    x_joint_indices_0b: Sequence[int],
    seglen: Sequence[float],
    state: FitState,
) -> np.ndarray:
    """Compute per-x-joint twist. Returns twist values without base_twist (for FK/geometry use).
    """
    ...


_SAMPLE_NUMBER = 200
_HINT_RADIUS = 10


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
    last_UP: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute transform to roughly align the curve with the robot's head seg.
    fplist: fplist[0]=[0,0,0] ensured
    Returns (ftrans, new_UP).
    """
    # get scale to match the length
    diffs = np.diff(fplist, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    scale = totlen / seg_lengths.sum()

    # get tangent around s=0: assume fplist is (N,3) and use it directly.
    pts = np.asarray(fplist, dtype=np.float64)
    base = pts[0]
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    for j in range(1, pts.shape[0]):
        dvec = pts[j] - base
        dist = float(np.linalg.norm(dvec))
        if dist > 1e-9:
            x_axis = dvec / dist
            break

    # get binormal around s=0
    j_found = None
    for j in range(1, pts.shape[0]):
        dvec = pts[j] - base
        if float(np.linalg.norm(dvec)) > 1e-9:
            j_found = j
            break

    binorm_proj = None
    if j_found is not None:
        v0j = pts[j_found] - base
        # find k
        k_found = None
        for k in range(j_found + 1, pts.shape[0]):
            vjk = pts[k] - pts[j_found]
            if float(np.linalg.norm(vjk)) <= 1e-9:
                continue
            # check collinearity of v0j and vjk via cross product norm
            cross = np.cross(v0j, vjk)
            if float(np.linalg.norm(cross)) > 1e-9:
                k_found = k
                break

        if k_found is not None:
            vjk = pts[k_found] - pts[j_found]
            binormal = np.cross(v0j, vjk)
            # project binormal onto plane orthogonal to x_axis
            binormal = np.asarray(binormal, dtype=np.float64)
            proj = binormal - (float(np.dot(binormal, x_axis))) * x_axis
            proj_norm = float(np.linalg.norm(proj))
            if proj_norm > 1e-9:
                binorm_proj = proj / proj_norm

    # fallback: project last_UP onto plane orthogonal to x_axis
    if binorm_proj is None:
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

    ftrans = float(scale) * np.column_stack([x_axis, y_axis, z_axis])
    new_UP = cur_UP
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
) -> None:
    twist_no_base = compute_twist(
        g_fn=g_fn,
        t=t,
        x_joint_indices_0b=x_joint_indices_0b,
        seglen=seglen,
        state=state,
    )

    fplist = get_fsample(f_fn, t, _SAMPLE_NUMBER)
    # ftrans is a 3*3 transform
    ftrans, state.last_UP = get_ftrans(fplist, totlen, joint_axes[0], state.last_UP)
    fplist = (ftrans @ fplist.T).T
    # Hint indices for closest curve point
    hint_i = get_hint(
        fplist=fplist,
        jn=n_joints,
        seglen=seglen,
        state=state,
    )
    backend_param = FitParam(
        jn=n_joints,
        fplist=fplist,
        hint_i=hint_i,
        twist_no_base=twist_no_base,
        joint_axes=joint_axes,
        joint_signs=joint_signs,
        seglen=seglen,
        hint_rad=_HINT_RADIUS,
    )
    v = backend_fn(
        state.joint_tar,
        backend_param,
    )
    state.joint_tar[:] = v[:]
