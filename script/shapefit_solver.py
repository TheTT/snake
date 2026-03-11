from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np

import jacob_backend
import linear_backend
import anneal_backend
import acf_backend
import step_backend
from shapefit_geometry import (
    _rot_axis,
    _rpy_to_mat3,
    assemble_joint_angles,
    forward_points_and_frames,
    segment_midpoint_s,
    smoothstep01,
    x_joint_s_intervals,
)
from shapefit_target import build_target_points, integrate_avg_g, target_point
from shapefit_types import CurveFn, FitConfig, FitState, TwistFn, Backend


def _build_fk_callbacks() -> dict[str, Callable[..., Any]]:
    cache: dict[str, Any] = {
        "initialized": False,
    }

    def fk_init(
        head_translation: Sequence[float] | np.ndarray,
        head_rpy: Sequence[float] | np.ndarray,
        joint_angles_base_x: Sequence[float] | np.ndarray,
        yz_init: Sequence[float] | np.ndarray,
        lengths_m: Sequence[float] | np.ndarray,
        joint_axes: Sequence[str],
    ) -> None:
        lengths = np.asarray(lengths_m, dtype=np.float64)
        n_joints = len(joint_axes)
        if lengths.size != n_joints + 1:
            raise ValueError("lengths_m length must be n_joints + 1")

        angles = np.asarray(joint_angles_base_x, dtype=np.float64).copy()
        if angles.size != n_joints:
            raise ValueError("joint_angles_base_x length must match joint count")
        yz = np.asarray(yz_init, dtype=np.float64)
        if yz.size == n_joints:
            angles[:] = yz
        elif yz.size > n_joints:
            raise ValueError("yz_init length cannot exceed joint count")
        elif yz.size > 0:
            angles[: yz.size] = yz

        points = np.zeros((n_joints + 2, 3), dtype=np.float64)
        frames = np.zeros((n_joints + 1, 3, 3), dtype=np.float64)

        points[0] = np.asarray(head_translation, dtype=np.float64)
        head_rot = _rpy_to_mat3(float(head_rpy[0]), float(head_rpy[1]), float(head_rpy[2]))

        cache.update(
            {
                "initialized": True,
                "n_joints": n_joints,
                "joint_axes": tuple(joint_axes),
                "lengths": lengths,
                "angles": angles,
                "points": points,
                "frames": frames,
                "head_rot": head_rot,
                "valid_upto_segment": -1,
            }
        )

    def fk_ensure_upto(i: int) -> None:
        if not cache.get("initialized", False):
            raise RuntimeError("fk_init must be called before fk_ensure_upto")

        n_joints = int(cache["n_joints"])
        seg_idx = max(0, min(int(i), n_joints))
        valid_upto = int(cache["valid_upto_segment"])
        if seg_idx <= valid_upto:
            return

        points = cache["points"]
        frames = cache["frames"]
        lengths = cache["lengths"]
        angles = cache["angles"]
        joint_axes = cache["joint_axes"]
        head_rot = cache["head_rot"]

        start_seg = valid_upto + 1
        for seg in range(start_seg, min(seg_idx, n_joints - 1) + 1):
            rot_before = head_rot if seg == 0 else frames[seg - 1]
            points[seg + 1] = points[seg] + rot_before @ np.array((float(lengths[seg]), 0.0, 0.0), dtype=np.float64)
            frames[seg] = rot_before @ _rot_axis(joint_axes[seg], float(angles[seg]))

        if seg_idx == n_joints:
            if n_joints > 0 and valid_upto < n_joints:
                tail_rot = frames[n_joints - 1]
                points[n_joints + 1] = points[n_joints] + tail_rot @ np.array(
                    (float(lengths[n_joints]), 0.0, 0.0), dtype=np.float64
                )
                frames[n_joints] = tail_rot

        cache["valid_upto_segment"] = seg_idx

    def fk_invalidate_from(i: int) -> None:
        if not cache.get("initialized", False):
            raise RuntimeError("fk_init must be called before fk_invalidate_from")
        seg_idx = max(0, min(int(i), int(cache["n_joints"])))
        cache["valid_upto_segment"] = min(int(cache["valid_upto_segment"]), seg_idx - 1)

    def fk_apply_delta(i: int, delta_rad: float) -> None:
        if not cache.get("initialized", False):
            raise RuntimeError("fk_init must be called before fk_apply_delta")
        j = int(i)
        n_joints = int(cache["n_joints"])
        if j < 0 or j >= n_joints:
            raise IndexError(f"joint index out of range: {j}")

        cache["angles"][j] += float(delta_rad)
        # Invalidate from j (not j+1): frames[j] depends on angles[j] and must be recomputed.
        fk_invalidate_from(j)

    def fk_get_midpoint(i: int) -> np.ndarray:
        if not cache.get("initialized", False):
            raise RuntimeError("fk_init must be called before fk_get_midpoint")

        seg_idx = int(i)
        n_segs = int(cache["n_joints"]) + 1
        if seg_idx < 0 or seg_idx >= n_segs:
            raise IndexError(f"segment index out of range: {seg_idx}")

        fk_ensure_upto(seg_idx)
        points = cache["points"]
        return 0.5 * (points[seg_idx] + points[seg_idx + 1])

    return {
        "fk_init": fk_init,
        "fk_apply_delta": fk_apply_delta,
        "fk_get_midpoint": fk_get_midpoint,
        "fk_invalidate_from": fk_invalidate_from,
        "fk_ensure_upto": fk_ensure_upto,
    }


def solve_shape_for_time(
    *,
    f_fn: CurveFn,
    g_fn: TwistFn,
    t: float,
    joint_axes: Sequence[str],
    joint_signs: Sequence[float],
    x_joint_indices_0b: Sequence[int],
    yz_joint_indices_0b: Sequence[int],
    lengths_m: Sequence[float],
    base_twist_rad: float,
    state: FitState,
    cfg: FitConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_joints = len(joint_axes)

    c1 = 1.0
    if t < cfg.startup_ramp_sec:
        c1 = smoothstep01(max(0.0, t) / cfg.startup_ramp_sec)

    intervals = x_joint_s_intervals(x_joint_indices_0b, lengths_m)
    twist_raw = np.array(
        [
            integrate_avg_g(g_fn, max(0.0, t), float(lo), float(hi), cfg.integration_samples)
            for lo, hi in intervals
        ],
        dtype=np.float64,
    )
    twist_target = base_twist_rad + c1 * twist_raw

    dt = 0.0 if state.last_t is None else max(0.0, float(t - state.last_t))
    if dt <= 0.0:
        alpha = 0.0
    else:
        alpha = math.exp(-dt / max(cfg.lowpass_tau_sec, 1e-6))
    state.twist_filtered = alpha * state.twist_filtered + (1.0 - alpha) * twist_target
    state.last_t = float(t)

    seg_mid_s = segment_midpoint_s(lengths_m)
    tgt = build_target_points(f_fn, lengths_m, t, seg_mid_s, cfg.startup_ramp_sec)
    fk_callbacks = _build_fk_callbacks()

    def residual(v: np.ndarray) -> np.ndarray:
        yz_vars = v[: len(yz_joint_indices_0b)]
        head = v[len(yz_joint_indices_0b):]
        head_translation = np.asarray(head[:3], dtype=np.float64)
        head_rpy = np.asarray(head[3:], dtype=np.float64)

        joint_angles = assemble_joint_angles(
            x_joint_indices_0b,
            yz_joint_indices_0b,
            joint_signs,
            state.twist_filtered,
            yz_vars,
            n_joints,
        )
        points, _frames = forward_points_and_frames(
            joint_angles=np.asarray(joint_angles, dtype=np.float64),
            joint_axes=joint_axes,
            lengths_m=lengths_m,
            head_translation=head_translation,
            head_rpy=head_rpy,
        )
        mids = 0.5 * (points[:-1] + points[1:])
        return (mids - tgt).reshape(-1)

    # Swap this backend import to anneal_backend.solve_with_anneal_placeholder
    # without touching residual construction logic.
    precomp = {
        "lengths_m": np.asarray(lengths_m, dtype=np.float64),
        "seg_mid_s": np.asarray(seg_mid_s, dtype=np.float64),
        "tgt": np.asarray(tgt, dtype=np.float64),
        # curve_samples: Nx3 points sampled uniformly by arc-length (meters)
        # Prefer uniform arc-length sampling for downstream backends.
        # Default to 200 samples; can be adjusted later if needed.
        # s_m: absolute arc-length positions in meters
        "curve_samples_n": 200,
        "curve_samples": None,
        "curve_samples_arc": None,
        "align_func": linear_backend._align_targets,
        "twist_filtered": np.asarray(state.twist_filtered, dtype=np.float64).copy(),
        "joint_axes": tuple(joint_axes),
        "joint_signs": np.asarray(joint_signs, dtype=np.float64),
        "x_joint_indices_0b": tuple(int(i) for i in x_joint_indices_0b),
        "yz_joint_indices_0b": tuple(int(i) for i in yz_joint_indices_0b),
        "fk_callbacks": fk_callbacks,
    }

    backend_map = {
        Backend.JACOB: jacob_backend.solve_with_jacob_least_squares,
        Backend.LINEAR: linear_backend.solve_with_linear_placeholder,
        Backend.ANNEAL: anneal_backend.solve_with_anneal_placeholder,
        Backend.ACF: acf_backend.solve_with_acf_placeholder,
        Backend.STEP: step_backend.solve_with_step_placeholder,
    }

    backend_fn = backend_map.get(cfg.backend, jacob_backend.solve_with_jacob_least_squares)
    # Build uniform arc-length curve samples (in meters) for precomp.
    total_length = float(np.sum(lengths_m))
    n_samples = int(precomp.get("curve_samples_n", 200))

    s_m = np.linspace(0.0, total_length, max(2, int(n_samples)), dtype=np.float64)
    s_norm = s_m / max(total_length, 1e-12)
    curve_pts = np.array([target_point(f_fn, lengths_m, t, float(s), cfg.startup_ramp_sec) for s in s_norm], dtype=np.float64)
    # update precomp with concrete samples
    precomp["curve_samples"] = curve_pts
    precomp["curve_samples_arc"] = np.column_stack((s_m, curve_pts))

    v = backend_fn(
        residual_fn=residual,
        v0=state.yz_and_head,
        cfg=cfg,
        precomp=precomp,
    )

    state.yz_and_head = v

    yz_vars = v[: len(yz_joint_indices_0b)]
    head = v[len(yz_joint_indices_0b):]
    joint_angles = assemble_joint_angles(
        x_joint_indices_0b,
        yz_joint_indices_0b,
        joint_signs,
        state.twist_filtered,
        yz_vars,
        n_joints,
    )
    points, frames = forward_points_and_frames(
        joint_angles=np.asarray(joint_angles, dtype=np.float64),
        joint_axes=joint_axes,
        lengths_m=lengths_m,
        head_translation=np.asarray(head[:3], dtype=np.float64),
        head_rpy=np.asarray(head[3:], dtype=np.float64),
    )
    return joint_angles, points, frames, head
