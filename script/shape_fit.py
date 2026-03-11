"""Backward-compatible facade for shape fitting modules.

The implementation is split into:
- shapefit_types.py: dataclasses and public type aliases
- shapefit_geometry.py: FK and geometric helpers
- shapefit_target.py: target-curve and twist integration helpers
- shapefit_solver.py: shape solver entrypoint
- jacob_backend.py: least-squares backend and Jacobian implementations (finite differences)
- anneal_backend.py: annealing backend placeholder with same interface
"""

from __future__ import annotations

from shapefit_geometry import cumulative_s, forward_points_and_frames, smoothstep01, x_joint_s_intervals
from shapefit_solver import solve_shape_for_time
from shapefit_types import CurveFn, FitConfig, FitState, TwistFn, Vec3, create_initial_state

__all__ = [
    "Vec3",
    "CurveFn",
    "TwistFn",
    "FitConfig",
    "FitState",
    "create_initial_state",
    "solve_shape_for_time",
    "smoothstep01",
    "cumulative_s",
    "x_joint_s_intervals",
    "forward_points_and_frames",
]
