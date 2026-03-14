"""Backward-compatible facade for shape fitting modules.

The implementation is split into:
- shapefit_types.py: dataclasses and public type aliases
- shapefit_geometry.py: FK and geometric helpers
- shapefit_solver.py: shape solver entrypoint
- jacob_backend.py: least-squares backend and Jacobian implementations (finite differences)
- anneal_backend.py: annealing backend placeholder with same interface
"""

from __future__ import annotations

from shapefit_solver import solve_shape_for_time
from shapefit_types import CurveFn, DebugInfo, FitState, TwistFn, create_initial_state

__all__ = [
    "CurveFn",
    "TwistFn",
    "DebugInfo",
    "FitState",
    "create_initial_state",
    "solve_shape_for_time",
]
