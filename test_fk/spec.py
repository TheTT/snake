from __future__ import annotations

from typing import List

from fwd_kine import Axis

N_JOINTS = 18

SEGMENT_LENGTHS_M: List[float] = [
    0.083,
    0.049, 0.0805, 0.076,
    0.049, 0.0805, 0.076,
    0.049, 0.0805, 0.076,
    0.049, 0.0805, 0.076,
    0.049, 0.0805, 0.076,
    0.049, 0.0805,
    0.113,
]

JOINT_AXES: List[Axis] = [
    Axis.Z, Axis.X, Axis.Y,
    Axis.Z, Axis.X, Axis.Y,
    Axis.Z, Axis.X, Axis.Y,
    Axis.Z, Axis.X, Axis.Y,
    Axis.Z, Axis.X, Axis.Y,
    Axis.Z, Axis.X, Axis.Z,
]

JOINT_SIGNS: List[float] = [
    +1, -1, +1,
    +1, -1, +1,
    +1, -1, +1,
    +1, -1, +1,
    +1, -1, +1,
    +1, -1, +1,
]

if len(SEGMENT_LENGTHS_M) != N_JOINTS + 1:
    raise ValueError("SEGMENT_LENGTHS_M length must be N_JOINTS+1")
if len(JOINT_AXES) != N_JOINTS:
    raise ValueError("JOINT_AXES length must be N_JOINTS")
if len(JOINT_SIGNS) != N_JOINTS:
    raise ValueError("JOINT_SIGNS length must be N_JOINTS")
