from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from common import RESOLUTIONS
from spec import N_JOINTS


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    required = [
        "xml",
        "output",
        "fps",
        "resolution",
        "sim_timestep",
        "camera_distance",
        "camera_elevation_deg",
        "ramp_duration_s",
        "hold_duration_s",
        "render_duration_s",
        "fk_separate_offset",
        "target_angles_rad",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing config keys: {missing}")

    if int(cfg["fps"]) <= 0:
        raise ValueError("fps must be > 0")
    if float(cfg["sim_timestep"]) <= 0.0:
        raise ValueError("sim_timestep must be > 0")

    if not isinstance(cfg["target_angles_rad"], list) or len(cfg["target_angles_rad"]) != N_JOINTS:
        raise ValueError(f"target_angles_rad must be a list of {N_JOINTS} values")

    fk_offset = cfg["fk_separate_offset"]
    if not isinstance(fk_offset, list) or len(fk_offset) != 3:
        raise ValueError("fk_separate_offset must be [x, y, z]")

    return cfg


def parse_resolution(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        if value not in RESOLUTIONS:
            raise ValueError("Unsupported string resolution, use 480p/720p/1080p or [w,h]")
        return RESOLUTIONS[value]

    if isinstance(value, list) and len(value) == 2:
        w = int(value[0])
        h = int(value[1])
        if w <= 0 or h <= 0:
            raise ValueError("resolution list values must be positive")
        return w, h

    raise ValueError("resolution must be a string preset or [w, h]")
