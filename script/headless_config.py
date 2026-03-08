"""Configuration loading and validation for headless rendering."""

from __future__ import annotations

import json
from typing import Any, Dict

from headless_common import HEADLESS_CONFIG_PATH, RESOLUTIONS


def load_config() -> Dict[str, Any]:
    """Load headless render config from script/headless.json."""
    with HEADLESS_CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = json.load(f)

    required = [
        "xml",
        "duration",
        "settle",
        "fps",
        "resolution",
        "camera_distance",
        "camera_azimuth",
        "camera_elevation",
        "sim_timestep",
        "video_speed",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing keys in {HEADLESS_CONFIG_PATH}: {missing}")

    if cfg["resolution"] not in RESOLUTIONS:
        raise ValueError(f"Invalid resolution: {cfg['resolution']}")

    if float(cfg["sim_timestep"]) <= 0.0:
        raise ValueError("sim_timestep must be > 0")
    if float(cfg["video_speed"]) <= 0.0:
        raise ValueError("video_speed must be > 0")
    if int(cfg["fps"]) <= 0:
        raise ValueError("fps must be > 0")

    cfg.setdefault("free_space_mode", False)
    cfg.setdefault("view_fov_scale", 1.0)
    cfg.setdefault("fsm_split_x_ratio", 0.5)
    cfg.setdefault("fsm_split_y_ratio", 0.5)

    if float(cfg["view_fov_scale"]) <= 0.0:
        raise ValueError("view_fov_scale must be > 0")

    split_x = float(cfg["fsm_split_x_ratio"])
    split_y = float(cfg["fsm_split_y_ratio"])
    if not (0.0 < split_x < 1.0):
        raise ValueError("fsm_split_x_ratio must be in (0, 1)")
    if not (0.0 < split_y < 1.0):
        raise ValueError("fsm_split_y_ratio must be in (0, 1)")

    return cfg
