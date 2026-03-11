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

    cfg.setdefault("view_fov_scale", 1.0)

    # Backwards-compatible support: allow either top-level fsm_* keys or
    # a nested `fsm` dict. Normalize into cfg["fsm"].
    fsm_block = cfg.get("fsm")
    if fsm_block is None:
        # Collect legacy top-level keys if present, else apply defaults.
        fsm_block = {
            "split_x_ratio": float(cfg.get("fsm_split_x_ratio", 0.5)),
            "split_y_ratio": float(cfg.get("fsm_split_y_ratio", 0.5)),
            "show_f_curve_overlay": bool(cfg.get("fsm_show_f_curve_overlay", False)),
            "show_local_axes": bool(cfg.get("fsm_show_local_axes", False)),
            "show_shell_overlay": bool(cfg.get("fsm_show_shell_overlay", False)),
        }
        cfg["fsm"] = fsm_block
    else:
        # Ensure keys exist with defaults when provided as block.
        fsm_block.setdefault("split_x_ratio", 0.5)
        fsm_block.setdefault("split_y_ratio", 0.5)
        fsm_block.setdefault("show_f_curve_overlay", False)
        fsm_block.setdefault("show_local_axes", False)
        fsm_block.setdefault("show_shell_overlay", False)

    if float(cfg["view_fov_scale"]) <= 0.0:
        raise ValueError("view_fov_scale must be > 0")

    split_x = float(cfg["fsm"]["split_x_ratio"])
    split_y = float(cfg["fsm"]["split_y_ratio"])
    if not (0.0 < split_x < 1.0):
        raise ValueError("fsm_split_x_ratio must be in (0, 1)")
    if not (0.0 < split_y < 1.0):
        raise ValueError("fsm_split_y_ratio must be in (0, 1)")

    if not isinstance(cfg["fsm"]["show_f_curve_overlay"], bool):
        raise ValueError("fsm.show_f_curve_overlay must be a boolean")
    if not isinstance(cfg["fsm"]["show_local_axes"], bool):
        raise ValueError("fsm.show_local_axes must be a boolean")
    if not isinstance(cfg["fsm"]["show_shell_overlay"], bool):
        raise ValueError("fsm.show_shell_overlay must be a boolean")

    return cfg
