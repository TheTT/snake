"""Shared constants and tiny utilities for headless rendering."""

from __future__ import annotations

import sys
from pathlib import Path


RESOLUTIONS = {
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2160p": (3840, 2160),
}

SCRIPT_DIR = Path(__file__).resolve().parent
HEADLESS_CONFIG_PATH = SCRIPT_DIR / "headless.json"
RES_DIR = (SCRIPT_DIR / "../res").resolve()


def print_progress(current: int, total: int, *, width: int = 32) -> None:
    """Render an in-place terminal progress bar."""
    if total <= 0:
        return

    current = max(0, min(current, total))
    if current == 0:
        return
    ratio = current / total
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    msg = f"\rRendering video: [{bar}] {current}/{total} ({ratio * 100:5.1f}%)"
    print(msg, end="", file=sys.stdout, flush=True)


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path)
