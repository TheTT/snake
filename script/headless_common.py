"""Shared constants and tiny utilities for headless rendering."""

from __future__ import annotations

import sys
from pathlib import Path
import time


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

_PP_START: float | None = None


def print_progress(current: int, total: int, *, width: int = 32) -> None:
    """Render an in-place terminal progress bar."""
    if total <= 0 or current <= 0:
        return

    global _PP_START
    if _PP_START is None:
        _PP_START = time.time()

    def _fmt_seconds(s: float) -> str:
        s = int(max(0, round(s)))
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        if h > 0:
            return f"{h}:{m:02d}:{sec:02d}"
        else:
            return f"{m}:{sec:02d}"

    ratio = current / total
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    msg = f"\rRendering video: [{bar}] {current}/{total} ({ratio * 100:5.1f}%)"
    elapsed = time.time() - _PP_START
    est_total = elapsed * (float(total) / float(current)) if current > 0 else 0.0
    msg = msg + f" | {_fmt_seconds(elapsed)}/{_fmt_seconds(est_total)}"
    print(msg, end="", file=sys.stdout, flush=True)


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path)
