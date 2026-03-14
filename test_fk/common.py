from __future__ import annotations

from pathlib import Path

import numpy as np

RESOLUTIONS = {
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}


def print_progress(current: int, total: int, width: int = 36) -> None:
    if total <= 0:
        return
    ratio = float(current) / float(total)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\rRendering: [{bar}] {current}/{total} ({100.0 * ratio:5.1f}%)", end="", flush=True)


def resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def compose_lr(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h = int(left.shape[0])
    w = int(left.shape[1])
    out = np.zeros((h, w * 2, 3), dtype=left.dtype)
    out[:, :w] = left
    out[:, w:] = right
    return out
