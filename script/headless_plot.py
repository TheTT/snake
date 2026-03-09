"""Plot helpers for headless rendering outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def plot_joint_angles_mod3(
    times: Sequence[float],
    joint_angles: Sequence[Sequence[float]],
    output_path: Path,
    segment_colors: Sequence[np.ndarray],
) -> None:
    """Plot joint angle trajectories with colors by joint-index mod 3."""
    if not times or not joint_angles:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skip angle plot: matplotlib import failed ({exc})")
        return

    times_arr = np.asarray(times, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    for idx, angles in enumerate(joint_angles):
        if not angles:
            continue
        y = np.asarray(angles, dtype=np.float64)
        c_rgba = np.asarray(segment_colors[idx % len(segment_colors)], dtype=np.float64)
        c_rgb = tuple(np.clip(c_rgba[:3], 0.0, 1.0).tolist())
        ax.plot(times_arr, y, color=c_rgb, linewidth=1.0, alpha=0.95)

    ax.set_xlim(float(times_arr[0]), float(times_arr[-1]))
    ax.set_xlabel("time (s)", color="white")
    ax.set_ylabel("angle (rad)", color="white")
    ax.set_title("Joint Angles Colored by Joint Index mod 3", color="white")

    for spine in ax.spines.values():
        spine.set_color("white")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    ax.grid(color="#333333", linewidth=0.6, alpha=0.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Angle plot saved: {output_path}")
