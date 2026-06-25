"""Generate a synthetic test video with lightning-like flashes at known frames."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import av
import numpy as np


@dataclass
class ClipManifest:
    path: str
    fps: float
    width: int
    height: int
    n_frames: int
    flash_groups: list[list[int]]
    flash_peaks: list[int]
    drift_range: tuple[int, int]


def generate_test_clip(out_path, fps=30, width=320, height=180, n_frames=300):
    """Write an H.264 mp4 with injected flashes. Returns a ClipManifest.

    Injected content:
      - frame 50:        single-frame full flash
      - frames 120..124: multi-frame flicker, peak at 122 (one strike)
      - frame 180:       ~12px bright vertical bolt (small bright area)
      - frames 220..280: slow brightness drift (decoy; must NOT be detected)
    """
    flash_groups = [[50], [120, 121, 122, 123, 124], [180]]
    flash_peaks = [50, 122, 180]
    drift_range = (220, 280)

    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator())
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18"}

    for i in range(n_frames):
        img = np.full((height, width, 3), 8, dtype=np.uint8)
        if i == 50:
            img[:] = 235
        elif 120 <= i <= 124:
            img[:] = [120, 200, 250, 190, 110][i - 120]
        elif i == 180:
            x0 = width // 2
            img[:, x0 - 6:x0 + 6] = 255
        elif drift_range[0] <= i <= drift_range[1]:
            frac = (i - drift_range[0]) / (drift_range[1] - drift_range[0])
            img[:] = int(8 + 60 * frac)

        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    return ClipManifest(
        path=str(out_path), fps=float(fps), width=width, height=height,
        n_frames=n_frames, flash_groups=flash_groups, flash_peaks=flash_peaks,
        drift_range=drift_range,
    )


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "test_clip.mp4"
    m = generate_test_clip(out)
    print(f"Wrote {m.path}: {m.n_frames} frames @ {m.fps}fps, peaks {m.flash_peaks}")
