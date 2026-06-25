"""Frame-accurate video access: metadata, iteration, exact native-res frames."""
from __future__ import annotations

import io
import threading
from dataclasses import dataclass

import av
import numpy as np
from PIL import Image


@dataclass
class VideoMeta:
    fps: float
    frame_count: int
    duration: float
    width: int
    height: int
    vfr: bool


class VideoFile:
    def __init__(self, path):
        self.path = str(path)
        self._meta: VideoMeta | None = None
        self._pts: list[int | None] = []
        self._indexed = False
        self._index_lock = threading.Lock()

    @property
    def meta(self) -> VideoMeta:
        if self._meta is None:
            with av.open(self.path) as c:
                s = c.streams.video[0]
                fps = float(s.average_rate) if s.average_rate else 0.0
                frames = s.frames or 0
                if s.duration is not None and s.time_base is not None:
                    duration = float(s.duration * s.time_base)
                elif c.duration is not None:
                    duration = c.duration / av.time_base
                else:
                    duration = (frames / fps) if fps else 0.0
                if not frames and fps and duration:
                    frames = int(round(duration * fps))
                self._meta = VideoMeta(
                    fps=fps, frame_count=frames, duration=duration,
                    width=s.codec_context.width, height=s.codec_context.height,
                    vfr=False,
                )
        return self._meta

    def iter_frames(self, downscale_height=None):
        """Yield (index, time_seconds, rgb_ndarray) for every frame, in order.

        Records pts per index so frame() can be exact afterward. If
        downscale_height is set, frames are scaled down for cheap analysis.
        """
        pts_local = []
        with av.open(self.path) as c:
            s = c.streams.video[0]
            tb = s.time_base
            idx = 0
            for frame in c.decode(s):
                pts_local.append(int(frame.pts) if frame.pts is not None else None)
                if frame.pts is not None and tb is not None:
                    t = float(frame.pts * tb)
                else:
                    t = idx / (self.meta.fps or 30.0)
                if downscale_height:
                    w = max(1, int(frame.width * downscale_height / frame.height))
                    img = frame.reformat(width=w, height=downscale_height,
                                         format="rgb24").to_ndarray()
                else:
                    img = frame.to_ndarray(format="rgb24")
                yield idx, t, img
                idx += 1
        with self._index_lock:
            self._pts = pts_local
            self._indexed = True
            if self._meta is not None and idx and idx != self._meta.frame_count:
                self._meta.frame_count = idx

    def frame(self, index) -> np.ndarray:
        """Return the exact frame at `index` as native-res RGB uint8 ndarray.

        Fast by design. If a full PTS index already exists (built by a prior
        scan via iter_frames), seek by the recorded PTS for exactness even on
        variable-frame-rate footage. Otherwise seek by estimated time
        (index/fps) and decode forward to the target frame: frame-accurate for
        constant-frame-rate video without a full decode pass. We never build the
        whole index here — doing so would stall the first precision-mode frame
        for minutes on a long 4K video whose events were loaded from cache.
        """
        fc = self.meta.frame_count
        if fc and not (0 <= index < fc):
            raise IndexError(f"frame {index} out of range (0..{fc - 1})")

        with av.open(self.path) as c:
            s = c.streams.video[0]
            tb = s.time_base

            # Exact path: the per-index PTS is already known (a scan ran).
            if (self._indexed and 0 <= index < len(self._pts)
                    and self._pts[index] is not None):
                target = self._pts[index]
                c.seek(target, stream=s, any_frame=False, backward=True)
                for frame in c.decode(s):
                    if frame.pts is None:
                        continue
                    if frame.pts >= target:
                        return frame.to_ndarray(format="rgb24")
                raise IndexError(f"frame {index} not found")

            # Time-based path: seek near index/fps and decode forward. No full
            # index build, so the first frame request is fast even on long 4K.
            fps = self.meta.fps or 30.0
            target_t = index / fps
            seek_pts = int(target_t / tb) if tb else 0
            c.seek(seek_pts, stream=s, any_frame=False, backward=True)
            best = None
            for frame in c.decode(s):
                best = frame
                ft = float(frame.pts * tb) if (frame.pts is not None and tb) else 0.0
                if ft >= target_t - 1e-6:
                    return frame.to_ndarray(format="rgb24")
            if best is not None:
                return best.to_ndarray(format="rgb24")
        raise IndexError(f"frame {index} not found")

    def frame_png_bytes(self, index) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(self.frame(index)).save(buf, format="PNG")
        return buf.getvalue()

    def thumb_jpeg_bytes(self, index, height=120) -> bytes:
        im = Image.fromarray(self.frame(index))
        w = max(1, int(im.width * height / im.height))
        im = im.resize((w, height), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=80)
        return buf.getvalue()
