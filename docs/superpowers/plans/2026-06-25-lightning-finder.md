# Lightning Finder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local tool that scans a video for lightning flashes, lets the user review each flash in a browser, step frame-by-frame to the exact peak, and extract that frame at full native quality — plus export a timestamps file.

**Architecture:** A Python FastAPI backend does brightness-spike detection and decodes exact frames from the file (PyAV, frame-accurate); a dependency-free browser frontend is the scrubber. The frontend uses native `<video>` for fast scrubbing and switches to backend-decoded exact frames in "precision mode" so what you see is what you grab, losslessly.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, PyAV (`av`), NumPy, Pillow; vanilla HTML/CSS/JS frontend; pytest + httpx for tests.

## Global Constraints

- Python **3.10+** (plan uses `X | None` type syntax).
- Dependencies, exact: `fastapi>=0.110`, `uvicorn>=0.29`, `av>=12.0`, `numpy>=1.26`, `Pillow>=10.0`, `pytest>=8.0`, `httpx>=0.27`.
- **PyAV is the single decoder** for both the brightness scan and exact-frame extraction. Frame indexing must be consistent everywhere.
- Server binds **127.0.0.1 only** (local tool, never exposed).
- **No frontend build step** — plain `.html`/`.css`/`.js` served as static files.
- Frame export default = **PNG, lossless, native resolution**. TIFF optional. Output path: `output/<video_stem>/frame_<index>_<HH-MM-SS.mmm>.png`.
- Timestamp export default = **CSV and JSON**.
- Frame timestamps derive from **PTS** (decoded frame presentation time), not `index × 1/fps`, so VFR footage stays correct.
- All modules live at the project root and are imported flat (`import detector`, `import video`); a root `conftest.py` puts the root on `sys.path` for tests. Run `pytest` from the project root.

---

### Task 1: Project scaffolding + synthetic test clip generator

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `conftest.py`
- Create: `generate_test_clip.py`
- Test: `tests/test_generate_clip.py`

**Interfaces:**
- Produces: `generate_test_clip(out_path, fps=30, width=320, height=180, n_frames=300) -> ClipManifest`. `ClipManifest` is a dataclass with fields `path:str, fps:float, width:int, height:int, n_frames:int, flash_groups:list[list[int]], flash_peaks:list[int], drift_range:tuple[int,int]`. Injected content: full flash at frame 50; flicker frames 120–124 (peak 122); a ~12px-wide bright bolt at frame 180; a slow brightness drift over frames 220–280 (decoy).
- Produces (test fixture): `conftest.py` exposes a session-scoped `clip` fixture returning a `ClipManifest` for a generated `test_clip.mp4`.

- [ ] **Step 1: Create `requirements.txt`**

```
fastapi>=0.110
uvicorn>=0.29
av>=12.0
numpy>=1.26
Pillow>=10.0
pytest>=8.0
httpx>=0.27
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.pyc
.lf_cache_*.json
output/
test_clip.mp4
.pytest_cache/
```

- [ ] **Step 3: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: all packages install; `python -c "import av, fastapi, numpy, PIL"` prints nothing and exits 0.

- [ ] **Step 4: Write the failing test** — `tests/test_generate_clip.py`

```python
import av
import numpy as np


def _brightness(rgb):
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return 0.5 * luma.mean() + 0.5 * np.percentile(luma, 99)


def _read_brightness(path):
    vals = []
    with av.open(path) as c:
        for frame in c.decode(c.streams.video[0]):
            vals.append(_brightness(frame.to_ndarray(format="rgb24")))
    return np.asarray(vals)


def test_manifest_fields(clip):
    assert clip.width == 320 and clip.height == 180
    assert clip.n_frames == 300
    assert clip.flash_peaks == [50, 122, 180]
    assert clip.drift_range == (220, 280)


def test_injected_flashes_are_bright(clip):
    b = _read_brightness(clip.path)
    baseline = np.median(b[:40])
    for peak in clip.flash_peaks:
        assert b[peak] > baseline + 40, f"frame {peak} not bright enough"


def test_drift_is_gradual_not_a_spike(clip):
    b = _read_brightness(clip.path)
    lo, hi = clip.drift_range
    # no single-frame jump inside the drift exceeds 12 brightness units
    diffs = np.abs(np.diff(b[lo:hi + 1]))
    assert diffs.max() < 12, "drift contains a spike it should not"
```

- [ ] **Step 5: Run test to verify it fails**

Run: `pytest tests/test_generate_clip.py -v`
Expected: FAIL — `conftest.py`/`generate_test_clip` do not exist (collection or import error).

- [ ] **Step 6: Create `conftest.py`**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest
from generate_test_clip import generate_test_clip


@pytest.fixture(scope="session")
def clip(tmp_path_factory):
    out = tmp_path_factory.mktemp("clips") / "test_clip.mp4"
    return generate_test_clip(str(out))
```

- [ ] **Step 7: Create `generate_test_clip.py`**

```python
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
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_generate_clip.py -v`
Expected: 3 passed.

- [ ] **Step 9: Commit**

```bash
git add requirements.txt .gitignore conftest.py generate_test_clip.py tests/test_generate_clip.py
git commit -m "feat: scaffolding + synthetic lightning test clip generator"
```

---

### Task 2: `video.py` — metadata, frame iteration, exact native-res frames

**Files:**
- Create: `video.py`
- Test: `tests/test_extraction.py`

**Interfaces:**
- Consumes: a video file path (e.g. the test clip from Task 1).
- Produces:
  - `VideoMeta` dataclass: `fps:float, frame_count:int, duration:float, width:int, height:int, vfr:bool`.
  - `VideoFile(path)` with: `.meta -> VideoMeta`; `.iter_frames(downscale_height=None)` yielding `(index:int, time:float, rgb:np.ndarray)` and building an internal PTS index; `.frame(index) -> np.ndarray` (native-res RGB uint8, frame-accurate); `.frame_png_bytes(index) -> bytes`; `.thumb_jpeg_bytes(index, height=120) -> bytes`.

- [ ] **Step 1: Write the failing test** — `tests/test_extraction.py`

```python
import io

import numpy as np
from PIL import Image

from video import VideoFile


def _mean(rgb):
    return float(rgb.mean())


def test_meta(clip):
    v = VideoFile(clip.path)
    m = v.meta
    assert m.width == 320 and m.height == 180
    assert abs(m.fps - clip.fps) < 0.5
    assert abs(m.frame_count - clip.n_frames) <= 1


def test_frame_native_resolution(clip):
    v = VideoFile(clip.path)
    f = v.frame(50)
    assert f.shape == (180, 320, 3)
    assert f.dtype == np.uint8


def test_dark_and_bright_frames(clip):
    v = VideoFile(clip.path)
    assert _mean(v.frame(0)) < 40
    assert _mean(v.frame(50)) > 150


def test_frame_accuracy_neighbors(clip):
    v = VideoFile(clip.path)
    # frame 122 is the flicker peak (250); 119 is dark pre-flicker (8)
    assert _mean(v.frame(122)) > _mean(v.frame(119)) + 50
    # peak 122 (250) brighter than 123 (190)
    assert _mean(v.frame(122)) > _mean(v.frame(123))


def test_frame_png_bytes(clip):
    v = VideoFile(clip.path)
    data = v.frame_png_bytes(50)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    im = Image.open(io.BytesIO(data))
    assert im.size == (320, 180)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_extraction.py -v`
Expected: FAIL — `No module named 'video'`.

- [ ] **Step 3: Create `video.py`**

```python
"""Frame-accurate video access: metadata, iteration, exact native-res frames."""
from __future__ import annotations

import io
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
        self._pts = []
        with av.open(self.path) as c:
            s = c.streams.video[0]
            tb = s.time_base
            idx = 0
            for frame in c.decode(s):
                self._pts.append(int(frame.pts) if frame.pts is not None else None)
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
        self._indexed = True
        if self._meta is not None and idx and idx != self._meta.frame_count:
            self._meta.frame_count = idx

    def frame(self, index) -> np.ndarray:
        """Return the exact frame at `index` as native-res RGB uint8 ndarray."""
        with av.open(self.path) as c:
            s = c.streams.video[0]
            tb = s.time_base
            if self._indexed and 0 <= index < len(self._pts) and self._pts[index] is not None:
                target = self._pts[index]
                c.seek(target, stream=s, any_frame=False, backward=True)
                for frame in c.decode(s):
                    if frame.pts is not None and frame.pts >= target:
                        return frame.to_ndarray(format="rgb24")
            else:
                fps = self.meta.fps or 30.0
                target_t = index / fps
                target = int(target_t / tb) if tb else 0
                c.seek(target, stream=s, any_frame=False, backward=True)
                best = None
                for frame in c.decode(s):
                    best = frame
                    ft = float(frame.pts * tb) if (frame.pts is not None and tb) else 0.0
                    if ft >= target_t:
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
        im = im.resize((w, height))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=80)
        return buf.getvalue()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_extraction.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add video.py tests/test_extraction.py
git commit -m "feat: frame-accurate video access and native-res frame extraction"
```

---

### Task 3: `detector.py` — brightness scan, flash detection, caching

**Files:**
- Create: `detector.py`
- Test: `tests/test_detector.py`

**Interfaces:**
- Consumes: `VideoFile` (Task 2) for `.iter_frames()` and `.meta.fps`.
- Produces:
  - `FlashEvent` dataclass: `start_frame:int, end_frame:int, peak_frame:int, peak_time:float, brightness:float`.
  - `frame_brightness(rgb) -> float`.
  - `scan_brightness(video, downscale_height=120, progress=None) -> (np.ndarray brightness, np.ndarray times)`.
  - `detect_flashes(brightness, times, fps, sensitivity=0.5, baseline_window=15, merge_gap=3, min_delta=12.0) -> list[FlashEvent]`.
  - `load_events(path, params) -> list[FlashEvent] | None`, `save_events(path, params, events)`, `cache_path(path, params)`.

- [ ] **Step 1: Write the failing test** — `tests/test_detector.py`

```python
from video import VideoFile
import detector


def _events(clip, sensitivity=0.5):
    v = VideoFile(clip.path)
    b, t = detector.scan_brightness(v)
    return detector.detect_flashes(b, t, v.meta.fps, sensitivity=sensitivity), b, t


def test_finds_all_three_strikes(clip):
    events, _, _ = _events(clip)
    peaks = sorted(e.peak_frame for e in events)
    for expected in clip.flash_peaks:           # [50, 122, 180]
        assert any(abs(p - expected) <= 1 for p in peaks), f"missed {expected}"


def test_flicker_is_one_event(clip):
    events, _, _ = _events(clip)
    assert len(events) == 3, f"expected 3 events, got {len(events)}"


def test_ignores_slow_drift(clip):
    events, _, _ = _events(clip)
    lo, hi = clip.drift_range
    for e in events:
        assert not (lo <= e.peak_frame <= hi), "drift wrongly detected as flash"


def test_cache_roundtrip(clip):
    events, _, _ = _events(clip)
    params = {"sensitivity": 0.5}
    detector.save_events(clip.path, params, events)
    loaded = detector.load_events(clip.path, params)
    assert loaded is not None
    assert [e.peak_frame for e in loaded] == [e.peak_frame for e in events]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_detector.py -v`
Expected: FAIL — `No module named 'detector'`.

- [ ] **Step 3: Create `detector.py`**

```python
"""Lightning flash detection from per-frame brightness."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class FlashEvent:
    start_frame: int
    end_frame: int
    peak_frame: int
    peak_time: float
    brightness: float


def frame_brightness(rgb: np.ndarray) -> float:
    """Mean luma plus emphasis on the brightest pixels.

    The 99th-percentile term lets a small bright bolt in an otherwise dark
    frame register even though its mean luma is low.
    """
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return float(0.5 * luma.mean() + 0.5 * np.percentile(luma, 99))


def scan_brightness(video, downscale_height=120, progress=None):
    """Return (brightness[n], times[n]) over every frame via one decode pass."""
    vals, times = [], []
    total = video.meta.frame_count or 0
    for index, t, img in video.iter_frames(downscale_height=downscale_height):
        vals.append(frame_brightness(img))
        times.append(t)
        if progress and total:
            progress(index + 1, total)
    return np.asarray(vals, dtype=np.float64), np.asarray(times, dtype=np.float64)


def detect_flashes(brightness, times, fps, sensitivity=0.5,
                   baseline_window=15, merge_gap=3, min_delta=12.0):
    """Detect sudden upward brightness spikes and group them into events.

    A frame is 'lit' when its brightness exceeds a short rolling-median
    baseline of preceding frames by more than max(min_delta, k * spread).
    Keying off a *rise above recent baseline* (not an absolute level) is what
    makes slow drift get ignored. sensitivity in [0,1]; higher = more sensitive.
    """
    n = len(brightness)
    if n == 0:
        return []
    k = 6.0 * (1.0 - sensitivity) + 1.0          # sensitivity 1 -> 1, 0 -> 7
    lit = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - baseline_window)
        window = brightness[lo:i] if i > lo else brightness[lo:i + 1]
        baseline = np.median(window)
        spread = np.median(np.abs(window - baseline)) + 1e-6
        thresh = baseline + max(min_delta, k * spread)
        lit[i] = brightness[i] > thresh

    events = []
    i = 0
    while i < n:
        if not lit[i]:
            i += 1
            continue
        start, end, gap, j = i, i, 0, i + 1
        while j < n:
            if lit[j]:
                end, gap = j, 0
            else:
                gap += 1
                if gap > merge_gap:
                    break
            j += 1
        seg = brightness[start:end + 1]
        peak = start + int(np.argmax(seg))
        events.append(FlashEvent(start, end, peak, float(times[peak]),
                                 float(brightness[peak])))
        i = end + 1
    return events


def _cache_key(path, params):
    st = os.stat(path)
    raw = (f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}|"
           f"{json.dumps(params, sort_keys=True)}")
    return hashlib.sha1(raw.encode()).hexdigest()


def cache_path(path, params):
    folder = os.path.dirname(os.path.abspath(path)) or "."
    return os.path.join(folder, f".lf_cache_{_cache_key(path, params)}.json")


def load_events(path, params):
    cp = cache_path(path, params)
    if os.path.exists(cp):
        with open(cp) as f:
            return [FlashEvent(**e) for e in json.load(f)]
    return None


def save_events(path, params, events):
    with open(cache_path(path, params), "w") as f:
        json.dump([asdict(e) for e in events], f)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_detector.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: all tests from Tasks 1–3 pass (12 total).

- [ ] **Step 6: Commit**

```bash
git add detector.py tests/test_detector.py
git commit -m "feat: lightning flash detection with rolling-baseline spikes + caching"
```

---

### Task 4: `app.py` — FastAPI backend (meta, events, video range, frame, grab, export, markers)

**Files:**
- Create: `app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `VideoFile` (Task 2), `detector` (Task 3).
- Produces:
  - `create_app(video_path=None) -> FastAPI` (falls back to `LF_VIDEO` env var).
  - Routes: `GET /` (HTML), `GET /static/*`, `GET /api/video/meta`, `GET /api/events`, `POST /api/scan {sensitivity}`, `GET /video` (range), `GET /api/frame/{index}.png`, `GET /api/thumb/{index}.jpg`, `POST /api/grab {index, fmt}`, `GET/POST /api/markers`, `GET /api/export?fmt=csv|json`.
  - Event JSON shape: `{start_frame, end_frame, peak_frame, peak_time, timecode, brightness, label, status}`.
- Note: `GET /` reads `static/index.html`, which is created in Task 5. For this task's tests, create a minimal placeholder `static/index.html` containing the string `Lightning Finder` so `GET /` returns 200; Task 5 replaces it.

- [ ] **Step 1: Create placeholder `static/index.html`**

```html
<!doctype html><html><head><meta charset="utf-8"><title>Lightning Finder</title></head>
<body>Lightning Finder</body></html>
```

- [ ] **Step 2: Write the failing test** — `tests/test_app.py`

```python
import io

from fastapi.testclient import TestClient
from PIL import Image

from app import create_app


def client(clip):
    return TestClient(create_app(clip.path))


def test_meta(clip):
    r = client(clip).get("/api/video/meta")
    assert r.status_code == 200
    assert r.json()["width"] == 320


def test_events(clip):
    r = client(clip).get("/api/events")
    assert r.status_code == 200
    assert len(r.json()["events"]) >= 3


def test_frame_png(clip):
    r = client(clip).get("/api/frame/50.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(r.content)).size == (320, 180)


def test_grab_writes_native_res_file(clip, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = TestClient(create_app(clip.path))
    r = c.post("/api/grab", json={"index": 50})
    assert r.status_code == 200
    import os
    from PIL import Image as I
    path = r.json()["path"]
    assert os.path.exists(path)
    assert I.open(path).size == (320, 180)


def test_video_range(clip):
    r = client(clip).get("/video", headers={"Range": "bytes=0-1023"})
    assert r.status_code == 206
    assert len(r.content) == 1024
    assert "content-range" in {k.lower() for k in r.headers}


def test_export_csv(clip):
    r = client(clip).get("/api/export?fmt=csv")
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    assert "peak_frame" in r.text


def test_markers_roundtrip(clip):
    c = client(clip)
    assert c.post("/api/markers", json={"peak_frame": 50, "label": "bolt",
                                        "status": "confirmed"}).status_code == 200
    assert c.get("/api/markers").json()["50"]["label"] == "bolt"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_app.py -v`
Expected: FAIL — `No module named 'app'`.

- [ ] **Step 4: Create `app.py`**

```python
"""FastAPI backend for Lightning Finder."""
from __future__ import annotations

import csv
import io
import mimetypes
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               Response, StreamingResponse)
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

import detector
from video import VideoFile

STATIC_DIR = Path(__file__).parent / "static"


def _fmt_ts(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


class AppState:
    def __init__(self, video_path):
        self.video = VideoFile(video_path)
        self.video_path = str(video_path)
        self.sensitivity = 0.5
        self.events = []
        self.brightness = None
        self.times = None
        self.markers: dict[int, dict] = {}
        self.output_dir = Path("output") / Path(video_path).stem
        self.lock = threading.Lock()

    def _params(self):
        return {"sensitivity": self.sensitivity}

    def ensure_events(self):
        with self.lock:
            if self.events:
                return self.events
            cached = detector.load_events(self.video_path, self._params())
            if cached is not None:
                self.events = cached
                return self.events
            self.brightness, self.times = detector.scan_brightness(self.video)
            self.events = detector.detect_flashes(
                self.brightness, self.times, self.video.meta.fps, self.sensitivity)
            detector.save_events(self.video_path, self._params(), self.events)
            return self.events

    def rescan(self, sensitivity):
        with self.lock:
            self.sensitivity = sensitivity
            if self.brightness is None:
                self.brightness, self.times = detector.scan_brightness(self.video)
            self.events = detector.detect_flashes(
                self.brightness, self.times, self.video.meta.fps, sensitivity)
            detector.save_events(self.video_path, self._params(), self.events)
            return self.events


def _ev_json(state, e):
    mk = state.markers.get(e.peak_frame, {})
    return {"start_frame": e.start_frame, "end_frame": e.end_frame,
            "peak_frame": e.peak_frame, "peak_time": e.peak_time,
            "timecode": _fmt_ts(e.peak_time), "brightness": e.brightness,
            "label": mk.get("label", ""), "status": mk.get("status", "")}


def create_app(video_path=None) -> FastAPI:
    video_path = video_path or os.environ.get("LF_VIDEO")
    if not video_path:
        raise RuntimeError("No video provided (set LF_VIDEO or pass video_path)")
    state = AppState(video_path)
    app = FastAPI()
    app.state.lf = state
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/video/meta")
    def meta():
        m = state.video.meta
        return {"fps": m.fps, "frame_count": m.frame_count,
                "duration": m.duration, "width": m.width, "height": m.height}

    @app.get("/api/events")
    def events():
        return {"events": [_ev_json(state, e) for e in state.ensure_events()]}

    class ScanBody(BaseModel):
        sensitivity: float = 0.5

    @app.post("/api/scan")
    def scan(body: ScanBody):
        return {"events": [_ev_json(state, e) for e in state.rescan(body.sensitivity)]}

    @app.get("/video")
    def video(request: Request):
        path = state.video_path
        size = os.path.getsize(path)
        media = mimetypes.guess_type(path)[0] or "video/mp4"
        rng = request.headers.get("range")
        if not rng:
            return FileResponse(path, media_type=media)
        _, _, spec = rng.partition("=")
        start_s, _, end_s = spec.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else size - 1
        end = min(end, size - 1)
        length = end - start + 1

        def iterfile():
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        headers = {"Content-Range": f"bytes {start}-{end}/{size}",
                   "Accept-Ranges": "bytes", "Content-Length": str(length)}
        return StreamingResponse(iterfile(), status_code=206, headers=headers,
                                 media_type=media)

    @app.get("/api/frame/{index}.png")
    def frame_png(index: int):
        try:
            data = state.video.frame_png_bytes(index)
        except IndexError:
            raise HTTPException(404, "frame out of range")
        return Response(content=data, media_type="image/png")

    @app.get("/api/thumb/{index}.jpg")
    def thumb(index: int):
        return Response(content=state.video.thumb_jpeg_bytes(index),
                        media_type="image/jpeg")

    class GrabBody(BaseModel):
        index: int
        fmt: str = "png"

    @app.post("/api/grab")
    def grab(body: GrabBody):
        state.output_dir.mkdir(parents=True, exist_ok=True)
        t = body.index / (state.video.meta.fps or 30.0)
        ts = _fmt_ts(t).replace(":", "-")
        ext = "tiff" if body.fmt.lower() in ("tiff", "tif") else "png"
        out = state.output_dir / f"frame_{body.index}_{ts}.{ext}"
        Image.fromarray(state.video.frame(body.index)).save(out)
        return {"path": str(out)}

    class MarkerBody(BaseModel):
        peak_frame: int
        label: str = ""
        status: str = "confirmed"

    @app.get("/api/markers")
    def get_markers():
        return {str(k): v for k, v in state.markers.items()}

    @app.post("/api/markers")
    def set_marker(body: MarkerBody):
        state.markers[body.peak_frame] = {"label": body.label, "status": body.status}
        return {"ok": True}

    @app.get("/api/export")
    def export(fmt: str = "csv"):
        evs = state.ensure_events()
        if fmt == "json":
            return JSONResponse([_ev_json(state, e) for e in evs])
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["peak_frame", "time", "timecode", "brightness", "label", "status"])
        for e in evs:
            mk = state.markers.get(e.peak_frame, {})
            w.writerow([e.peak_frame, f"{e.peak_time:.3f}", _fmt_ts(e.peak_time),
                        f"{e.brightness:.1f}", mk.get("label", ""), mk.get("status", "")])
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=flashes.csv"})

    return app


if __name__ == "__main__":
    import sys
    import webbrowser

    import uvicorn

    if len(sys.argv) < 2:
        print("usage: python app.py <video_path> [port]")
        sys.exit(1)
    os.environ["LF_VIDEO"] = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    application = create_app(sys.argv[1])
    webbrowser.open(f"http://127.0.0.1:{port}")
    uvicorn.run(application, host="127.0.0.1", port=port)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_app.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add app.py static/index.html tests/test_app.py
git commit -m "feat: FastAPI backend — meta, events, ranged video, frame, grab, export"
```

---

### Task 5: Frontend skeleton — layout, video load, playback, meta

**Files:**
- Modify: `static/index.html` (replace placeholder)
- Create: `static/style.css`
- Create: `static/app.js`

**Interfaces:**
- Consumes: `GET /api/video/meta`, `GET /video`.
- Produces (DOM contract relied on by Tasks 6–8): element ids `#video`, `#frame-img`, `#stage`, `#timeline`, `#flash-list`, `#time-label`, `#res-label`, `#mode-label`, `#sensitivity`, and buttons `#btn-play`, `#btn-prev-flash`, `#btn-next-flash`, `#btn-frame-back`, `#btn-frame-fwd`, `#btn-grab`, `#btn-rescan`, `#btn-export-csv`, `#btn-export-json`. JS globals: `state` object, `seekToFrame(f)`, `curFrameFromVideo()`, `fmtTime(s)`, `updateTimeLabel(f)`.

- [ ] **Step 1: Replace `static/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lightning Finder</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <header>
    <h1>⚡ Lightning Finder</h1>
    <span id="res-label" class="muted"></span>
    <span id="mode-label" class="badge">playback</span>
  </header>

  <main>
    <section id="viewer">
      <div id="stage">
        <video id="video" preload="metadata"></video>
        <img id="frame-img" alt="exact frame" hidden>
      </div>
      <div id="timeline" title="click to seek"></div>
      <div id="controls">
        <button id="btn-prev-flash" title="Prev flash (Shift+←)">⏮ flash</button>
        <button id="btn-frame-back" title="Frame back (←)">◀ frame</button>
        <button id="btn-play" title="Play/Pause (Space)">▶ / ⏸</button>
        <button id="btn-frame-fwd" title="Frame fwd (→)">frame ▶</button>
        <button id="btn-next-flash" title="Next flash (Shift+→)">flash ⏭</button>
        <span id="time-label" class="muted">frame 0 · 00:00:00.000</span>
        <span class="spacer"></span>
        <label>sensitivity
          <input id="sensitivity" type="range" min="0" max="1" step="0.05" value="0.5">
        </label>
        <button id="btn-rescan">Rescan</button>
        <button id="btn-grab" class="primary" title="Grab full-quality (G)">⬇ Grab</button>
        <button id="btn-export-csv">CSV</button>
        <button id="btn-export-json">JSON</button>
      </div>
    </section>

    <aside id="sidebar">
      <h2>Flashes</h2>
      <ol id="flash-list"></ol>
    </aside>
  </main>

  <div id="toast" hidden></div>
  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `static/style.css`**

```css
:root { --bg:#0c0f16; --panel:#161b26; --line:#2a3346; --text:#e6ebf5;
        --muted:#8a93a6; --accent:#ffd23f; --mark:#ffd23f; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font:14px/1.4 system-ui, sans-serif; }
header { display:flex; align-items:center; gap:12px; padding:10px 16px;
         border-bottom:1px solid var(--line); }
header h1 { font-size:16px; margin:0; }
.muted { color:var(--muted); }
.badge { margin-left:auto; padding:2px 10px; border:1px solid var(--line);
         border-radius:999px; color:var(--accent); font-size:12px; }
main { display:grid; grid-template-columns:1fr 280px; gap:12px; padding:12px; }
#stage { position:relative; background:#000; border:1px solid var(--line);
         border-radius:8px; display:flex; align-items:center; justify-content:center;
         min-height:50vh; overflow:hidden; }
#video, #frame-img { max-width:100%; max-height:70vh; display:block; }
#timeline { position:relative; height:34px; margin:10px 0; background:var(--panel);
            border:1px solid var(--line); border-radius:6px; cursor:pointer; }
#timeline .marker { position:absolute; top:0; width:2px; height:100%;
                    background:var(--mark); }
#timeline .playhead { position:absolute; top:-3px; width:2px; height:40px;
                      background:#fff; }
#controls { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
button { background:var(--panel); color:var(--text); border:1px solid var(--line);
         border-radius:6px; padding:6px 10px; cursor:pointer; }
button:hover { border-color:var(--accent); }
button.primary { background:var(--accent); color:#000; font-weight:600; }
.spacer { flex:1; }
#sidebar { background:var(--panel); border:1px solid var(--line); border-radius:8px;
           padding:10px; max-height:84vh; overflow:auto; }
#sidebar h2 { font-size:13px; text-transform:uppercase; color:var(--muted); }
#flash-list { list-style:none; margin:0; padding:0; }
#flash-list li { padding:6px 8px; border-radius:6px; cursor:pointer;
                 display:flex; justify-content:space-between; gap:6px; }
#flash-list li:hover { background:#1f2636; }
#flash-list li.active { background:#2a3346; }
#flash-list li.rejected { opacity:.45; text-decoration:line-through; }
#toast { position:fixed; bottom:16px; left:50%; transform:translateX(-50%);
         background:#000; border:1px solid var(--accent); color:var(--text);
         padding:8px 14px; border-radius:8px; }
```

- [ ] **Step 3: Create `static/app.js`** (skeleton — meta, playback, scrub)

```javascript
const state = { meta: null, fps: 30, duration: 0, frameCount: 0,
                mode: "playback", curFrame: 0, events: [], markers: {} };

const $ = (id) => document.getElementById(id);
const video = $("video");
const frameImg = $("frame-img");

function fmtTime(s) {
  if (!isFinite(s)) s = 0;
  const ms = Math.round((s - Math.floor(s)) * 1000);
  let x = Math.floor(s);
  const h = Math.floor(x / 3600); x %= 3600;
  const m = Math.floor(x / 60); const sec = x % 60;
  const p = (n, w = 2) => String(n).padStart(w, "0");
  return `${p(h)}:${p(m)}:${p(sec)}.${p(ms, 3)}`;
}

function curFrameFromVideo() { return Math.round(video.currentTime * state.fps); }

function updateTimeLabel(f) {
  state.curFrame = f;
  $("time-label").textContent = `frame ${f} · ${fmtTime(f / state.fps)}`;
}

function seekToFrame(f) {
  f = Math.max(0, Math.min((state.frameCount || 1) - 1, f));
  if (state.mode === "precision") { showFrame(f); }
  else { video.currentTime = f / state.fps; updateTimeLabel(f); }
}

// showFrame/precision are defined in Task 7; provide a fallback for the skeleton.
function showFrame(f) { updateTimeLabel(f); }

function togglePlay() {
  if (typeof exitPrecision === "function") exitPrecision();
  if (video.paused) video.play(); else video.pause();
}

async function initMeta() {
  state.meta = await (await fetch("/api/video/meta")).json();
  state.fps = state.meta.fps || 30;
  state.duration = state.meta.duration || 0;
  state.frameCount = state.meta.frame_count || 0;
  $("res-label").textContent =
    `${state.meta.width}×${state.meta.height} · ${state.fps.toFixed(2)} fps · ${state.frameCount} frames`;
  video.src = "/video";
  video.addEventListener("timeupdate", () => {
    if (state.mode === "playback") updateTimeLabel(curFrameFromVideo());
  });
}

function wireSkeleton() {
  $("btn-play").addEventListener("click", togglePlay);
  $("timeline").addEventListener("click", (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const frac = (e.clientX - r.left) / r.width;
    seekToFrame(Math.round(frac * (state.frameCount || 1)));
  });
}

function toast(msg) {
  const t = $("toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), 2500);
}

async function main() {
  await initMeta();
  wireSkeleton();
  if (typeof loadEvents === "function") await loadEvents();   // Task 6
  if (typeof wireKeyboard === "function") wireKeyboard();     // Task 8
  if (typeof wirePrecision === "function") wirePrecision();   // Task 7
  if (typeof wireExport === "function") wireExport();         // Task 8
}
main();
```

- [ ] **Step 4: Verify backend still serves the page**

Run: `pytest tests/test_app.py::test_meta -v`
Expected: PASS (the `static/index.html` replacement still returns 200 via `GET /`; confirm with the next manual step).

- [ ] **Step 5: Manual/automated browser smoke test**

Start the server against the test clip:
```bash
python generate_test_clip.py test_clip.mp4
python app.py test_clip.mp4 8000
```
Then verify `http://127.0.0.1:8000` (using the Playwright MCP `browser_navigate` + `browser_snapshot`/`browser_take_screenshot`, or manually):
- The page shows the ⚡ header and the resolution label `320×180 · 30.00 fps · 300 frames`.
- The video element is present; pressing ▶ / ⏸ plays and pauses.
- Clicking on the timeline strip moves the time label.

Expected: all three observations hold. Stop the server (Ctrl+C) when done.

- [ ] **Step 6: Commit**

```bash
git add static/index.html static/style.css static/app.js
git commit -m "feat: frontend skeleton — layout, video load, playback, scrub"
```

---

### Task 6: Frontend — timeline markers, flash list, flash navigation

**Files:**
- Modify: `static/app.js` (add events loading, marker/list rendering, flash nav)

**Interfaces:**
- Consumes: `GET /api/events`; `state`, `seekToFrame`, `fmtTime` (Task 5).
- Produces: `loadEvents()`, `renderTimeline()`, `renderList()`, `jumpToFlash(dir)`, `setPlayhead(frac)`. Relied on by Task 5's `main()` (`loadEvents`) and Task 8 (keyboard `jumpToFlash`).

- [ ] **Step 1: Append events + rendering + navigation to `static/app.js`**

```javascript
// ---- Task 6: events, timeline markers, flash list, navigation ----------
async function loadEvents() {
  const r = await (await fetch("/api/events")).json();
  state.events = r.events;
  renderTimeline();
  renderList();
}

function setPlayhead(frac) {
  let ph = document.querySelector("#timeline .playhead");
  if (!ph) {
    ph = document.createElement("div");
    ph.className = "playhead";
    $("timeline").appendChild(ph);
  }
  ph.style.left = `${Math.max(0, Math.min(1, frac)) * 100}%`;
}

function renderTimeline() {
  const tl = $("timeline");
  tl.querySelectorAll(".marker").forEach((m) => m.remove());
  const dur = state.duration || (state.frameCount / state.fps) || 1;
  for (const e of state.events) {
    const m = document.createElement("div");
    m.className = "marker";
    m.style.left = `${(e.peak_time / dur) * 100}%`;
    m.title = `frame ${e.peak_frame} · ${e.timecode}`;
    m.addEventListener("click", (ev) => { ev.stopPropagation(); seekToFrame(e.peak_frame); });
    tl.appendChild(m);
  }
}

function renderList() {
  const ol = $("flash-list");
  ol.innerHTML = "";
  state.events.forEach((e, i) => {
    const li = document.createElement("li");
    li.dataset.peak = e.peak_frame;
    if (e.status === "rejected") li.classList.add("rejected");
    li.innerHTML = `<span>#${i + 1} · ${e.timecode}</span>` +
                   `<span class="muted">f${e.peak_frame}</span>`;
    li.addEventListener("click", () => seekToFrame(e.peak_frame));
    ol.appendChild(li);
  });
}

function highlightActive(frame) {
  document.querySelectorAll("#flash-list li").forEach((li) =>
    li.classList.toggle("active", Number(li.dataset.peak) === frame));
}

function jumpToFlash(dir) {
  if (!state.events.length) return;
  const cur = state.curFrame;
  const peaks = state.events.map((e) => e.peak_frame);
  let target = null;
  if (dir > 0) target = peaks.find((p) => p > cur);
  else target = [...peaks].reverse().find((p) => p < cur);
  if (target == null) target = dir > 0 ? peaks[0] : peaks[peaks.length - 1];
  seekToFrame(target);
  highlightActive(target);
}
```

- [ ] **Step 2: Hook the playhead into time updates** — modify the `timeupdate` listener in `initMeta()` so the playhead tracks playback. Replace the listener body added in Task 5 with:

```javascript
  video.addEventListener("timeupdate", () => {
    if (state.mode === "playback") {
      const f = curFrameFromVideo();
      updateTimeLabel(f);
      setPlayhead((state.duration ? video.currentTime / state.duration : 0));
    }
  });
```

- [ ] **Step 3: Verify (Playwright MCP or manual)**

Restart the server (`python app.py test_clip.mp4 8000`) and load the page:
- The timeline shows **3 markers** (near 1.67s, 4.07s, 6.0s for the 30fps clip).
- The sidebar lists **3 flashes** with timecodes; clicking entry #2 moves the time label to ~frame 122.
- Clicking the "flash ⏭" button from the start jumps to ~frame 50.

Expected: all observations hold.

- [ ] **Step 4: Commit**

```bash
git add static/app.js
git commit -m "feat: timeline markers, flash list, and flash-to-flash navigation"
```

---

### Task 7: Frontend — precision mode, frame stepping, full-quality grab

**Files:**
- Modify: `static/app.js` (add precision mode, frame stepping, grab; real `showFrame`)

**Interfaces:**
- Consumes: `GET /api/frame/{n}.png`, `POST /api/grab`; `state`, `updateTimeLabel`, `toast`, `highlightActive`.
- Produces: `enterPrecision(f)`, `exitPrecision()`, real `showFrame(f)` (overrides the skeleton fallback), `stepFrame(d)`, `stepTime(dt)`, `grab()`, `wirePrecision()`. Relied on by Task 8 keyboard handlers (`stepFrame`, `stepTime`, `grab`).

- [ ] **Step 1: Append precision/stepping/grab to `static/app.js`**

```javascript
// ---- Task 7: precision mode, frame stepping, full-quality grab ----------
function setModeLabel() { $("mode-label").textContent = state.mode; }

function enterPrecision(f) {
  state.mode = "precision";
  video.pause();
  video.hidden = true;
  frameImg.hidden = false;
  setModeLabel();
  showFrame(f);
}

function exitPrecision() {
  if (state.mode !== "precision") return;
  state.mode = "playback";
  frameImg.hidden = true;
  video.hidden = false;
  setModeLabel();
  video.currentTime = state.curFrame / state.fps;
}

// Real implementation; overrides the Task 5 skeleton fallback because this
// script runs later in the same file.
function showFrame(f) {
  f = Math.max(0, Math.min((state.frameCount || 1) - 1, f));
  state.curFrame = f;
  frameImg.src = `/api/frame/${f}.png`;
  updateTimeLabel(f);
  setPlayhead(state.duration ? (f / state.fps) / state.duration : 0);
  highlightActive(f);
}

function stepFrame(d) {
  if (state.mode !== "precision") enterPrecision(curFrameFromVideo());
  else showFrame(state.curFrame + d);
}

function stepTime(dt) {
  const f = Math.round((state.curFrame / state.fps + dt) * state.fps);
  if (state.mode !== "precision") enterPrecision(f); else showFrame(f);
}

async function grab() {
  const r = await fetch("/api/grab", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ index: state.curFrame }),
  });
  const data = await r.json();
  toast(`Saved ${data.path}`);
}

function wirePrecision() {
  $("btn-frame-back").addEventListener("click", () => stepFrame(-1));
  $("btn-frame-fwd").addEventListener("click", () => stepFrame(1));
  $("btn-grab").addEventListener("click", grab);
}
```

- [ ] **Step 2: Verify (Playwright MCP or manual)**

Restart the server and load the page:
- Jump to flash #1 (frame ~50), then click "frame ▶": the mode badge flips to **precision** and the displayed image swaps to the backend frame; the time label increments one frame per click.
- Stepping back and forth shows the exact neighbor frames (frame 122 visibly brighter than 123).
- Click **Grab**; a toast shows a saved path under `output/test_clip/`. Confirm the file exists and is `320×180`:
  ```bash
  python -c "from PIL import Image; import glob; p=glob.glob('output/test_clip/*.png')[-1]; print(p, Image.open(p).size)"
  ```
  Expected: prints the path and `(320, 180)` — full native resolution.

- [ ] **Step 3: Commit**

```bash
git add static/app.js
git commit -m "feat: precision mode, frame stepping, and full-quality frame grab"
```

---

### Task 8: Frontend — export, keyboard shortcuts, confirm/reject; README + final integration

**Files:**
- Modify: `static/app.js` (export wiring, keyboard map, marker status)
- Create: `README.md`

**Interfaces:**
- Consumes: `GET /api/export`, `POST /api/markers`, `POST /api/scan`; all functions from Tasks 5–7.
- Produces: `wireExport()`, `wireKeyboard()`, `setStatus(peakFrame, status)`, sensitivity rescan wiring.

- [ ] **Step 1: Append export, keyboard, markers, rescan to `static/app.js`**

```javascript
// ---- Task 8: export, keyboard, markers, rescan -------------------------
function wireExport() {
  $("btn-export-csv").addEventListener("click", () => { window.location = "/api/export?fmt=csv"; });
  $("btn-export-json").addEventListener("click", () => { window.location = "/api/export?fmt=json"; });

  $("btn-prev-flash").addEventListener("click", () => jumpToFlash(-1));
  $("btn-next-flash").addEventListener("click", () => jumpToFlash(1));

  $("btn-rescan").addEventListener("click", async () => {
    const sensitivity = parseFloat($("sensitivity").value);
    const r = await (await fetch("/api/scan", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sensitivity }),
    })).json();
    state.events = r.events;
    renderTimeline();
    renderList();
    toast(`Found ${state.events.length} flashes`);
  });
}

async function setStatus(peakFrame, status) {
  await fetch("/api/markers", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ peak_frame: peakFrame, status }),
  });
  const e = state.events.find((x) => x.peak_frame === peakFrame);
  if (e) e.status = status;
  renderList();
}

function wireKeyboard() {
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    switch (e.key) {
      case "ArrowLeft":  e.preventDefault(); e.shiftKey ? jumpToFlash(-1) : stepFrame(-1); break;
      case "ArrowRight": e.preventDefault(); e.shiftKey ? jumpToFlash(1)  : stepFrame(1);  break;
      case " ":          e.preventDefault(); togglePlay(); break;
      case "g": case "G": grab(); break;
      case ",": stepTime(-0.1); break;
      case ".": stepTime(0.1); break;
      case "x": case "X": if (state.events.length) setStatus(state.curFrame, "rejected"); break;
      case "c": case "C": if (state.events.length) setStatus(state.curFrame, "confirmed"); break;
    }
  });
}
```

- [ ] **Step 2: Create `README.md`**

```markdown
# ⚡ Lightning Finder

Scan a video for lightning flashes, review each one in the browser, step
frame-by-frame to the exact peak, and grab that frame at full native quality.

## Install
```
pip install -r requirements.txt
```

## Run
```
python app.py path/to/your_video.mp4
```
A browser opens at http://127.0.0.1:8000. First open scans the whole video
(progress in the terminal) and caches results next to the file.

## Use
- Sidebar lists detected flashes — click to jump. `Shift+←/→` jumps flash to flash.
- `←/→` steps one frame (enters precision mode = exact decoded frames).
- `,` / `.` nudge by 0.1s. `Space` plays/pauses.
- `G` grabs the current frame to `output/<video>/` as a lossless native-res PNG.
- `C` / `X` confirm/reject the current flash.
- **CSV** / **JSON** buttons export the timestamps.
- Adjust **sensitivity** + **Rescan** if flashes are missed or over-detected.

## Test
```
python generate_test_clip.py test_clip.mp4
pytest -v
```

## Not in v1
Multi-frame stacking, batch processing, auto-pick-best-frame-within-a-strike.
```

- [ ] **Step 3: Full test suite**

Run: `pytest -v`
Expected: all backend tests (Tasks 1–4) pass — 19 total.

- [ ] **Step 4: Final end-to-end verification (Playwright MCP or manual)**

Restart `python app.py test_clip.mp4 8000` and confirm the full loop:
- Keyboard: `Shift+→` jumps to next flash; `→` steps one frame into precision; `G` saves a frame; `X` strikes through the active flash in the list.
- Move the sensitivity slider down and click **Rescan**; the flash count toast updates and markers re-render.
- Click **CSV**: a `flashes.csv` downloads containing a `peak_frame` header and 3 data rows. Click **JSON**: valid JSON array of events.

Expected: every step behaves as described.

- [ ] **Step 5: Validate on real footage**

When the user drops their real video into the folder, run `python app.py "<their video>"` and confirm with them that the detected flash list looks right (no obvious misses / false hits); tune the default `sensitivity` in `app.py` (`AppState.sensitivity`) if needed.

- [ ] **Step 6: Commit**

```bash
git add static/app.js README.md
git commit -m "feat: export, keyboard shortcuts, confirm/reject, README; v1 complete"
```

---

## Self-Review

**1. Spec coverage**

| Spec requirement | Task |
| --- | --- |
| Detect lightning as brightness spikes vs dark baseline | Task 3 (`detect_flashes`) |
| Never miss single-frame flashes (decode every frame, downscale) | Task 3 (`scan_brightness`, frame 50/180 tests) |
| Ignore slow drift (dawn/exposure) | Task 3 (`test_ignores_slow_drift`) |
| Group one flickering strike as one event | Task 3 (`test_flicker_is_one_event`) |
| Cache results to a sidecar | Task 3 (`load_events`/`save_events`) |
| Metadata (fps, frame count, duration, resolution, VFR) | Task 2 (`VideoMeta`) |
| Frame-accurate exact-frame extraction via PTS | Task 2 (`frame`, `iter_frames` pts index) |
| Full-quality native-res grab (PNG/TIFF) | Task 4 (`/api/grab`), Task 7 (UI) |
| Serve video with HTTP range for `<video>` | Task 4 (`/video`, `test_video_range`) |
| Browser scrubber: player + timeline + markers + flash list | Tasks 5–6 |
| Precision mode = backend-decoded exact frames | Task 7 |
| Frame-step + small time-nudge navigation | Task 7 (`stepFrame`, `stepTime`) |
| Prev/next flash + keyboard map | Tasks 6, 8 |
| Sensitivity slider + live rescan | Tasks 4 (`/api/scan`), 8 |
| Confirm/reject/label flashes | Tasks 4 (`/api/markers`), 8 |
| Export timestamps CSV + JSON | Task 4 (`/api/export`), Task 8 (UI) |
| Synthetic test clip with known flashes incl. decoy drift | Task 1 |
| Run pointed at a video (CLI + auto-open browser) | Task 4 (`__main__`) |

No gaps found. Multi-frame stacking, batch, auto-pick-best are intentionally out of v1 (spec "out of scope").

**2. Placeholder scan**

The skeleton `showFrame` in Task 5 is a deliberate, documented fallback that Task 7 overrides (later definition in the same file wins); both are shown in full, not placeholders. No "TBD"/"add error handling"/"similar to Task N" remain.

**3. Type consistency**

- `FlashEvent` fields (`start_frame, end_frame, peak_frame, peak_time, brightness`) are produced in Task 3 and consumed identically in Task 4 `_ev_json` (which adds `timecode`, `label`, `status` for the wire format); the frontend reads exactly those wire fields.
- `VideoFile.frame`/`frame_png_bytes`/`thumb_jpeg_bytes`/`iter_frames`/`meta` signatures match between Task 2 (defs) and Tasks 3–4 (calls).
- JS function names are consistent across tasks: `seekToFrame`, `showFrame`, `enterPrecision`/`exitPrecision`, `stepFrame`, `stepTime`, `grab`, `jumpToFlash`, `loadEvents`, `renderTimeline`, `renderList`, `setPlayhead`, `highlightActive`, `wireExport`/`wireKeyboard`/`wirePrecision`.
- Endpoint paths match between backend (Task 4) and frontend callers (Tasks 5–8): `/api/video/meta`, `/api/events`, `/api/scan`, `/video`, `/api/frame/{n}.png`, `/api/grab`, `/api/markers`, `/api/export`.

No inconsistencies found.
