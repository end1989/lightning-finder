"""FastAPI backend for Lightning Finder."""
from __future__ import annotations

import csv
import io
import mimetypes
import os
import string
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               Response, StreamingResponse)
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

import bolt
import detector
from video import VideoFile

STATIC_DIR = Path(__file__).parent / "static"

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
              ".mpg", ".mpeg", ".wmv"}


def _fmt_ts(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    s = total_ms // 1000
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _scan_progress(done, total):
    """Print scan progress to the terminal for long videos (quiet for short clips)."""
    if not total or total < 1500:
        return
    step = max(1, total // 20)  # ~every 5%
    if done % step == 0 or done == total:
        print(f"[scan] {done}/{total} frames ({100 * done // total}%)", flush=True)


class AppState:
    def __init__(self, video_path=None):
        self.lock = threading.Lock()
        self._gen = 0
        self._reset_empty()
        if video_path:
            self.load(video_path)

    def _reset_empty(self):
        self.video = None
        self.video_path = None
        self.sensitivity = 0.5
        self.events = []
        self.brightness = None
        self.times = None
        self.markers: dict[int, dict] = {}
        self.output_dir = None
        self.bolt_results = None
        self.bolt_status = "idle"
        self.bolt_progress = (0, 0)
        self._bolt_thread = None

    def load(self, video_path):
        with self.lock:
            self._reset_empty()
            self._gen += 1
            self.video = VideoFile(video_path)
            self.video_path = str(video_path)
            self.output_dir = Path("output") / Path(video_path).stem

    @property
    def loaded(self):
        return self.video is not None

    def _params(self):
        return {"sensitivity": self.sensitivity}

    def _ensure_brightness(self):
        """Load the brightness/times arrays from cache, or scan once and cache.

        Brightness is sensitivity-independent, so caching it makes reopening a
        video AND every rescan instant — only the cheap detect step re-runs per
        sensitivity. (Previously rescan re-decoded the whole video because the
        brightness was never restored after a cache-hit load.)
        """
        if self.brightness is not None:
            return
        cached = detector.load_brightness(self.video_path)
        if cached is not None:
            self.brightness, self.times = cached
            return
        self.brightness, self.times = detector.scan_brightness(
            self.video, progress=_scan_progress)
        detector.save_brightness(self.video_path, self.brightness, self.times)

    def ensure_events(self):
        with self.lock:
            if self.events:
                return self.events
            self._ensure_brightness()
            self.events = detector.detect_flashes(
                self.brightness, self.times, self.video.meta.fps, self.sensitivity)
            return self.events

    def rescan(self, sensitivity):
        with self.lock:
            self.sensitivity = sensitivity
            self._ensure_brightness()
            self.events = detector.detect_flashes(
                self.brightness, self.times, self.video.meta.fps, sensitivity)
            # the event set changed, so any prior bolt refine is stale
            self.bolt_results = None
            self.bolt_status = "idle"
            self.bolt_progress = (0, 0)
            return self.events

    def start_bolt_refine(self):
        """Kick off (or load from cache) the 'best shots' bolt refine pass."""
        self.ensure_events()
        with self.lock:
            if self.bolt_status == "running":
                return
            cached = bolt.load_bolt(self.video_path, self._params())
            if cached is not None and len(cached) == len(self.events):
                self.bolt_results = cached
                self.bolt_status = "done"
                self.bolt_progress = (len(cached), len(cached))
                return
            self.bolt_status = "running"
            self.bolt_progress = (0, len(self.events))
            self._bolt_thread = threading.Thread(target=self._run_bolt_refine,
                                                  daemon=True)
            self._bolt_thread.start()

    def _run_bolt_refine(self):
        # Snapshot everything tied to the current video UNDER THE LOCK, so a
        # concurrent /api/open (which mutates these fields under the same lock)
        # can't be observed half-applied. If the user opens a different video
        # mid-refine, this thread keeps decoding (and caching to) the ORIGINAL
        # video; its in-memory write is gated by the _gen check below.
        with self.lock:
            gen = self._gen
            vid = self.video
            video_path = self.video_path
            params = self._params()
            events = list(self.events)
        if vid is None:
            return
        thumb_dir = bolt.thumbs_dir(video_path, params)

        def prog(done, total):
            if self._gen == gen:
                self.bolt_progress = (done, total)

        try:
            results = bolt.refine_events(vid, events, progress=prog, thumb_dir=thumb_dir)
            bolt.save_bolt(video_path, params, results)   # correct original-video data -> its own cache
            with self.lock:
                if self._gen != gen:        # a different video is now current; leave its state alone
                    return
                self.bolt_results = results
                self.bolt_status = "done"
        except Exception as exc:            # pragma: no cover - defensive
            with self.lock:
                if self._gen == gen:
                    self.bolt_status = "error"
            print(f"[bolt] refine failed: {exc}", flush=True)


def _ev_json(state, e):
    mk = state.markers.get(e.peak_frame, {})
    return {"start_frame": e.start_frame, "end_frame": e.end_frame,
            "peak_frame": e.peak_frame, "peak_time": e.peak_time,
            "timecode": _fmt_ts(e.peak_time), "brightness": e.brightness,
            "label": mk.get("label", ""), "status": mk.get("status", "")}


def _bolt_status_json(state):
    """Refine progress, plus the 'best shots' ranking once done."""
    done, total = state.bolt_progress
    out = {"status": state.bolt_status, "done": done, "total": total}
    if state.bolt_status == "done" and state.bolt_results is not None:
        by_peak = {r.peak_frame: r for r in state.bolt_results}
        items = []
        for e in state.ensure_events():
            j = _ev_json(state, e)
            r = by_peak.get(e.peak_frame)
            if r is not None:
                j.update({"bolt_frame": r.bolt_frame, "bolt_score": r.bolt_score,
                          "vertical_px": r.vertical_px, "is_bolt": r.is_bolt,
                          "bolt_timecode": _fmt_ts(r.bolt_frame / (state.video.meta.fps or 30))})
            items.append(j)
        items.sort(key=lambda x: x.get("bolt_score", 0.0), reverse=True)
        out["events"] = items
        out["n_bolts"] = sum(1 for x in items if x.get("is_bolt"))
    return out


class ScanBody(BaseModel):
    sensitivity: float = 0.5


class GrabBody(BaseModel):
    index: int
    fmt: str = "png"


class MarkerBody(BaseModel):
    peak_frame: int
    label: str = ""
    status: str = "confirmed"


class OpenBody(BaseModel):
    path: str


def create_app(video_path=None) -> FastAPI:
    video_path = video_path or os.environ.get("LF_VIDEO")
    state = AppState(video_path)        # empty if no path/env
    app = FastAPI()
    app.state.lf = state
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    def need_video():
        if state.video is None:
            raise HTTPException(409, "No video loaded")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    @app.get("/api/state")
    def app_state():
        if state.video is None:
            return {"loaded": False}
        m = state.video.meta
        return {"loaded": True, "name": Path(state.video_path).name,
                "width": m.width, "height": m.height, "fps": m.fps,
                "frame_count": m.frame_count, "duration": m.duration}

    @app.get("/api/browse")
    def browse(path: str = ""):
        if not path:
            if os.name == "nt":
                drives = [f"{d}:\\" for d in string.ascii_uppercase
                          if os.path.exists(f"{d}:\\")]
                return {"path": "", "parent": None, "dirs": drives, "videos": []}
            path = os.path.expanduser("~")
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            raise HTTPException(400, "not a directory")
        dirs, videos = [], []
        try:
            entries = sorted(os.listdir(path), key=str.lower)
        except PermissionError:
            raise HTTPException(403, "permission denied")
        for name in entries:
            full = os.path.join(path, name)
            try:
                if os.path.isdir(full):
                    dirs.append(full)
                elif os.path.splitext(name)[1].lower() in VIDEO_EXTS:
                    videos.append(full)
            except OSError:
                continue
        up = os.path.dirname(path)
        parent = "" if up == path else up        # "" -> drive list on Windows
        return {"path": path, "parent": parent, "dirs": dirs, "videos": videos}

    @app.post("/api/open")
    def open_video(body: OpenBody):
        if not os.path.isfile(body.path):
            raise HTTPException(400, "not a file")
        try:
            state.load(body.path)
            m = state.video.meta
        except Exception:
            with state.lock:
                state._reset_empty()
            raise HTTPException(400, "not a valid video file")
        return {"ok": True, "name": Path(body.path).name,
                "width": m.width, "height": m.height, "fps": m.fps,
                "frame_count": m.frame_count, "duration": m.duration}

    @app.get("/api/video/meta")
    def meta():
        need_video()
        m = state.video.meta
        return {"fps": m.fps, "frame_count": m.frame_count,
                "duration": m.duration, "width": m.width, "height": m.height}

    @app.get("/api/events")
    def events():
        need_video()
        return {"events": [_ev_json(state, e) for e in state.ensure_events()]}

    @app.post("/api/scan")
    def scan(body: ScanBody):
        need_video()
        return {"events": [_ev_json(state, e) for e in state.rescan(body.sensitivity)]}

    @app.post("/api/refine-bolts")
    def refine_bolts():
        need_video()
        state.start_bolt_refine()
        return _bolt_status_json(state)

    @app.get("/api/refine-bolts")
    def refine_bolts_status():
        need_video()
        return _bolt_status_json(state)

    @app.get("/api/bolt-thumb/{frame}.jpg")
    def bolt_thumb(frame: int):
        need_video()
        p = bolt.thumb_path(state.video_path, state._params(), frame)
        if os.path.exists(p):
            return FileResponse(p, media_type="image/jpeg")
        try:
            data = bolt.make_thumbnail(state.video.frame(frame))
        except IndexError:
            raise HTTPException(404, "frame out of range")
        os.makedirs(bolt.thumbs_dir(state.video_path, state._params()), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
        return Response(content=data, media_type="image/jpeg")

    @app.get("/video")
    def video(request: Request):
        need_video()
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
        need_video()
        try:
            data = state.video.frame_png_bytes(index)
        except IndexError:
            raise HTTPException(404, "frame out of range")
        return Response(content=data, media_type="image/png")

    @app.get("/api/frame/{index}.jpg")
    def frame_jpg(index: int):
        need_video()
        # Fast native-res JPEG preview for precision-mode display (grab uses .png).
        try:
            data = state.video.frame_jpeg_bytes(index)
        except IndexError:
            raise HTTPException(404, "frame out of range")
        return Response(content=data, media_type="image/jpeg")

    @app.get("/api/thumb/{index}.jpg")
    def thumb(index: int):
        need_video()
        return Response(content=state.video.thumb_jpeg_bytes(index),
                        media_type="image/jpeg")

    @app.post("/api/grab")
    def grab(body: GrabBody):
        need_video()
        state.output_dir.mkdir(parents=True, exist_ok=True)
        t = body.index / (state.video.meta.fps or 30.0)
        ts = _fmt_ts(t).replace(":", "-")
        ext = "tiff" if body.fmt.lower() in ("tiff", "tif") else "png"
        out = state.output_dir / f"frame_{body.index}_{ts}.{ext}"
        Image.fromarray(state.video.frame(body.index)).save(out)
        return {"path": str(out)}

    @app.get("/api/markers")
    def get_markers():
        need_video()
        return {str(k): v for k, v in state.markers.items()}

    @app.post("/api/markers")
    def set_marker(body: MarkerBody):
        need_video()
        state.markers[body.peak_frame] = {"label": body.label, "status": body.status}
        return {"ok": True}

    @app.get("/api/export")
    def export(fmt: str = "csv"):
        need_video()
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

    path = sys.argv[1] if len(sys.argv) > 1 else None     # optional shortcut
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    application = create_app(path)
    webbrowser.open(f"http://127.0.0.1:{port}")
    uvicorn.run(application, host="127.0.0.1", port=port)
