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
            return self.events


def _ev_json(state, e):
    mk = state.markers.get(e.peak_frame, {})
    return {"start_frame": e.start_frame, "end_frame": e.end_frame,
            "peak_frame": e.peak_frame, "peak_time": e.peak_time,
            "timecode": _fmt_ts(e.peak_time), "brightness": e.brightness,
            "label": mk.get("label", ""), "status": mk.get("status", "")}


class ScanBody(BaseModel):
    sensitivity: float = 0.5


class GrabBody(BaseModel):
    index: int
    fmt: str = "png"


class MarkerBody(BaseModel):
    peak_frame: int
    label: str = ""
    status: str = "confirmed"


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

    @app.get("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    @app.get("/api/video/meta")
    def meta():
        m = state.video.meta
        return {"fps": m.fps, "frame_count": m.frame_count,
                "duration": m.duration, "width": m.width, "height": m.height}

    @app.get("/api/events")
    def events():
        return {"events": [_ev_json(state, e) for e in state.ensure_events()]}

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

    @app.get("/api/frame/{index}.jpg")
    def frame_jpg(index: int):
        # Fast native-res JPEG preview for precision-mode display (grab uses .png).
        try:
            data = state.video.frame_jpeg_bytes(index)
        except IndexError:
            raise HTTPException(404, "frame out of range")
        return Response(content=data, media_type="image/jpeg")

    @app.get("/api/thumb/{index}.jpg")
    def thumb(index: int):
        return Response(content=state.video.thumb_jpeg_bytes(index),
                        media_type="image/jpeg")

    @app.post("/api/grab")
    def grab(body: GrabBody):
        state.output_dir.mkdir(parents=True, exist_ok=True)
        t = body.index / (state.video.meta.fps or 30.0)
        ts = _fmt_ts(t).replace(":", "-")
        ext = "tiff" if body.fmt.lower() in ("tiff", "tif") else "png"
        out = state.output_dir / f"frame_{body.index}_{ts}.{ext}"
        Image.fromarray(state.video.frame(body.index)).save(out)
        return {"path": str(out)}

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
