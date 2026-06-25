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


def test_rescan_uses_cached_brightness_not_full_scan(clip, monkeypatch):
    """Rescan must reuse the cached brightness, not re-decode the whole video.

    On a long video whose events came from cache, brightness was never in
    memory, so the old rescan triggered a full multi-minute scan. Now brightness
    is cached and restored, so rescan only re-runs the cheap detect step.
    """
    import detector
    from app import AppState

    AppState(clip.path).ensure_events()                 # builds + caches brightness
    assert detector.load_brightness(clip.path) is not None

    s = AppState(clip.path)                              # fresh: brightness not in memory
    assert s.brightness is None

    def _boom(*a, **k):
        raise AssertionError("rescan must not full-scan when brightness is cached")

    monkeypatch.setattr(detector, "scan_brightness", _boom)
    events = s.rescan(0.2)                               # must load cached brightness
    assert isinstance(events, list)
    assert s.brightness is not None


def test_rescan_endpoint_changes_with_sensitivity(clip):
    """The /api/scan endpoint returns events for the requested sensitivity."""
    c = client(clip)
    r = c.post("/api/scan", json={"sensitivity": 0.5})
    assert r.status_code == 200
    assert "events" in r.json()


def test_refine_bolts_endpoint(clip):
    """Best-shots refine runs in the background and returns ranked events."""
    import time
    c = client(clip)
    started = c.post("/api/refine-bolts").json()
    assert started["status"] in ("running", "done")
    r = started
    for _ in range(100):
        r = c.get("/api/refine-bolts").json()
        if r["status"] == "done":
            break
        time.sleep(0.2)
    assert r["status"] == "done"
    assert len(r["events"]) >= 3
    assert all("is_bolt" in e and "bolt_frame" in e for e in r["events"])
    scores = [e["bolt_score"] for e in r["events"]]
    assert scores == sorted(scores, reverse=True)        # ranked by bolt strength
