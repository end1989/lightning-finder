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
