import cv2
import numpy as np

import bolt
import detector
from video import VideoFile


def test_line_score_detects_vertical_bolt_over_glow():
    base = np.full((1080, 1920), 20, np.uint8)
    glow = np.full((1080, 1920), 95, np.uint8)             # uniform brightening, no line
    strike = glow.copy()
    cv2.line(strike, (900, 70), (930, 690), 255, 4)         # thin tall vertical channel
    s_glow, h_glow = bolt.line_score(glow, base)
    s_bolt, h_bolt = bolt.line_score(strike, base)
    assert h_bolt >= bolt.BOLT_MIN_VERTICAL                 # real channel
    assert h_glow < bolt.BOLT_MIN_VERTICAL                  # glow is not
    assert s_bolt > s_glow * 3


def test_line_score_rejects_horizontal_line():
    base = np.full((1080, 1920), 20, np.uint8)
    horiz = np.full((1080, 1920), 95, np.uint8)
    cv2.line(horiz, (100, 300), (1800, 305), 255, 4)        # long horizontal city/horizon
    _s, h = bolt.line_score(horiz, base)
    assert h < bolt.BOLT_MIN_VERTICAL                        # rejected by the vertical bias


def test_refine_events_returns_aligned_results(clip):
    v = VideoFile(clip.path)
    b, t = detector.scan_brightness(v)
    events = detector.detect_flashes(b, t, v.meta.fps, 0.5)
    results = bolt.refine_events(v, events)
    assert len(results) == len(events)
    for r, e in zip(results, events):
        assert r.peak_frame == e.peak_frame
        assert e.start_frame - 4 <= r.bolt_frame <= e.end_frame + 4


def test_bolt_cache_roundtrip(clip):
    results = [bolt.BoltResult(50, 49, 1.2, 100, True),
               bolt.BoltResult(122, 122, 0.0, 0, False)]
    params = {"sensitivity": 0.5}
    bolt.save_bolt(clip.path, params, results)
    loaded = bolt.load_bolt(clip.path, params)
    assert loaded == results
