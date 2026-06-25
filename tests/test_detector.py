import numpy as np

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


def test_brightness_cache_roundtrip(clip):
    """Brightness/times are cached (sensitivity-independent) so rescan is cheap."""
    v = VideoFile(clip.path)
    b, t = detector.scan_brightness(v)
    detector.save_brightness(clip.path, b, t)
    loaded = detector.load_brightness(clip.path)
    assert loaded is not None
    lb, lt = loaded
    assert np.allclose(lb, b)
    assert np.allclose(lt, t)
