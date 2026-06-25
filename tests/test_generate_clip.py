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
