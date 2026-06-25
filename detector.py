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
        with open(cp, encoding="utf-8") as f:
            return [FlashEvent(**e) for e in json.load(f)]
    return None


def save_events(path, params, events):
    with open(cache_path(path, params), "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in events], f)


def brightness_cache_path(path):
    """Sidecar path for the cached brightness/times arrays.

    Keyed by path + size + mtime only (NOT sensitivity): the per-frame
    brightness is the expensive, sensitivity-independent artifact, so caching it
    makes both reopening a video and changing sensitivity (rescan) instant.
    """
    st = os.stat(path)
    raw = f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"
    key = hashlib.sha1(raw.encode()).hexdigest()
    folder = os.path.dirname(os.path.abspath(path)) or "."
    return os.path.join(folder, f".lf_bright_{key}.npz")


def load_brightness(path):
    """Return (brightness, times) from the sidecar cache, or None if absent."""
    cp = brightness_cache_path(path)
    if not os.path.exists(cp):
        return None
    d = np.load(cp)
    try:
        return np.array(d["brightness"]), np.array(d["times"])
    finally:
        d.close()


def save_brightness(path, brightness, times):
    np.savez(brightness_cache_path(path), brightness=brightness, times=times)
