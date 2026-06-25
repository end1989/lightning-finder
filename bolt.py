"""Lightning-channel ('bolt') detection.

Brightness finds *when* a flash happens; this finds the frame where a real
lightning *channel* is visible — a thin, tall, vertical/diagonal bright line in
the sky — versus diffuse cloud glow with no channel. The trick is temporal
differencing against a per-pixel baseline of the flash window, which cancels the
static scene (foreground, city lights, horizon) and leaves only the transient
strike.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass

import cv2
import numpy as np

SKY_FRACTION = 0.72        # search only the upper part of the frame (skip city/foreground)
TOPHAT_KERNEL = 11         # isolate bright structures thinner than this (~1080p scale)
DIFF_THRESH = 16           # min brightening over baseline to count
ANALYZE_HEIGHT = 1080      # analyse at ~1080p for speed; bolts stay visible
BOLT_MIN_VERTICAL = 90     # a real channel spans at least this many px vertically (@1080)


@dataclass
class BoltResult:
    peak_frame: int        # original brightness peak of the event
    bolt_frame: int        # frame with the strongest channel
    bolt_score: float      # channel strength (for ranking)
    vertical_px: int       # vertical extent of the channel (for classification)
    is_bolt: bool          # True if a real channel was found (else glow-only)


def to_gray(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if g.shape[0] > ANALYZE_HEIGHT + 20:
        s = ANALYZE_HEIGHT / g.shape[0]
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return g


def line_score(gray, baseline):
    """Return (score, vertical_px) for the strongest thin, vertically-extended
    bright structure in the sky region of (gray - baseline).

    High for a lightning channel; ~0 for diffuse glow or horizontal
    city/horizon/rail structures (which are rejected by the vertical bias).
    """
    diff = cv2.subtract(gray, baseline)
    sky = diff[: int(diff.shape[0] * SKY_FRACTION)]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL, TOPHAT_KERNEL))
    tophat = cv2.morphologyEx(sky, cv2.MORPH_TOPHAT, k)
    _, mask = cv2.threshold(tophat, DIFF_THRESH, 255, cv2.THRESH_BINARY)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    best_score, best_h = 0.0, 0
    for i in range(1, n):
        _x, _y, w, h, area = stats[i]
        if area < 5 or h < w:                      # skip tiny + horizontal-dominant blobs
            continue
        score = float(h) * (h / (w + 1.0)) ** 0.3  # tall, and mildly favour thin
        if score > best_score:
            best_score, best_h = score, int(h)
    return best_score, best_h


def refine_event(video, start_frame, end_frame, peak_frame, window=4):
    """Find the best channel frame within a flash window. Returns BoltResult."""
    lo, hi = max(0, start_frame - window), end_frame + window
    grays = {idx: to_gray(rgb)
             for idx, rgb in video.iter_range(lo, hi, downscale_height=ANALYZE_HEIGHT)}
    if not grays:
        return BoltResult(peak_frame, peak_frame, 0.0, 0, False)
    baseline = np.min(np.stack(list(grays.values())), axis=0)
    best_idx, best_score, best_h = peak_frame, 0.0, 0
    for idx, g in grays.items():
        sc, h = line_score(g, baseline)
        if sc > best_score:
            best_idx, best_score, best_h = idx, sc, h
    return BoltResult(peak_frame, best_idx, best_score, best_h,
                      best_h >= BOLT_MIN_VERTICAL)


def refine_events(video, events, window=4, progress=None):
    """Refine every event; return list[BoltResult] aligned with `events`."""
    results = []
    total = len(events)
    for i, e in enumerate(events):
        results.append(refine_event(video, e.start_frame, e.end_frame,
                                     e.peak_frame, window=window))
        if progress:
            progress(i + 1, total)
    return results


# ---- caching (keyed by video + the sensitivity that produced the events) -----
def _bolt_cache_path(path, params):
    st = os.stat(path)
    raw = (f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}|"
           f"{json.dumps(params, sort_keys=True)}")
    key = hashlib.sha1(raw.encode()).hexdigest()
    folder = os.path.dirname(os.path.abspath(path)) or "."
    return os.path.join(folder, f".lf_bolt_{key}.json")


def load_bolt(path, params):
    cp = _bolt_cache_path(path, params)
    if os.path.exists(cp):
        with open(cp, encoding="utf-8") as f:
            return [BoltResult(**r) for r in json.load(f)]
    return None


def save_bolt(path, params, results):
    with open(_bolt_cache_path(path, params), "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f)
