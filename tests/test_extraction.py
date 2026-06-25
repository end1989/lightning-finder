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


def test_thumb_jpeg_bytes(clip):
    v = VideoFile(clip.path)
    data = v.thumb_jpeg_bytes(50, height=120)
    assert data[:3] == b"\xff\xd8\xff"          # JPEG magic
    im = Image.open(io.BytesIO(data))
    assert im.height == 120


def test_frame_does_not_build_full_index(clip):
    """frame() must NOT force a full-video PTS index build.

    When events load from cache, iter_frames never runs, so _indexed stays
    False. If frame() forced a full decode pass to build the index, the first
    precision-mode frame would stall for minutes on a long 4K video. frame()
    must instead seek by time without building the whole index.
    """
    v = VideoFile(clip.path)
    assert v._indexed is False
    f = v.frame(50)
    assert f.shape == (180, 320, 3)
    assert _mean(f) > 150                         # still the correct bright frame
    assert v._indexed is False, "frame() must not force a full PTS index build"


def test_frame_uses_prebuilt_index_when_available(clip):
    """When a scan already built the index, frame() stays frame-accurate."""
    v = VideoFile(clip.path)
    list(v.iter_frames())                          # builds the PTS index
    assert v._indexed is True
    assert _mean(v.frame(122)) > _mean(v.frame(123))   # peak 122 > 123
