# Lightning Finder — Design Spec

**Date:** 2026-06-25
**Status:** Approved (design); pending implementation plan

## Purpose

A local tool that scans a video (typically 4K and/or timelapse) for **lightning
flashes** — sudden brightness spikes against an otherwise dark baseline — then
lets the user review each flash in a browser, nudge frame-by-frame to the exact
peak of the strike, and extract that frame at the video's **full native
quality** (not a screenshot). It also exports a timestamps file of every
detected flash.

Primary use: capturing the single perfect lightning frame, plus a reviewable
list of when every strike happened.

## Success criteria

1. Pointed at a real storm video, the tool lists the lightning flashes with
   timestamps the user agrees are correct (no obvious misses, few false hits).
2. The user can jump flash-to-flash, step ±1 frame to the exact peak, and grab
   that frame **at native resolution, losslessly** — and the saved image is
   provably the exact frame shown.
3. Single-frame flashes (common in timelapse) are not missed.
4. Slow brightness changes (dawn, exposure drift) do **not** register as flashes.

## Architecture

Two pieces, no frontend build step:

- **Python backend** — FastAPI + uvicorn. Runs detection, serves the video
  (HTTP range requests), and decodes **exact frames** from the file at native
  resolution.
- **Browser frontend** — plain HTML/CSS/JS. The scrubber UI.

### Why "decode exact frames" matters (the full-quality guarantee)

A browser `<video>` element and any OS "screenshot" only ever yield
display-resolution, re-compressed pixels — and `<video>` seeking can land ~1
frame off depending on the codec/keyframe layout. So the app has two viewing
modes:

- **Playback mode** — native `<video>`, smooth scrubbing to roughly find a
  strike.
- **Precision mode** — as soon as the user frame-steps, the on-screen image
  becomes the **backend-decoded exact frame** (`GET /api/frame/{n}.png`, native
  resolution, PyAV frame-accurate decode). What is shown *is* the real frame, so
  Grab saves exactly that, losslessly. Stepping ±1 fetches the true adjacent
  frame.

## Components

```
lightning_finder/
  app.py              # FastAPI server + routes
  detector.py         # brightness-spike detection
  video.py            # frame-accurate decode + native-res extraction + meta
  generate_test_clip.py
  static/  index.html  app.js  style.css
  tests/   test_detector.py  test_extraction.py
  output/  (grabbed frames + timestamp exports; created at runtime)
```

### `video.py` — frame service

- Reads video metadata: fps (and whether variable frame rate), frame count,
  duration, native resolution.
- `frame(index) -> native-resolution image`: PyAV seeks to the nearest keyframe
  and decodes forward to the exact frame index. Used for precision-mode display
  and full-quality grab.
- `thumb(index)`: small downscaled JPEG for fast previews.
- This is the single source of truth for frame-accuracy; the detector and the UI
  both rely on the same frame indexing.

### `detector.py` — lightning detection

- **Decodes every frame** (no skipping) so single-frame flashes can't slip
  through, but **downscales each frame during decode** so the brightness
  computation is cheap. Decoding is the dominant cost; downscaling keeps it
  bounded. Long 4K scans report progress and the result is cached to a JSON
  sidecar (keyed by file path + size + mtime) so reopening is instant.
- **Brightness metric** per frame: mean luma, combined with a high-percentile
  term so a small bright bolt in an otherwise dark frame still registers.
- **Flash rule:** brightness rising *sharply* above a short **rolling baseline**
  (e.g. median over a recent window). Requiring a sharp *rise* — not just a high
  absolute value — is what makes slow dawn/exposure drift get ignored while
  sudden spikes count. Threshold is adaptive (baseline + k·spread) with a
  sensitivity control.
- **Event grouping:** consecutive above-threshold frames merge into one event;
  the **brightest frame** in the run is the representative `peak_frame`. A small
  merge-gap keeps one flickering strike (which can flicker over several frames)
  as a single event rather than many.
- Output per event: `{ start_frame, end_frame, peak_frame, peak_time,
  brightness }`.

### `app.py` — FastAPI routes

- `GET /` — serves the UI.
- `GET /api/video/meta` — fps, frame count, duration, resolution.
- `GET /api/events` — detected flash events (returns cache or triggers scan).
- `POST /api/scan` — (re)run detection with `{ sensitivity, ... }`.
- `GET /video` — serves the media file with HTTP range support for `<video>`.
- `GET /api/frame/{index}.png` — exact native-resolution frame (precision view).
- `GET /api/thumb/{index}.jpg` — small preview.
- `POST /api/grab` — save exact frame `n` as full-quality PNG (or TIFF) into the
  output folder; returns the saved path.
- `GET/POST /api/markers` — user confirm/reject/label flashes; export timestamps
  (CSV/JSON, optional SRT).

### Frontend (`static/`)

Single page:

- Video / exact-frame view (top-left).
- Timeline scrubber with **flash markers** drawn from events; current-position
  indicator; click a marker to seek.
- A side **flash list** (click an entry to jump to that strike).
- Control bar: prev-flash · frame− · play/pause · frame+ · next-flash ·
  **Grab full-quality** · sensitivity slider + Rescan · mode indicator
  (playback / precision).
- Keyboard: `←/→` step one frame · `Shift+←/→` jump prev/next flash · `Space`
  play/pause · `G` grab · `,` / `.` time-nudge.

## Data flow

1. Launch the app pointed at a video (CLI arg, or pick/confirm in the UI).
2. Backend reads metadata, runs detection (or loads cache), returns events.
3. UI renders player + timeline markers + flash list.
4. User jumps to a flash (`Shift+→`), enters precision mode, steps ±1 frame to
   the exact peak.
5. **Grab** → backend decodes that exact frame at native resolution, writes PNG
   (or TIFF) to `output/`, UI confirms the saved path.
6. User exports the timestamps (CSV + JSON; optional SRT) of all/confirmed
   flashes.

## Output formats (defaults)

- **Frames:** PNG, lossless, native resolution →
  `lightning_finder/output/<video_stem>/frame_<index>_<timestamp>.png`.
  TIFF available as an option.
- **Timestamps:** CSV **and** JSON, each row = `frame_index, time, brightness,
  label`. SRT / NLE edit-marker export is an optional extra.

## Dependencies

- Backend: `fastapi`, `uvicorn`, `av` (PyAV — frame-accurate decode + native
  extraction), `numpy`, `Pillow`.
- Frontend: none (vanilla HTML/CSS/JS).
- PyAV is the single decoder for both the brightness scan and exact-frame
  extraction, so frame indexing is consistent everywhere. (OpenCV is not
  required; if it turns out materially faster for the bulk brightness scan it may
  be added for that path only, but PyAV remains the source of truth for frame
  indices.)

## Testing strategy

- **`generate_test_clip.py`** produces a black video at a known fps with bright
  flashes injected at **known frame indices**, including:
  - a single-frame flash,
  - a multi-frame flicker (one strike, several frames),
  - a slow brightness drift across the clip as a decoy.
- **`test_detector.py`** asserts the detector finds exactly the injected flashes
  (peak within ±1 frame), groups the flicker as one event, and does **not**
  report the drift.
- **`test_extraction.py`** extracts a known frame and verifies it is at native
  resolution and its pixels match the injected flash content.
- After automated tests pass, validate on the user's real footage and confirm
  the flash list looks right.

## Performance notes / honest trade-offs

- Catching a 1-frame flash requires decoding every frame; there is no shortcut
  (keyframe/scene-cut heuristics would miss single-frame flashes). Cost is
  bounded by downscaling during decode and by caching results. Timelapse has few
  frames anyway; long real-time 4K is a heavier one-time scan with a progress
  bar.

## Explicitly out of scope for v1 (future enhancements)

- **Multi-frame stacking / denoise-combine** to produce an even cleaner shot
  ("keep it beautiful for now").
- Batch processing of multiple videos.
- Auto-picking the best frame within a strike.

## Open defaults the user may still override

- Frame export default = PNG (TIFF optional).
- Timestamp export default = CSV + JSON (SRT optional).
- Keyboard map as listed above.
