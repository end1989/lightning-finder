# ⚡ Lightning Finder

Scan a video for lightning flashes, review each one in the browser, step
frame-by-frame to the exact peak, and grab that frame at full native quality.

## Install
```
pip install -r requirements.txt
```

## Run
```
python app.py path/to/your_video.mp4
```
A browser opens at http://127.0.0.1:8000. First open scans the whole video
(progress in the terminal) and caches results next to the file.

## Use
- Sidebar lists detected flashes — click to jump. `Shift+←/→` jumps flash to flash.
- `←/→` steps one frame (enters precision mode = exact decoded frames).
- `,` / `.` nudge by 0.1s. `Space` plays/pauses.
- `G` grabs the current frame to `output/<video>/` as a lossless native-res PNG.
- `C` / `X` confirm/reject the current flash.
- **CSV** / **JSON** buttons export the timestamps.
- Adjust **sensitivity** + **Rescan** if flashes are missed or over-detected.
- **⚡ Best Shots** ranks the flashes by *lightning-channel* strength instead of
  brightness. It scans each flash for a real, thin, vertical bolt in the sky
  (using temporal differencing so the static scene — buildings, city lights,
  horizon — cancels out), re-points each to its true bolt frame, and pushes the
  flashes that are only diffuse glow (no channel) to the bottom. Click an entry
  to jump straight to that exact frame. The pass takes a few minutes the first
  time and is then cached.

> Note: the plain flash list uses the *brightest* frame, which is often the sky
> glow a frame or two after the strike — use **Best Shots** to land on the
> actual lightning channel.

## Test
```
python generate_test_clip.py test_clip.mp4
pytest -v
```

## Not yet
Multi-frame stacking, batch processing, motion-stabilised differencing.
