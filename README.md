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

## Test
```
python generate_test_clip.py test_clip.mp4
pytest -v
```

## Not in v1
Multi-frame stacking, batch processing, auto-pick-best-frame-within-a-strike.
