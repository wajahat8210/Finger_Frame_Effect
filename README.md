# Finger Frame Effect (Python + OpenCV)

A Python/OpenCV clone of [sophiamyang/finger-frame-effect](https://github.com/sophiamyang/finger-frame-effect):
hold up both hands in an "L" shape like a photographer framing a shot, and a live visual effect (pixelate,
blur, invert, noir, glitch, toon, or a Van Gogh-style painterly filter) is applied only inside the frame your
fingers make — tracked live from your webcam.

Originally built in JavaScript (browser + MediaPipe WASM + Canvas). This version reproduces the same idea
in Python using OpenCV for video/image processing and MediaPipe's HandLandmarker for hand tracking.

## Features

- Live webcam feed with real-time hand tracking (up to 2 hands)
- Custom gesture detection for the "L" shape (thumb + index extended, other fingers curled)
- Smoothed, jitter-free frame tracking (EMA smoothing + flicker debounce)
- 7 switchable effects, clipped to the tracked frame:
  - **1** — Pixelate
  - **2** — Blur
  - **3** — Invert
  - **4** — Noir
  - **5** — Glitch
  - **6** — Toon
  - **7** — Van Gogh (painterly stylization)
- On-screen FPS counter and help overlay
- Downscaled detection input for better performance on modest hardware

## Requirements

- Python **3.9–3.12** (MediaPipe does not yet support 3.13+)
- A webcam
- macOS, Linux, or Windows

## Setup

```bash
# 1. Clone or download this project, then move into its folder
cd finger-frame-effect-py

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\Activate.ps1       # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the hand-tracking model (must sit in the project folder)
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

> Windows PowerShell, instead of `curl`:
> ```powershell
> Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "hand_landmarker.task"
> ```

## Usage

```bash
python main.py
```

Hold up both hands and form an "L" shape with each — thumb and index finger spread apart, other three
fingers curled in — to open the frame. Move your hands to move the frame around.

**Controls**

| Key | Action |
|---|---|
| `1`–`7` | Switch the active effect |
| `h` | Toggle the on-screen help overlay |
| `q` | Quit |

## Project structure

```
finger-frame-effect-py/
├── main.py              # Entry point: webcam loop, FPS counter, key handling
├── gesture.py            # MediaPipe hand tracking + "L shape" detection + smoothing
├── effects.py             # The 7 effects + mask-based compositing into the frame
├── requirements.txt
└── hand_landmarker.task  # MediaPipe model file (downloaded, not committed to git)
```

## Troubleshooting

- **`Model file not found`** — make sure `hand_landmarker.task` is in the same folder as `main.py`.
- **`pip install mediapipe` fails** — check your Python version is 3.9–3.12 (`python --version`).
- **Low FPS / laggy** — close other apps using the camera, or lower `detection_width` in
  `GestureTracker(...)` inside `gesture.py` (e.g. from `480` to `360`).
- **Frame flickers on/off** — raise `lost_grace_frames` in `GestureTracker(...)` for more tolerance to
  brief missed detections.
- **The frame never appears** — spread your thumb and index further apart and curl the other three fingers
  in more tightly; the detection thresholds are generous but not infinite.

## Credits

- Inspired by [sophiamyang/finger-frame-effect](https://github.com/sophiamyang/finger-frame-effect)
- Hand tracking via [Google MediaPipe](https://github.com/google-ai-edge/mediapipe)
- Built with [OpenCV](https://opencv.org/) and [NumPy](https://numpy.org/)
