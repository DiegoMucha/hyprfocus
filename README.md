# Hyprfocus

Headless sleep-state and closed-fist detection for Raspberry Pi 5 using a
Raspberry Pi Camera, MediaPipe Face Landmarker, and Gesture Recognizer.

This is an attention heuristic, not a medical sleep detector. A missing or
occluded face is reported as `NO_FACE`, never as sleeping.

## Raspberry Pi setup

Use 64-bit Raspberry Pi OS and verify the camera first:

```bash
rpicam-hello -n -t 3000
```

Install Picamera2 and environment tools:

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-venv
```

From this repository, create a uv environment with Raspberry Pi OS's system
Python. Using `/usr/bin/python3` is important because Picamera2 is installed for
that interpreter:

```bash
uv venv --clear --python /usr/bin/python3 --system-site-packages
source .venv/bin/activate
uv sync --active --python /usr/bin/python3
python -c "from picamera2 import Picamera2; print('Picamera2 OK')"
```

The models are expected at `./face_landmarker.task` and
`./gesture_recognizer.task`, as included in this repo.

## Run

```bash
hyprfocus
```

The program runs without a desktop and emits JSON status lines:

```json
{"state": "AWAKE", "sleeping": false, "ear": 0.284, "eyes_closed_seconds": 0.0, "closed_fist": false, "closed_fist_score": null, "closed_fist_hand": null}
{"state": "SLEEPING", "sleeping": true, "ear": 0.139, "eyes_closed_seconds": 3.07, "closed_fist": true, "closed_fist_score": 0.91, "closed_fist_hand": "Left"}
```

Stop it with `Ctrl+C`.

Useful calibration options:

```bash
hyprfocus --closed-threshold 0.20 --drowsy-after 1 --sleeping-after 3
```

Watch the `ear` value with your eyes naturally open and closed. Set
`--closed-threshold` between those values. Larger thresholds classify eyes as
closed more readily. Glasses, camera angle, lighting, and distance affect the
measurement.

Only MediaPipe's `Closed_Fist` category is enabled and exposed. Adjust its
minimum confidence if needed:

```bash
hyprfocus --gesture-threshold 0.70
```

For development with a USB webcam instead of Picamera2:

```bash
hyprfocus --camera opencv --device 0
```

## Detection states

- `NO_FACE`: no usable face landmarks; the closed-eye timer is reset.
- `AWAKE`: eyes are open, or have only just closed.
- `DROWSY`: eyes have remained closed for `--drowsy-after` seconds.
- `SLEEPING`: eyes have remained closed for `--sleeping-after` seconds.

## Tests

```bash
uv sync --all-groups
uv run pytest -q
```
