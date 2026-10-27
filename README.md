# Hyprfocus

Headless sleep-state detection for Raspberry Pi 5 using a Raspberry Pi Camera,
MediaPipe Face Landmarker, and sustained eye closure.

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

From this repository, create an environment that can see the system Picamera2
package and install the project:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The model is expected at `./face_landmarker.task`, as included in this repo.

## Run

```bash
hyprfocus
```

The program runs without a desktop and emits JSON status lines:

```json
{"state": "AWAKE", "sleeping": false, "ear": 0.284, "eyes_closed_seconds": 0.0}
{"state": "DROWSY", "sleeping": false, "ear": 0.142, "eyes_closed_seconds": 1.04}
{"state": "SLEEPING", "sleeping": true, "ear": 0.139, "eyes_closed_seconds": 3.07}
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
