from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .camera import create_camera
from .sleep_detector import SleepDetector, SleepState


LOGGER = logging.getLogger("hyprfocus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect sustained eye closure with MediaPipe Face Landmarker."
    )
    parser.add_argument("--model", type=Path, default=Path("face_landmarker.task"))
    parser.add_argument(
        "--camera", choices=("picamera2", "opencv"), default="picamera2"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--closed-threshold", type=float, default=0.20)
    parser.add_argument("--drowsy-after", type=float, default=1.0)
    parser.add_argument("--sleeping-after", type=float, default=3.0)
    parser.add_argument(
        "--report-interval",
        type=float,
        default=0.5,
        help="Seconds between status lines (state changes print immediately).",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise FileNotFoundError(f"Face Landmarker model not found: {args.model}")

    detector = SleepDetector(
        closed_threshold=args.closed_threshold,
        drowsy_after=args.drowsy_after,
        sleeping_after=args.sleeping_after,
    )
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.model.resolve())),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    started = time.monotonic()
    last_report = float("-inf")
    previous_state: SleepState | None = None

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        with create_camera(
            args.camera, args.width, args.height, args.fps, args.device
        ) as camera:
            LOGGER.info("Camera started; press Ctrl+C to stop")
            while True:
                frame = camera.read_rgb()
                now = time.monotonic()
                timestamp_ms = int((now - started) * 1000)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                result = landmarker.detect_for_video(image, timestamp_ms)
                landmarks = result.face_landmarks[0] if result.face_landmarks else None
                sleep = detector.update(landmarks, now)

                if (
                    sleep.state != previous_state
                    or now - last_report >= args.report_interval
                ):
                    print(
                        json.dumps(
                            {
                                "state": sleep.state,
                                "sleeping": sleep.state == SleepState.SLEEPING,
                                "ear": (
                                    round(sleep.eye_aspect_ratio, 3)
                                    if sleep.eye_aspect_ratio is not None
                                    else None
                                ),
                                "eyes_closed_seconds": round(
                                    sleep.eyes_closed_seconds, 2
                                ),
                            }
                        ),
                        flush=True,
                    )
                    previous_state = sleep.state
                    last_report = now


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run(parse_args())
    except KeyboardInterrupt:
        LOGGER.info("Stopped")


if __name__ == "__main__":
    main()
