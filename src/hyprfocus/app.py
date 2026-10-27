from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

from .camera import create_camera
from .gesture_detector import CLOSED_FIST_LABEL, closed_fist_from_result
from .sleep_detector import SleepDetector, SleepState


LOGGER = logging.getLogger("hyprfocus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect sleep state and closed fists with MediaPipe."
    )
    parser.add_argument(
        "--face-model", type=Path, default=Path("face_landmarker.task")
    )
    parser.add_argument(
        "--gesture-model", type=Path, default=Path("gesture_recognizer.task")
    )
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
    parser.add_argument("--gesture-threshold", type=float, default=0.60)
    parser.add_argument(
        "--report-interval",
        type=float,
        default=0.5,
        help="Seconds between status lines (state changes print immediately).",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if not args.face_model.is_file():
        raise FileNotFoundError(f"Face Landmarker model not found: {args.face_model}")
    if not args.gesture_model.is_file():
        raise FileNotFoundError(
            f"Gesture Recognizer model not found: {args.gesture_model}"
        )

    detector = SleepDetector(
        closed_threshold=args.closed_threshold,
        drowsy_after=args.drowsy_after,
        sleeping_after=args.sleeping_after,
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(args.face_model.resolve())
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    gesture_options = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(args.gesture_model.resolve())
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        canned_gesture_classifier_options=processors.ClassifierOptions(
            score_threshold=args.gesture_threshold,
            category_allowlist=[CLOSED_FIST_LABEL],
        ),
    )

    started = time.monotonic()
    last_report = float("-inf")
    previous_state: SleepState | None = None
    previous_closed_fist: bool | None = None

    with vision.FaceLandmarker.create_from_options(face_options) as landmarker:
        with vision.GestureRecognizer.create_from_options(
            gesture_options
        ) as gesture_recognizer:
            _run_camera_loop(
                args,
                detector,
                landmarker,
                gesture_recognizer,
                started,
                last_report,
                previous_state,
                previous_closed_fist,
            )


def _run_camera_loop(
    args: argparse.Namespace,
    detector: SleepDetector,
    landmarker: vision.FaceLandmarker,
    gesture_recognizer: vision.GestureRecognizer,
    started: float,
    last_report: float,
    previous_state: SleepState | None,
    previous_closed_fist: bool | None,
) -> None:
        with create_camera(
            args.camera, args.width, args.height, args.fps, args.device
        ) as camera:
            LOGGER.info("Camera and both MediaPipe models started; press Ctrl+C to stop")
            while True:
                frame = camera.read_rgb()
                now = time.monotonic()
                timestamp_ms = int((now - started) * 1000)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                face_result = landmarker.detect_for_video(image, timestamp_ms)
                gesture_result = gesture_recognizer.recognize_for_video(
                    image, timestamp_ms
                )
                landmarks = (
                    face_result.face_landmarks[0]
                    if face_result.face_landmarks
                    else None
                )
                sleep = detector.update(landmarks, now)
                closed_fist = closed_fist_from_result(gesture_result)

                if (
                    sleep.state != previous_state
                    or closed_fist.detected != previous_closed_fist
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
                                "closed_fist": closed_fist.detected,
                                "closed_fist_score": (
                                    round(closed_fist.score, 3)
                                    if closed_fist.score is not None
                                    else None
                                ),
                                "closed_fist_hand": closed_fist.handedness,
                            }
                        ),
                        flush=True,
                    )
                    previous_state = sleep.state
                    previous_closed_fist = closed_fist.detected
                    last_report = now


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run(parse_args())
    except KeyboardInterrupt:
        LOGGER.info("Stopped")


if __name__ == "__main__":
    main()
