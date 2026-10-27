from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


class Camera(ABC):
    @abstractmethod
    def read_rgb(self) -> np.ndarray:
        """Return one uint8 RGB frame."""

    @abstractmethod
    def close(self) -> None:
        """Release camera resources."""

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class PiCamera(Camera):
    def __init__(self, width: int, height: int, fps: int) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2 is unavailable. Install it with: "
                "sudo apt install python3-picamera2"
            ) from exc

        self._camera = Picamera2()
        config = self._camera.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": fps},
            buffer_count=4,
        )
        self._camera.configure(config)
        self._camera.start()

    def read_rgb(self) -> np.ndarray:
        # Picamera2 RGB888 is laid out as BGR bytes for OpenCV-style arrays.
        bgr = self._camera.capture_array("main")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self._camera.stop()
        self._camera.close()


class OpenCVCamera(Camera):
    def __init__(self, device: int, width: int, height: int, fps: int) -> None:
        self._capture = cv2.VideoCapture(device)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_FPS, fps)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open camera device {device}")

    def read_rgb(self) -> np.ndarray:
        ok, bgr = self._capture.read()
        if not ok:
            raise RuntimeError("Camera stopped returning frames")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self._capture.release()


def create_camera(
    backend: str, width: int, height: int, fps: int, device: int
) -> Camera:
    if backend == "picamera2":
        return PiCamera(width, height, fps)
    if backend == "opencv":
        return OpenCVCamera(device, width, height, fps)
    raise ValueError(f"Unknown camera backend: {backend}")
