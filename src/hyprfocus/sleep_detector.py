from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import hypot
from typing import Sequence


LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)


class SleepState(StrEnum):
    NO_FACE = "NO_FACE"
    AWAKE = "AWAKE"
    DROWSY = "DROWSY"
    SLEEPING = "SLEEPING"


@dataclass(frozen=True)
class SleepResult:
    state: SleepState
    eye_aspect_ratio: float | None
    eyes_closed_seconds: float


def _distance(a: object, b: object) -> float:
    return hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def eye_aspect_ratio(landmarks: Sequence[object], indices: Sequence[int]) -> float:
    """Calculate eye openness independent of face size."""
    p1, p2, p3, p4, p5, p6 = (landmarks[index] for index in indices)
    horizontal = 2.0 * _distance(p1, p4)
    if horizontal < 1e-8:
        return 0.0
    return (_distance(p2, p6) + _distance(p3, p5)) / horizontal


def mean_eye_aspect_ratio(landmarks: Sequence[object]) -> float:
    return (
        eye_aspect_ratio(landmarks, LEFT_EYE)
        + eye_aspect_ratio(landmarks, RIGHT_EYE)
    ) / 2.0


class SleepDetector:
    def __init__(
        self,
        closed_threshold: float = 0.20,
        drowsy_after: float = 1.0,
        sleeping_after: float = 3.0,
    ) -> None:
        if drowsy_after < 0 or sleeping_after <= drowsy_after:
            raise ValueError("sleeping_after must be greater than drowsy_after")
        self.closed_threshold = closed_threshold
        self.drowsy_after = drowsy_after
        self.sleeping_after = sleeping_after
        self._closed_since: float | None = None

    def update(
        self, landmarks: Sequence[object] | None, now: float
    ) -> SleepResult:
        if not landmarks:
            self._closed_since = None
            return SleepResult(SleepState.NO_FACE, None, 0.0)

        ear = mean_eye_aspect_ratio(landmarks)
        if ear >= self.closed_threshold:
            self._closed_since = None
            return SleepResult(SleepState.AWAKE, ear, 0.0)

        if self._closed_since is None:
            self._closed_since = now
        closed_for = max(0.0, now - self._closed_since)

        if closed_for >= self.sleeping_after:
            state = SleepState.SLEEPING
        elif closed_for >= self.drowsy_after:
            state = SleepState.DROWSY
        else:
            state = SleepState.AWAKE
        return SleepResult(state, ear, closed_for)
