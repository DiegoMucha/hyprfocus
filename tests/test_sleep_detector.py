from dataclasses import dataclass

import pytest

from hyprfocus.sleep_detector import SleepDetector, SleepState


@dataclass
class Point:
    x: float
    y: float


def landmarks(ear: float) -> list[Point]:
    points = [Point(0.0, 0.0) for _ in range(478)]
    eyes = (
        ((33, 160, 158, 133, 153, 144), 0.0),
        ((362, 385, 387, 263, 373, 380), 3.0),
    )
    for indices, offset in eyes:
        p1, p2, p3, p4, p5, p6 = indices
        points[p1] = Point(offset, 0.0)
        points[p4] = Point(offset + 2.0, 0.0)
        points[p2] = Point(offset + 0.5, ear)
        points[p6] = Point(offset + 0.5, -ear)
        points[p3] = Point(offset + 1.5, ear)
        points[p5] = Point(offset + 1.5, -ear)
    return points


def test_sustained_closure_transitions_to_sleeping() -> None:
    detector = SleepDetector(closed_threshold=0.2, drowsy_after=1, sleeping_after=3)
    assert detector.update(landmarks(0.1), 10).state == SleepState.AWAKE
    assert detector.update(landmarks(0.1), 11).state == SleepState.DROWSY
    result = detector.update(landmarks(0.1), 13)
    assert result.state == SleepState.SLEEPING
    assert result.eyes_closed_seconds == pytest.approx(3)


def test_open_eyes_reset_timer() -> None:
    detector = SleepDetector()
    detector.update(landmarks(0.1), 10)
    assert detector.update(landmarks(0.3), 12).state == SleepState.AWAKE
    assert detector.update(landmarks(0.1), 13).eyes_closed_seconds == 0


def test_missing_face_is_not_sleeping() -> None:
    detector = SleepDetector()
    assert detector.update(None, 10).state == SleepState.NO_FACE
