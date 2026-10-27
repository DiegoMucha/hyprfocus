from dataclasses import dataclass

from hyprfocus.gesture_detector import closed_fist_from_result


@dataclass
class Category:
    category_name: str
    score: float


@dataclass
class Result:
    gestures: list[list[Category]]
    handedness: list[list[Category]]


def test_closed_fist_is_returned() -> None:
    result = Result(
        gestures=[[Category("Closed_Fist", 0.91)]],
        handedness=[[Category("Left", 0.98)]],
    )
    closed_fist = closed_fist_from_result(result)
    assert closed_fist.detected is True
    assert closed_fist.score == 0.91
    assert closed_fist.handedness == "Left"


def test_other_gestures_are_ignored() -> None:
    result = Result(
        gestures=[[Category("Thumb_Up", 0.99)]],
        handedness=[[Category("Right", 0.98)]],
    )
    closed_fist = closed_fist_from_result(result)
    assert closed_fist.detected is False
    assert closed_fist.score is None
    assert closed_fist.handedness is None


def test_missing_hand_is_not_a_closed_fist() -> None:
    closed_fist = closed_fist_from_result(Result(gestures=[], handedness=[]))
    assert closed_fist.detected is False
