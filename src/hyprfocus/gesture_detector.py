from __future__ import annotations

from dataclasses import dataclass


CLOSED_FIST_LABEL = "Closed_Fist"


@dataclass(frozen=True)
class ClosedFistResult:
    detected: bool
    score: float | None
    handedness: str | None


def closed_fist_from_result(result: object) -> ClosedFistResult:
    """Extract only Closed_Fist from a MediaPipe GestureRecognizerResult."""
    gestures = getattr(result, "gestures", ())
    handedness = getattr(result, "handedness", ())

    best_score: float | None = None
    best_hand: str | None = None
    for hand_index, hand_gestures in enumerate(gestures):
        for category in hand_gestures:
            if category.category_name != CLOSED_FIST_LABEL:
                continue
            score = float(category.score)
            if best_score is None or score > best_score:
                best_score = score
                best_hand = _hand_name(handedness, hand_index)

    return ClosedFistResult(
        detected=best_score is not None,
        score=best_score,
        handedness=best_hand,
    )


def _hand_name(handedness: object, hand_index: int) -> str | None:
    if hand_index >= len(handedness) or not handedness[hand_index]:
        return None
    return str(handedness[hand_index][0].category_name)
