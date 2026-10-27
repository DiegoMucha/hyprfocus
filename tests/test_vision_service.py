import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from services.vision_service import (  # noqa: E402
    DROWSY_ALERT_MESSAGE,
    SLEEPING_ALERT_MESSAGE,
    VisionService,
)


def detector_line(state: str) -> str:
    return json.dumps(
        {
            "state": state,
            "sleeping": state == "SLEEPING",
            "ear": 0.1,
            "eyes_closed_seconds": 3.0,
            "closed_fist": False,
        }
    )


def test_drowsy_and_sleeping_have_distinct_alerts() -> None:
    alerts: list[str] = []
    service = VisionService(on_vision_alert_callback=alerts.append)
    service.process = SimpleNamespace(
        stdout=io.StringIO(
            "\n".join(
                [
                    detector_line("AWAKE"),
                    detector_line("DROWSY"),
                    detector_line("DROWSY"),
                    detector_line("SLEEPING"),
                ]
            )
        )
    )

    service._read_output_loop()

    assert alerts == [DROWSY_ALERT_MESSAGE, SLEEPING_ALERT_MESSAGE]


def test_same_state_does_not_repeat_alert_on_every_report() -> None:
    alerts: list[str] = []
    service = VisionService(on_vision_alert_callback=alerts.append)
    service.process = SimpleNamespace(
        stdout=io.StringIO(
            "\n".join(detector_line("DROWSY") for _ in range(3))
        )
    )

    service._read_output_loop()

    assert alerts == [DROWSY_ALERT_MESSAGE]
