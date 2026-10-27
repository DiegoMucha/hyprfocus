import logging
import math
import shutil
import subprocess
from pathlib import Path

from config import Config


logger = logging.getLogger(__name__)


class MicrophoneService:
    """Records a Whisper-compatible WAV from the Pi's ALSA microphone."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or Config.MICROPHONE_DEVICE

    def record(self, output_path: Path, duration_seconds: float) -> Path:
        arecord = shutil.which("arecord")
        if not arecord:
            raise RuntimeError(
                "arecord is not installed. Install it with: sudo apt install alsa-utils"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            arecord,
            "-q",
            "-D",
            self.device,
            "-f",
            "S16_LE",
            "-r",
            "16000",
            "-c",
            "1",
            "-d",
            str(max(1, math.ceil(duration_seconds))),
            str(output_path),
        ]
        logger.info(
            "Grabando micrófono ALSA '%s' durante %.1f segundos",
            self.device,
            duration_seconds,
        )
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(5.0, duration_seconds + 5.0),
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "arecord terminó con error")
        if not output_path.exists() or output_path.stat().st_size <= 44:
            raise RuntimeError("El micrófono no produjo audio")
        return output_path
