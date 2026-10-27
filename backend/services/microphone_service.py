import logging
import math
import re
import shutil
import subprocess
import wave
from array import array
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
        errors: list[str] = []
        for device in self._candidate_devices(arecord):
            output_path.unlink(missing_ok=True)
            command = [
                arecord,
                "-q",
                "-D",
                device,
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
                device,
                duration_seconds,
            )
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=max(5.0, duration_seconds + 5.0),
            )
            if (
                result.returncode == 0
                and output_path.exists()
                and output_path.stat().st_size > 44
            ):
                self._log_recording_levels(output_path)
                if device != self.device:
                    logger.info("Micrófono ALSA seleccionado automáticamente: %s", device)
                    self.device = device
                return output_path
            errors.append(f"{device}: {result.stderr.strip() or 'sin audio'}")

        details = " | ".join(errors) or "arecord no encontró dispositivos de captura"
        raise RuntimeError(
            f"No se pudo abrir un micrófono ALSA. {details}. "
            "Ejecuta 'arecord -l' y configura MICROPHONE_DEVICE=plughw:CARD,DEVICE."
        )

    def _candidate_devices(self, arecord: str) -> list[str]:
        """Return configured device followed by physical ALSA capture devices."""
        candidates = [self.device]
        listing = subprocess.run(
            [arecord, "-l"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        for card, device in re.findall(
            r"card\s+(\d+):.*?device\s+(\d+):",
            listing.stdout,
            flags=re.IGNORECASE,
        ):
            candidate = f"plughw:{card},{device}"
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _log_recording_levels(output_path: Path) -> None:
        """Log enough WAV signal information to distinguish silence from STT failure."""
        try:
            with wave.open(str(output_path), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                frame_count = wav_file.getnframes()
                frames = wav_file.readframes(frame_count)

            if sample_width != 2:
                logger.warning(
                    "No se pudo medir el nivel del micrófono: WAV usa %s bits",
                    sample_width * 8,
                )
                return

            samples = array("h")
            samples.frombytes(frames)
            if not samples:
                logger.warning("La grabación del micrófono no contiene muestras de audio")
                return

            peak = max(abs(sample) for sample in samples)
            rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
            duration = frame_count / sample_rate if sample_rate else 0.0
            logger.info(
                "Audio capturado: %.2fs, %sHz, %s canal(es), RMS %.1f%%, pico %.1f%%",
                duration,
                sample_rate,
                channels,
                rms / 32768 * 100,
                peak / 32768 * 100,
            )
            if peak < 328:
                logger.warning(
                    "La grabación está prácticamente en silencio (pico menor a 1%%)"
                )
        except (OSError, EOFError, wave.Error) as exc:
            logger.warning("No se pudo analizar el WAV capturado: %s", exc)
