import logging
import subprocess
import sys
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from services.microphone_service import MicrophoneService  # noqa: E402
from services.stt_service import STTService  # noqa: E402


def test_microphone_logs_non_silent_signal_level(tmp_path, caplog) -> None:
    wav_path = tmp_path / "voice.wav"
    samples = array("h", [0, 4096, -4096, 8192, -8192] * 3200)
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(samples.tobytes())

    with caplog.at_level(logging.INFO):
        MicrophoneService._log_recording_levels(wav_path)

    assert "Audio capturado: 1.00s, 16000Hz, 1 canal(es)" in caplog.text
    assert "prácticamente en silencio" not in caplog.text


def test_stt_reports_whisper_stderr_on_failure(tmp_path, monkeypatch) -> None:
    cli_path = tmp_path / "whisper-cli"
    model_path = tmp_path / "model.bin"
    wav_path = tmp_path / "voice.wav"
    for path in (cli_path, model_path, wav_path):
        path.touch()

    service = STTService.__new__(STTService)
    service.cli_path = str(cli_path)
    service.model_path = str(model_path)
    service.threads = 4
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="failed to load whisper model",
        ),
    )

    result = service.transcribe(str(wav_path))

    assert result == (
        "Error: whisper-cli terminó con código 1: failed to load whisper model"
    )
