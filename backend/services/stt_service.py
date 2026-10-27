import os
import subprocess
import logging
import shutil
from pathlib import Path
from typing import Optional
from config import Config

logger = logging.getLogger(__name__)

class STTService:
    def __init__(self, cli_path: Optional[str] = None, model_path: Optional[str] = None, threads: int = 4):
        self.cli_path = self._resolve_cli(cli_path or Config.WHISPER_CLI_PATH)
        self.model_path = self._resolve_model(model_path or Config.WHISPER_MODEL_PATH)
        self.threads = threads or Config.WHISPER_THREADS

    @staticmethod
    def _resolve_cli(configured_path: str) -> str:
        candidates = [Path(configured_path), Path(Config.DEFAULT_WHISPER_CLI)]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        global_cli = shutil.which("whisper-cli")
        return global_cli or configured_path

    @staticmethod
    def _resolve_model(configured_path: str) -> str:
        candidates = [Path(configured_path), Path(Config.DEFAULT_WHISPER_MODEL)]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return configured_path

    def convert_to_wav(self, input_path: str) -> str:
        """
        Convierte cualquier formato de audio recibido (webm, ogg, mp3) a WAV 16kHz mono,
        que es el formato óptimo requerido por Whisper.
        """
        in_p = Path(input_path)
        if in_p.suffix.lower() == ".wav":
            return input_path

        out_wav = in_p.with_suffix(".16k.wav")
        if shutil.which("ffmpeg"):
            cmd = [
                "ffmpeg", "-y", "-i", str(in_p),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                str(out_wav)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return str(out_wav)
        return input_path

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe un archivo de audio a texto en español utilizando whisper-cli.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Archivo de audio no encontrado: {audio_path}")

        wav_path = self.convert_to_wav(audio_path)

        # Verificar si existe el binario de whisper-cli
        if not shutil.which(self.cli_path) and not os.path.exists(self.cli_path):
            logger.warning(f"Binario whisper-cli no encontrado en: {self.cli_path}")
            # Si no está instalado, intentar comando global 'whisper-cli' o 'whisper'
            if shutil.which("whisper-cli"):
                self.cli_path = "whisper-cli"
            else:
                return "Error: whisper-cli no está instalado o configurado en la Raspberry Pi."

        if not os.path.isfile(self.model_path):
            return f"Error: modelo Whisper no encontrado en {self.model_path}"

        # Ejecutar whisper.cpp cli
        cmd = [
            str(self.cli_path),
            "-m", str(self.model_path),
            "-f", str(wav_path),
            "-l", "es",
            "-t", str(self.threads),
            "--no-timestamps"
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=45
            )
            if result.returncode != 0:
                details = " ".join(result.stderr.split())[-1200:]
                message = (
                    f"whisper-cli terminó con código {result.returncode}: "
                    f"{details or 'sin detalles en stderr'}"
                )
                logger.error(message)
                return f"Error: {message}"
            raw_text = result.stdout.strip()
            # Limpiar líneas de logs de whisper.cpp (e.g. system_info, whisper_print_timings)
            lines = [line.strip() for line in raw_text.splitlines() if line.strip() and not line.startswith("whisper_") and not line.startswith("system_info")]
            transcribed = " ".join(lines).strip()
            if not transcribed:
                logger.warning(
                    "whisper-cli terminó correctamente pero no detectó texto; "
                    "revisa el nivel RMS/pico registrado justo antes de esta línea"
                )
            logger.info(f"Transcripción obtenida: '{transcribed}'")
            return transcribed
        except subprocess.TimeoutExpired:
            message = "whisper-cli excedió el tiempo máximo de 45 segundos"
            logger.error(message)
            return f"Error: {message}"
        except Exception as exc:
            message = f"no se pudo ejecutar whisper-cli: {exc}"
            logger.exception(message)
            return f"Error: {message}"
