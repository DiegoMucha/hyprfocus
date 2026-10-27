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
        self.cli_path = cli_path or Config.WHISPER_CLI_PATH
        self.model_path = model_path or Config.WHISPER_MODEL_PATH
        self.threads = threads or Config.WHISPER_THREADS

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
            raw_text = result.stdout.strip()
            # Limpiar líneas de logs de whisper.cpp (e.g. system_info, whisper_print_timings)
            lines = [line.strip() for line in raw_text.splitlines() if line.strip() and not line.startswith("whisper_") and not line.startswith("system_info")]
            transcribed = " ".join(lines).strip()
            logger.info(f"Transcripción obtenida: '{transcribed}'")
            return transcribed
        except Exception as e:
            logger.error(f"Error ejecutando whisper-cli: {e}")
            return ""
