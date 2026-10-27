import os
import subprocess
import logging
import uuid
import shutil
from pathlib import Path
from typing import Optional
from config import Config

logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        self.model_path = model_path or Config.PIPER_MODEL_PATH
        self.config_path = config_path or Config.PIPER_CONFIG_PATH
        self.piper_bin = Config.PIPER_BIN
        self.output_dir = Config.AUDIO_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(self, text: str, play_locally: Optional[bool] = None) -> Optional[str]:
        """
        Sintetiza texto a voz usando Piper ONNX.
        Devuelve el path absoluto del archivo .wav generado.
        Si play_locally es True (o por defecto según Config), lo reproduce en el parlante de la Raspberry Pi.
        """
        if not text or not text.strip():
            return None

        # Limpiar texto de caracteres problemáticos para CLI
        clean_text = text.replace('"', '').replace('\n', ' ').strip()
        filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
        output_file = self.output_dir / filename

        # 1. Intentar con el ejecutable piper (CLI estándar en Linux / Raspberry Pi)
        cmd = [
            self.piper_bin,
            "--model", str(self.model_path),
            "--output_file", str(output_file),
            "--", clean_text
        ]

        success = False
        try:
            # En Linux/Pi se suele usar echo text | piper ... o argumento directo
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20
            )
            if process.returncode == 0 and output_file.exists():
                success = True
            else:
                logger.warning(f"Piper CLI retorno código {process.returncode}: {process.stderr}")
        except Exception as e:
            logger.warning(f"Error ejecutando piper CLI ({e}), intentando via piping...")

        # Fallback a echo | piper si el argumento directo falló
        if not success:
            try:
                p = subprocess.Popen(
                    [self.piper_bin, "--model", str(self.model_path), "--output_file", str(output_file)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = p.communicate(input=clean_text, timeout=20)
                if p.returncode == 0 and output_file.exists():
                    success = True
                else:
                    logger.error(f"Fallback piper fallo: {stderr}")
            except Exception as e2:
                logger.error(f"Error total en sintesis TTS con Piper: {e2}")

        if not success or not output_file.exists():
            logger.error("No se pudo generar el archivo de audio con Piper.")
            return None

        # Reproducir localmente en el altavoz de la Raspberry Pi si está habilitado
        should_play = play_locally if play_locally is not None else Config.PLAY_ON_PI_SPEAKER
        if should_play:
            self.play_audio_locally(str(output_file))

        return str(output_file)

    def play_audio_locally(self, audio_path: str):
        """
        Reproduce un archivo WAV en la salida de audio de la Raspberry Pi usando aplay o paplay.
        Se ejecuta en segundo plano para no bloquear.
        """
        try:
            if shutil.which("aplay"):
                subprocess.Popen(["aplay", "-q", audio_path])
            elif shutil.which("paplay"):
                subprocess.Popen(["paplay", audio_path])
            elif shutil.which("pw-play"):
                subprocess.Popen(["pw-play", audio_path])
            else:
                logger.info(f"No se encontro reproductor nativo (aplay/paplay) para: {audio_path}")
        except Exception as err:
            logger.warning(f"No se pudo reproducir audio en la salida local: {err}")
