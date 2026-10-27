import os
import sys
import json
import time
import shutil
import logging
import asyncio
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from config import Config

logger = logging.getLogger(__name__)

class VisionService:
    def __init__(
        self,
        on_vision_alert_callback: Optional[Callable[[str], Any]] = None,
        on_closed_fist_callback: Optional[Callable[[], Any]] = None,
    ):
        self.on_vision_alert_callback = on_vision_alert_callback
        self.on_closed_fist_callback = on_closed_fist_callback

        # Estado del detector
        self.is_running: bool = False
        self.process: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()
        self.last_error: Optional[str] = None
        self.last_stderr: Optional[str] = None

        # Datos en tiempo real
        self.state: str = "NO_FACE"
        self.sleeping: bool = False
        self.ear: Optional[float] = None
        self.eyes_closed_seconds: float = 0.0
        self.closed_fist: bool = False
        self.closed_fist_score: Optional[float] = None
        self.closed_fist_hand: Optional[str] = None
        self.last_fist_detected_time: float = 0.0
        self.last_fist_trigger_time: float = 0.0
        self._previous_closed_fist: bool = False

        # Alertas automáticas de somnolencia
        self.last_drowsy_alert_time: float = 0.0
        self.alert_cooldown_seconds: float = 25.0

        # Simulación manual (para pruebas o desarrollo sin cámara física)
        self.simulated_fist: bool = False
        self.simulated_fist_time: float = 0.0

    def is_voice_authorized(self) -> Dict[str, Any]:
        """
        Verifica si el usuario tiene permitido ejecutar comandos por voz.
        Regla: Debe mostrar un puño cerrado para activar el micrófono y los comandos por voz.
        Se otorga una ventana de gracia (FIST_ACTIVATION_TIMEOUT) para que pueda hablar con naturalidad.
        """
        if not Config.REQUIRE_CLOSED_FIST:
            return {"authorized": True, "reason": "Verificación de puño desactivada"}

        now = time.time()
        # Verificar detección real o simulada
        fist_active = self.closed_fist or self.simulated_fist
        time_since_fist = now - max(self.last_fist_detected_time, self.simulated_fist_time)

        if fist_active or (time_since_fist <= Config.FIST_ACTIVATION_TIMEOUT):
            remaining = max(0.0, Config.FIST_ACTIVATION_TIMEOUT - time_since_fist) if not fist_active else Config.FIST_ACTIVATION_TIMEOUT
            return {
                "authorized": True,
                "fist_detected": fist_active,
                "remaining_seconds": round(remaining, 1),
                "reason": "Puño cerrado verificado"
            }

        return {
            "authorized": False,
            "fist_detected": False,
            "remaining_seconds": 0.0,
            "reason": "Debes mostrar un puño cerrado a la cámara para activar el micrófono y los comandos por voz."
        }

    def simulate_fist_gesture(self, duration_seconds: float = 10.0):
        """Simula la detección de un puño cerrado para pruebas desde el frontend"""
        self.simulated_fist = True
        self.simulated_fist_time = time.time()
        logger.info(f"Puño cerrado simulado activado por {duration_seconds}s")
        if self.on_closed_fist_callback:
            self.on_closed_fist_callback()
        timer = threading.Timer(duration_seconds, self._clear_simulated_fist)
        timer.daemon = True
        timer.start()

    def _clear_simulated_fist(self):
        self.simulated_fist = False

    def get_status(self) -> Dict[str, Any]:
        """Retorna el estado actual de visión y autorización de voz"""
        auth_info = self.is_voice_authorized()
        return {
            "is_running": self.is_running,
            "ready": self._ready_event.is_set(),
            "last_error": self.last_error,
            "last_stderr": self.last_stderr,
            "state": self.state,
            "sleeping": self.sleeping,
            "ear": self.ear,
            "eyes_closed_seconds": self.eyes_closed_seconds,
            "closed_fist": self.closed_fist or self.simulated_fist,
            "closed_fist_score": self.closed_fist_score,
            "closed_fist_hand": self.closed_fist_hand,
            "voice_authorized": auth_info["authorized"],
            "authorization_details": auth_info
        }

    def start(self, startup_timeout: Optional[float] = None) -> bool:
        """Inicia el proceso de detección de hyprfocus"""
        if self.is_running:
            return True

        self._ready_event.clear()
        self.last_error = None
        self.last_stderr = None
        if startup_timeout is None:
            startup_timeout = Config.VISION_STARTUP_TIMEOUT

        face_task = Path(Config.HYPRFOCUS_FACE_MODEL)
        gesture_task = Path(Config.HYPRFOCUS_GESTURE_MODEL)

        if not face_task.exists() or not gesture_task.exists():
            logger.warning("Modelos de MediaPipe (.task) no encontrados en hyprfocus.")
            return False

        # Comando para ejecutar hyprfocus
        cmd = [
            sys.executable,
            "-m", "hyprfocus.app",
            "--camera", Config.HYPRFOCUS_CAMERA,
            "--device", str(Config.HYPRFOCUS_DEVICE),
            "--face-model", str(face_task),
            "--gesture-model", str(gesture_task),
            "--report-interval", "0.2"
        ]

        # Añadir la carpeta src de hyprfocus al PYTHONPATH
        env = os.environ.copy()
        src_path = str(Config.HYPRFOCUS_DIR / "src")
        python_paths = [src_path]
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            python_paths.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        env["HYPRFOCUS_SYSTEM_PYTHON_PACKAGES"] = str(
            Config.SYSTEM_PYTHON_DIST_PACKAGES
        )

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
                cwd=str(Config.HYPRFOCUS_DIR)
            )
            self.is_running = True
            self._reader_thread = threading.Thread(target=self._read_output_loop, daemon=True)
            self._stderr_thread = threading.Thread(target=self._read_stderr_loop, daemon=True)
            self._reader_thread.start()
            self._stderr_thread.start()

            if not self._ready_event.wait(timeout=startup_timeout):
                return_code = self.process.poll()
                reason = self.last_error or (
                    f"hyprfocus terminó con código {return_code}"
                    if return_code is not None
                    else "hyprfocus no produjo datos dentro del tiempo esperado"
                )
                logger.error("No se pudo iniciar visión: %s", reason)
                self.stop()
                self.last_error = reason
                return False

            logger.info("Módulo de Visión hyprfocus listo y produciendo datos.")
            return True
        except Exception as e:
            logger.error(f"Error al iniciar hyprfocus: {e}")
            self.is_running = False
            return False

    def stop(self):
        """Detiene el proceso de visión"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
        self.is_running = False
        self._ready_event.clear()
        self._previous_closed_fist = False
        logger.info("Módulo de Visión hyprfocus detenido.")

    def _read_output_loop(self):
        """Lee las líneas JSON emitidas por hyprfocus en tiempo real"""
        if not self.process or not self.process.stdout:
            return

        for line in self.process.stdout:
            line_str = line.strip()
            if not line_str or not line_str.startswith("{"):
                continue

            try:
                data = json.loads(line_str)
                self._ready_event.set()
                self.state = data.get("state", "NO_FACE")
                self.sleeping = data.get("sleeping", False)
                self.ear = data.get("ear")
                self.eyes_closed_seconds = data.get("eyes_closed_seconds", 0.0)

                # Detección de Puño Cerrado
                fist_detected = data.get("closed_fist", False)
                self.closed_fist = bool(fist_detected)
                self.closed_fist_score = data.get("closed_fist_score")
                self.closed_fist_hand = data.get("closed_fist_hand")

                now = time.time()
                if self.closed_fist:
                    self.last_fist_detected_time = now

                is_new_fist = self.closed_fist and not self._previous_closed_fist
                cooldown_passed = (
                    now - self.last_fist_trigger_time >= Config.FIST_TRIGGER_COOLDOWN
                )
                if is_new_fist and cooldown_passed:
                    self.last_fist_trigger_time = now
                    logger.info(
                        "✊ Nuevo puño cerrado (%s, score=%s). Activando micrófono.",
                        self.closed_fist_hand,
                        self.closed_fist_score,
                    )
                    if self.on_closed_fist_callback:
                        try:
                            self.on_closed_fist_callback()
                        except Exception as fist_err:
                            logger.error(
                                "Error despachando activación por puño: %s", fist_err
                            )
                self._previous_closed_fist = self.closed_fist

                # Detección de Somnolencia / Sueño
                if self.state in ("DROWSY", "SLEEPING"):
                    if now - self.last_drowsy_alert_time > self.alert_cooldown_seconds:
                        self.last_drowsy_alert_time = now
                        desc = f"Se detectó somnolencia en el estudiante (ojos cerrados durante {self.eyes_closed_seconds} segundos)."
                        logger.warning(f"⚠️ Alerta de visión: {desc}")
                        if self.on_vision_alert_callback:
                            try:
                                self.on_vision_alert_callback(desc)
                            except Exception as alert_err:
                                logger.error(f"Error despachando alerta de visión: {alert_err}")

            except json.JSONDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error procesando frame de visión: {e}")

        self.is_running = False
        self._ready_event.clear()
        self._previous_closed_fist = False

    def _read_stderr_loop(self):
        """Drain and expose MediaPipe, Picamera2 and child-process errors."""
        if not self.process or not self.process.stderr:
            return

        for line in self.process.stderr:
            message = line.strip()
            if not message:
                continue
            self.last_stderr = message
            is_error = any(
                marker in message
                for marker in (
                    "ERROR",
                    "Traceback",
                    "RuntimeError",
                    "ModuleNotFoundError",
                    "FileNotFoundError",
                )
            )
            if is_error:
                self.last_error = message
                logger.error("hyprfocus stderr: %s", message)
            else:
                logger.debug("hyprfocus stderr: %s", message)
