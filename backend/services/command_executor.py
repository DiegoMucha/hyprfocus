import asyncio
import logging
import uuid
import time
from typing import Dict, Any, List, Optional
from services.tts_service import TTSService
from services.music_service import MusicService
from services.openai_service import OpenAIService
from services.document_service import DocumentService

logger = logging.getLogger(__name__)

class CommandExecutor:
    def __init__(
        self,
        openai_service: OpenAIService,
        tts_service: TTSService,
        music_service: MusicService,
        document_service: DocumentService
    ):
        self.openai_service = openai_service
        self.tts_service = tts_service
        self.music_service = music_service
        self.document_service = document_service

        # Comandos activos (para cumplir con la regla de contexto de funcionalidades.md)
        self.active_commands: Dict[str, Dict[str, Any]] = {}
        # Estado de aclaración pendiente
        self.pending_clarification: Optional[Dict[str, Any]] = None
        # Últimos resultados generados (resúmenes y flashcards)
        self.latest_summary: Optional[Dict[str, Any]] = None
        self.latest_flashcards: Optional[Dict[str, Any]] = None
        # Tareas asyncio de timers
        self._timer_tasks: Dict[str, asyncio.Task] = {}

    def get_active_commands_list(self) -> List[Dict[str, Any]]:
        """Devuelve comandos activos para alimentar al modelo de OpenAI."""
        active = []
        for cmd_id, data in self.active_commands.items():
            active.append({
                "ID_comando": cmd_id,
                "Comando": data.get("Comando"),
                "Detalle": data.get("Detalle")
            })
        if self.music_service.is_playing():
            active.append({
                "ID_comando": "CMD_MUSIC",
                "Comando": "Reproducir Musica",
                "Detalle": self.music_service.current_track
            })
        return active

    async def execute(self, command_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta la acción correspondiente al JSON de Hyper y genera
        el audio de voz de respuesta si aplica.
        """
        cmd_type = command_json.get("Comando")
        cmd_id = command_json.get("ID_comando", f"CMD_{uuid.uuid4().hex[:4].upper()}")
        response_data = dict(command_json)
        response_data["execution_status"] = "success"

        # 1. Determinar el mensaje de voz a sintetizar para el usuario
        speech_text = (
            command_json.get("Mensaje para responder al usuario") or
            command_json.get("Mensaje para el usuario") or
            ""
        )

        # 2. Procesar según el comando
        if cmd_type == "Timer":
            duration = int(command_json.get("Tiempo", 0))
            end_message = command_json.get("Mensaje cuando termine el timer", "El temporizador ha terminado.")
            if duration > 0:
                self.active_commands[cmd_id] = {
                    "Comando": "Timer",
                    "Tiempo_total": duration,
                    "Iniciado_en": time.time(),
                    "Detalle": f"Timer de {duration} segundos"
                }
                # Iniciar tarea en segundo plano
                task = asyncio.create_task(self._run_timer(cmd_id, duration, end_message))
                self._timer_tasks[cmd_id] = task
                self.pending_clarification = None

        elif cmd_type == "Timer periodico":
            interval = int(command_json.get("Intervalo", 0))
            repeats = command_json.get("Repeticiones")
            interval_msg = command_json.get("Mensaje al cumplirse el intervalo", "Recordatorio periódico cumplido.")
            if interval > 0:
                self.active_commands[cmd_id] = {
                    "Comando": "Timer periodico",
                    "Intervalo": interval,
                    "Repeticiones": repeats,
                    "Iniciado_en": time.time(),
                    "Detalle": f"Periódico cada {interval}s"
                }
                task = asyncio.create_task(self._run_periodic_timer(cmd_id, interval, repeats, interval_msg))
                self._timer_tasks[cmd_id] = task
                self.pending_clarification = None

        elif cmd_type == "Detener comando":
            target_cmd = command_json.get("Comando a detener", "")
            target_id = command_json.get("ID_objetivo")

            stopped = False
            if target_cmd == "Reproducir Musica" or (not target_cmd and self.music_service.is_playing()):
                stopped = self.music_service.stop()
            else:
                # Detener temporizador
                if target_id and target_id in self._timer_tasks:
                    self._timer_tasks[target_id].cancel()
                    self._timer_tasks.pop(target_id, None)
                    self.active_commands.pop(target_id, None)
                    stopped = True
                else:
                    # Cancelar el primer temporizador activo
                    for tid, ttask in list(self._timer_tasks.items()):
                        ttask.cancel()
                        self._timer_tasks.pop(tid, None)
                        self.active_commands.pop(tid, None)
                        stopped = True
                        break
            response_data["action_executed"] = "stopped" if stopped else "none_active"

        elif cmd_type == "Reproducir Musica":
            song = command_json.get("Cancion", "")
            artist = command_json.get("Artista")
            started = self.music_service.play(song, artist)
            response_data["music_found"] = started
            if not started:
                speech_text = f"No encontré la canción {song} en la biblioteca local de la Raspberry Pi."
                response_data["Mensaje para responder al usuario"] = speech_text
            self.pending_clarification = None

        elif cmd_type == "Resumen":
            # Generar el resumen de contenido con OpenAI
            requested_text = command_json.get("Texto", "")
            instructions = command_json.get("Instrucciones")

            # Si el usuario hace referencia al documento cargado o no da texto largo
            active_doc = self.document_service.get_active_document()
            text_to_summarize = requested_text
            if active_doc and (not requested_text or "documento" in requested_text.lower() or len(requested_text) < 30):
                text_to_summarize = active_doc["content"]

            full_summary = self.openai_service.generate_full_summary(text_to_summarize, instructions)
            self.latest_summary = {
                "id": cmd_id,
                "tema": requested_text or (active_doc["filename"] if active_doc else "General"),
                "summary": full_summary,
                "timestamp": time.time()
            }
            response_data["generated_summary"] = self.latest_summary
            self.pending_clarification = None

        elif cmd_type == "Flashcards":
            topic = command_json.get("Tema", "")
            count = int(command_json.get("Cantidad", 5))
            active_doc = self.document_service.get_active_document()
            topic_or_context = topic
            if active_doc and (not topic or "documento" in topic.lower()):
                topic_or_context = f"{topic} basado en el archivo: {active_doc['content'][:8000]}"

            flashcards_list = self.openai_service.generate_flashcards_content(topic_or_context, count)
            self.latest_flashcards = {
                "id": cmd_id,
                "tema": topic,
                "cards": flashcards_list,
                "count": len(flashcards_list),
                "timestamp": time.time()
            }
            response_data["generated_flashcards"] = self.latest_flashcards
            self.pending_clarification = None

        elif cmd_type == "Aclaracion":
            # Guardar el estado para la siguiente consulta del usuario
            self.pending_clarification = {
                "Comando_pendiente": command_json.get("Comando pendiente"),
                "Dato_faltante": command_json.get("Dato faltante")
            }

        elif cmd_type == "Alerta":
            # Alerta directa de visión
            pass

        # 3. Síntesis de voz con Piper local
        audio_file_path = None
        if speech_text:
            try:
                audio_file_path = self.tts_service.synthesize(speech_text)
                if audio_file_path:
                    response_data["audio_url"] = f"/api/audio/{uuid.UUID(hex=uuid.uuid4().hex).hex}_{cmd_id}.wav"
                    response_data["audio_local_path"] = audio_file_path
            except Exception as e:
                logger.error(f"Error generando síntesis TTS: {e}")

        return response_data

    async def _run_timer(self, cmd_id: str, duration: int, end_message: str):
        """Tarea asíncrona para temporizador simple"""
        try:
            await asyncio.sleep(duration)
            logger.info(f"Timer {cmd_id} finalizado!")
            # Sintetizar mensaje de fin
            self.tts_service.synthesize(end_message, play_locally=True)
        except asyncio.CancelledError:
            logger.info(f"Timer {cmd_id} cancelado por el usuario.")
        finally:
            self.active_commands.pop(cmd_id, None)
            self._timer_tasks.pop(cmd_id, None)

    async def _run_periodic_timer(self, cmd_id: str, interval: int, repeats: Optional[int], interval_msg: str):
        """Tarea asíncrona para temporizador periódico"""
        count = 0
        try:
            while True:
                await asyncio.sleep(interval)
                count += 1
                logger.info(f"Timer periódico {cmd_id} - Alerta {count}")
                self.tts_service.synthesize(interval_msg, play_locally=True)
                if repeats is not None and count >= repeats:
                    break
        except asyncio.CancelledError:
            logger.info(f"Timer periódico {cmd_id} detenido.")
        finally:
            self.active_commands.pop(cmd_id, None)
            self._timer_tasks.pop(cmd_id, None)
