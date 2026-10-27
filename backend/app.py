import os
import uuid
import shutil
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from config import Config
from services.openai_service import OpenAIService
from services.tts_service import TTSService
from services.stt_service import STTService
from services.document_service import DocumentService
from services.music_service import MusicService
from services.command_executor import CommandExecutor
from services.vision_service import VisionService
from services.microphone_service import MicrophoneService

# Configuración de logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("HyperBackend")

app = FastAPI(
    title="Hyper Assistant Backend",
    description="Backend para Raspberry Pi: Whisper STT, Piper TTS, OpenAI, Visión Hyprfocus y Ejecución de Comandos",
    version="1.1.0"
)

# Permitir CORS desde cualquier origen (útil para app de escritorio y clientes web en la red local)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar servicios
openai_service = OpenAIService()
tts_service = TTSService()
stt_service = STTService()
document_service = DocumentService()
music_service = MusicService()
command_executor = CommandExecutor(openai_service, tts_service, music_service, document_service)
microphone_service = MicrophoneService()

# Cache de archivos de audio generados para streaming
audio_cache: Dict[str, str] = {}
voice_capture_state: Dict[str, Any] = {
    "status": "idle",
    "last_transcription": None,
    "last_error": None,
    "last_result": None,
    "microphone_device": microphone_service.device,
    "whisper_cli": stt_service.cli_path,
    "whisper_model": stt_service.model_path,
}
voice_capture_lock: Optional[asyncio.Lock] = None

def handle_vision_alert(alert_description: str):
    """Callback invocado por VisionService cuando hyprfocus detecta somnolencia/ojos cerrados"""
    logger.info(f"Despachando evento de visión automático: {alert_description}")
    asyncio.run_coroutine_threadsafe(
        trigger_automatic_vision_alert(alert_description),
        main_event_loop
    )

async def trigger_automatic_vision_alert(alert_text: str):
    cmd_id = f"CMD_VIS_{uuid.uuid4().hex[:4].upper()}"
    command_json = openai_service.interpret_user_command(
        user_message="Alerta de visión computacional",
        command_id=cmd_id,
        vision_event=alert_text
    )
    result = await command_executor.execute(command_json)
    audio_file_path = result.get("audio_local_path")
    if audio_file_path and os.path.exists(audio_file_path):
        audio_id = f"{cmd_id}.wav"
        audio_cache[audio_id] = audio_file_path


def handle_closed_fist():
    """Called by VisionService's reader thread on a new Closed_Fist edge."""
    if main_event_loop is None or main_event_loop.is_closed():
        logger.error("No hay event loop disponible para activar el micrófono")
        return
    asyncio.run_coroutine_threadsafe(capture_fist_voice_command(), main_event_loop)


async def capture_fist_voice_command():
    """Record and execute one local voice command after a fist trigger."""
    global voice_capture_lock
    if voice_capture_lock is None:
        voice_capture_lock = asyncio.Lock()
    if voice_capture_lock.locked():
        logger.info("Se ignoró el gesto: ya hay una captura de voz en progreso")
        return

    async with voice_capture_lock:
        voice_capture_state.update(
            status="recording", last_error=None, last_transcription=None
        )
        recording = Config.AUDIO_OUTPUT_DIR / "temp" / f"fist_{uuid.uuid4().hex[:8]}.wav"
        try:
            await asyncio.to_thread(
                microphone_service.record,
                recording,
                Config.MICROPHONE_RECORD_SECONDS,
            )
            voice_capture_state["microphone_device"] = microphone_service.device
            voice_capture_state["status"] = "transcribing"
            transcription = await asyncio.to_thread(stt_service.transcribe, str(recording))
            if not transcription or transcription.startswith("Error:"):
                raise RuntimeError(transcription or "Whisper no detectó voz")

            voice_capture_state["last_transcription"] = transcription
            voice_capture_state["status"] = "executing"
            cmd_id = f"CMD_FIST_{uuid.uuid4().hex[:4].upper()}"
            active_doc = document_service.get_active_document()
            doc_context = (
                f"Archivo activo: {active_doc['filename']}" if active_doc else None
            )
            command_json = await asyncio.to_thread(
                openai_service.interpret_user_command,
                user_message=transcription,
                command_id=cmd_id,
                active_commands=command_executor.get_active_commands_list(),
                music_catalog=music_service.get_catalog(),
                pending_clarification=command_executor.pending_clarification,
                document_context=doc_context,
            )
            result = await command_executor.execute(command_json)
            voice_capture_state["last_result"] = result

            audio_file_path = result.get("audio_local_path")
            if audio_file_path and os.path.exists(audio_file_path):
                audio_cache[f"{cmd_id}.wav"] = audio_file_path
            voice_capture_state["status"] = "idle"
            logger.info("Comando por puño ejecutado: %s", transcription)
        except Exception as exc:
            voice_capture_state.update(status="error", last_error=str(exc))
            logger.exception("Falló la captura de voz activada por puño")
        finally:
            try:
                recording.unlink(missing_ok=True)
            except Exception:
                pass

# Servicio de visión (Hyprfocus: MediaPipe Face Landmarker + Gesture Recognizer)
vision_service = VisionService(
    on_vision_alert_callback=handle_vision_alert,
    on_closed_fist_callback=handle_closed_fist,
)

main_event_loop = None

@app.on_event("startup")
async def startup_event():
    global main_event_loop, voice_capture_lock
    main_event_loop = asyncio.get_running_loop()
    voice_capture_lock = asyncio.Lock()
    if Config.AUTO_START_VISION:
        started = await asyncio.to_thread(vision_service.start)
        if not started:
            logger.error("El backend inició, pero el servicio de visión no está listo")

@app.on_event("shutdown")
def shutdown_event():
    vision_service.stop()

class ChatRequest(BaseModel):
    message: str

class VisionEventRequest(BaseModel):
    event: str

class SetActiveDocRequest(BaseModel):
    document_id: str

@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "Hyper Voice & Study Assistant",
        "model": Config.OPENAI_MODEL,
        "active_document": document_service.get_active_document()["filename"] if document_service.get_active_document() else None,
        "active_commands_count": len(command_executor.active_commands),
        "vision_running": vision_service.is_running,
        "require_closed_fist": Config.REQUIRE_CLOSED_FIST
    }

@app.post("/api/chat")
async def process_chat_message(request: ChatRequest):
    """
    Procesa un mensaje de texto del usuario.
    Ejecuta el reconocimiento de intención con OpenAI y activa la función correspondiente.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")

    cmd_id = f"CMD_{uuid.uuid4().hex[:4].upper()}"
    active_doc = document_service.get_active_document()
    doc_context = f"Archivo activo: {active_doc['filename']} (longitud: {active_doc['text_length']} caracteres)" if active_doc else None

    # Interpretar con OpenAI
    command_json = openai_service.interpret_user_command(
        user_message=request.message,
        command_id=cmd_id,
        active_commands=command_executor.get_active_commands_list(),
        music_catalog=music_service.get_catalog(),
        pending_clarification=command_executor.pending_clarification,
        document_context=doc_context
    )

    # Ejecutar acción
    result = await command_executor.execute(command_json)

    # Si se generó audio, registrar en cache
    audio_file_path = result.get("audio_local_path")
    if audio_file_path and os.path.exists(audio_file_path):
        audio_id = f"{cmd_id}.wav"
        audio_cache[audio_id] = audio_file_path
        result["audio_url"] = f"/api/audio/{audio_id}"

    return {
        "user_message": request.message,
        "command": result,
        "active_document": active_doc["filename"] if active_doc else None
    }

@app.post("/api/voice")
async def process_voice_audio(file: UploadFile = File(...)):
    """
    Recibe un audio de voz del usuario (WAV, WEBM, OGG):
    SEGURIDAD / GESTO:
    Para que el usuario pueda ejecutar comandos por voz, debe mostrar un PUÑO CERRADO
    a la cámara (reconocido por hyprfocus).
    """
    # 0. Verificación del gesto de puño cerrado
    auth = vision_service.is_voice_authorized()
    if not auth["authorized"]:
        deny_msg = "Debes mostrar un puño cerrado a la cámara para activar el micrófono y los comandos por voz."
        cmd_id = f"CMD_DENY_{uuid.uuid4().hex[:4].upper()}"
        
        # Sintetizar mensaje de advertencia
        audio_path = tts_service.synthesize(deny_msg, play_locally=True)
        if audio_path and os.path.exists(audio_path):
            audio_cache[f"{cmd_id}.wav"] = audio_path

        return JSONResponse(
            status_code=403,
            content={
                "voice_authorized": False,
                "error": "fist_required",
                "message": deny_msg,
                "command": {
                    "ID_comando": cmd_id,
                    "Comando": "Alerta",
                    "Mensaje para responder al usuario": deny_msg,
                    "audio_url": f"/api/audio/{cmd_id}.wav" if audio_path else None
                }
            }
        )

    temp_dir = Config.AUDIO_OUTPUT_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "audio.wav").suffix or ".wav"
    temp_input = temp_dir / f"in_{uuid.uuid4().hex[:6]}{ext}"

    try:
        content = await file.read()
        with open(temp_input, "wb") as f:
            f.write(content)

        # 1. Transcripción local STT
        transcription = stt_service.transcribe(str(temp_input))
        if not transcription:
            transcription = "No se pudo transcribir el audio."

        # 2. Interpretación de comandos y ejecución
        cmd_id = f"CMD_{uuid.uuid4().hex[:4].upper()}"
        active_doc = document_service.get_active_document()
        doc_context = f"Archivo activo: {active_doc['filename']}" if active_doc else None

        command_json = openai_service.interpret_user_command(
            user_message=transcription,
            command_id=cmd_id,
            active_commands=command_executor.get_active_commands_list(),
            music_catalog=music_service.get_catalog(),
            pending_clarification=command_executor.pending_clarification,
            document_context=doc_context
        )

        result = await command_executor.execute(command_json)

        # Audio URL
        audio_file_path = result.get("audio_local_path")
        if audio_file_path and os.path.exists(audio_file_path):
            audio_id = f"{cmd_id}.wav"
            audio_cache[audio_id] = audio_file_path
            result["audio_url"] = f"/api/audio/{audio_id}"

        return {
            "voice_authorized": True,
            "transcription": transcription,
            "command": result,
            "active_document": active_doc["filename"] if active_doc else None
        }
    finally:
        if temp_input.exists():
            try:
                temp_input.unlink()
            except Exception:
                pass

# --- ENDPOINTS DE VISIÓN COMPUTACIONAL (HYPRFOCUS) ---

@app.get("/api/vision/status")
def get_vision_status():
    """Devuelve el estado en tiempo real de la cámara, somnolencia y puño cerrado"""
    return {
        **vision_service.get_status(),
        "voice_capture": dict(voice_capture_state),
    }

@app.post("/api/vision/start")
def start_vision():
    """Inicia el detector hyprfocus en la Raspberry Pi"""
    success = vision_service.start()
    return {"status": "started" if success else "failed", "is_running": vision_service.is_running}

@app.post("/api/vision/stop")
def stop_vision():
    """Detiene el detector hyprfocus"""
    vision_service.stop()
    return {"status": "stopped", "is_running": False}

@app.post("/api/vision/simulate-fist")
def simulate_fist(duration: float = 10.0):
    """Simula la detección de puño cerrado por N segundos (útil para pruebas y desarrollo)"""
    vision_service.simulate_fist_gesture(duration_seconds=duration)
    return {"status": "simulated", "duration": duration, "voice_authorized": True}

@app.post("/api/vision-event")
async def process_vision_event(req: VisionEventRequest):
    """
    Recibe una alerta del sistema de visión computacional (ej. postura o distancia a la pantalla).
    Tiene máxima prioridad según funcionalidades.md.
    """
    cmd_id = f"CMD_VIS_{uuid.uuid4().hex[:4].upper()}"
    command_json = openai_service.interpret_user_command(
        user_message="Alerta de visión computacional",
        command_id=cmd_id,
        vision_event=req.event
    )
    result = await command_executor.execute(command_json)
    audio_file_path = result.get("audio_local_path")
    if audio_file_path and os.path.exists(audio_file_path):
        audio_id = f"{cmd_id}.wav"
        audio_cache[audio_id] = audio_file_path
        result["audio_url"] = f"/api/audio/{audio_id}"

    return {"event": req.event, "command": result}

# --- GESTIÓN DE ARCHIVOS Y ESTUDIO ---

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Permite subir archivos (PDF, TXT, MD, etc.) para que Hyper pueda generar resúmenes
    y flashcards de su contenido.
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    doc_info = document_service.save_and_process_upload(file.filename or "archivo.txt", content)
    return {
        "status": "success",
        "document": {
            "id": doc_info["id"],
            "filename": doc_info["filename"],
            "text_length": doc_info["text_length"],
            "preview": doc_info["preview"]
        }
    }

@app.get("/api/documents")
def get_documents():
    return {
        "documents": document_service.list_documents(),
        "active_document": document_service.get_active_document()
    }

@app.post("/api/documents/set-active")
def set_active_document(req: SetActiveDocRequest):
    success = document_service.set_active_document(req.document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Documento no encontrado.")
    return {"status": "success", "active_document": req.document_id}

# --- TEMPORIZADORES Y RESULTADOS ---

@app.get("/api/timers")
def get_active_timers():
    """Devuelve los temporizadores activos en la Raspberry Pi"""
    return {
        "timers": command_executor.active_commands
    }

@app.post("/api/timers/{cmd_id}/cancel")
async def cancel_timer(cmd_id: str):
    cmd_json = {
        "Comando": "Detener comando",
        "Comando a detener": "Timer",
        "ID_objetivo": cmd_id,
        "Mensaje para responder al usuario": f"Temporizador {cmd_id} cancelado manualmente."
    }
    result = await command_executor.execute(cmd_json)
    return result

@app.get("/api/results")
def get_latest_results():
    """Devuelve el último resumen y lote de flashcards generados"""
    return {
        "summary": command_executor.latest_summary,
        "flashcards": command_executor.latest_flashcards
    }

@app.get("/api/music/catalog")
def get_music_catalog():
    return {
        "catalog": music_service.get_catalog(),
        "is_playing": music_service.is_playing(),
        "current_track": music_service.current_track
    }

@app.post("/api/music/stop")
def stop_music():
    stopped = music_service.stop()
    return {"status": "stopped" if stopped else "not_playing"}

@app.get("/api/audio/{audio_id}")
def stream_audio(audio_id: str):
    """Permite al frontend reproducir el archivo de audio sintetizado por Piper"""
    path = audio_cache.get(audio_id)
    if not path or not os.path.exists(path):
        direct_path = Config.AUDIO_OUTPUT_DIR / audio_id
        if direct_path.exists():
            path = str(direct_path)
        else:
            raise HTTPException(status_code=404, detail="Audio no encontrado.")
    return FileResponse(path, media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=Config.HOST, port=Config.PORT, reload=True)
