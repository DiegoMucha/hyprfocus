import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent          # hyprfocus/backend
HYPRFOCUS_ROOT = BASE_DIR.parent                   # hyprfocus
WORKSPACE_ROOT = HYPRFOCUS_ROOT.parent             # Directorio raíz del proyecto / usuario

# Cargar .env desde el backend o desde hyprfocus
load_dotenv(BASE_DIR / ".env")
load_dotenv(HYPRFOCUS_ROOT / ".env")
load_dotenv(WORKSPACE_ROOT / ".env")

class Config:
    # Google Gemini
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    # Local STT (whisper.cpp)
    # Por defecto busca en la ruta de la Pi o local
    WHISPER_ROOT_CANDIDATES = [
        HYPRFOCUS_ROOT / "whisper.cpp",
        WORKSPACE_ROOT / "whisper.cpp",
        Path("/home/juanjose/whisper.cpp"),
    ]
    WHISPER_ROOT = next(
        (path for path in WHISPER_ROOT_CANDIDATES if path.exists()),
        WHISPER_ROOT_CANDIDATES[0],
    )
    DEFAULT_WHISPER_CLI = WHISPER_ROOT / "build" / "bin" / "whisper-cli"
    WHISPER_CLI_PATH = os.getenv("WHISPER_CLI_PATH", str(DEFAULT_WHISPER_CLI))
    DEFAULT_WHISPER_MODEL = WHISPER_ROOT / "models" / "ggml-base.bin"
    WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", str(DEFAULT_WHISPER_MODEL))
    WHISPER_THREADS = int(os.getenv("WHISPER_THREADS", "4"))

    # Local TTS (Piper)
    # Busca en tts/ relativo o en la ruta de usuario
    TTS_CANDIDATES = [
        WORKSPACE_ROOT / "tts" / "es_AR-daniela-high.onnx",
        HYPRFOCUS_ROOT / "tts" / "es_AR-daniela-high.onnx",
        Path("/home/juanjose/tts/es_AR-daniela-high.onnx"),
        WORKSPACE_ROOT / "tts" / "es_MX-ald-medium.onnx"
    ]
    FOUND_ONNX = next((p for p in TTS_CANDIDATES if p.exists()), TTS_CANDIDATES[0])
    PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", str(FOUND_ONNX))
    PIPER_CONFIG_PATH = os.getenv("PIPER_CONFIG_PATH", f"{PIPER_MODEL_PATH}.json")
    PIPER_BIN = os.getenv("PIPER_BIN", "piper")

    # Directorios de datos
    UPLOADS_DIR = BASE_DIR / "uploads"
    MUSIC_DIR = BASE_DIR / "music"
    AUDIO_OUTPUT_DIR = BASE_DIR / "generated_audio"

    # Red
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))

    # Reproducción de audio local en la Raspberry Pi
    PLAY_ON_PI_SPEAKER = os.getenv("PLAY_ON_PI_SPEAKER", "true").lower() in ("1", "true", "yes")

    # Hyprfocus (Visión Computacional)
    HYPRFOCUS_DIR = HYPRFOCUS_ROOT
    HYPRFOCUS_FACE_MODEL = os.getenv("HYPRFOCUS_FACE_MODEL", str(HYPRFOCUS_ROOT / "face_landmarker.task"))
    HYPRFOCUS_GESTURE_MODEL = os.getenv("HYPRFOCUS_GESTURE_MODEL", str(HYPRFOCUS_ROOT / "gesture_recognizer.task"))
    HYPRFOCUS_CAMERA = os.getenv("HYPRFOCUS_CAMERA", "picamera2")
    HYPRFOCUS_DEVICE = int(os.getenv("HYPRFOCUS_DEVICE", "0"))
    SYSTEM_PYTHON_DIST_PACKAGES = Path(
        os.getenv("SYSTEM_PYTHON_DIST_PACKAGES", "/usr/lib/python3/dist-packages")
    )
    REQUIRE_CLOSED_FIST = os.getenv("REQUIRE_CLOSED_FIST", "true").lower() in ("1", "true", "yes")
    FIST_ACTIVATION_TIMEOUT = float(os.getenv("FIST_ACTIVATION_TIMEOUT", "8.0"))
    FIST_TRIGGER_COOLDOWN = float(os.getenv("FIST_TRIGGER_COOLDOWN", "2.0"))
    AUTO_START_VISION = os.getenv("AUTO_START_VISION", "true").lower() in ("1", "true", "yes")
    VISION_STARTUP_TIMEOUT = float(os.getenv("VISION_STARTUP_TIMEOUT", "30.0"))

    # Captura local del micrófono activada por el gesto Closed_Fist
    MICROPHONE_DEVICE = os.getenv("MICROPHONE_DEVICE", "default")
    MICROPHONE_RECORD_SECONDS = float(os.getenv("MICROPHONE_RECORD_SECONDS", "5.0"))

# Crear directorios necesarios
Config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
Config.MUSIC_DIR.mkdir(parents=True, exist_ok=True)
Config.AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
