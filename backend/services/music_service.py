import os
import subprocess
import logging
import shutil
from pathlib import Path
from typing import List, Optional
from config import Config

logger = logging.getLogger(__name__)

class MusicService:
    def __init__(self, music_dir: Optional[Path] = None):
        self.music_dir = music_dir or Config.MUSIC_DIR
        self.music_dir.mkdir(parents=True, exist_ok=True)
        self.current_process: Optional[subprocess.Popen] = None
        self.current_track: Optional[str] = None

    def get_catalog(self) -> List[str]:
        """Obtiene la lista de canciones disponibles en la carpeta de música"""
        tracks = []
        for ext in ("*.mp3", "*.wav", "*.ogg", "*.flac"):
            for f in self.music_dir.glob(ext):
                tracks.append(f.stem)
        return tracks

    def find_track(self, song_name: str, artist: Optional[str] = None) -> Optional[Path]:
        """Busca una pista que coincida con el nombre de la canción y/o artista"""
        song_lower = song_name.lower().strip()
        artist_lower = (artist or "").lower().strip()

        for ext in ("*.mp3", "*.wav", "*.ogg", "*.flac"):
            for f in self.music_dir.glob(ext):
                filename_lower = f.stem.lower()
                if song_lower in filename_lower:
                    if not artist_lower or artist_lower in filename_lower:
                        return f
        # Si no encontró coincidencia exacta, devolver la primera que contenga alguna palabra clave
        for f in self.music_dir.iterdir():
            if f.suffix.lower() in (".mp3", ".wav", ".ogg", ".flac"):
                if any(w in f.stem.lower() for w in song_lower.split()):
                    return f
        return None

    def play(self, song_name: str, artist: Optional[str] = None) -> bool:
        """Detiene cualquier canción activa y reproduce la solicitada"""
        self.stop()
        track_path = self.find_track(song_name, artist)
        if not track_path:
            logger.warning(f"Canción '{song_name}' no encontrada en catálogo local.")
            return False

        try:
            # Buscar reproductor disponible en la Raspberry Pi
            if shutil.which("mpv"):
                self.current_process = subprocess.Popen(["mpv", "--no-video", str(track_path)])
            elif shutil.which("mpg123") and track_path.suffix.lower() == ".mp3":
                self.current_process = subprocess.Popen(["mpg123", "-q", str(track_path)])
            elif shutil.which("aplay") and track_path.suffix.lower() == ".wav":
                self.current_process = subprocess.Popen(["aplay", "-q", str(track_path)])
            elif shutil.which("ffplay"):
                self.current_process = subprocess.Popen(["ffplay", "-nodisp", "-autoexit", str(track_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                logger.error("No se encontró ningún reproductor multimedia (mpv, mpg123, aplay, ffplay)")
                return False

            self.current_track = track_path.stem
            return True
        except Exception as e:
            logger.error(f"Error al reproducir música: {e}")
            return False

    def stop(self) -> bool:
        """Detiene la reproducción activa"""
        if self.current_process:
            try:
                self.current_process.terminate()
                self.current_process.wait(timeout=2)
            except Exception:
                try:
                    self.current_process.kill()
                except Exception:
                    pass
            self.current_process = None
            self.current_track = None
            return True
        return False

    def is_playing(self) -> bool:
        if self.current_process and self.current_process.poll() is None:
            return True
        self.current_process = None
        self.current_track = None
        return False
