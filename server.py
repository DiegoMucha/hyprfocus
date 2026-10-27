import sys
from pathlib import Path

# Asegurar que backend y hyprfocus/src estén en el path
HYPRFOCUS_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = HYPRFOCUS_ROOT / "backend"
SRC_DIR = HYPRFOCUS_ROOT / "src"

for d in (BACKEND_DIR, SRC_DIR):
    d_str = str(d)
    if d_str not in sys.path:
        sys.path.insert(0, d_str)

from app import app
from config import Config

def main():
    import uvicorn
    print(f"Iniciando Hyper Backend desde hyprfocus en {Config.HOST}:{Config.PORT}...")
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)

if __name__ == "__main__":
    main()
