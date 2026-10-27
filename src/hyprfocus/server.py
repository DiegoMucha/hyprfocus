from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"

backend_path = str(BACKEND_DIR)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)


def main() -> None:
    import uvicorn

    from app import app
    from config import Config

    print(f"Iniciando Hyper Backend en {Config.HOST}:{Config.PORT}...")
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)


if __name__ == "__main__":
    main()
