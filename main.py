"""Workspace-root entrypoint for running the nested crop disease app with Uvicorn."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "crop_disease_detection"
if not PROJECT_DIR.is_dir():
    raise RuntimeError(f"Crop disease project directory was not found: {PROJECT_DIR}")

# The application imports `backend` and `ml` as top-level packages.
project_dir_text = str(PROJECT_DIR)
if project_dir_text not in sys.path:
    sys.path.insert(0, project_dir_text)

from crop_disease_detection.main import app  # noqa: E402

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
