"""Single-service public demo: the API under ``/api`` and the built dashboard at ``/``.

Build the dashboard with ``VITE_API_BASE_URL=/api``, then run
``uvicorn api.demo:app``.  Set ``DEMO_READ_ONLY=1`` for a public deployment.
"""
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.main import app as api

DIST = Path(os.getenv("DASHBOARD_DIST", Path(__file__).resolve().parents[1] / "dashboard" / "dist")).resolve()

app = FastAPI(title="CI/CD AI Optimizer demo", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/api", api)
app.mount("/assets", StaticFiles(directory=DIST / "assets", check_dir=False), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def dashboard(path: str) -> FileResponse:
    """Static files by path; every other route is the single-page app's."""
    file = (DIST / path).resolve()
    if path and file.is_file() and DIST in file.parents:
        return FileResponse(file)
    return FileResponse(DIST / "index.html")
