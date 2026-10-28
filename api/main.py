"""FastAPI application entry point."""

import os
import re

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.demo_mode import read_only
from api.routers.pipelines import router as pipelines_router
from api.routers.webhooks import router as webhooks_router
from api.routers.predict import router as predict_router


# Writes still allowed in read-only mode: stateless what-if scoring, and build
# submission, which scores against stored history but does not save (see create_build).
READ_ONLY_ALLOWED = re.compile(r"^/predict/|^/pipelines/.+/builds$")

app = FastAPI(title="CI/CD AI Optimizer API")


@app.middleware("http")
async def block_writes_in_read_only_mode(request: Request, call_next):
    # Match the path relative to where the API is mounted (api.demo serves it under /api).
    path = request.url.path.removeprefix(request.scope.get("root_path", ""))
    if read_only() and request.method not in ("GET", "HEAD", "OPTIONS") and not READ_ONLY_ALLOWED.search(path):
        return JSONResponse(status_code=403, content={"detail": "This is a read-only demo; nothing can be changed."})
    return await call_next(request)


# Added last so it runs first: CORS headers reach the read-only 403 too.
cors_origins = [origin.strip() for origin in os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5173").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/status")
def status() -> dict:
    return {"read_only": read_only()}


app.include_router(pipelines_router)
app.include_router(webhooks_router)
app.include_router(predict_router)
