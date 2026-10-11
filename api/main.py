"""FastAPI application entry point."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.auth import router as auth_router
from api.routers.pipelines import router as pipelines_router
from api.routers.webhooks import router as webhooks_router
from api.routers.predict import router as predict_router


app = FastAPI(title="CI/CD AI Optimizer API")
cors_origins = [origin.strip() for origin in os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5173").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(pipelines_router)
app.include_router(webhooks_router)
app.include_router(predict_router)
