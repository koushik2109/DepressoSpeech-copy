"""MindScope Backend – FastAPI application factory."""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from config.settings import get_settings
from database import init_db

settings = get_settings()
logger = logging.getLogger("mindscope")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    # Startup: create tables and storage dir
    logger.info("Initializing database...")
    await init_db()
    Path(settings.STORAGE_LOCAL_PATH).mkdir(parents=True, exist_ok=True)
    logger.info("MindScope backend ready — listening on port %s", settings.APP_PORT)
    yield
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="MindScope API",
        description="Depression screening backend – PHQ-8 with voice analysis",
        version="1.0.0",
        lifespan=lifespan,
    )

    # GZip — compress JSON responses over 500 bytes for faster transfer
    app.add_middleware(GZipMiddleware, minimum_size=500)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request metrics (now batched — no per-request DB write)
    from src.middleware.metrics import MetricsMiddleware
    app.add_middleware(MetricsMiddleware)

    # Register routers
    from src.routes import (
        auth_router,
        assessments_router,
        audio_router,
        doctor_router,
        admin_router,
    )

    prefix = settings.API_V1_PREFIX
    app.include_router(auth_router, prefix=prefix)
    app.include_router(assessments_router, prefix=prefix)
    app.include_router(audio_router, prefix=prefix)
    app.include_router(doctor_router, prefix=prefix)
    app.include_router(admin_router, prefix=prefix)

    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "mindscope-backend"}

    return app


app = create_app()
