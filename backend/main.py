"""FastAPI application entry-point for the Code Flow Visualizer."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import get_settings
from backend.database import init_db

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup / shutdown lifecycle hook.

    On startup:
    - Ensure required data directories exist.
    - Initialise the database (create tables if needed).
    """
    logger.info("Starting Code Flow Visualizer v%s", _app.version)
    settings.ensure_directories()
    await init_db()
    logger.info("Database initialised, data dirs ready")
    yield
    logger.info("Shutting down Code Flow Visualizer")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Code Flow Visualizer",
    version="0.1.0",
    lifespan=lifespan,
)

# -- CORS (permissive for local development) --------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- API Routers ------------------------------------------------------------

def _register_routers() -> None:
    """Import and include API routers.

    Wrapped in a function so import errors surface clearly during startup
    rather than at module-import time.
    """
    try:
        from backend.api import router as api_router  # noqa: WPS433
        from backend.api.ws import router as ws_router  # noqa: WPS433

        app.include_router(api_router, prefix="/api")
        # Mount WebSocket router at root level so /ws/... paths work
        # without the /api prefix (matching frontend proxy config).
        app.include_router(ws_router)
    except ImportError:
        logger.warning(
            "No API routers found in backend.api -- "
            "the /api prefix will have no endpoints."
        )


_register_routers()


# -- Root health-check endpoint ---------------------------------------------

@app.get("/", tags=["health"])
async def root() -> JSONResponse:
    """Minimal health-check returning application status and version."""
    return JSONResponse(
        content={"status": "ok", "version": app.version},
    )
