"""API router aggregation for the Code Flow Visualizer.

Combines all endpoint sub-routers into a single ``router`` that is mounted
by the main application under the ``/api`` prefix.  The WebSocket router
is exported separately so that ``main.py`` can mount it without the
``/api`` prefix if desired.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.api.repos import router as repos_router
from backend.api.analysis import router as analysis_router
from backend.api.graph import router as graph_router
from backend.api.features import router as features_router
from backend.api.dead_code import router as dead_code_router
from backend.api.ws import router as ws_router

router = APIRouter()

router.include_router(repos_router, prefix="/repos", tags=["repositories"])
router.include_router(analysis_router, prefix="/repos", tags=["analysis"])
router.include_router(graph_router, prefix="/repos", tags=["graph"])
router.include_router(features_router, prefix="/repos", tags=["features"])
router.include_router(dead_code_router, prefix="/repos", tags=["dead-code"])
router.include_router(ws_router, tags=["websocket"])
