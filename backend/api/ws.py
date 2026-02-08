"""WebSocket endpoint for real-time analysis progress streaming.

Streams :class:`AnalysisStatusResponse` updates to connected clients at
a configurable interval while an analysis is running.  The connection is
closed gracefully once the analysis completes or fails.

Routing note
------------
The WebSocket path is ``/ws/analysis/{repo_id}``.  Because ``main.py``
mounts the combined API router under ``/api``, this endpoint will be
reachable at ``/api/ws/analysis/{repo_id}``.  If a bare ``/ws`` prefix
is preferred, mount ``ws_router`` directly on the application instead.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.api.analysis import get_analysis_progress

logger = logging.getLogger(__name__)
router = APIRouter()

# Interval in seconds between progress pushes to connected clients.
_PUSH_INTERVAL: float = 0.5


@router.websocket("/ws/analysis/{repo_id}")
async def analysis_progress_ws(websocket: WebSocket, repo_id: str) -> None:
    """Stream analysis progress updates over a WebSocket connection.

    The server sends a JSON-serialised :class:`AnalysisStatusResponse`
    every ``_PUSH_INTERVAL`` seconds while an analysis is in progress
    for the given *repo_id*.

    The connection is closed with code **1000** (normal closure) when the
    analysis reaches a terminal state (``completed`` or ``failed``).
    If no analysis entry exists for the repo, a single ``not_found``
    message is sent before closing.
    """
    await websocket.accept()

    progress_store = get_analysis_progress()

    try:
        # If there is no progress entry at all, notify and close.
        progress = progress_store.get(repo_id)
        if progress is None:
            await websocket.send_json({
                "snapshot_id": "",
                "status": "not_found",
                "progress": 0.0,
                "current_stage": "No analysis in progress for this repository",
            })
            await websocket.close(code=1000)
            return

        # Stream updates until the analysis reaches a terminal state.
        while True:
            progress = progress_store.get(repo_id)
            if progress is None:
                await websocket.close(code=1000)
                return

            await websocket.send_json(progress.model_dump())

            if progress.status in ("completed", "failed"):
                await websocket.close(code=1000)
                return

            await asyncio.sleep(_PUSH_INTERVAL)

    except WebSocketDisconnect:
        logger.debug(
            "WebSocket client disconnected for repo_id=%s", repo_id,
        )
    except Exception:
        logger.exception(
            "Unexpected error in analysis WebSocket for repo_id=%s",
            repo_id,
        )
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
