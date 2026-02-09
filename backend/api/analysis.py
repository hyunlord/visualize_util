"""Analysis trigger and status API endpoints.

Provides endpoints to start a full code-flow analysis on a repository
and to query the progress of a running or completed analysis run.

Analysis runs are executed as background ``asyncio`` tasks.  Progress
state is tracked in a module-level dictionary so it can be read by the
status endpoint and the WebSocket progress stream.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import async_session_factory, get_db
from backend.git_ops.diff import get_current_sha
from backend.models.db_models import AnalysisSnapshot, Repository
from backend.models.schemas import AnalysisRequest, AnalysisStatusResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Module-level progress store, keyed by repo_id.
# Imported by the WebSocket module to stream updates.
_analysis_progress: dict[str, AnalysisStatusResponse] = {}


def get_analysis_progress() -> dict[str, AnalysisStatusResponse]:
    """Return a reference to the shared progress dictionary.

    Exposed as a function so other modules can import it without
    creating circular import issues.
    """
    return _analysis_progress


# ---------------------------------------------------------------------------
# Background analysis runner
# ---------------------------------------------------------------------------


async def _run_analysis(repo_id: str, snapshot_id: str, language: str = "en") -> None:
    """Execute a full analysis in the background.

    Uses its own database session (not the request session) because
    the originating HTTP request has already returned.
    """
    # Lazy import to avoid circular dependency -- the engine module
    # may import models that reference the same registry.
    from backend.analyzer.engine import AnalysisEngine
    from backend.analyzer.language_registry import get_default_registry

    progress = _analysis_progress.get(repo_id)
    if progress is None:
        return

    async with async_session_factory() as session:
        try:
            repo = await session.get(Repository, repo_id)
            snapshot = await session.get(AnalysisSnapshot, snapshot_id)
            if repo is None or snapshot is None:
                logger.error(
                    "Analysis aborted: repo or snapshot not found "
                    "(repo_id=%s, snapshot_id=%s)",
                    repo_id,
                    snapshot_id,
                )
                _analysis_progress[repo_id] = AnalysisStatusResponse(
                    snapshot_id=snapshot_id,
                    status="failed",
                    progress=0.0,
                    current_stage="Repository or snapshot not found",
                )
                return

            snapshot.status = "running"
            await session.commit()

            _analysis_progress[repo_id] = AnalysisStatusResponse(
                snapshot_id=snapshot_id,
                status="running",
                progress=0.0,
                current_stage="Initialising analysis engine",
            )

            def progress_callback(stage: str, pct: float) -> None:
                """Update the shared progress state.

                Called by the analysis engine at each significant step.
                Signature matches engine: (stage_name, percentage).
                """
                _analysis_progress[repo_id] = AnalysisStatusResponse(
                    snapshot_id=snapshot_id,
                    status="running",
                    progress=round(min(pct, 100.0), 1),
                    current_stage=stage,
                )

            registry = get_default_registry()
            engine = AnalysisEngine(session, registry)
            await engine.run_full_analysis(repo, snapshot, progress_callback, language=language)

            snapshot.status = "completed"
            repo.last_analyzed_at = datetime.now(timezone.utc)
            repo.last_commit_sha = snapshot.commit_sha
            await session.commit()

            _analysis_progress[repo_id] = AnalysisStatusResponse(
                snapshot_id=snapshot_id,
                status="completed",
                progress=100.0,
                current_stage="Analysis complete",
            )

        except Exception:
            logger.exception(
                "Analysis failed for repo_id=%s snapshot_id=%s",
                repo_id,
                snapshot_id,
            )
            try:
                if snapshot is not None:
                    snapshot.status = "failed"
                    await session.commit()
            except Exception:
                logger.exception("Failed to mark snapshot as failed")

            _analysis_progress[repo_id] = AnalysisStatusResponse(
                snapshot_id=snapshot_id,
                status="failed",
                progress=_analysis_progress.get(
                    repo_id, AnalysisStatusResponse(
                        snapshot_id=snapshot_id,
                        status="failed",
                        progress=0.0,
                        current_stage="",
                    ),
                ).progress,
                current_stage="Analysis failed due to an internal error",
            )


# ---------------------------------------------------------------------------
# POST /repos/{repo_id}/analyze  --  start a new analysis
# ---------------------------------------------------------------------------


@router.post(
    "/{repo_id}/analyze",
    response_model=AnalysisStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a full code-flow analysis",
)
async def start_analysis(
    repo_id: str,
    body: AnalysisRequest | None = None,
    db: AsyncSession = Depends(get_db),
) -> AnalysisStatusResponse:
    """Trigger a full analysis of the repository.

    Creates a new :class:`AnalysisSnapshot` and launches the analysis
    engine as a background task.  Returns immediately with the snapshot
    id and initial status so the client can poll or connect via
    WebSocket for progress updates.
    """
    repo = await db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository not found: {repo_id}",
        )

    # Prevent starting a new analysis while one is already running
    existing = _analysis_progress.get(repo_id)
    if existing is not None and existing.status == "running":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An analysis is already running for this repository",
        )

    # Read current HEAD SHA from the repo on disk
    try:
        commit_sha = get_current_sha(repo.local_path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read current commit SHA: {exc}",
        ) from exc

    snapshot = AnalysisSnapshot(
        repo_id=repo_id,
        commit_sha=commit_sha,
        status="pending",
    )
    db.add(snapshot)
    await db.flush()
    await db.refresh(snapshot)

    progress = AnalysisStatusResponse(
        snapshot_id=snapshot.id,
        status="pending",
        progress=0.0,
        current_stage="Queued for analysis",
    )
    _analysis_progress[repo_id] = progress

    # Launch the background task -- it manages its own DB session
    language = body.language if body else "en"
    asyncio.create_task(
        _run_analysis(repo_id, snapshot.id, language=language),
        name=f"analysis-{repo_id}",
    )

    return progress


# ---------------------------------------------------------------------------
# GET /repos/{repo_id}/analyze/status  --  latest analysis status
# ---------------------------------------------------------------------------


@router.get(
    "/{repo_id}/analyze/status",
    response_model=AnalysisStatusResponse,
    summary="Get latest analysis status",
)
async def get_analysis_status(
    repo_id: str,
    db: AsyncSession = Depends(get_db),
) -> AnalysisStatusResponse:
    """Return the progress of the most recent analysis for a repository.

    If a background analysis is currently running, the live progress
    from the in-memory store is returned.  Otherwise the status is
    derived from the latest :class:`AnalysisSnapshot` in the database.
    """
    # Check in-memory progress first (covers running analyses)
    live_progress = _analysis_progress.get(repo_id)
    if live_progress is not None:
        return live_progress

    # Fall back to the most recent snapshot in the database
    repo = await db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository not found: {repo_id}",
        )

    result = await db.execute(
        select(AnalysisSnapshot)
        .where(AnalysisSnapshot.repo_id == repo_id)
        .order_by(AnalysisSnapshot.analyzed_at.desc())
        .limit(1)
    )
    snapshot = result.scalar_one_or_none()

    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analysis has been run for this repository",
        )

    progress = 100.0 if snapshot.status == "completed" else 0.0
    stage = (
        "Analysis complete"
        if snapshot.status == "completed"
        else "Analysis failed" if snapshot.status == "failed"
        else "Pending"
    )

    return AnalysisStatusResponse(
        snapshot_id=snapshot.id,
        status=snapshot.status,
        progress=progress,
        current_stage=stage,
    )
