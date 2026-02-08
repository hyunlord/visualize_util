"""Dead code analysis API endpoint.

Queries the latest completed analysis snapshot for code nodes that have
been flagged as potentially dead (unreferenced) code and returns them
in the ``DeadCodeResponse`` format.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.database import get_db
from backend.models.db_models import (
    AnalysisSnapshot,
    CodeNode,
    Repository,
)
from backend.models.schemas import DeadCodeItem, DeadCodeResponse

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_latest_completed_snapshot(
    db: AsyncSession,
    repo_id: str,
) -> AnalysisSnapshot | None:
    """Return the most recent completed snapshot for a repository."""
    result = await db.execute(
        select(AnalysisSnapshot)
        .where(
            AnalysisSnapshot.repo_id == repo_id,
            AnalysisSnapshot.status == "completed",
        )
        .order_by(AnalysisSnapshot.analyzed_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


def _node_to_dead_code_item(node: CodeNode) -> DeadCodeItem:
    """Convert a dead-code :class:`CodeNode` to a :class:`DeadCodeItem`."""
    metadata = node.metadata_ or {}
    return DeadCodeItem(
        node_id=node.id,
        file_path=node.file_path,
        name=node.name,
        node_type=node.node_type,
        line_start=node.line_start,
        line_end=node.line_end,
        reason=metadata.get("dead_code_reason", "No incoming references detected"),
        confidence=metadata.get("dead_code_confidence", 0.5),
        llm_explanation=metadata.get("dead_code_llm_explanation"),
        suggested_feature=metadata.get("dead_code_suggested_feature"),
    )


# ---------------------------------------------------------------------------
# GET /repos/{repo_id}/dead-code  --  list dead code items
# ---------------------------------------------------------------------------


@router.get(
    "/{repo_id}/dead-code",
    response_model=DeadCodeResponse,
    summary="Get dead code analysis results",
)
async def get_dead_code(
    repo_id: str,
    db: AsyncSession = Depends(get_db),
) -> DeadCodeResponse:
    """Return all code nodes flagged as potentially dead (unreferenced)
    from the latest completed analysis snapshot.

    Dead code items include contextual information such as the reason
    for the flag, confidence score, and optional LLM-generated
    explanation.
    """
    repo = await db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository not found: {repo_id}",
        )

    snapshot = await _get_latest_completed_snapshot(db, repo_id)
    if snapshot is None:
        return DeadCodeResponse(items=[], total_count=0)

    result = await db.execute(
        select(CodeNode)
        .where(
            CodeNode.snapshot_id == snapshot.id,
            CodeNode.is_dead_code.is_(True),
        )
        .options(selectinload(CodeNode.feature))
        .order_by(CodeNode.file_path, CodeNode.line_start)
    )
    dead_nodes = list(result.scalars().all())

    items = [_node_to_dead_code_item(n) for n in dead_nodes]
    return DeadCodeResponse(items=items, total_count=len(items))
