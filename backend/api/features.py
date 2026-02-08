"""Feature management API endpoints.

Provides endpoints to list detected features for a repository and to
update feature metadata (name, description, colour).
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.models.db_models import CodeNode, Feature, Repository
from backend.models.schemas import FeatureInfo

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request schemas local to this module
# ---------------------------------------------------------------------------


class FeatureUpdateRequest(BaseModel):
    """Payload for updating a feature's mutable properties."""

    name: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _build_feature_info(
    db: AsyncSession,
    feature: Feature,
) -> FeatureInfo:
    """Construct a :class:`FeatureInfo` with a live node count."""
    result = await db.execute(
        select(func.count(CodeNode.id)).where(
            CodeNode.feature_id == feature.id,
        )
    )
    node_count: int = result.scalar_one()

    return FeatureInfo(
        id=feature.id,
        name=feature.name,
        description=feature.description,
        color=feature.color,
        node_count=node_count,
        auto_detected=feature.auto_detected,
    )


# ---------------------------------------------------------------------------
# GET /repos/{repo_id}/features  --  list all features
# ---------------------------------------------------------------------------


@router.get(
    "/{repo_id}/features",
    response_model=list[FeatureInfo],
    summary="List all features for a repository",
)
async def list_features(
    repo_id: str,
    db: AsyncSession = Depends(get_db),
) -> list[FeatureInfo]:
    """Return every feature associated with a repository, including a count
    of nodes assigned to each feature.
    """
    repo = await db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository not found: {repo_id}",
        )

    # Fetch all features for the repo
    result = await db.execute(
        select(Feature)
        .where(Feature.repo_id == repo_id)
        .order_by(Feature.name)
    )
    features = list(result.scalars().all())

    # Build node counts in a single query
    counts_result = await db.execute(
        select(
            CodeNode.feature_id,
            func.count(CodeNode.id).label("node_count"),
        )
        .where(
            CodeNode.feature_id.in_([f.id for f in features]),
        )
        .group_by(CodeNode.feature_id)
    )
    counts_map: dict[str, int] = {
        row.feature_id: row.node_count for row in counts_result.all()
    }

    return [
        FeatureInfo(
            id=f.id,
            name=f.name,
            description=f.description,
            color=f.color,
            node_count=counts_map.get(f.id, 0),
            auto_detected=f.auto_detected,
        )
        for f in features
    ]


# ---------------------------------------------------------------------------
# PUT /repos/{repo_id}/features/{feature_id}  --  update a feature
# ---------------------------------------------------------------------------


@router.put(
    "/{repo_id}/features/{feature_id}",
    response_model=FeatureInfo,
    summary="Update feature details",
)
async def update_feature(
    repo_id: str,
    feature_id: str,
    body: FeatureUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> FeatureInfo:
    """Update the mutable properties of a feature.

    Only fields present in the request body are changed; omitted fields
    retain their current values.
    """
    repo = await db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository not found: {repo_id}",
        )

    feature = await db.get(Feature, feature_id)
    if feature is None or feature.repo_id != repo_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature not found: {feature_id}",
        )

    if body.name is not None:
        feature.name = body.name
    if body.description is not None:
        feature.description = body.description
    if body.color is not None:
        feature.color = body.color

    await db.flush()
    await db.refresh(feature)

    return await _build_feature_info(db, feature)
