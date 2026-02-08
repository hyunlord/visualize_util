"""Graph data API endpoint for the React Flow frontend.

Converts the internal :class:`CodeNode` and :class:`CodeEdge` ORM objects
from the latest completed analysis snapshot into the ``GraphResponse``
schema that the frontend consumes directly.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.database import get_db
from backend.models.db_models import (
    AnalysisSnapshot,
    CodeEdge,
    CodeNode,
    Feature,
    Repository,
)
from backend.models.schemas import (
    FeatureInfo,
    GraphEdge,
    GraphNode,
    GraphResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Map node_type to React Flow custom node types.
_RF_NODE_TYPE_MAP: dict[str, str] = {
    "function": "functionNode",
    "class": "classNode",
    "method": "methodNode",
    "file": "fileNode",
    "module": "moduleNode",
}

# Edge types that should animate in React Flow (indicates active data flow).
_ANIMATED_EDGE_TYPES: frozenset[str] = frozenset({"calls", "instantiates"})


def _code_node_to_graph_node(node: CodeNode) -> GraphNode:
    """Convert a :class:`CodeNode` ORM object to a :class:`GraphNode` schema."""
    return GraphNode(
        id=node.id,
        type=_RF_NODE_TYPE_MAP.get(node.node_type, "functionNode"),
        position={"x": 0.0, "y": 0.0},
        data={
            "label": node.name,
            "nodeType": node.node_type,
            "filePath": node.file_path,
            "language": node.language,
            "lineStart": node.line_start,
            "lineEnd": node.line_end,
            "sourceCode": node.source_code,
            "docstring": node.docstring,
            "description": node.description,
            "featureId": node.feature_id,
            "featureName": node.feature.name if node.feature else None,
            "featureColor": node.feature.color if node.feature else None,
            "isEntryPoint": node.is_entry_point,
            "isDeadCode": node.is_dead_code,
            "metadata": node.metadata_,
        },
    )


def _code_edge_to_graph_edge(edge: CodeEdge) -> GraphEdge:
    """Convert a :class:`CodeEdge` ORM object to a :class:`GraphEdge` schema."""
    return GraphEdge(
        id=edge.id,
        source=edge.source_node_id,
        target=edge.target_node_id,
        type=edge.edge_type,
        animated=edge.edge_type in _ANIMATED_EDGE_TYPES,
        data={
            "edgeType": edge.edge_type,
            "lineNumber": edge.line_number,
            "isLlmInferred": edge.is_llm_inferred,
            "metadata": edge.metadata_,
        },
    )


def _feature_to_info(feature: Feature, node_count: int) -> FeatureInfo:
    """Build a :class:`FeatureInfo` schema from a Feature ORM object."""
    return FeatureInfo(
        id=feature.id,
        name=feature.name,
        description=feature.description,
        color=feature.color,
        node_count=node_count,
        auto_detected=feature.auto_detected,
    )


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


# ---------------------------------------------------------------------------
# GET /repos/{repo_id}/graph  --  full graph for React Flow
# ---------------------------------------------------------------------------


@router.get(
    "/{repo_id}/graph",
    response_model=GraphResponse,
    summary="Get the code-flow graph for React Flow",
)
async def get_graph(
    repo_id: str,
    db: AsyncSession = Depends(get_db),
) -> GraphResponse:
    """Return the complete code-flow graph from the latest completed analysis.

    Nodes and edges are formatted for direct consumption by React Flow.
    Position data is set to ``{x: 0, y: 0}`` -- the frontend uses
    dagre or a similar layout engine to compute final positions.
    """
    repo = await db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository not found: {repo_id}",
        )

    snapshot = await _get_latest_completed_snapshot(db, repo_id)
    if snapshot is None:
        return GraphResponse(nodes=[], edges=[], features=[])

    # Load nodes with their feature relationship eagerly
    nodes_result = await db.execute(
        select(CodeNode)
        .where(CodeNode.snapshot_id == snapshot.id)
        .options(selectinload(CodeNode.feature))
    )
    code_nodes = list(nodes_result.scalars().all())

    # Load edges
    edges_result = await db.execute(
        select(CodeEdge).where(CodeEdge.snapshot_id == snapshot.id)
    )
    code_edges = list(edges_result.scalars().all())

    # Build feature info with node counts
    feature_counts_result = await db.execute(
        select(
            CodeNode.feature_id,
            func.count(CodeNode.id).label("node_count"),
        )
        .where(
            CodeNode.snapshot_id == snapshot.id,
            CodeNode.feature_id.isnot(None),
        )
        .group_by(CodeNode.feature_id)
    )
    feature_counts: dict[str, int] = {
        row.feature_id: row.node_count
        for row in feature_counts_result.all()
    }

    features_result = await db.execute(
        select(Feature).where(Feature.repo_id == repo_id)
    )
    features = list(features_result.scalars().all())

    graph_nodes = [_code_node_to_graph_node(n) for n in code_nodes]
    graph_edges = [_code_edge_to_graph_edge(e) for e in code_edges]
    feature_infos = [
        _feature_to_info(f, feature_counts.get(f.id, 0)) for f in features
    ]

    return GraphResponse(
        nodes=graph_nodes,
        edges=graph_edges,
        features=feature_infos,
    )
