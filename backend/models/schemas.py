"""Pydantic v2 request/response schemas for the Code Flow Visualizer API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class RepoCreateRequest(BaseModel):
    """Payload for creating or registering a repository."""

    url: Optional[str] = None
    local_path: Optional[str] = None
    branch: str = "main"

    @model_validator(mode="after")
    def _require_url_or_path(self) -> RepoCreateRequest:
        if not self.url and not self.local_path:
            raise ValueError(
                "At least one of 'url' or 'local_path' must be provided."
            )
        return self


class RepoResponse(BaseModel):
    """Serialised representation of a tracked repository."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    url: Optional[str] = None
    local_path: str
    branch: str
    last_commit_sha: Optional[str] = None
    last_analyzed_at: Optional[datetime] = None
    created_at: datetime


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

# Supported analysis languages (extensible)
SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "ko": "Korean",
}


class AnalysisRequest(BaseModel):
    """Payload for starting a new analysis."""
    language: str = Field(default="en", description="Language for feature names and descriptions")


class AnalysisStatusResponse(BaseModel):
    """Progress update for a running or completed analysis."""

    snapshot_id: str
    status: str
    progress: float = Field(ge=0.0, le=100.0, default=0.0)
    current_stage: str = ""


# ---------------------------------------------------------------------------
# Graph Visualisation
# ---------------------------------------------------------------------------

class GraphNode(BaseModel):
    """A single node in the visual code-flow graph."""

    id: str
    type: str
    position: dict[str, float] = Field(
        default_factory=lambda: {"x": 0.0, "y": 0.0},
    )
    data: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """A single edge in the visual code-flow graph."""

    id: str
    source: str
    target: str
    type: str
    animated: bool = False
    data: dict[str, Any] = Field(default_factory=dict)


class FeatureInfo(BaseModel):
    """Summary information about a detected or manual feature group."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: Optional[str] = None
    color: str
    node_count: int = 0
    auto_detected: bool = True
    flow_summary: Optional[str] = None


class FlowStep(BaseModel):
    """A single step in a feature's code execution flow."""
    order: int
    node_id: Optional[str] = None
    file: str
    function: str
    description: str
    source_code: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    calls_next: list[str] = Field(default_factory=list)


class FeatureFlowResponse(BaseModel):
    """Complete flow data for a single feature."""
    feature: FeatureInfo
    flow_summary: Optional[str] = None
    flow_steps: list[FlowStep] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class GraphResponse(BaseModel):
    """Full graph payload returned to the frontend."""

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    features: list[FeatureInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Dead Code
# ---------------------------------------------------------------------------

class DeadCodeItem(BaseModel):
    """A single potentially-dead code element."""

    node_id: str
    file_path: str
    name: str
    node_type: str
    line_start: int
    line_end: int
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    llm_explanation: Optional[str] = None
    suggested_feature: Optional[str] = None


class DeadCodeResponse(BaseModel):
    """Aggregated dead-code analysis results."""

    items: list[DeadCodeItem] = Field(default_factory=list)
    total_count: int = 0
