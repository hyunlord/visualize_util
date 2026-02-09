"""SQLAlchemy ORM models for the Code Flow Visualizer."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


def _uuid() -> str:
    """Generate a new UUID4 string suitable for use as a primary key."""
    return str(uuid4())


def _utcnow() -> datetime:
    """Return the current UTC timestamp (timezone-aware)."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class Repository(Base):
    """A tracked Git repository."""

    __tablename__ = "repositories"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid,
    )
    url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    local_path: Mapped[str] = mapped_column(String(4096), nullable=False)
    branch: Mapped[str] = mapped_column(String(256), default="main")
    last_commit_sha: Mapped[str | None] = mapped_column(
        String(40), nullable=True,
    )
    last_analyzed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow,
    )

    # -- relationships --
    snapshots: Mapped[list[AnalysisSnapshot]] = relationship(
        back_populates="repository",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    features: Mapped[list[Feature]] = relationship(
        back_populates="repository",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Repository id={self.id!r} path={self.local_path!r}>"


# ---------------------------------------------------------------------------
# AnalysisSnapshot
# ---------------------------------------------------------------------------

class AnalysisSnapshot(Base):
    """A point-in-time analysis of a repository at a specific commit."""

    __tablename__ = "analysis_snapshots"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid,
    )
    repo_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
    )
    commit_sha: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), default="pending",
    )  # pending | running | completed | failed
    analyzed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow,
    )
    stats: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    language: Mapped[str] = mapped_column(String(8), default="en")

    # -- relationships --
    repository: Mapped[Repository] = relationship(back_populates="snapshots")
    nodes: Mapped[list[CodeNode]] = relationship(
        back_populates="snapshot",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    edges: Mapped[list[CodeEdge]] = relationship(
        back_populates="snapshot",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<AnalysisSnapshot id={self.id!r} "
            f"repo_id={self.repo_id!r} status={self.status!r}>"
        )


# ---------------------------------------------------------------------------
# Feature
# ---------------------------------------------------------------------------

class Feature(Base):
    """A logical feature grouping detected or manually defined in a repo."""

    __tablename__ = "features"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid,
    )
    repo_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    color: Mapped[str] = mapped_column(String(32), nullable=False)
    auto_detected: Mapped[bool] = mapped_column(Boolean, default=True)
    verification_status: Mapped[str] = mapped_column(
        String(16), default="pending",
    )  # pending | verified | rejected
    verification_notes: Mapped[str | None] = mapped_column(
        Text, nullable=True,
    )
    flow_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # -- relationships --
    repository: Mapped[Repository] = relationship(back_populates="features")
    nodes: Mapped[list[CodeNode]] = relationship(
        back_populates="feature",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Feature id={self.id!r} name={self.name!r}>"


# ---------------------------------------------------------------------------
# CodeNode
# ---------------------------------------------------------------------------

class CodeNode(Base):
    """A discrete code element (file, function, class, method, module)."""

    __tablename__ = "code_nodes"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid,
    )
    snapshot_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("analysis_snapshots.id", ondelete="CASCADE"),
        nullable=False,
    )
    file_path: Mapped[str] = mapped_column(String(4096), nullable=False)
    node_type: Mapped[str] = mapped_column(
        String(16), nullable=False,
    )  # file | function | class | method | module
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    language: Mapped[str] = mapped_column(String(32), nullable=False)
    line_start: Mapped[int] = mapped_column(Integer, nullable=False)
    line_end: Mapped[int] = mapped_column(Integer, nullable=False)
    source_code: Mapped[str] = mapped_column(Text, nullable=False)
    docstring: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata", JSON, nullable=True,
    )
    feature_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("features.id", ondelete="SET NULL"),
        nullable=True,
    )
    is_entry_point: Mapped[bool] = mapped_column(Boolean, default=False)
    is_dead_code: Mapped[bool] = mapped_column(Boolean, default=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    flow_order: Mapped[int | None] = mapped_column(Integer, nullable=True)
    flow_description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # -- relationships --
    snapshot: Mapped[AnalysisSnapshot] = relationship(
        back_populates="nodes",
    )
    feature: Mapped[Feature | None] = relationship(back_populates="nodes")
    outgoing_edges: Mapped[list[CodeEdge]] = relationship(
        foreign_keys="CodeEdge.source_node_id",
        back_populates="source_node",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    incoming_edges: Mapped[list[CodeEdge]] = relationship(
        foreign_keys="CodeEdge.target_node_id",
        back_populates="target_node",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<CodeNode id={self.id!r} type={self.node_type!r} "
            f"name={self.name!r}>"
        )


# ---------------------------------------------------------------------------
# CodeEdge
# ---------------------------------------------------------------------------

class CodeEdge(Base):
    """A directed relationship between two :class:`CodeNode` instances."""

    __tablename__ = "code_edges"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid,
    )
    snapshot_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("analysis_snapshots.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_node_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("code_nodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    target_node_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("code_nodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    edge_type: Mapped[str] = mapped_column(
        String(16), nullable=False,
    )  # imports | calls | inherits | instantiates
    line_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata", JSON, nullable=True,
    )
    is_llm_inferred: Mapped[bool] = mapped_column(Boolean, default=False)

    # -- relationships --
    snapshot: Mapped[AnalysisSnapshot] = relationship(
        back_populates="edges",
    )
    source_node: Mapped[CodeNode] = relationship(
        foreign_keys=[source_node_id],
        back_populates="outgoing_edges",
    )
    target_node: Mapped[CodeNode] = relationship(
        foreign_keys=[target_node_id],
        back_populates="incoming_edges",
    )

    def __repr__(self) -> str:
        return (
            f"<CodeEdge id={self.id!r} type={self.edge_type!r} "
            f"{self.source_node_id!r} -> {self.target_node_id!r}>"
        )
