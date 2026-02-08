"""Five-stage analysis pipeline orchestrator for the Code Flow Visualizer.

Coordinates the full analysis lifecycle: code parsing, call-graph construction,
feature discovery, dead-code detection, and LLM-powered enrichment.  Results
are persisted to the database through the provided SQLAlchemy async session.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable, Coroutine
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from backend.analyzer.base_analyzer import AnalysisResult, RawEdge, RawNode
from backend.analyzer.language_registry import (
    LanguageRegistry,
    get_default_registry,
)
from backend.models.db_models import (
    AnalysisSnapshot,
    CodeEdge,
    CodeNode,
    Feature,
    Repository,
)
from backend.utils.file_discovery import discover_files, get_file_content

logger = logging.getLogger(__name__)

# Type alias for the optional progress callback.
ProgressCallback = Callable[[str, float], Coroutine[Any, Any, None]] | Callable[[str, float], None] | None

# Stage weights for progress calculation (must sum to 1.0).
_STAGE_WEIGHTS: list[tuple[str, float]] = [
    ("code_parsing", 0.25),
    ("feature_discovery", 0.15),
    ("feature_matching", 0.15),
    ("unmatched_analysis", 0.20),
    ("verification", 0.25),
]

_MAX_VERIFICATION_ITERATIONS = 3


def _uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid4())


def _utcnow() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


class AnalysisEngine:
    """Orchestrates the five-stage code analysis pipeline.

    Stages:
        1. **Code Parsing** -- discover files, run language analyzers, collect
           raw nodes and edges.
        2. **Feature Discovery** -- build call graph, detect entry points,
           cluster nodes into feature groups via prefix/directory heuristics
           and optional LLM assistance.
        3. **Feature-Code Matching** -- BFS from entry points to tag every
           reachable node to its owning feature.
        4. **Unmatched Code Analysis** -- classify nodes that belong to no
           feature (shared utilities, dead code).
        5. **Verification Loop** -- AST-vs-LLM cross-check with up to three
           refinement iterations.

    After all stages complete (or on failure), results are persisted through
    the supplied SQLAlchemy session.
    """

    def __init__(
        self,
        session: AsyncSession,
        registry: LanguageRegistry | None = None,
    ) -> None:
        self._session = session
        self._registry = registry or get_default_registry()

        # Internal state accumulated across stages.
        self._analysis_results: list[AnalysisResult] = []
        self._raw_nodes: dict[str, RawNode] = {}  # fqn -> RawNode
        self._raw_edges: list[RawEdge] = []
        self._node_feature_map: dict[str, str] = {}  # fqn -> feature_id
        self._entry_point_fqns: set[str] = set()
        self._dead_code_fqns: set[str] = set()
        self._descriptions: dict[str, str] = {}  # fqn -> description
        self._llm_edges: list[RawEdge] = []  # edges discovered by LLM

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_full_analysis(
        self,
        repo: Repository,
        snapshot: AnalysisSnapshot,
        progress_callback: ProgressCallback = None,
    ) -> None:
        """Execute the full five-stage analysis pipeline.

        On success the snapshot status is set to ``"completed"`` and all
        discovered nodes, edges, and features are written to the database.
        On failure the snapshot status is set to ``"failed"`` with partial
        results saved where possible.
        """
        snapshot.status = "running"
        self._session.add(snapshot)
        await self._session.flush()

        stats: dict[str, Any] = {
            "files_discovered": 0,
            "files_analyzed": 0,
            "nodes": 0,
            "edges": 0,
            "features": 0,
            "dead_code_nodes": 0,
            "errors": [],
        }

        db_features: list[Feature] = []

        try:
            # ---- Stage 1: Code Parsing ----
            await self._report_progress(progress_callback, "code_parsing", 0.0)
            await self._stage_code_parsing(repo, stats)
            await self._report_progress(progress_callback, "code_parsing", 1.0)

            # ---- Stage 2: Feature Discovery ----
            await self._report_progress(progress_callback, "feature_discovery", 0.0)
            call_graph, feature_result = await self._stage_feature_discovery(
                repo, stats,
            )
            await self._report_progress(progress_callback, "feature_discovery", 1.0)

            # ---- Stage 3: Feature-Code Matching ----
            await self._report_progress(progress_callback, "feature_matching", 0.0)
            db_features = await self._stage_feature_matching(
                repo, call_graph, feature_result, stats,
            )
            await self._report_progress(progress_callback, "feature_matching", 1.0)

            # ---- Stage 4: Unmatched Code Analysis ----
            await self._report_progress(progress_callback, "unmatched_analysis", 0.0)
            await self._stage_unmatched_analysis(call_graph, feature_result, stats)
            await self._report_progress(progress_callback, "unmatched_analysis", 1.0)

            # ---- Stage 5: Verification Loop ----
            await self._report_progress(progress_callback, "verification", 0.0)
            await self._stage_verification(call_graph, stats, progress_callback)
            await self._report_progress(progress_callback, "verification", 1.0)

            # ---- Persist results ----
            await self._persist_results(repo, snapshot, db_features, stats)
            snapshot.status = "completed"

        except Exception as exc:
            error_msg = f"Analysis failed: {exc}"
            logger.exception(error_msg)
            stats["errors"].append(error_msg)
            stats["traceback"] = traceback.format_exc()

            # Attempt to persist partial results.
            try:
                await self._persist_results(repo, snapshot, db_features, stats)
            except Exception:
                logger.exception("Failed to persist partial results")

            snapshot.status = "failed"

        finally:
            snapshot.stats = stats
            snapshot.analyzed_at = _utcnow()
            repo.last_analyzed_at = _utcnow()
            self._session.add(snapshot)
            self._session.add(repo)
            await self._session.flush()

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    async def _stage_code_parsing(
        self,
        repo: Repository,
        stats: dict[str, Any],
    ) -> None:
        """Stage 1: Discover source files and run language analyzers."""
        repo_path = repo.local_path
        discovered = discover_files(repo_path, self._registry)
        stats["files_discovered"] = len(discovered)

        for discovered_file in discovered:
            source = get_file_content(discovered_file.path)
            if not source:
                continue

            # Compute the relative file path for storage.
            rel_path = discovered_file.path
            if rel_path.startswith(repo_path):
                rel_path = rel_path[len(repo_path) :].lstrip("/\\")

            result = self._registry.analyze_file(rel_path, source, repo_path)
            if result is None:
                continue

            self._analysis_results.append(result)
            stats["files_analyzed"] = stats.get("files_analyzed", 0) + 1

            # Merge nodes keyed by fully-qualified name.
            for node in result.nodes:
                fqn = node.fully_qualified_name
                self._raw_nodes[fqn] = node

            # Collect edges.
            self._raw_edges.extend(result.edges)

            # Track entry points.
            for ep in result.entry_points:
                self._entry_point_fqns.add(ep.node_name)

        stats["nodes"] = len(self._raw_nodes)
        stats["edges"] = len(self._raw_edges)

        logger.info(
            "Stage 1 complete: %d files -> %d nodes, %d edges",
            stats["files_analyzed"],
            stats["nodes"],
            stats["edges"],
        )

    async def _stage_feature_discovery(
        self,
        repo: Repository,
        stats: dict[str, Any],
    ) -> tuple[Any, Any]:
        """Stage 2: Build call graph and group features.

        Returns the call graph and feature grouping result for use in
        subsequent stages.
        """
        from backend.analyzer.call_graph import build_call_graph
        from backend.analyzer.feature_grouper import group_features

        call_graph = build_call_graph(self._analysis_results)

        # Attempt LLM-assisted feature grouping if available.
        llm = self._get_llm_client()
        feature_result = await group_features(call_graph, llm)

        stats["features"] = len(feature_result.features)
        logger.info(
            "Stage 2 complete: %d features discovered",
            stats["features"],
        )
        return call_graph, feature_result

    async def _stage_feature_matching(
        self,
        repo: Repository,
        call_graph: Any,
        feature_result: Any,
        stats: dict[str, Any],
    ) -> list[Feature]:
        """Stage 3: BFS from entry points to tag nodes to features.

        Creates Feature DB records and populates ``_node_feature_map``.
        Returns the list of Feature ORM instances for later persistence.
        """
        db_features: list[Feature] = []

        for fg in feature_result.features:
            feature_id = _uuid()
            db_feature = Feature(
                id=feature_id,
                repo_id=repo.id,
                name=fg.name,
                description=fg.description,
                color=fg.color,
                auto_detected=True,
                verification_status="pending",
            )
            db_features.append(db_feature)

            # BFS from the feature's entry points to discover the full
            # reachable sub-graph and assign nodes to this feature.
            reachable: set[str] = set()
            for ep_fqn in fg.entry_points:
                reachable |= call_graph.reachable_from(ep_fqn)

            # Also include nodes explicitly listed in the feature group.
            reachable |= set(fg.node_fqns)

            for fqn in reachable:
                # First-match wins: if already assigned, skip.
                if fqn not in self._node_feature_map:
                    self._node_feature_map[fqn] = feature_id

        matched_count = len(self._node_feature_map)
        total_nodes = len(self._raw_nodes)
        logger.info(
            "Stage 3 complete: %d / %d nodes matched to features",
            matched_count,
            total_nodes,
        )
        return db_features

    async def _stage_unmatched_analysis(
        self,
        call_graph: Any,
        feature_result: Any,
        stats: dict[str, Any],
    ) -> None:
        """Stage 4: Classify unmatched nodes, detect dead code."""
        from backend.analyzer.dead_code_detector import detect_dead_code

        llm = self._get_llm_client()
        dead_items = await detect_dead_code(call_graph, feature_result, llm)

        for item in dead_items:
            self._dead_code_fqns.add(item.node_fqn)

        stats["dead_code_nodes"] = len(self._dead_code_fqns)
        logger.info(
            "Stage 4 complete: %d dead-code nodes detected",
            stats["dead_code_nodes"],
        )

    async def _stage_verification(
        self,
        call_graph: Any,
        stats: dict[str, Any],
        progress_callback: ProgressCallback,
    ) -> None:
        """Stage 5: AST-vs-LLM cross-check with up to 3 iterations.

        Uses the LLM to enrich node descriptions and discover dynamic
        call relationships that static analysis may have missed.
        """
        llm = self._get_llm_client()
        if llm is None:
            logger.info("Stage 5 skipped: no LLM client available")
            return

        from backend.analyzer.llm_enricher import (
            discover_dynamic_calls,
            enrich_descriptions,
        )

        nodes_list = list(self._raw_nodes.values())

        for iteration in range(1, _MAX_VERIFICATION_ITERATIONS + 1):
            pct = iteration / _MAX_VERIFICATION_ITERATIONS
            await self._report_progress(
                progress_callback, "verification", pct * 0.9,
            )

            # Enrich descriptions for nodes that lack them.
            nodes_needing_desc = [
                n for n in nodes_list
                if n.fully_qualified_name not in self._descriptions
            ]
            if nodes_needing_desc:
                new_descriptions = await enrich_descriptions(nodes_needing_desc, llm)
                self._descriptions.update(new_descriptions)

            # Discover dynamic call edges the AST could not find.
            new_edges = await discover_dynamic_calls(nodes_list, call_graph, llm)
            if not new_edges:
                logger.info(
                    "Verification iteration %d: no new edges, stopping early",
                    iteration,
                )
                break

            # Deduplicate against already-known edges.
            existing_edge_keys = {
                (e.source, e.target, e.edge_type) for e in self._raw_edges
            }
            existing_llm_keys = {
                (e.source, e.target, e.edge_type) for e in self._llm_edges
            }
            novel_edges = [
                e for e in new_edges
                if (e.source, e.target, e.edge_type) not in existing_edge_keys
                and (e.source, e.target, e.edge_type) not in existing_llm_keys
            ]

            if not novel_edges:
                logger.info(
                    "Verification iteration %d: all edges already known, stopping",
                    iteration,
                )
                break

            self._llm_edges.extend(novel_edges)
            logger.info(
                "Verification iteration %d: discovered %d new edges",
                iteration,
                len(novel_edges),
            )

        stats["llm_descriptions"] = len(self._descriptions)
        stats["llm_edges"] = len(self._llm_edges)
        stats["verification_iterations"] = min(
            iteration, _MAX_VERIFICATION_ITERATIONS,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _persist_results(
        self,
        repo: Repository,
        snapshot: AnalysisSnapshot,
        db_features: list[Feature],
        stats: dict[str, Any],
    ) -> None:
        """Write all discovered features, nodes, and edges to the database."""
        # Persist features.
        for feature in db_features:
            self._session.add(feature)
        await self._session.flush()

        # Build a mapping from RawNode FQN -> CodeNode DB id for edge FK resolution.
        fqn_to_db_id: dict[str, str] = {}

        # Persist CodeNode records.
        for fqn, raw_node in self._raw_nodes.items():
            node_id = _uuid()
            fqn_to_db_id[fqn] = node_id

            db_node = CodeNode(
                id=node_id,
                snapshot_id=snapshot.id,
                file_path=raw_node.file_path,
                node_type=raw_node.node_type,
                name=raw_node.name,
                language=raw_node.language,
                line_start=raw_node.line_start,
                line_end=raw_node.line_end,
                source_code=raw_node.source_code,
                docstring=raw_node.docstring,
                metadata_=raw_node.metadata or None,
                feature_id=self._node_feature_map.get(fqn),
                is_entry_point=fqn in self._entry_point_fqns,
                is_dead_code=fqn in self._dead_code_fqns,
                description=self._descriptions.get(fqn),
            )
            self._session.add(db_node)

        await self._session.flush()

        # Persist CodeEdge records for static edges.
        edge_count = 0
        for raw_edge in self._raw_edges:
            source_db_id = fqn_to_db_id.get(raw_edge.source)
            target_db_id = fqn_to_db_id.get(raw_edge.target)
            if source_db_id is None or target_db_id is None:
                # Edge references an external or unresolved node; skip.
                continue

            db_edge = CodeEdge(
                id=_uuid(),
                snapshot_id=snapshot.id,
                source_node_id=source_db_id,
                target_node_id=target_db_id,
                edge_type=raw_edge.edge_type,
                line_number=raw_edge.line_number or None,
                metadata_=raw_edge.metadata or None,
                is_llm_inferred=False,
            )
            self._session.add(db_edge)
            edge_count += 1

        # Persist LLM-inferred edges.
        for raw_edge in self._llm_edges:
            source_db_id = fqn_to_db_id.get(raw_edge.source)
            target_db_id = fqn_to_db_id.get(raw_edge.target)
            if source_db_id is None or target_db_id is None:
                continue

            db_edge = CodeEdge(
                id=_uuid(),
                snapshot_id=snapshot.id,
                source_node_id=source_db_id,
                target_node_id=target_db_id,
                edge_type=raw_edge.edge_type,
                line_number=raw_edge.line_number or None,
                metadata_=raw_edge.metadata or None,
                is_llm_inferred=True,
            )
            self._session.add(db_edge)
            edge_count += 1

        await self._session.flush()

        # Update repo metadata.
        repo.last_commit_sha = snapshot.commit_sha
        repo.last_analyzed_at = _utcnow()
        self._session.add(repo)
        await self._session.flush()

        stats["persisted_nodes"] = len(fqn_to_db_id)
        stats["persisted_edges"] = edge_count
        logger.info(
            "Persisted %d nodes and %d edges to database",
            len(fqn_to_db_id),
            edge_count,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_llm_client() -> Any | None:
        """Attempt to obtain the LLM client singleton.

        Returns ``None`` if the LLM is not configured, allowing the
        pipeline to degrade gracefully without LLM enrichment.
        """
        try:
            from backend.llm.client import LLMClient

            return LLMClient.get_instance()
        except Exception:
            logger.debug("LLM client unavailable; enrichment will be skipped")
            return None

    @staticmethod
    async def _report_progress(
        callback: ProgressCallback,
        stage: str,
        progress: float,
    ) -> None:
        """Invoke the progress callback if one was provided.

        Handles both sync and async callbacks transparently.
        """
        if callback is None:
            return

        try:
            result = callback(stage, progress)
            # If the callback is a coroutine, await it.
            if hasattr(result, "__await__"):
                await result
        except Exception:
            logger.debug(
                "Progress callback raised an exception (stage=%s, pct=%.2f)",
                stage,
                progress,
                exc_info=True,
            )
