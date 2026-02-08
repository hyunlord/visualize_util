"""Detect unused and dead code using AST graph analysis with LLM verification.

Combines structural reachability analysis (call-graph traversal, edge
inspection) with an optional LLM second pass that reclassifies candidates
to reduce false positives and suggest feature assignments for code that
turns out to be alive.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from backend.analyzer.base_analyzer import RawNode
from backend.analyzer.call_graph import CallGraph
from backend.analyzer.feature_grouper import FeatureGroupResult

logger = logging.getLogger(__name__)

# Valid classifications for dead code items (AST pass and LLM pass)
_AST_CLASSIFICATIONS = frozenset({
    "no_references",
    "unreachable",
    "self_referential",
    "imported_not_called",
})

_LLM_CLASSIFICATIONS = frozenset({
    "truly_dead",
    "stub_incomplete",
    "debug_test_utility",
    "alive_misclassified",
})

_DEFAULT_CONFIDENCE_THRESHOLD = 0.5
_LLM_BATCH_SIZE = 10


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------


@dataclass(slots=True)
class DeadCodeItem:
    """A code node identified as potentially dead or unused.

    Attributes:
        node_fqn: Fully-qualified name of the dead code node.
        file_path: Source file containing the node.
        name: Short name of the function/class/method.
        node_type: Type of the code entity (function, class, method, module).
        line_start: Starting line number in the source file.
        line_end: Ending line number in the source file.
        reason: Human-readable explanation of why this is considered dead.
        confidence: Score from 0.0 to 1.0 expressing certainty of deadness.
        classification: AST-derived classification category.
        llm_explanation: LLM-provided reasoning (empty if LLM not used).
        suggested_feature: Feature name the LLM suggests this belongs to.
    """

    node_fqn: str
    file_path: str
    name: str
    node_type: str
    line_start: int
    line_end: int
    reason: str
    confidence: float
    classification: str
    llm_explanation: str = ""
    suggested_feature: str = ""


# ------------------------------------------------------------------
# AST first pass
# ------------------------------------------------------------------


def _ast_first_pass(
    call_graph: CallGraph,
    features: FeatureGroupResult,
) -> list[DeadCodeItem]:
    """Classify dead-code candidates using structural graph analysis.

    Classification rules:
    1. **no_references** (confidence 0.95): No incoming edges and not in
       any feature group.
    2. **unreachable** (confidence 0.8): Has incoming edges but is not
       reachable from any entry point.
    3. **self_referential** (confidence 0.9): The only incoming edges
       come from the node itself (recursive with no external callers).
    4. **imported_not_called** (confidence 0.7): Appears as the target
       of an import edge but has no call/instantiate/inherit edges.
    """
    candidates: list[DeadCodeItem] = []

    # Pre-compute sets for efficient lookup
    feature_fqns: set[str] = set()
    for feat in features.features:
        feature_fqns |= feat.node_fqns
    feature_fqns |= features.shared_utils

    unreachable_fqns = call_graph.get_unreachable_nodes()

    # Collect all edge targets/sources by type for import-but-not-called detection
    import_targets: set[str] = set()
    call_targets: set[str] = set()
    for edge in call_graph.edges:
        if edge.edge_type == "imports":
            import_targets.add(edge.target)
        elif edge.edge_type in ("calls", "instantiates", "inherits"):
            call_targets.add(edge.target)

    for fqn, node in call_graph.nodes.items():
        # Skip module-type nodes (they are structural, not callable)
        if node.node_type == "module":
            continue

        incoming = call_graph.adjacency_reverse.get(fqn, set())
        in_feature = fqn in feature_fqns
        is_unreachable = fqn in unreachable_fqns

        # Rule 1: No references AND not in any feature
        if not incoming and not in_feature:
            candidates.append(_make_item(
                node=node,
                fqn=fqn,
                reason=(
                    f"No incoming edges (calls, imports, or inheritance) "
                    f"and not assigned to any feature group"
                ),
                confidence=0.95,
                classification="no_references",
            ))
            continue

        # Rule 3: Only self-references (check before unreachable since
        # self-referential is more specific)
        if incoming and call_graph.has_only_self_references(fqn) and not in_feature:
            candidates.append(_make_item(
                node=node,
                fqn=fqn,
                reason=(
                    f"Only called by itself (recursive) with no external callers"
                ),
                confidence=0.9,
                classification="self_referential",
            ))
            continue

        # Rule 2: Has incoming edges but unreachable from entry points
        if incoming and is_unreachable and not in_feature:
            candidates.append(_make_item(
                node=node,
                fqn=fqn,
                reason=(
                    f"Has {len(incoming)} incoming edge(s) but is not reachable "
                    f"from any entry point via transitive call chain"
                ),
                confidence=0.8,
                classification="unreachable",
            ))
            continue

        # Rule 4: Imported but never called
        if fqn in import_targets and fqn not in call_targets and not in_feature:
            candidates.append(_make_item(
                node=node,
                fqn=fqn,
                reason=(
                    f"Imported by other modules but never invoked via call, "
                    f"instantiation, or inheritance"
                ),
                confidence=0.7,
                classification="imported_not_called",
            ))

    return candidates


def _make_item(
    *,
    node: RawNode,
    fqn: str,
    reason: str,
    confidence: float,
    classification: str,
) -> DeadCodeItem:
    """Helper to construct a ``DeadCodeItem`` from a node and analysis result."""
    return DeadCodeItem(
        node_fqn=fqn,
        file_path=node.file_path,
        name=node.name,
        node_type=node.node_type,
        line_start=node.line_start,
        line_end=node.line_end,
        reason=reason,
        confidence=confidence,
        classification=classification,
    )


# ------------------------------------------------------------------
# LLM second pass
# ------------------------------------------------------------------


async def _llm_second_pass(
    candidates: list[DeadCodeItem],
    call_graph: CallGraph,
    features: FeatureGroupResult,
    llm: Any,
) -> list[DeadCodeItem]:
    """Refine dead-code candidates using LLM analysis.

    For each candidate, ask the LLM to classify as:
    - ``truly_dead``: Confirmed dead code with no runtime usage.
    - ``stub_incomplete``: Incomplete implementation or placeholder.
    - ``debug_test_utility``: Debug/test helper that may be intentionally unused.
    - ``alive_misclassified``: Not actually dead; suggest a feature it belongs to.

    Adjusts confidence scores based on LLM response and filters out
    candidates the LLM identifies as alive.
    """
    from backend.llm.prompts import classify_unmatched_code_prompt

    refined: list[DeadCodeItem] = []

    # Process in batches to limit API calls
    for batch_start in range(0, len(candidates), _LLM_BATCH_SIZE):
        batch = candidates[batch_start:batch_start + _LLM_BATCH_SIZE]
        tasks = [
            _classify_single_candidate(item, call_graph, features, llm)
            for item in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for item, result in zip(batch, results):
            if isinstance(result, Exception):
                logger.warning(
                    "LLM classification failed for %s: %s",
                    item.node_fqn,
                    result,
                )
                # Keep original classification on failure
                refined.append(item)
            elif result is not None:
                refined.append(result)
            else:
                refined.append(item)

    return refined


async def _classify_single_candidate(
    item: DeadCodeItem,
    call_graph: CallGraph,
    features: FeatureGroupResult,
    llm: Any,
) -> DeadCodeItem:
    """Ask the LLM to classify a single dead-code candidate.

    Returns a new ``DeadCodeItem`` with updated confidence, classification,
    and LLM explanation.
    """
    from backend.llm.prompts import classify_unmatched_code_prompt

    node = call_graph.nodes.get(item.node_fqn)
    if node is None:
        return item

    # Gather surrounding function names for context
    surrounding: list[str] = []
    for fqn, n in call_graph.nodes.items():
        if n.file_path == node.file_path and fqn != item.node_fqn:
            surrounding.append(n.name)
    surrounding = surrounding[:10]

    prompt = classify_unmatched_code_prompt(
        source_code=node.source_code,
        language=node.language or "python",
        file_path=node.file_path,
        surrounding_functions=surrounding if surrounding else None,
    )

    result = await llm.complete_json(prompt, temperature=0.1)
    if not result:
        return item

    llm_classification = result.get("classification", "")
    llm_reason = result.get("reason", "")
    suggested_owner = result.get("suggested_owner")

    # Map LLM classifications to confidence adjustments
    if llm_classification == "dead_code":
        # LLM confirms it is dead
        return DeadCodeItem(
            node_fqn=item.node_fqn,
            file_path=item.file_path,
            name=item.name,
            node_type=item.node_type,
            line_start=item.line_start,
            line_end=item.line_end,
            reason=item.reason,
            confidence=min(item.confidence + 0.05, 1.0),
            classification=item.classification,
            llm_explanation=llm_reason,
            suggested_feature="",
        )

    if llm_classification in ("module_init", "global_config", "constant", "type_definition"):
        # LLM says this is structural code, not truly dead -- lower confidence
        return DeadCodeItem(
            node_fqn=item.node_fqn,
            file_path=item.file_path,
            name=item.name,
            node_type=item.node_type,
            line_start=item.line_start,
            line_end=item.line_end,
            reason=item.reason,
            confidence=max(item.confidence - 0.3, 0.1),
            classification=item.classification,
            llm_explanation=f"LLM reclassified as '{llm_classification}': {llm_reason}",
            suggested_feature=_resolve_suggested_feature(suggested_owner, features),
        )

    if llm_classification == "decorator":
        # Decorators are typically alive but missed by static analysis
        return DeadCodeItem(
            node_fqn=item.node_fqn,
            file_path=item.file_path,
            name=item.name,
            node_type=item.node_type,
            line_start=item.line_start,
            line_end=item.line_end,
            reason=item.reason,
            confidence=max(item.confidence - 0.4, 0.05),
            classification=item.classification,
            llm_explanation=f"LLM identified as decorator: {llm_reason}",
            suggested_feature=_resolve_suggested_feature(suggested_owner, features),
        )

    if llm_classification == "generated":
        # Auto-generated code; keep moderate confidence
        return DeadCodeItem(
            node_fqn=item.node_fqn,
            file_path=item.file_path,
            name=item.name,
            node_type=item.node_type,
            line_start=item.line_start,
            line_end=item.line_end,
            reason=item.reason,
            confidence=item.confidence * 0.7,
            classification=item.classification,
            llm_explanation=f"LLM identified as generated code: {llm_reason}",
            suggested_feature="",
        )

    # Default: keep original confidence, add LLM explanation
    return DeadCodeItem(
        node_fqn=item.node_fqn,
        file_path=item.file_path,
        name=item.name,
        node_type=item.node_type,
        line_start=item.line_start,
        line_end=item.line_end,
        reason=item.reason,
        confidence=item.confidence,
        classification=item.classification,
        llm_explanation=llm_reason,
        suggested_feature=_resolve_suggested_feature(suggested_owner, features),
    )


def _resolve_suggested_feature(
    suggested_owner: str | None,
    features: FeatureGroupResult,
) -> str:
    """Map an LLM-suggested owner to a feature name.

    Tries matching the suggested owner against feature node sets.
    Returns the feature name or an empty string.
    """
    if not suggested_owner:
        return ""

    for feat in features.features:
        # Check if the suggested owner is in this feature
        for fqn in feat.node_fqns:
            if suggested_owner in fqn or fqn.endswith(f"::{suggested_owner}"):
                return feat.name

    # No match found; return the raw suggestion as a hint
    return suggested_owner


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


async def detect_dead_code(
    call_graph: CallGraph,
    features: FeatureGroupResult,
    llm: Any | None = None,
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[DeadCodeItem]:
    """Detect dead/unused code using AST analysis with optional LLM refinement.

    Parameters:
        call_graph: Resolved call graph for the codebase.
        features: Feature grouping result (nodes in features are excluded).
        llm: Optional ``LLMClient`` for second-pass classification.
        confidence_threshold: Minimum confidence to include in results.

    Returns:
        Dead code items sorted by confidence descending, filtered by threshold.
    """
    # Step 1: AST first pass
    candidates = _ast_first_pass(call_graph, features)
    logger.info("AST first pass found %d dead-code candidates", len(candidates))

    if not candidates:
        return []

    # Step 2: LLM second pass
    if llm is not None:
        candidates = await _llm_second_pass(candidates, call_graph, features, llm)
        logger.info("LLM second pass refined %d candidates", len(candidates))

    # Step 3: Filter by confidence threshold
    filtered = [
        item for item in candidates
        if item.confidence >= confidence_threshold
    ]

    # Step 4: Sort by confidence descending
    filtered.sort(key=lambda item: item.confidence, reverse=True)

    logger.info(
        "Dead code detection complete: %d items above threshold %.2f",
        len(filtered),
        confidence_threshold,
    )

    return filtered
