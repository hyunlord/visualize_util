"""LLM complement analysis that enhances AST results at each pipeline stage.

Provides batch-aware async functions to:
- Generate human-readable descriptions for functions and methods.
- Discover dynamic dispatch patterns invisible to static analysis.
- Suggest feature assignments for unmatched code.
- Cross-verify AST-derived structure against LLM semantic understanding.

All functions gracefully degrade when the LLM is unavailable or returns
errors, returning empty or unchanged results instead of raising.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from backend.analyzer.base_analyzer import RawEdge, RawNode
from backend.analyzer.call_graph import CallGraph
from backend.analyzer.feature_grouper import FeatureGroupResult

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 10


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------


@dataclass(slots=True)
class VerificationResult:
    """Outcome of cross-verifying AST structure against LLM analysis.

    Attributes:
        is_converged: True if AST and LLM agree on the current grouping.
        changes: List of recommended changes (dicts with ``type``,
            ``node_fqn``, ``from_feature``, ``to_feature``, ``reason``).
        iteration: The verification iteration that produced this result.
    """

    is_converged: bool
    changes: list[dict[str, str]] = field(default_factory=list)
    iteration: int = 0


# ------------------------------------------------------------------
# Description enrichment
# ------------------------------------------------------------------


async def enrich_descriptions(
    nodes: list[RawNode],
    llm: Any,
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> dict[str, str]:
    """Generate concise 1-line descriptions for functions and methods.

    Parameters:
        nodes: Code nodes to describe.
        llm: ``LLMClient`` instance for generating descriptions.
        batch_size: Maximum number of concurrent LLM requests per batch.

    Returns:
        Mapping from fully-qualified name to generated description.
        Nodes that fail or are skipped are omitted from the dict.
    """
    from backend.llm.prompts import describe_function_prompt

    descriptions: dict[str, str] = {}

    # Filter to describable node types
    describable = [
        n for n in nodes
        if n.node_type in ("function", "method")
    ]

    for batch_start in range(0, len(describable), batch_size):
        batch = describable[batch_start:batch_start + batch_size]
        tasks = [
            _describe_single_node(node, llm, describe_function_prompt)
            for node in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for node, result in zip(batch, results):
            if isinstance(result, Exception):
                logger.warning(
                    "Failed to describe %s: %s",
                    node.fully_qualified_name,
                    result,
                )
                continue
            if result:
                descriptions[node.fully_qualified_name] = result

    logger.info(
        "Generated descriptions for %d/%d nodes",
        len(descriptions),
        len(describable),
    )
    return descriptions


async def _describe_single_node(
    node: RawNode,
    llm: Any,
    prompt_fn: Any,
) -> str:
    """Generate a description for a single code node.

    Returns the description string, or empty string on failure.
    """
    prompt = prompt_fn(
        function_name=node.name,
        source_code=node.source_code,
        language=node.language or "python",
        file_path=node.file_path,
    )

    result = await llm.complete(prompt, temperature=0.3, max_tokens=200)
    if result is None:
        return ""

    # Clean up the response: take first line, strip quotes
    description = result.strip().split("\n")[0].strip()
    if description.startswith('"') and description.endswith('"'):
        description = description[1:-1]

    return description


# ------------------------------------------------------------------
# Dynamic call discovery
# ------------------------------------------------------------------


async def discover_dynamic_calls(
    nodes: list[RawNode],
    call_graph: CallGraph,
    llm: Any,
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> list[RawEdge]:
    """Discover dynamic dispatch patterns that static analysis missed.

    Examines functions for patterns like ``getattr()``, ``importlib.import_module()``,
    string-based lookups, callback registration, and plugin loading.

    Parameters:
        nodes: Code nodes to analyze for dynamic dispatch.
        call_graph: Existing call graph (used to extract known symbols).
        llm: ``LLMClient`` for analyzing dynamic patterns.
        batch_size: Maximum number of concurrent LLM requests per batch.

    Returns:
        New ``RawEdge`` instances for discovered dynamic call relationships.
    """
    from backend.llm.prompts import analyze_dynamic_calls_prompt

    discovered_edges: list[RawEdge] = []

    # Only analyze functions and methods that could contain dynamic dispatch
    analyzable = [
        n for n in nodes
        if n.node_type in ("function", "method")
        and _has_dynamic_indicators(n.source_code)
    ]

    if not analyzable:
        return []

    # Extract known symbol names from the call graph for reference
    known_symbols = sorted(call_graph.nodes.keys())[:100]

    for batch_start in range(0, len(analyzable), batch_size):
        batch = analyzable[batch_start:batch_start + batch_size]
        tasks = [
            _analyze_dynamic_calls_single(
                node, known_symbols, llm, analyze_dynamic_calls_prompt
            )
            for node in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for node, result in zip(batch, results):
            if isinstance(result, Exception):
                logger.warning(
                    "Dynamic call analysis failed for %s: %s",
                    node.fully_qualified_name,
                    result,
                )
                continue
            discovered_edges.extend(result)

    logger.info("Discovered %d dynamic call edges", len(discovered_edges))
    return discovered_edges


_DYNAMIC_INDICATORS = frozenset({
    "getattr",
    "setattr",
    "importlib",
    "import_module",
    "__import__",
    "globals()",
    "locals()",
    "eval(",
    "exec(",
    "dispatch",
    "registry",
    "handler_map",
    "callback",
    "plugin",
    "hook",
})


def _has_dynamic_indicators(source_code: str) -> bool:
    """Quick check whether source code contains patterns suggesting dynamic dispatch."""
    lower = source_code.lower()
    return any(indicator in lower for indicator in _DYNAMIC_INDICATORS)


async def _analyze_dynamic_calls_single(
    node: RawNode,
    known_symbols: list[str],
    llm: Any,
    prompt_fn: Any,
) -> list[RawEdge]:
    """Analyze a single node for dynamic call patterns.

    Returns a list of newly discovered ``RawEdge`` instances.
    """
    prompt = prompt_fn(
        source_code=node.source_code,
        language=node.language or "python",
        file_path=node.file_path,
        function_name=node.name,
        known_symbols=known_symbols,
    )

    result = await llm.complete_json(prompt, temperature=0.1)
    if not result:
        return []

    calls = result.get("calls", [])
    if not isinstance(calls, list):
        return []

    edges: list[RawEdge] = []
    for call_info in calls:
        target = call_info.get("target", "")
        confidence = call_info.get("confidence", 0.0)
        mechanism = call_info.get("mechanism", "dynamic")

        if not target or confidence < 0.5:
            continue

        edges.append(RawEdge(
            source=node.fully_qualified_name,
            target=target,
            edge_type="calls",
            line_number=node.line_start,
            metadata={
                "dynamic": True,
                "mechanism": mechanism,
                "confidence": confidence,
                "llm_discovered": True,
            },
        ))

    return edges


# ------------------------------------------------------------------
# Missing connection suggestions
# ------------------------------------------------------------------


async def suggest_missing_connections(
    unmatched: set[str],
    call_graph: CallGraph,
    features: FeatureGroupResult,
    llm: Any,
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Suggest feature assignments for unmatched code nodes.

    For each unmatched node, asks the LLM to analyze its purpose and
    suggest which feature it should belong to, based on surrounding
    code context.

    Parameters:
        unmatched: Set of fully-qualified names of unmatched nodes.
        call_graph: The call graph containing all nodes.
        features: Current feature grouping result.
        llm: ``LLMClient`` for classification.
        batch_size: Maximum number of concurrent LLM requests per batch.

    Returns:
        List of suggestion dicts, each containing ``node_fqn``,
        ``suggested_feature``, ``reason``, and ``confidence``.
    """
    from backend.llm.prompts import classify_unmatched_code_prompt

    suggestions: list[dict[str, Any]] = []

    unmatched_nodes = [
        (fqn, call_graph.nodes[fqn])
        for fqn in unmatched
        if fqn in call_graph.nodes
    ]

    if not unmatched_nodes:
        return []

    # Build feature name index for matching
    feature_names = [f.name for f in features.features]

    for batch_start in range(0, len(unmatched_nodes), batch_size):
        batch = unmatched_nodes[batch_start:batch_start + batch_size]
        tasks = [
            _suggest_single_connection(
                fqn, node, call_graph, feature_names, llm, classify_unmatched_code_prompt
            )
            for fqn, node in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (fqn, _), result in zip(batch, results):
            if isinstance(result, Exception):
                logger.warning(
                    "Missing connection suggestion failed for %s: %s",
                    fqn,
                    result,
                )
                continue
            if result is not None:
                suggestions.append(result)

    logger.info(
        "Generated %d feature suggestions for %d unmatched nodes",
        len(suggestions),
        len(unmatched_nodes),
    )
    return suggestions


async def _suggest_single_connection(
    fqn: str,
    node: RawNode,
    call_graph: CallGraph,
    feature_names: list[str],
    llm: Any,
    prompt_fn: Any,
) -> dict[str, Any] | None:
    """Suggest a feature for a single unmatched node.

    Returns a suggestion dict or None if no suggestion could be made.
    """
    # Gather surrounding functions in the same file for context
    surrounding: list[str] = []
    for other_fqn, other_node in call_graph.nodes.items():
        if other_node.file_path == node.file_path and other_fqn != fqn:
            surrounding.append(other_node.name)
    surrounding = surrounding[:10]

    prompt = prompt_fn(
        source_code=node.source_code,
        language=node.language or "python",
        file_path=node.file_path,
        surrounding_functions=surrounding if surrounding else None,
    )

    result = await llm.complete_json(prompt, temperature=0.1)
    if not result:
        return None

    classification = result.get("classification", "")
    reason = result.get("reason", "")
    suggested_owner = result.get("suggested_owner")

    # Skip nodes classified as truly structural (not features)
    if classification in ("module_init", "global_config", "constant"):
        return {
            "node_fqn": fqn,
            "suggested_feature": "__structural__",
            "reason": reason,
            "confidence": 0.6,
            "classification": classification,
        }

    # Try to match suggested_owner to a known feature
    matched_feature = ""
    if suggested_owner:
        matched_feature = _match_to_feature(suggested_owner, call_graph, feature_names)

    if not matched_feature and classification not in ("dead_code", "other"):
        # Fallback: try to infer from file path proximity
        matched_feature = _infer_feature_from_file(node.file_path, call_graph, feature_names)

    return {
        "node_fqn": fqn,
        "suggested_feature": matched_feature,
        "reason": reason,
        "confidence": 0.7 if matched_feature else 0.3,
        "classification": classification,
    }


def _match_to_feature(
    suggested_owner: str,
    call_graph: CallGraph,
    feature_names: list[str],
) -> str:
    """Match an LLM-suggested owner to a feature name by checking node membership."""
    # Direct feature name match
    owner_lower = suggested_owner.lower().replace(" ", "_")
    for name in feature_names:
        if name.lower() == owner_lower:
            return name

    # Check if the suggested owner is a node that belongs to a feature
    # (this requires importing FeatureGroupResult, but we work with feature_names)
    for fqn in call_graph.nodes:
        if suggested_owner in fqn or fqn.endswith(f"::{suggested_owner}"):
            # Found the node; the caller will need to map it to a feature
            return suggested_owner

    return ""


def _infer_feature_from_file(
    file_path: str,
    call_graph: CallGraph,
    feature_names: list[str],
) -> str:
    """Infer a feature assignment by file path similarity.

    Finds the feature whose name best matches the file path components.
    Returns empty string if no reasonable match is found.
    """
    path_parts = set(
        file_path.replace("\\", "/")
        .replace(".py", "")
        .replace(".ts", "")
        .replace(".js", "")
        .split("/")
    )

    best_match = ""
    best_score = 0

    for name in feature_names:
        name_parts = set(name.split("_"))
        overlap = len(path_parts & name_parts)
        if overlap > best_score:
            best_score = overlap
            best_match = name

    # Require at least one meaningful overlap
    return best_match if best_score >= 1 else ""


# ------------------------------------------------------------------
# Cross-verification
# ------------------------------------------------------------------


async def verify_cross_check(
    features: FeatureGroupResult,
    call_graph: CallGraph,
    llm: Any,
    *,
    max_iterations: int = 3,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> VerificationResult:
    """Cross-verify AST-derived feature grouping against LLM semantic analysis.

    Runs an iterative convergence loop:
    1. For each feature, ask the LLM if the grouping is semantically coherent.
    2. Collect recommended changes (node moves between features).
    3. Check if AST structure supports the LLM recommendations.
    4. Converge when no more changes are recommended.

    Parameters:
        features: Current feature grouping result.
        call_graph: The call graph for structural validation.
        llm: ``LLMClient`` for semantic analysis.
        max_iterations: Maximum verification rounds before declaring convergence.
        batch_size: Maximum number of concurrent LLM requests per batch.

    Returns:
        A ``VerificationResult`` indicating convergence status and recommended changes.
    """
    from backend.llm.prompts import verify_feature_grouping_prompt

    all_changes: list[dict[str, str]] = []

    for iteration in range(1, max_iterations + 1):
        # Build feature summaries for the LLM
        feature_dicts: list[dict[str, object]] = []
        for feat in features.features:
            sample_fqns = sorted(feat.node_fqns)[:15]
            ep_names = [ep.node_name for ep in feat.entry_points]
            feature_dicts.append({
                "name": feat.name,
                "entry_points": ep_names,
                "functions": sample_fqns,
            })

        if not feature_dicts:
            return VerificationResult(is_converged=True, iteration=iteration)

        # Ask LLM to verify groupings in batches
        iteration_changes: list[dict[str, str]] = []

        for batch_start in range(0, len(feature_dicts), batch_size):
            batch = feature_dicts[batch_start:batch_start + batch_size]

            prompt = verify_feature_grouping_prompt(features=batch)
            result = await llm.complete_json(prompt, temperature=0.1)

            if not result:
                continue

            for feat_result in result.get("features", []):
                if feat_result.get("valid", True):
                    continue

                feat_name = feat_result.get("name", "")
                reason = feat_result.get("reason", "")
                suggested_split = feat_result.get("suggested_split")

                if suggested_split:
                    for split_name in suggested_split:
                        iteration_changes.append({
                            "type": "split",
                            "node_fqn": "",
                            "from_feature": feat_name,
                            "to_feature": split_name,
                            "reason": reason,
                        })
                else:
                    iteration_changes.append({
                        "type": "review",
                        "node_fqn": "",
                        "from_feature": feat_name,
                        "to_feature": "",
                        "reason": reason,
                    })

        # Validate LLM recommendations against AST structure
        validated_changes = _validate_changes_structurally(
            iteration_changes, call_graph, features
        )
        all_changes.extend(validated_changes)

        if not validated_changes:
            logger.info(
                "Cross-verification converged at iteration %d", iteration
            )
            return VerificationResult(
                is_converged=True,
                changes=all_changes,
                iteration=iteration,
            )

        logger.info(
            "Cross-verification iteration %d: %d changes recommended",
            iteration,
            len(validated_changes),
        )

    # Did not converge within max iterations
    return VerificationResult(
        is_converged=False,
        changes=all_changes,
        iteration=max_iterations,
    )


def _validate_changes_structurally(
    changes: list[dict[str, str]],
    call_graph: CallGraph,
    features: FeatureGroupResult,
) -> list[dict[str, str]]:
    """Filter LLM-recommended changes against AST structural constraints.

    Removes changes that would violate call-graph connectivity
    (e.g. splitting a feature whose nodes are tightly connected).
    """
    valid_changes: list[dict[str, str]] = []
    feature_map = {f.name: f for f in features.features}

    for change in changes:
        change_type = change.get("type", "")
        from_feature_name = change.get("from_feature", "")

        if change_type == "split":
            from_feature = feature_map.get(from_feature_name)
            if from_feature is None:
                continue

            # A split is valid if the feature has multiple disconnected
            # components (entry points whose reachable sets don't overlap).
            if len(from_feature.entry_points) < 2:
                # Single entry point -- split doesn't make structural sense
                continue

            # Check if entry points reach distinct subsets
            reachable_sets: list[set[str]] = []
            for ep in from_feature.entry_points:
                resolved = call_graph._resolve_entry_point_fqn(ep.node_name)
                if resolved is not None:
                    reachable_sets.append(call_graph.reachable_from(resolved))

            if len(reachable_sets) >= 2:
                # Check overlap between first two sets
                pairwise_overlap = reachable_sets[0] & reachable_sets[1]
                total_nodes = reachable_sets[0] | reachable_sets[1]
                if total_nodes and len(pairwise_overlap) / len(total_nodes) < 0.5:
                    valid_changes.append(change)

        elif change_type == "review":
            # Review changes are always passed through as advisory
            valid_changes.append(change)

    return valid_changes
