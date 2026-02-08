"""Group code nodes into features based on entry points and call-graph traversal.

Uses a combination of AST-derived structure (entry point clustering, BFS
reachability) and optional LLM verification to produce semantically
coherent feature groups.  Shared utility code reachable from multiple
features is separated into a dedicated bucket.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from backend.analyzer.base_analyzer import EntryPoint
from backend.analyzer.call_graph import CallGraph

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Color palette for feature groups
# ------------------------------------------------------------------

_FEATURE_COLORS: list[str] = [
    "#4C72B0",  # steel blue
    "#DD8452",  # sandy brown
    "#55A868",  # medium sea green
    "#C44E52",  # indian red
    "#8172B3",  # medium purple
    "#937860",  # dark tan
    "#DA8BC3",  # plum
    "#8C8C8C",  # gray
    "#CCB974",  # dark khaki
    "#64B5CD",  # sky blue
    "#A1D99B",  # light green
    "#FC9272",  # salmon
    "#BCBDDC",  # lavender
    "#FDD0A2",  # peach
    "#C7E9C0",  # honeydew
    "#FDAE6B",  # light orange
]

_SHARED_UTILS_COLOR = "#AAAAAA"

_MAX_VERIFICATION_ITERATIONS = 3


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------


@dataclass(slots=True)
class FeatureGroup:
    """A coherent set of code nodes representing a user-facing feature.

    Attributes:
        name: Machine-readable feature name in snake_case.
        description: Human-readable one-sentence summary.
        entry_points: Entry points that anchor this feature.
        node_fqns: Fully-qualified names of all nodes belonging to this feature.
        color: Hex color assigned to this feature for visualization.
    """

    name: str
    description: str
    entry_points: list[EntryPoint]
    node_fqns: set[str] = field(default_factory=set)
    color: str = ""


@dataclass(slots=True)
class FeatureGroupResult:
    """Aggregated output of the feature grouping pipeline.

    Attributes:
        features: Named feature groups with their constituent nodes.
        shared_utils: fqns reachable from two or more features.
        unmatched: fqns not reachable from any entry point.
    """

    features: list[FeatureGroup] = field(default_factory=list)
    shared_utils: set[str] = field(default_factory=set)
    unmatched: set[str] = field(default_factory=set)


# ------------------------------------------------------------------
# Step 1: Cluster entry points
# ------------------------------------------------------------------


def _cluster_entry_points(
    entry_points: list[EntryPoint],
) -> list[list[EntryPoint]]:
    """Group entry points by shared URL prefix, directory, or decorator pattern.

    Clustering heuristics (applied in order):
    1. HTTP routes sharing a URL path prefix (e.g. ``/api/users``).
    2. CLI commands sharing a command group.
    3. Entry points in the same directory.
    4. Remaining entry points each form their own cluster.
    """
    if not entry_points:
        return []

    # Separate by entry type first
    http_entries: list[EntryPoint] = []
    ws_entries: list[EntryPoint] = []
    cli_entries: list[EntryPoint] = []
    other_entries: list[EntryPoint] = []

    for ep in entry_points:
        if ep.entry_type == "http_route":
            http_entries.append(ep)
        elif ep.entry_type == "websocket":
            ws_entries.append(ep)
        elif ep.entry_type == "cli":
            cli_entries.append(ep)
        else:
            other_entries.append(ep)

    clusters: list[list[EntryPoint]] = []

    # Cluster HTTP routes by URL prefix
    if http_entries:
        clusters.extend(_cluster_by_url_prefix(http_entries))

    # Cluster WebSocket endpoints by URL prefix
    if ws_entries:
        clusters.extend(_cluster_by_url_prefix(ws_entries))

    # Cluster CLI commands by command group
    if cli_entries:
        clusters.extend(_cluster_by_cli_group(cli_entries))

    # Cluster remaining entries by directory
    if other_entries:
        clusters.extend(_cluster_by_directory(other_entries))

    return clusters


def _cluster_by_url_prefix(entries: list[EntryPoint]) -> list[list[EntryPoint]]:
    """Group HTTP/WS entries by their first two URL path segments."""
    prefix_groups: dict[str, list[EntryPoint]] = defaultdict(list)

    for ep in entries:
        url = ep.route_info.get("url", "")
        prefix = _extract_url_prefix(url)
        prefix_groups[prefix].append(ep)

    return list(prefix_groups.values())


def _extract_url_prefix(url: str) -> str:
    """Extract a clustering key from a URL path.

    Takes the first two non-parameter segments::

        /api/v1/users/{id}/orders  ->  /api/v1/users
        /health                    ->  /health
        /                          ->  /
    """
    segments = [s for s in url.strip("/").split("/") if s]
    # Filter out path parameter segments like {id} or :id
    meaningful = [
        s for s in segments
        if not (s.startswith("{") or s.startswith(":"))
    ]
    if not meaningful:
        return url or "/"
    # Use up to 2 meaningful segments as the prefix
    prefix_parts = meaningful[:2]
    return "/" + "/".join(prefix_parts)


def _cluster_by_cli_group(entries: list[EntryPoint]) -> list[list[EntryPoint]]:
    """Group CLI entries by command group (first word of command name)."""
    groups: dict[str, list[EntryPoint]] = defaultdict(list)
    for ep in entries:
        cmd = ep.route_info.get("command", ep.node_name)
        group_name = cmd.split()[0] if cmd else ep.node_name
        groups[group_name].append(ep)
    return list(groups.values())


def _cluster_by_directory(entries: list[EntryPoint]) -> list[list[EntryPoint]]:
    """Group entries by the directory of their source node."""
    dir_groups: dict[str, list[EntryPoint]] = defaultdict(list)
    for ep in entries:
        # Extract directory from the node_name (which contains file path info)
        node_name = ep.node_name
        if "::" in node_name:
            file_part = node_name.split("::")[0]
        elif ":" in node_name:
            file_part = node_name.split(":")[0]
        else:
            file_part = node_name

        directory = os.path.dirname(file_part) or "root"
        dir_groups[directory].append(ep)

    return list(dir_groups.values())


# ------------------------------------------------------------------
# Step 2: BFS from clusters to tag reachable nodes
# ------------------------------------------------------------------


def _bfs_tag_nodes(
    clusters: list[list[EntryPoint]],
    call_graph: CallGraph,
) -> dict[int, set[str]]:
    """For each cluster index, BFS from its entry points and collect reachable fqns.

    Returns a mapping from cluster index to the set of reachable node fqns.
    """
    cluster_nodes: dict[int, set[str]] = {}

    for idx, cluster in enumerate(clusters):
        reachable: set[str] = set()
        for ep in cluster:
            resolved = call_graph._resolve_entry_point_fqn(ep.node_name)
            if resolved is not None:
                reachable |= call_graph.reachable_from(resolved)
        cluster_nodes[idx] = reachable

    return cluster_nodes


# ------------------------------------------------------------------
# Step 3: Separate shared utils
# ------------------------------------------------------------------


def _separate_shared(
    cluster_nodes: dict[int, set[str]],
    all_node_fqns: set[str],
) -> tuple[dict[int, set[str]], set[str], set[str]]:
    """Split nodes into exclusive feature members, shared utilities, and unmatched.

    A node appearing in 2+ clusters is moved to ``shared_utils``.
    Nodes not in any cluster are ``unmatched``.

    Returns:
        ``(exclusive_clusters, shared_utils, unmatched)``
    """
    # Count cluster membership for each node
    membership_count: dict[str, int] = defaultdict(int)
    for fqns in cluster_nodes.values():
        for fqn in fqns:
            membership_count[fqn] += 1

    shared_utils: set[str] = {
        fqn for fqn, count in membership_count.items()
        if count >= 2
    }

    # Remove shared from each cluster
    exclusive: dict[int, set[str]] = {
        idx: fqns - shared_utils
        for idx, fqns in cluster_nodes.items()
    }

    # Find unmatched (not in any cluster)
    all_claimed = set()
    for fqns in cluster_nodes.values():
        all_claimed |= fqns

    unmatched = all_node_fqns - all_claimed

    return exclusive, shared_utils, unmatched


# ------------------------------------------------------------------
# Step 4 & 5: LLM naming, verification, and re-grouping
# ------------------------------------------------------------------


def _strip_path_params(url: str) -> str:
    """Remove path parameter segments like ``{id}`` or ``:id`` from a URL."""
    segments = url.strip("/").split("/")
    return "/".join(
        s for s in segments
        if s and not s.startswith("{") and not s.startswith(":")
    )


def _generate_default_name(cluster: list[EntryPoint], index: int) -> tuple[str, str]:
    """Generate a default feature name from cluster entry points when LLM is unavailable.

    Returns ``(name, description)``.
    """
    if not cluster:
        return f"feature_{index}", f"Feature group {index}"

    # Collect function names for fallback
    func_names = []
    for ep in cluster:
        short = ep.node_name.split("::")[-1].split(":")[-1]
        if short:
            func_names.append(short)

    # Try to derive a name from URL prefixes (strip path params first)
    urls = [ep.route_info.get("url", "") for ep in cluster if ep.route_info.get("url")]
    cleaned_urls = [_strip_path_params(u) for u in urls]
    cleaned_urls = [u for u in cleaned_urls if u]  # filter empty

    if cleaned_urls:
        common = os.path.commonprefix(cleaned_urls).strip("/")
        if common:
            # Convert URL path to readable name: "dead-code" → "Dead Code"
            readable = common.replace("/", " ").replace("-", " ").replace("_", " ")
            readable = " ".join(w.capitalize() for w in readable.split())
            snake = re.sub(r"[^a-zA-Z0-9]+", "_", common).strip("_").lower()
            if snake:
                methods = {
                    ep.route_info.get("method", "").upper()
                    for ep in cluster if ep.route_info.get("method")
                }
                method_str = "/".join(sorted(methods)) if methods else ""
                desc = f"{method_str} /{common}" if method_str else f"Handles /{common}"
                return readable, desc

    # Try using function names to create a meaningful name
    if func_names:
        # Group by common prefix of function names (e.g. list_repos, create_repo -> Repo)
        if len(func_names) == 1:
            readable = func_names[0].replace("_", " ").title()
            return readable, f"Endpoint: {func_names[0]}"
        # Use the common noun: create_repo, list_repos → "Repos"
        words_sets = [set(fn.split("_")) for fn in func_names]
        common_words = words_sets[0]
        for ws in words_sets[1:]:
            common_words &= ws
        # Remove common verbs
        common_words -= {"get", "set", "list", "create", "update", "delete", "post", "put"}
        if common_words:
            readable = " ".join(w.capitalize() for w in sorted(common_words))
            return readable, f"Endpoints: {', '.join(func_names)}"

    # Try CLI command group
    commands = [ep.route_info.get("command", "") for ep in cluster if ep.route_info.get("command")]
    if commands:
        name = re.sub(r"[^a-zA-Z0-9]+", "_", commands[0]).strip("_").lower()
        if name:
            return name, f"CLI command group: {commands[0]}"

    # Try entry type for non-generic types
    types = {ep.entry_type for ep in cluster}
    if len(types) == 1:
        entry_type = next(iter(types))
        if entry_type == "websocket":
            ws_urls = [_strip_path_params(ep.route_info.get("url", "")) for ep in cluster]
            ws_url = ws_urls[0] if ws_urls else ""
            readable = ws_url.replace("/", " ").replace("-", " ").strip().title() or "WebSocket"
            return f"{readable} (WS)", f"WebSocket endpoint: {ws_url or func_names[0] if func_names else 'unknown'}"
        if entry_type == "cli":
            return f"CLI {index}", f"CLI entry point group"

    # Fallback: use first function name or combine function names
    if func_names:
        if len(func_names) == 1:
            readable = func_names[0].replace("_", " ").title()
            return readable, f"Endpoint: {func_names[0]}"
        # Try to find the dominant noun across functions
        all_words: list[str] = []
        for fn in func_names:
            all_words.extend(fn.split("_"))
        # Remove common verbs and find most frequent noun
        verbs = {"get", "set", "list", "create", "update", "delete", "post",
                 "put", "start", "stop", "run", "is", "has", "do"}
        nouns = [w for w in all_words if w.lower() not in verbs and len(w) > 2]
        if nouns:
            # Pick the most common noun
            from collections import Counter
            noun_counts = Counter(nouns)
            best_noun = noun_counts.most_common(1)[0][0]
            readable = best_noun.replace("_", " ").title()
            return readable, f"Endpoints: {', '.join(func_names)}"
        # Last resort: join first two function names
        readable = " & ".join(fn.replace("_", " ").title() for fn in func_names[:2])
        return readable, f"Endpoints: {', '.join(func_names)}"

    return f"feature_{index}", f"Feature group {index}"


async def _llm_name_features(
    clusters: list[list[EntryPoint]],
    features: list[FeatureGroup],
    llm: Any,
) -> None:
    """Use LLM to generate meaningful names for feature groups.

    Mutates ``features`` in place with LLM-suggested names/descriptions.
    Falls back to default names on LLM failure.
    """
    from backend.llm.prompts import name_feature_prompt

    for idx, (cluster, feature) in enumerate(zip(clusters, features)):
        ep_dicts = []
        for ep in cluster:
            ep_dict: dict[str, str] = {"name": ep.node_name}
            if ep.route_info.get("url"):
                ep_dict["route"] = ep.route_info["url"]
            if ep.route_info.get("method"):
                ep_dict["method"] = ep.route_info["method"]
            ep_dicts.append(ep_dict)

        if not ep_dicts:
            continue

        prompt = name_feature_prompt(entry_points=ep_dicts)
        result = await llm.complete_json(prompt, temperature=0.1)

        if result and result.get("name"):
            feature.name = str(result["name"])
            feature.description = str(result.get("description", feature.description))


async def _llm_verify_grouping(
    features: list[FeatureGroup],
    call_graph: CallGraph,
    llm: Any,
) -> dict[str, Any]:
    """Ask LLM to validate whether feature groupings are semantically coherent.

    Returns the parsed verification result, or an empty dict on failure.
    """
    from backend.llm.prompts import verify_feature_grouping_prompt

    feature_dicts: list[dict[str, object]] = []
    for feat in features:
        # Sample representative function names for the prompt
        func_names = sorted(feat.node_fqns)[:10]
        ep_names = [ep.node_name for ep in feat.entry_points]
        feature_dicts.append({
            "name": feat.name,
            "entry_points": ep_names,
            "functions": func_names,
        })

    if not feature_dicts:
        return {}

    prompt = verify_feature_grouping_prompt(features=feature_dicts)
    return await llm.complete_json(prompt, temperature=0.1)


def _structural_check(features: list[FeatureGroup], call_graph: CallGraph) -> list[str]:
    """AST-based structural checks on feature groupings.

    Returns a list of issue descriptions.  An empty list means the
    grouping passes structural validation.
    """
    issues: list[str] = []

    for feat in features:
        if not feat.entry_points:
            issues.append(f"Feature '{feat.name}' has no entry points")
            continue

        if not feat.node_fqns:
            issues.append(f"Feature '{feat.name}' has no code nodes")
            continue

        # Check connectivity: all nodes should be reachable from at least
        # one of the feature's entry points.
        reachable: set[str] = set()
        for ep in feat.entry_points:
            resolved = call_graph._resolve_entry_point_fqn(ep.node_name)
            if resolved is not None:
                reachable |= call_graph.reachable_from(resolved)

        unreachable_in_feature = feat.node_fqns - reachable
        if unreachable_in_feature:
            pct = len(unreachable_in_feature) / max(len(feat.node_fqns), 1)
            if pct > 0.5:
                issues.append(
                    f"Feature '{feat.name}': {len(unreachable_in_feature)}/{len(feat.node_fqns)} "
                    f"nodes are unreachable from its entry points"
                )

    return issues


async def _verification_loop(
    features: list[FeatureGroup],
    clusters: list[list[EntryPoint]],
    call_graph: CallGraph,
    llm: Any,
) -> list[FeatureGroup]:
    """Run up to ``_MAX_VERIFICATION_ITERATIONS`` rounds of verification.

    Each round:
    1. AST structural check.
    2. LLM semantic check (if LLM available).
    3. Re-group if issues found.

    Returns the final list of features.
    """
    for iteration in range(1, _MAX_VERIFICATION_ITERATIONS + 1):
        # AST structural check
        structural_issues = _structural_check(features, call_graph)
        if structural_issues:
            logger.info(
                "Verification iteration %d: %d structural issues",
                iteration,
                len(structural_issues),
            )
            for issue in structural_issues:
                logger.debug("  %s", issue)

        # LLM semantic check
        llm_issues: list[dict[str, Any]] = []
        if llm is not None:
            verification = await _llm_verify_grouping(features, call_graph, llm)
            for feat_result in verification.get("features", []):
                if not feat_result.get("valid", True):
                    llm_issues.append(feat_result)

        if not structural_issues and not llm_issues:
            logger.info("Verification converged at iteration %d", iteration)
            break

        # Attempt re-grouping based on LLM suggestions
        if llm_issues:
            features = _apply_split_suggestions(features, llm_issues, call_graph)

    return features


def _apply_split_suggestions(
    features: list[FeatureGroup],
    llm_issues: list[dict[str, Any]],
    call_graph: CallGraph,
) -> list[FeatureGroup]:
    """Apply LLM-suggested splits to invalid feature groups.

    If the LLM suggests splitting a feature, create new sub-features
    by re-running BFS from entry point subsets.  Features the LLM
    validated are kept unchanged.
    """
    invalid_names = {
        issue["name"] for issue in llm_issues
        if issue.get("suggested_split")
    }

    result: list[FeatureGroup] = []
    color_idx = 0

    for feat in features:
        if feat.name not in invalid_names:
            result.append(feat)
            color_idx += 1
            continue

        # Find the corresponding LLM suggestion
        suggestion = next(
            (i for i in llm_issues if i["name"] == feat.name),
            None,
        )
        if suggestion is None or not suggestion.get("suggested_split"):
            result.append(feat)
            color_idx += 1
            continue

        # Split the feature's entry points across suggested sub-features.
        # Simple heuristic: divide entry points evenly among suggested names.
        suggested_names: list[str] = suggestion["suggested_split"]
        ep_count = len(feat.entry_points)
        chunk_size = max(1, ep_count // len(suggested_names))

        for i, sub_name in enumerate(suggested_names):
            start = i * chunk_size
            end = start + chunk_size if i < len(suggested_names) - 1 else ep_count
            sub_eps = feat.entry_points[start:end]

            if not sub_eps:
                continue

            sub_reachable: set[str] = set()
            for ep in sub_eps:
                resolved = call_graph._resolve_entry_point_fqn(ep.node_name)
                if resolved is not None:
                    sub_reachable |= call_graph.reachable_from(resolved)

            sub_feature = FeatureGroup(
                name=sub_name,
                description=f"Split from '{feat.name}'",
                entry_points=sub_eps,
                node_fqns=sub_reachable & feat.node_fqns,
                color=_FEATURE_COLORS[color_idx % len(_FEATURE_COLORS)],
            )
            result.append(sub_feature)
            color_idx += 1

    return result


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


async def group_features(
    call_graph: CallGraph,
    llm: Any | None = None,
) -> FeatureGroupResult:
    """Group code nodes into features based on entry points and call-graph traversal.

    Parameters:
        call_graph: The resolved call graph to partition.
        llm: Optional ``LLMClient`` for LLM-assisted naming and verification.

    Returns:
        A ``FeatureGroupResult`` containing named features, shared utilities,
        and unmatched code nodes.
    """
    return await _group_features_async(call_graph, llm)


async def _group_features_async(
    call_graph: CallGraph,
    llm: Any | None = None,
) -> FeatureGroupResult:
    """Async implementation of the feature grouping pipeline.

    Steps:
    1. Cluster entry points by URL prefix / directory / decorator pattern.
    2. BFS from each cluster's entry points to tag reachable nodes.
    3. Separate shared nodes (reachable from 2+ features).
    4. If LLM available, name features and run verification loop.
    5. Assign colors.
    """
    # Step 1: Cluster entry points
    clusters = _cluster_entry_points(call_graph.entry_points)
    logger.info("Formed %d entry-point clusters", len(clusters))

    if not clusters:
        return FeatureGroupResult(
            unmatched=set(call_graph.nodes.keys()),
        )

    # Step 2: BFS tagging
    cluster_nodes = _bfs_tag_nodes(clusters, call_graph)

    # Step 3: Separate shared utilities
    all_node_fqns = set(call_graph.nodes.keys())
    exclusive, shared_utils, unmatched = _separate_shared(cluster_nodes, all_node_fqns)

    # Build initial feature groups
    features: list[FeatureGroup] = []
    for idx, cluster in enumerate(clusters):
        name, description = _generate_default_name(cluster, idx)
        feature = FeatureGroup(
            name=name,
            description=description,
            entry_points=list(cluster),
            node_fqns=exclusive.get(idx, set()),
            color=_FEATURE_COLORS[idx % len(_FEATURE_COLORS)],
        )
        features.append(feature)

    # Step 4: LLM naming and verification
    if llm is not None:
        # Name features with LLM
        await _llm_name_features(clusters, features, llm)

        # Step 5: Verification loop (AST + LLM)
        features = await _verification_loop(features, clusters, call_graph, llm)

    # Re-assign colors after potential splits
    for idx, feature in enumerate(features):
        feature.color = _FEATURE_COLORS[idx % len(_FEATURE_COLORS)]

    # Remove empty features
    features = [f for f in features if f.node_fqns or f.entry_points]

    # Deduplicate names by appending a qualifier from entry points
    name_counts: dict[str, int] = defaultdict(int)
    for feat in features:
        name_counts[feat.name] += 1
    seen_names: dict[str, int] = defaultdict(int)
    for feat in features:
        if name_counts[feat.name] > 1:
            seen_names[feat.name] += 1
            # Try to differentiate using function names from entry points
            funcs = [
                ep.node_name.split("::")[-1].split(":")[-1]
                for ep in feat.entry_points
            ]
            if funcs:
                qualifier = funcs[0].replace("_", " ").title()
                feat.name = f"{feat.name} ({qualifier})"
            else:
                feat.name = f"{feat.name} {seen_names[feat.name]}"

    logger.info(
        "Feature grouping complete: %d features, %d shared utils, %d unmatched",
        len(features),
        len(shared_utils),
        len(unmatched),
    )

    return FeatureGroupResult(
        features=features,
        shared_utils=shared_utils,
        unmatched=unmatched,
    )
