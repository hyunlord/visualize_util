"""Build and query a function-level call graph from merged analysis results.

Merges per-file ``AnalysisResult`` objects into a single directed graph,
resolves symbolic call targets to concrete ``RawNode`` instances, and
provides traversal helpers used by feature grouping and dead-code detection.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

from backend.analyzer.base_analyzer import (
    AnalysisResult,
    EntryPoint,
    RawEdge,
    RawNode,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------


@dataclass(slots=True)
class CallGraph:
    """Fully-resolved call graph spanning one or more source files.

    Attributes:
        nodes: Mapping from fully-qualified name to the corresponding node.
        edges: All resolved edges (source and target are valid fqn keys).
        adjacency_forward: ``fqn -> set[fqn]`` for outgoing call targets.
        adjacency_reverse: ``fqn -> set[fqn]`` for incoming callers.
        entry_points: Externally-reachable entry points from all files.
        unresolved_edges: Edges whose target could not be matched to a node.
    """

    nodes: dict[str, RawNode] = field(default_factory=dict)
    edges: list[RawEdge] = field(default_factory=list)
    adjacency_forward: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    adjacency_reverse: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    entry_points: list[EntryPoint] = field(default_factory=list)
    unresolved_edges: list[RawEdge] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def reachable_from(self, entry_fqn: str) -> set[str]:
        """Return the set of node fqns reachable from *entry_fqn* via BFS.

        Follows ``adjacency_forward`` edges.  The *entry_fqn* itself is
        included in the returned set if it exists in the graph.
        """
        visited: set[str] = set()
        queue: deque[str] = deque()

        if entry_fqn in self.nodes:
            queue.append(entry_fqn)
            visited.add(entry_fqn)

        while queue:
            current = queue.popleft()
            for neighbour in self.adjacency_forward.get(current, ()):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)

        return visited

    def reachable_from_dfs(self, entry_fqn: str) -> set[str]:
        """Return the set of node fqns reachable from *entry_fqn* via DFS.

        Useful when traversal order matters (e.g. topological analysis).
        """
        visited: set[str] = set()
        stack: list[str] = []

        if entry_fqn in self.nodes:
            stack.append(entry_fqn)

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for neighbour in self.adjacency_forward.get(current, ()):
                if neighbour not in visited:
                    stack.append(neighbour)

        return visited

    def get_unreachable_nodes(self) -> set[str]:
        """Return fqns of nodes not reachable from any entry point.

        These are candidates for dead-code analysis.
        """
        all_reachable: set[str] = set()
        for ep in self.entry_points:
            resolved_fqn = self._resolve_entry_point_fqn(ep.node_name)
            if resolved_fqn is not None:
                all_reachable |= self.reachable_from(resolved_fqn)

        return set(self.nodes.keys()) - all_reachable

    def callers_of(self, fqn: str) -> set[str]:
        """Return the set of nodes that directly call *fqn*."""
        return set(self.adjacency_reverse.get(fqn, ()))

    def callees_of(self, fqn: str) -> set[str]:
        """Return the set of nodes directly called by *fqn*."""
        return set(self.adjacency_forward.get(fqn, ()))

    def has_only_self_references(self, fqn: str) -> bool:
        """Return True if the only incoming edge to *fqn* is from itself."""
        callers = self.adjacency_reverse.get(fqn, set())
        return callers == {fqn}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_entry_point_fqn(self, node_name: str) -> str | None:
        """Map an entry point's ``node_name`` to a graph fqn.

        Entry points may use a ``module:function`` qualified-name format
        while graph nodes use ``file_path::name``.  This method attempts
        both direct lookup and the resolution strategies.
        """
        if node_name in self.nodes:
            return node_name

        # Try matching by short name or qualified suffix
        for fqn in self.nodes:
            if fqn.endswith(f"::{node_name}") or fqn.endswith(f":{node_name}"):
                return fqn
            # module:Class.method -> file_path::method
            if ":" in node_name:
                _, short = node_name.rsplit(":", 1)
                if fqn.endswith(f"::{short}"):
                    return fqn

        return None


# ------------------------------------------------------------------
# Target resolution
# ------------------------------------------------------------------


class _TargetResolver:
    """Resolve symbolic call-edge targets to concrete node fqns.

    Uses a cascade of resolution strategies ordered from most to least
    specific.  The first match wins.
    """

    def __init__(self, nodes: dict[str, RawNode]) -> None:
        self._nodes = nodes

        # Pre-compute lookup indexes for efficient resolution.
        # short_name -> list of fqns (for ambiguous short names)
        self._short_name_index: dict[str, list[str]] = defaultdict(list)
        # module_prefix::name -> fqn
        self._module_index: dict[str, str] = {}
        # class_method patterns: ClassName.method_name -> fqn
        self._class_method_index: dict[str, str] = {}

        for fqn, node in nodes.items():
            # Short name index
            self._short_name_index[node.name].append(fqn)

            # Module index: strip file extension from file_path prefix
            # e.g. "backend/app.py::start_server" can match "app.start_server"
            parts = fqn.split("::")
            if len(parts) == 2:
                file_part, name_part = parts
                # Create dotted module path without extension
                module_dotted = file_part.replace("/", ".").replace("\\", ".")
                if module_dotted.endswith(".py"):
                    module_dotted = module_dotted[:-3]
                self._module_index[f"{module_dotted}::{name_part}"] = fqn
                # Also index just the last module segment
                last_module = module_dotted.rsplit(".", 1)[-1]
                self._module_index[f"{last_module}::{name_part}"] = fqn

            # Class method index from qualified_name metadata
            qname = node.metadata.get("qualified_name", "")
            if "." in qname and ":" in qname:
                # e.g. "backend.app:MyClass.my_method" -> "MyClass.my_method"
                _, class_method = qname.rsplit(":", 1)
                self._class_method_index[class_method] = fqn

    def resolve(self, target: str, source_fqn: str) -> str | None:
        """Attempt to resolve a symbolic *target* to a known node fqn.

        Resolution strategies (in order):
        1. Exact match in node dict
        2. Module-qualified match (``module.function``)
        3. Class.method match
        4. Same-file match (prefer targets in the same file as source)
        5. Short name match (unambiguous only)

        Returns ``None`` if no resolution is possible.
        """
        # Strategy 1: Exact match
        if target in self._nodes:
            return target

        # Strategy 2: Module-qualified match
        # Target might be "config.get_settings" -> try "config::get_settings"
        if "." in target:
            dotted_key = target.replace(".", "::", 1)
            if dotted_key in self._module_index:
                return self._module_index[dotted_key]

            # Try the last dotted segment as module prefix
            parts = target.rsplit(".", 1)
            if len(parts) == 2:
                module_part, name_part = parts
                key = f"{module_part}::{name_part}"
                if key in self._module_index:
                    return self._module_index[key]

        # Strategy 3: Class.method match
        if target in self._class_method_index:
            return self._class_method_index[target]

        # Also try "self.method" -> look up the method in the source's class
        if target.startswith("self."):
            method_name = target[5:]  # strip "self."
            source_file = source_fqn.split("::")[0] if "::" in source_fqn else ""
            # Find the class context from source_fqn metadata
            source_node = self._nodes.get(source_fqn)
            if source_node:
                source_qname = source_node.metadata.get("qualified_name", "")
                if ":" in source_qname:
                    module_part, rest = source_qname.rsplit(":", 1)
                    if "." in rest:
                        class_name = rest.rsplit(".", 1)[0]
                        class_method_key = f"{class_name}.{method_name}"
                        if class_method_key in self._class_method_index:
                            return self._class_method_index[class_method_key]

            # Fallback: search for any method with that name in the same file
            for fqn in self._short_name_index.get(method_name, []):
                node = self._nodes[fqn]
                if node.file_path == source_file and node.node_type == "method":
                    return fqn

        # Strategy 4: Same-file match
        source_file = source_fqn.split("::")[0] if "::" in source_fqn else ""
        short_name = target.rsplit(".", 1)[-1] if "." in target else target
        candidates = self._short_name_index.get(short_name, [])
        same_file_candidates = [
            fqn for fqn in candidates
            if self._nodes[fqn].file_path == source_file
        ]
        if len(same_file_candidates) == 1:
            return same_file_candidates[0]

        # Strategy 5: Short name match (unambiguous across project)
        if len(candidates) == 1:
            return candidates[0]

        return None


# ------------------------------------------------------------------
# Import edge resolution
# ------------------------------------------------------------------


def _resolve_import_edges(
    edges: list[RawEdge],
    nodes: dict[str, RawNode],
) -> tuple[list[RawEdge], list[RawEdge]]:
    """Refine import edges to point to specific imported symbols.

    For an import edge like ``module_a -> module_b`` that carries
    ``metadata["imported_names"] = ["func_x", "ClassY"]``, create
    additional edges from ``module_a`` to the specific target nodes
    (``module_b::func_x``, ``module_b::ClassY``) when they exist.

    Returns ``(resolved_edges, unresolved_edges)``.
    """
    resolved: list[RawEdge] = []
    unresolved: list[RawEdge] = []

    # Index nodes by file_path for efficient lookup
    file_nodes: dict[str, list[RawNode]] = defaultdict(list)
    for node in nodes.values():
        file_nodes[node.file_path].append(node)

    for edge in edges:
        if edge.edge_type != "imports":
            resolved.append(edge)
            continue

        imported_names = edge.metadata.get("imported_names", [])
        if not imported_names:
            # No specific names -- keep the module-level edge
            resolved.append(edge)
            continue

        # The target of an import edge is a module dotted path.
        # Find nodes whose file_path corresponds to that module.
        target_module = edge.target
        matched_any = False

        for name in imported_names:
            # Try direct fqn construction: target files might use
            # different path conventions, so search broadly.
            target_fqn_candidates = [
                fqn for fqn, node in nodes.items()
                if node.name == name and (
                    # Module path matches file path (dotted vs slash)
                    _module_matches_file(target_module, node.file_path)
                )
            ]

            if target_fqn_candidates:
                for target_fqn in target_fqn_candidates:
                    resolved.append(RawEdge(
                        source=edge.source,
                        target=target_fqn,
                        edge_type="imports",
                        line_number=edge.line_number,
                        metadata={**edge.metadata, "resolved_symbol": name},
                    ))
                matched_any = True

        if not matched_any:
            # Keep the original edge as unresolved
            unresolved.append(edge)
        else:
            # Also keep the original module-level edge for completeness
            resolved.append(edge)

    return resolved, unresolved


def _module_matches_file(module_path: str, file_path: str) -> bool:
    """Check whether a dotted module path corresponds to a file path.

    ``"backend.config"`` matches ``"backend/config.py"`` or
    ``"backend/config/__init__.py"``.
    """
    normalized_module = module_path.replace(".", "/")
    normalized_file = file_path.replace("\\", "/")

    # Strip .py extension and __init__ suffix
    if normalized_file.endswith(".py"):
        normalized_file = normalized_file[:-3]
    if normalized_file.endswith("/__init__"):
        normalized_file = normalized_file[: -len("/__init__")]

    return normalized_file == normalized_module or normalized_file.endswith(f"/{normalized_module}")


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def build_call_graph(all_results: list[AnalysisResult]) -> CallGraph:
    """Build a unified call graph from multiple per-file analysis results.

    Steps:
    1. Merge all nodes into a single dict keyed by fully-qualified name.
    2. Collect and deduplicate entry points.
    3. Resolve call-edge targets to concrete node fqns.
    4. Resolve import edges to specific imported symbols where possible.
    5. Build forward and reverse adjacency indexes.

    Parameters:
        all_results: Per-file analysis results to merge.

    Returns:
        A fully-populated ``CallGraph`` ready for traversal.
    """
    graph = CallGraph()

    # ------------------------------------------------------------------
    # Step 1: Merge nodes
    # ------------------------------------------------------------------
    for result in all_results:
        for node in result.nodes:
            fqn = node.fully_qualified_name
            if fqn in graph.nodes:
                logger.debug(
                    "Duplicate node fqn %s from %s (keeping first occurrence)",
                    fqn,
                    node.file_path,
                )
                continue
            graph.nodes[fqn] = node

    logger.info("Merged %d nodes from %d files", len(graph.nodes), len(all_results))

    # ------------------------------------------------------------------
    # Step 2: Collect entry points (deduplicate by node_name)
    # ------------------------------------------------------------------
    seen_entries: set[str] = set()
    for result in all_results:
        for ep in result.entry_points:
            if ep.node_name not in seen_entries:
                seen_entries.add(ep.node_name)
                graph.entry_points.append(ep)

    logger.info("Collected %d entry points", len(graph.entry_points))

    # ------------------------------------------------------------------
    # Step 3: Resolve call/instantiate/inherit edges
    # ------------------------------------------------------------------
    resolver = _TargetResolver(graph.nodes)
    all_edges: list[RawEdge] = []

    for result in all_results:
        all_edges.extend(result.edges)

    resolved_edges: list[RawEdge] = []
    import_edges: list[RawEdge] = []

    for edge in all_edges:
        if edge.edge_type == "imports":
            import_edges.append(edge)
            continue

        resolved_target = resolver.resolve(edge.target, edge.source)
        if resolved_target is not None:
            resolved_edges.append(RawEdge(
                source=edge.source,
                target=resolved_target,
                edge_type=edge.edge_type,
                line_number=edge.line_number,
                metadata=edge.metadata,
            ))
        else:
            graph.unresolved_edges.append(edge)

    # ------------------------------------------------------------------
    # Step 4: Resolve import edges to specific symbols
    # ------------------------------------------------------------------
    refined_imports, unresolved_imports = _resolve_import_edges(
        import_edges, graph.nodes
    )
    resolved_edges.extend(refined_imports)
    graph.unresolved_edges.extend(unresolved_imports)

    # ------------------------------------------------------------------
    # Step 5: Build adjacency and final edge list
    # ------------------------------------------------------------------
    # Ensure adjacency dicts use defaultdict behavior
    if not isinstance(graph.adjacency_forward, defaultdict):
        graph.adjacency_forward = defaultdict(set, graph.adjacency_forward)
    if not isinstance(graph.adjacency_reverse, defaultdict):
        graph.adjacency_reverse = defaultdict(set, graph.adjacency_reverse)

    for edge in resolved_edges:
        # Only build adjacency for edges whose both endpoints exist
        source_exists = edge.source in graph.nodes
        target_exists = edge.target in graph.nodes

        graph.edges.append(edge)

        if source_exists and target_exists:
            graph.adjacency_forward[edge.source].add(edge.target)
            graph.adjacency_reverse[edge.target].add(edge.source)
        elif source_exists:
            # Source exists but target is a module-level reference;
            # keep the edge but only in forward adjacency for reachability.
            graph.adjacency_forward[edge.source].add(edge.target)

    logger.info(
        "Call graph built: %d resolved edges, %d unresolved edges",
        len(graph.edges),
        len(graph.unresolved_edges),
    )

    return graph
