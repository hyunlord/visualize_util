"""Abstract base class and core data models for language analyzers.

Every language-specific analyzer (Python, TypeScript, Go, etc.) inherits
from ``BaseAnalyzer`` and populates the shared data structures defined
here.  The analyzer pipeline consumes ``AnalysisResult`` objects
regardless of the source language.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------


@dataclass(slots=True)
class RawNode:
    """A single code entity discovered during static analysis.

    Represents a function, class, method, or module-level block that
    can participate in call-graph edges and feature grouping.
    """

    name: str
    node_type: str  # "function" | "class" | "method" | "module"
    file_path: str
    line_start: int
    line_end: int
    source_code: str
    docstring: str | None = None
    language: str = ""
    metadata: dict = field(default_factory=dict)

    _VALID_NODE_TYPES = frozenset({"function", "class", "method", "module"})

    def __post_init__(self) -> None:
        if self.node_type not in self._VALID_NODE_TYPES:
            raise ValueError(
                f"Invalid node_type={self.node_type!r}. "
                f"Expected one of {sorted(self._VALID_NODE_TYPES)}"
            )

    @property
    def fully_qualified_name(self) -> str:
        """Derive a fully-qualified name from file path and node name."""
        return f"{self.file_path}::{self.name}"


@dataclass(slots=True)
class RawEdge:
    """A directed relationship between two code entities.

    Edges are resolved from call sites, import statements, inheritance
    declarations, or instantiation expressions found during analysis.
    """

    source: str  # fully qualified name of source node
    target: str  # fully qualified name or import reference of target
    edge_type: str  # "calls" | "imports" | "inherits" | "instantiates"
    line_number: int = 0
    metadata: dict = field(default_factory=dict)

    _VALID_EDGE_TYPES = frozenset({"calls", "imports", "inherits", "instantiates"})

    def __post_init__(self) -> None:
        if self.edge_type not in self._VALID_EDGE_TYPES:
            raise ValueError(
                f"Invalid edge_type={self.edge_type!r}. "
                f"Expected one of {sorted(self._VALID_EDGE_TYPES)}"
            )


@dataclass(slots=True)
class ImportInfo:
    """A single import statement with resolution metadata.

    Captures both absolute and relative imports so that the analyzer
    pipeline can resolve them to concrete file paths later.
    """

    module_path: str  # e.g., "backend.config"
    imported_names: list[str]  # e.g., ["Settings", "get_settings"]
    alias: str | None = None
    is_relative: bool = False
    relative_level: int = 0  # number of dots for relative import
    line_number: int = 0


@dataclass(slots=True)
class EntryPoint:
    """An externally-reachable entry into the code graph.

    Entry points anchor feature grouping -- every reachable sub-graph
    that starts from an entry point is considered part of the same
    feature.
    """

    node_name: str  # fully qualified function name
    entry_type: str  # "http_route" | "websocket" | "main" | "cli" | "export"
    route_info: dict = field(default_factory=dict)  # method, url, etc.

    _VALID_ENTRY_TYPES = frozenset(
        {"http_route", "websocket", "main", "cli", "export", "lifecycle"}
    )

    def __post_init__(self) -> None:
        if self.entry_type not in self._VALID_ENTRY_TYPES:
            raise ValueError(
                f"Invalid entry_type={self.entry_type!r}. "
                f"Expected one of {sorted(self._VALID_ENTRY_TYPES)}"
            )


@dataclass(slots=True)
class AnalysisResult:
    """Aggregated output of a single-file analysis pass.

    Collects all nodes, edges, entry points, imports, and any errors
    encountered while parsing a source file.
    """

    nodes: list[RawNode] = field(default_factory=list)
    edges: list[RawEdge] = field(default_factory=list)
    entry_points: list[EntryPoint] = field(default_factory=list)
    imports: list[ImportInfo] = field(default_factory=list)
    file_path: str = ""
    language: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


# ------------------------------------------------------------------
# Abstract analyzer
# ------------------------------------------------------------------


class BaseAnalyzer(ABC):
    """Interface that every language-specific analyzer must implement.

    The analyzer pipeline calls these methods in order:

    1. ``get_supported_extensions()`` -- used by the registry for routing.
    2. ``analyze_file()``            -- parse a single source file.
    3. ``detect_entry_points()``     -- identify externally-reachable nodes.
    4. ``resolve_import()``          -- map an import to a concrete file.
    """

    @abstractmethod
    def get_supported_extensions(self) -> list[str]:
        """Return file extensions this analyzer handles (e.g. ``['.py']``)."""
        ...

    @abstractmethod
    def analyze_file(
        self, file_path: str, source: str, repo_root: str
    ) -> AnalysisResult:
        """Parse *source* and return an ``AnalysisResult``.

        Parameters:
            file_path: Path of the file relative to *repo_root*.
            source: Full text content of the file.
            repo_root: Absolute path to the repository root.
        """
        ...

    @abstractmethod
    def resolve_import(
        self, import_info: ImportInfo, file_path: str, repo_root: str
    ) -> str | None:
        """Resolve an import to a concrete file path within the repo.

        Returns the resolved path relative to *repo_root*, or ``None``
        if the import refers to an external/third-party package.
        """
        ...

    @abstractmethod
    def detect_entry_points(
        self, nodes: list[RawNode], file_path: str
    ) -> list[EntryPoint]:
        """Scan *nodes* for externally-reachable entry points.

        The definition of "entry point" is language-specific -- HTTP route
        handlers for web frameworks, ``main()`` for CLI programs, etc.
        """
        ...
