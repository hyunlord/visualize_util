"""Comprehensive Python source-code analyzer built on the ``ast`` module.

Extracts functions, classes, methods, imports, call edges, and entry points
from Python files.  Designed to handle real-world patterns found in production
FastAPI / Flask applications including:

* Lazy imports inside async function bodies
* Module-level singleton instantiation (``job_queue = JobQueue()``)
* Conditional post-processing calls (guarded by ``if`` / ``elif`` / ``else``)
* Decorator-based HTTP routes, WebSocket endpoints, and CLI commands
* CLIP-style lazy model loading / unloading patterns
* Nested functions and closures
* Star imports and relative imports
"""

from __future__ import annotations

import ast
import logging
import os
import textwrap
from pathlib import Path
from typing import Any

from backend.analyzer.base_analyzer import (
    AnalysisResult,
    BaseAnalyzer,
    EntryPoint,
    ImportInfo,
    RawEdge,
    RawNode,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _file_to_module(file_path: str, repo_root: str) -> str:
    """Convert a file path to a dotted module name relative to *repo_root*.

    ``/repo/backend/app.py`` with repo_root ``/repo`` becomes ``backend.app``.
    ``__init__.py`` files yield the package name (no trailing ``.__init__``).
    """
    rel = os.path.relpath(file_path, repo_root)
    parts = Path(rel).with_suffix("").parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _get_end_lineno(node: ast.AST) -> int:
    """Return the end line number, falling back to start if unavailable."""
    return getattr(node, "end_lineno", None) or getattr(node, "lineno", 0)


def _get_source_segment(source_lines: list[str], start: int, end: int) -> str:
    """Extract source text for lines *start*..*end* (1-based, inclusive)."""
    if not source_lines or start < 1:
        return ""
    segment = source_lines[start - 1 : end]
    return textwrap.dedent("\n".join(segment))


def _get_docstring(node: ast.AST) -> str | None:
    """Return the docstring of a function / class node, or ``None``."""
    return ast.get_docstring(node)


def _unparse_annotation(node: ast.expr | None) -> str | None:
    """Best-effort unparse of a type-annotation AST node."""
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _extract_call_name(node: ast.Call) -> str | None:
    """Return a human-readable callee name from a ``Call`` node.

    Handles:
    * Simple:       ``func()``          -> ``"func"``
    * Attribute:    ``obj.method()``    -> ``"obj.method"``
    * Chained:      ``a.b.c()``         -> ``"a.b.c"``
    * Subscript:    ``d["k"]()``        -> ``None`` (too dynamic)
    """
    func = node.func
    parts: list[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
        parts.reverse()
        return ".".join(parts)
    return None


def _decorator_names(decorators: list[ast.expr]) -> list[dict[str, Any]]:
    """Extract decorator info (name + arguments) from a list of decorator nodes."""
    result: list[dict[str, Any]] = []
    for deco in decorators:
        info: dict[str, Any] = {"name": "", "args": []}
        node = deco
        # Handle @deco(...) call form
        if isinstance(node, ast.Call):
            info["args"] = [ast.unparse(a) for a in node.args]
            info["kwargs"] = {
                kw.arg: ast.unparse(kw.value)
                for kw in node.keywords
                if kw.arg is not None
            }
            node = node.func
        # Resolve the name (handles chained attributes like app.get)
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        parts.reverse()
        info["name"] = ".".join(parts)
        result.append(info)
    return result


def _is_inside_conditional(ancestors: list[ast.AST]) -> bool:
    """Return ``True`` if the ancestor chain contains an ``If`` node."""
    return any(isinstance(a, ast.If) for a in ancestors)


def _extract_function_args(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[dict[str, Any]]:
    """Extract argument names, defaults, and annotations from a function def."""
    args_info: list[dict[str, Any]] = []
    all_args = (
        node.args.posonlyargs + node.args.args + node.args.kwonlyargs
    )
    for arg in all_args:
        args_info.append({
            "name": arg.arg,
            "annotation": _unparse_annotation(arg.annotation),
        })
    if node.args.vararg:
        args_info.append({
            "name": f"*{node.args.vararg.arg}",
            "annotation": _unparse_annotation(node.args.vararg.annotation),
        })
    if node.args.kwarg:
        args_info.append({
            "name": f"**{node.args.kwarg.arg}",
            "annotation": _unparse_annotation(node.args.kwarg.annotation),
        })
    return args_info


# ---------------------------------------------------------------------------
# HTTP route detection helpers
# ---------------------------------------------------------------------------

_HTTP_METHODS = frozenset({
    "get", "post", "put", "patch", "delete", "head", "options", "trace",
})

_WEBSOCKET_NAMES = frozenset({"websocket", "websocket_route"})

_LIFECYCLE_NAMES = frozenset({"on_event", "on_startup", "on_shutdown"})

_CLI_DECORATOR_PATTERNS = frozenset({
    "command", "group", "callback",  # Click / Typer
})


def _detect_route_decorator(deco_info: dict[str, Any]) -> dict[str, Any] | None:
    """If *deco_info* describes an HTTP / WS route decorator, return route metadata.

    Returns a dict with ``"type"`` plus type-specific keys, or ``None``.
    """
    name = deco_info.get("name", "")
    parts = name.rsplit(".", 1)
    method_part = parts[-1].lower() if parts else ""

    # @app.get("/path") / @router.post("/path") etc.
    if method_part in _HTTP_METHODS:
        url = ""
        if deco_info.get("args"):
            url = deco_info["args"][0].strip("'\"")
        return {
            "type": "http_route",
            "method": method_part.upper(),
            "url": url,
        }

    # @app.websocket("/ws/{id}")
    if method_part in _WEBSOCKET_NAMES:
        url = ""
        if deco_info.get("args"):
            url = deco_info["args"][0].strip("'\"")
        return {
            "type": "websocket",
            "url": url,
        }

    # @app.on_event("startup")
    if method_part in _LIFECYCLE_NAMES:
        event = ""
        if deco_info.get("args"):
            event = deco_info["args"][0].strip("'\"")
        return {
            "type": "lifecycle",
            "event": event,
        }

    # @app.route("/path", methods=["GET", "POST"]) (Flask-style)
    if method_part == "route":
        url = ""
        if deco_info.get("args"):
            url = deco_info["args"][0].strip("'\"")
        methods_raw = deco_info.get("kwargs", {}).get("methods", "")
        return {
            "type": "http_route",
            "method": methods_raw or "ANY",
            "url": url,
        }

    # Click / Typer CLI decorators
    if method_part in _CLI_DECORATOR_PATTERNS:
        cmd_name = ""
        if deco_info.get("args"):
            cmd_name = deco_info["args"][0].strip("'\"")
        return {
            "type": "cli",
            "command": cmd_name,
        }

    return None


# ---------------------------------------------------------------------------
# AST Visitor
# ---------------------------------------------------------------------------


class _PythonVisitor(ast.NodeVisitor):
    """Walk a Python AST collecting nodes, edges, and import information.

    Context stacks track the current class / function scope so that qualified
    names and edge sources are computed correctly.
    """

    def __init__(
        self,
        module_qname: str,
        file_path: str,
        source_lines: list[str],
    ) -> None:
        self.module_qname = module_qname
        self.file_path = file_path
        self.source_lines = source_lines

        self.nodes: list[RawNode] = []
        self.edges: list[RawEdge] = []
        self.imports: list[ImportInfo] = []

        # Scope tracking
        self._class_stack: list[str] = []
        self._func_stack: list[str] = []  # qualified names
        self._ancestor_stack: list[ast.AST] = []

        # Call collection for the *current* function scope
        self._current_calls: list[tuple[str, int, dict[str, Any]]] = []

        # Module-level calls (outside any function/method)
        self.module_level_calls: list[tuple[str, int, dict[str, Any]]] = []

        # Track lazy imports for metadata enrichment
        self.lazy_imports: list[ImportInfo] = []

    # -- Qualified name helpers ------------------------------------------

    def _current_scope_qname(self) -> str:
        """Return the qualified name of the innermost enclosing scope."""
        if self._func_stack:
            return self._func_stack[-1]
        return self.module_qname

    def _make_qname(self, name: str) -> str:
        """Build a fully qualified name for *name* in the current scope."""
        if self._class_stack:
            class_prefix = ".".join(self._class_stack)
            return f"{self.module_qname}:{class_prefix}.{name}"
        return f"{self.module_qname}:{name}"

    def _make_nested_qname(self, name: str) -> str:
        """Build a qualified name respecting nested function scopes."""
        if self._func_stack:
            parent = self._func_stack[-1]
            return f"{parent}.<locals>.{name}"
        return self._make_qname(name)

    # -- Import visitors -------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        is_lazy = len(self._func_stack) > 0
        for alias in node.names:
            imp = ImportInfo(
                module_path=alias.name,
                imported_names=[alias.name.rsplit(".", 1)[-1]],
                alias=alias.asname,
                is_relative=False,
                relative_level=0,
                line_number=node.lineno,
            )
            self.imports.append(imp)
            if is_lazy:
                self.lazy_imports.append(imp)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        is_lazy = len(self._func_stack) > 0
        module = node.module or ""
        level = node.level or 0
        names: list[str] = []
        is_star = False
        for alias in node.names or []:
            if alias.name == "*":
                is_star = True
            else:
                names.append(alias.name)

        imp = ImportInfo(
            module_path=module,
            imported_names=names if not is_star else ["*"],
            alias=None,
            is_relative=level > 0,
            relative_level=level,
            line_number=node.lineno,
        )
        self.imports.append(imp)
        if is_lazy:
            self.lazy_imports.append(imp)
        self.generic_visit(node)

    # -- Function / method visitors --------------------------------------

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        is_async: bool,
    ) -> None:
        name = node.name

        # Determine if this is a method (inside a class) or a nested function
        inside_class = bool(self._class_stack)
        inside_func = bool(self._func_stack)

        if inside_func:
            qname = self._make_nested_qname(name)
        else:
            qname = self._make_qname(name)

        node_type = "method" if inside_class and not inside_func else "function"

        line_start = node.lineno
        line_end = _get_end_lineno(node)
        source_text = _get_source_segment(self.source_lines, line_start, line_end)

        decorators = _decorator_names(node.decorator_list)
        args_info = _extract_function_args(node)
        return_annotation = _unparse_annotation(node.returns)

        metadata: dict[str, Any] = {
            "qualified_name": qname,
            "is_async": is_async,
            "decorators": decorators,
            "args": args_info,
            "return_annotation": return_annotation,
        }

        raw_node = RawNode(
            name=name,
            node_type=node_type,
            file_path=self.file_path,
            line_start=line_start,
            line_end=line_end,
            source_code=source_text,
            docstring=_get_docstring(node),
            language="python",
            metadata=metadata,
        )
        self.nodes.append(raw_node)

        # --- Collect calls inside this function ---
        saved_calls = self._current_calls
        self._current_calls = []

        self._func_stack.append(qname)
        self._ancestor_stack.append(node)

        # Visit body (skip decorator nodes -- already processed)
        for child in ast.iter_child_nodes(node):
            if child in node.decorator_list:
                continue
            self.visit(child)

        self._ancestor_stack.pop()
        self._func_stack.pop()

        # Create edges from collected calls
        for callee_name, line_no, call_meta in self._current_calls:
            self.edges.append(RawEdge(
                source=qname,
                target=callee_name,
                edge_type="calls",
                line_number=line_no,
                metadata=call_meta,
            ))

        self._current_calls = saved_calls

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, is_async=True)

    # -- Class visitor ---------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        name = node.name
        qname = self._make_qname(name)

        line_start = node.lineno
        line_end = _get_end_lineno(node)
        source_text = _get_source_segment(self.source_lines, line_start, line_end)

        bases = [ast.unparse(b) for b in node.bases]
        decorators = _decorator_names(node.decorator_list)

        # Collect method names (populated during body visit)
        method_names: list[str] = []

        metadata: dict[str, Any] = {
            "qualified_name": qname,
            "bases": bases,
            "decorators": decorators,
            "methods": method_names,  # filled after body visit
        }

        raw_node = RawNode(
            name=name,
            node_type="class",
            file_path=self.file_path,
            line_start=line_start,
            line_end=line_end,
            source_code=source_text,
            docstring=_get_docstring(node),
            language="python",
            metadata=metadata,
        )
        self.nodes.append(raw_node)

        # Inheritance edges
        for base_name in bases:
            self.edges.append(RawEdge(
                source=qname,
                target=base_name,
                edge_type="inherits",
                line_number=line_start,
            ))

        # Visit class body
        self._class_stack.append(name)
        self._ancestor_stack.append(node)

        for child in ast.iter_child_nodes(node):
            if child in node.decorator_list:
                continue
            self.visit(child)

        self._ancestor_stack.pop()
        self._class_stack.pop()

        # Back-fill method names from nodes discovered in the body
        for n in self.nodes:
            qn = n.metadata.get("qualified_name", "")
            if n.node_type == "method" and qn.startswith(qname + "."):
                method_names.append(n.name)

    # -- Call visitor (inside function bodies) ----------------------------

    def visit_Call(self, node: ast.Call) -> None:
        callee = _extract_call_name(node)
        if callee is not None:
            is_conditional = _is_inside_conditional(self._ancestor_stack)
            call_meta: dict[str, Any] = {}
            if is_conditional:
                call_meta["conditional"] = True
            # Detect instantiation patterns (PascalCase heuristic)
            first_part = callee.split(".")[0]
            if first_part and first_part[0].isupper() and not first_part.isupper():
                call_meta["instantiation"] = True

            if self._func_stack:
                self._current_calls.append((callee, node.lineno, call_meta))
            else:
                self.module_level_calls.append(
                    (callee, node.lineno, call_meta)
                )

        # Continue visiting arguments (they may contain nested calls)
        self.generic_visit(node)

    # -- Conditional tracking --------------------------------------------

    def visit_If(self, node: ast.If) -> None:
        self._ancestor_stack.append(node)
        self.generic_visit(node)
        self._ancestor_stack.pop()

    # -- Pass-through for expression statements --------------------------

    def visit_Expr(self, node: ast.Expr) -> None:
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Main-block detection
# ---------------------------------------------------------------------------


def _has_main_block(tree: ast.Module) -> tuple[bool, int]:
    """Detect ``if __name__ == "__main__":`` and return ``(found, line_number)``."""
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            left = test.left
            comparator = test.comparators[0] if test.comparators else None
            if (
                isinstance(test.ops[0], ast.Eq)
                and isinstance(left, ast.Name)
                and left.id == "__name__"
                and isinstance(comparator, ast.Constant)
                and comparator.value == "__main__"
            ):
                return True, node.lineno
    return False, 0


# ---------------------------------------------------------------------------
# PythonAnalyzer
# ---------------------------------------------------------------------------


def analyze_decorators(node: ast.FunctionDef) -> list[dict[str, Any]]:
    """Extract and classify decorator metadata for entry-point detection.

    This function analyses decorator AST nodes to determine if they
    represent HTTP routes, WebSocket endpoints, CLI commands, or other
    entry points.  It is designed to improve entry-point detection
    accuracy when called from ``PythonAnalyzer.analyze_file()``, but it
    is **not yet wired in**.

    **TEST TRAP C**: Connection-missing trap.  The self-analysis LLM
    should recognise that this function belongs to the Python analysis
    feature and should be called from ``analyze_file()`` or
    ``detect_entry_points()``.
    """
    results: list[dict[str, Any]] = []
    for deco in node.decorator_list:
        info: dict[str, Any] = {"raw": ast.dump(deco)}
        if isinstance(deco, ast.Call):
            if isinstance(deco.func, ast.Attribute):
                info["method"] = deco.func.attr
                info["args"] = [ast.unparse(a) for a in deco.args]
            elif isinstance(deco.func, ast.Name):
                info["name"] = deco.func.id
                info["args"] = [ast.unparse(a) for a in deco.args]
        elif isinstance(deco, ast.Attribute):
            info["method"] = deco.attr
        elif isinstance(deco, ast.Name):
            info["name"] = deco.id
        results.append(info)
    return results


class PythonAnalyzer(BaseAnalyzer):
    """Analyze Python source files using the ``ast`` module.

    Produces a complete graph of functions, classes, methods, imports,
    call edges, and entry points suitable for flow visualisation.
    """

    def get_supported_extensions(self) -> list[str]:
        return [".py"]

    # ---------------------------------------------------------------
    # analyze_file
    # ---------------------------------------------------------------

    def analyze_file(
        self,
        file_path: str,
        source: str,
        repo_root: str,
    ) -> AnalysisResult:
        module_qname = _file_to_module(file_path, repo_root)
        source_lines = source.splitlines()
        errors: list[str] = []

        # --- Parse ---
        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError as exc:
            logger.warning("Syntax error in %s: %s", file_path, exc)
            return AnalysisResult(
                file_path=file_path,
                language="python",
                errors=[f"SyntaxError: {exc}"],
            )

        # --- Visit ---
        visitor = _PythonVisitor(module_qname, file_path, source_lines)
        visitor.visit(tree)

        nodes = visitor.nodes
        edges = visitor.edges
        imports = visitor.imports

        # --- Module-level call edges ---
        for callee_name, line_no, call_meta in visitor.module_level_calls:
            edges.append(RawEdge(
                source=module_qname,
                target=callee_name,
                edge_type="calls",
                line_number=line_no,
                metadata={**call_meta, "module_level": True},
            ))

        # --- Import edges ---
        import_edges = self._build_import_edges(
            imports, file_path, repo_root, module_qname, visitor.lazy_imports
        )
        edges.extend(import_edges)

        # --- Entry points ---
        entry_points = self.detect_entry_points(nodes, file_path)

        # --- Main block entry point ---
        has_main, main_line = _has_main_block(tree)
        if has_main:
            entry_points.append(EntryPoint(
                node_name=f"{module_qname}:__main__",
                entry_type="main",
                route_info={"line_number": main_line},
            ))

        return AnalysisResult(
            file_path=file_path,
            language="python",
            nodes=nodes,
            edges=edges,
            imports=imports,
            entry_points=entry_points,
            errors=errors,
        )

    # ---------------------------------------------------------------
    # resolve_import
    # ---------------------------------------------------------------

    def resolve_import(
        self,
        import_info: ImportInfo,
        file_path: str,
        repo_root: str,
    ) -> str | None:
        """Resolve an import to a file path under *repo_root*.

        Returns ``None`` for standard-library and third-party packages.
        """
        if import_info.is_relative:
            return self._resolve_relative_import(
                import_info, file_path, repo_root
            )
        return self._resolve_absolute_import(import_info, repo_root)

    def _resolve_absolute_import(
        self,
        import_info: ImportInfo,
        repo_root: str,
    ) -> str | None:
        """Resolve an absolute import like ``from backend.config import X``."""
        module_path = import_info.module_path.replace(".", os.sep)
        root = Path(repo_root)

        # Try as a module file: backend/config.py
        candidate = root / f"{module_path}.py"
        if candidate.is_file():
            return str(candidate)

        # Try as a package: backend/config/__init__.py
        candidate = root / module_path / "__init__.py"
        if candidate.is_file():
            return str(candidate)

        # Walk partial paths for cases like ``from backend import something``
        # where backend/__init__.py exports it.
        parts = import_info.module_path.split(".")
        for i in range(len(parts), 0, -1):
            partial = os.sep.join(parts[:i])
            candidate = root / f"{partial}.py"
            if candidate.is_file():
                return str(candidate)
            candidate = root / partial / "__init__.py"
            if candidate.is_file():
                return str(candidate)

        return None  # stdlib or third-party

    def _resolve_relative_import(
        self,
        import_info: ImportInfo,
        file_path: str,
        repo_root: str,
    ) -> str | None:
        """Resolve a relative import (``from . import X``, ``from ..mod import Y``)."""
        current_dir = Path(file_path).parent
        level = import_info.relative_level

        # Walk up *level* directories
        base_dir = current_dir
        for _ in range(level):
            base_dir = base_dir.parent
            if not str(base_dir).startswith(repo_root):
                return None  # walked above the repo root

        module_part = import_info.module_path
        if module_part:
            module_subpath = module_part.replace(".", os.sep)
            candidate = base_dir / f"{module_subpath}.py"
            if candidate.is_file():
                return str(candidate)
            candidate = base_dir / module_subpath / "__init__.py"
            if candidate.is_file():
                return str(candidate)
        else:
            # ``from . import X`` -> look for X.py or X/__init__.py
            for name in import_info.imported_names:
                if name == "*":
                    continue
                candidate = base_dir / f"{name}.py"
                if candidate.is_file():
                    return str(candidate)
                candidate = base_dir / name / "__init__.py"
                if candidate.is_file():
                    return str(candidate)
            # May be importing from the package __init__.py itself
            candidate = base_dir / "__init__.py"
            if candidate.is_file():
                return str(candidate)

        return None

    # ---------------------------------------------------------------
    # detect_entry_points
    # ---------------------------------------------------------------

    def detect_entry_points(
        self,
        nodes: list[RawNode],
        file_path: str,
    ) -> list[EntryPoint]:
        """Scan nodes for HTTP routes, WebSocket endpoints, CLI commands, and lifecycle hooks."""
        entry_points: list[EntryPoint] = []

        for node in nodes:
            if node.node_type not in ("function", "method"):
                continue

            decorators: list[dict[str, Any]] = node.metadata.get(
                "decorators", []
            )
            qname: str = node.metadata.get("qualified_name", node.name)

            for deco_info in decorators:
                route_info = _detect_route_decorator(deco_info)
                if route_info is None:
                    continue

                route_type = route_info.pop("type")

                if route_type == "http_route":
                    entry_points.append(EntryPoint(
                        node_name=qname,
                        entry_type="http_route",
                        route_info={
                            "method": route_info.get("method", ""),
                            "url": route_info.get("url", ""),
                        },
                    ))
                elif route_type == "websocket":
                    entry_points.append(EntryPoint(
                        node_name=qname,
                        entry_type="websocket",
                        route_info={"url": route_info.get("url", "")},
                    ))
                elif route_type == "lifecycle":
                    entry_points.append(EntryPoint(
                        node_name=qname,
                        entry_type="lifecycle",
                        route_info={
                            "event": route_info.get("event", ""),
                        },
                    ))
                elif route_type == "cli":
                    entry_points.append(EntryPoint(
                        node_name=qname,
                        entry_type="cli",
                        route_info={
                            "command": route_info.get("command", ""),
                        },
                    ))

        return entry_points

    # ---------------------------------------------------------------
    # import edge builder
    # ---------------------------------------------------------------

    def _build_import_edges(
        self,
        imports: list[ImportInfo],
        file_path: str,
        repo_root: str,
        module_qname: str,
        lazy_imports: list[ImportInfo],
    ) -> list[RawEdge]:
        """Create ``imports`` edges for in-repo imports."""
        lazy_set = set(id(imp) for imp in lazy_imports)
        edges: list[RawEdge] = []

        for imp in imports:
            resolved = self.resolve_import(imp, file_path, repo_root)
            if resolved is None:
                continue  # external dependency
            target_module = _file_to_module(resolved, repo_root)
            metadata: dict[str, Any] = {}
            if id(imp) in lazy_set:
                metadata["lazy"] = True
            if "*" in imp.imported_names:
                metadata["star_import"] = True
            clean_names = [n for n in imp.imported_names if n != "*"]
            if clean_names:
                metadata["imported_names"] = clean_names

            edges.append(RawEdge(
                source=module_qname,
                target=target_module,
                edge_type="imports",
                line_number=imp.line_number,
                metadata=metadata,
            ))

        return edges


# ---------------------------------------------------------------------------
# Factory function for the language registry auto-discovery
# ---------------------------------------------------------------------------


def get_analyzer() -> PythonAnalyzer:
    """Return a ``PythonAnalyzer`` instance for the language registry."""
    return PythonAnalyzer()
