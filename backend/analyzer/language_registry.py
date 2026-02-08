"""Language registry for auto-detection and analyzer management.

Maps file extensions to language identifiers and routes analysis
requests to the appropriate ``BaseAnalyzer`` implementation.
"""

from __future__ import annotations

import logging
import os
from pathlib import PurePosixPath

from backend.analyzer.base_analyzer import AnalysisResult, BaseAnalyzer

logger = logging.getLogger(__name__)

# Canonical extension -> language mapping.
# Analyzers may register additional extensions via ``register()``.
_DEFAULT_EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".swift": "swift",
    ".scala": "scala",
    ".ex": "elixir",
    ".exs": "elixir",
}


class LanguageRegistry:
    """Central registry that maps languages to their analyzers.

    Typical usage::

        registry = LanguageRegistry()
        registry.register(PythonAnalyzer())

        result = registry.analyze_file("app.py", source_code, "/repo")
    """

    def __init__(self) -> None:
        self._analyzers: dict[str, BaseAnalyzer] = {}
        self._extension_map: dict[str, str] = dict(_DEFAULT_EXTENSION_MAP)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, analyzer: BaseAnalyzer) -> None:
        """Register a language analyzer and map its extensions.

        If the analyzer supports extensions not yet in the default map,
        the first extension is used to derive the language name, and all
        extensions are added.
        """
        extensions = analyzer.get_supported_extensions()
        if not extensions:
            logger.warning(
                "Analyzer %s reports no supported extensions; skipping.",
                type(analyzer).__name__,
            )
            return

        # Determine the canonical language name from the first extension.
        language = self._extension_map.get(extensions[0])
        if language is None:
            # Derive from extension (e.g. ".lua" -> "lua").
            language = extensions[0].lstrip(".").lower()

        self._analyzers[language] = analyzer

        # Ensure every extension the analyzer supports is in the map.
        for ext in extensions:
            if ext not in self._extension_map:
                self._extension_map[ext] = language

        logger.info(
            "Registered %s analyzer for language=%s extensions=%s",
            type(analyzer).__name__,
            language,
            extensions,
        )

    # ------------------------------------------------------------------
    # Detection & lookup
    # ------------------------------------------------------------------

    def detect_language(self, file_path: str) -> str | None:
        """Detect the language of a file from its extension.

        Returns the language identifier (e.g. ``"python"``) or ``None``
        if the extension is not recognized.
        """
        ext = PurePosixPath(file_path).suffix.lower()
        return self._extension_map.get(ext)

    def get_analyzer(self, language: str) -> BaseAnalyzer | None:
        """Return the registered analyzer for *language*, or ``None``."""
        return self._analyzers.get(language)

    def get_supported_languages(self) -> list[str]:
        """Return a sorted list of languages with registered analyzers."""
        return sorted(self._analyzers.keys())

    def get_supported_extensions(self) -> list[str]:
        """Return all file extensions that map to a registered analyzer."""
        return sorted(
            ext
            for ext, lang in self._extension_map.items()
            if lang in self._analyzers
        )

    # ------------------------------------------------------------------
    # Convenience: detect + analyze in one call
    # ------------------------------------------------------------------

    def analyze_file(
        self, file_path: str, source: str, repo_root: str
    ) -> AnalysisResult | None:
        """Auto-detect language and analyze a source file.

        Returns ``None`` if the language is unrecognized or no analyzer
        is registered for it.
        """
        language = self.detect_language(file_path)
        if language is None:
            return None

        analyzer = self.get_analyzer(language)
        if analyzer is None:
            return None

        return analyzer.analyze_file(file_path, source, repo_root)


# ------------------------------------------------------------------
# Default singleton
# ------------------------------------------------------------------

_default_registry: LanguageRegistry | None = None


def get_default_registry() -> LanguageRegistry:
    """Return the application-wide default ``LanguageRegistry``.

    On first call, creates the registry and attempts to auto-register
    any analyzers found in ``backend.analyzer.languages``.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = LanguageRegistry()
        _auto_register(_default_registry)
    return _default_registry


def _auto_register(registry: LanguageRegistry) -> None:
    """Discover and register analyzer implementations.

    Looks for concrete ``BaseAnalyzer`` subclasses inside the
    ``backend.analyzer.languages`` package.  Each module in that
    package is expected to expose a top-level ``Analyzer`` class or
    a function ``get_analyzer() -> BaseAnalyzer``.
    """
    languages_dir = os.path.join(
        os.path.dirname(__file__), "languages"
    )
    if not os.path.isdir(languages_dir):
        return

    for filename in sorted(os.listdir(languages_dir)):
        if not filename.endswith(".py") or filename.startswith("_"):
            continue

        module_name = f"backend.analyzer.languages.{filename[:-3]}"
        try:
            import importlib

            module = importlib.import_module(module_name)
        except Exception:
            logger.debug("Could not import analyzer module %s", module_name, exc_info=True)
            continue

        # Prefer a factory function, fall back to an ``Analyzer`` class.
        analyzer: BaseAnalyzer | None = None
        if hasattr(module, "get_analyzer"):
            analyzer = module.get_analyzer()
        elif hasattr(module, "Analyzer"):
            analyzer = module.Analyzer()

        if analyzer is not None and isinstance(analyzer, BaseAnalyzer):
            registry.register(analyzer)
        else:
            logger.debug(
                "Module %s does not expose a BaseAnalyzer; skipping.", module_name
            )
