"""File discovery utility for repository scanning.

Walks a repository directory tree, identifies source files by language,
and provides content reading with robust encoding handling.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.analyzer.language_registry import LanguageRegistry

logger = logging.getLogger(__name__)

# Directories to skip unconditionally during discovery.
_SKIP_DIRS: frozenset[str] = frozenset(
    {
        "node_modules",
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        ".env",
        "dist",
        "build",
        ".next",
        ".nuxt",
        ".svelte-kit",
        ".output",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "egg-info",
        ".eggs",
        "target",  # Rust / Java build output
        "vendor",  # Go vendor
        "coverage",
        ".coverage",
        ".gradle",
        ".idea",
        ".vscode",
    }
)

# Maximum file size to consider (5 MB).  Anything larger is likely
# generated, minified, or a data file -- not useful for analysis.
_MAX_FILE_SIZE_BYTES: int = 5 * 1024 * 1024

# Encodings to attempt when reading source files, in order.
_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "latin-1")


@dataclass(slots=True)
class DiscoveredFile:
    """Metadata about a discovered source file."""

    path: str  # absolute path
    language: str
    size: int  # bytes


def discover_files(
    repo_path: str,
    registry: LanguageRegistry,
) -> list[DiscoveredFile]:
    """Walk *repo_path* and return metadata for every recognised source file.

    Directories listed in ``_SKIP_DIRS`` are pruned during the walk.
    Files larger than ``_MAX_FILE_SIZE_BYTES`` are silently skipped.

    Returns a list of ``DiscoveredFile`` sorted by path for deterministic
    ordering.
    """
    repo_path = os.path.abspath(repo_path)
    if not os.path.isdir(repo_path):
        logger.error("Repository path does not exist: %s", repo_path)
        return []

    results: list[DiscoveredFile] = []

    for dirpath, dirnames, filenames in os.walk(repo_path, topdown=True):
        # Prune skipped directories in-place so os.walk does not descend.
        dirnames[:] = [
            d
            for d in dirnames
            if d not in _SKIP_DIRS and not d.startswith(".")
        ]

        for filename in filenames:
            file_path = os.path.join(dirpath, filename)

            # Skip oversized files.
            try:
                size = os.path.getsize(file_path)
            except OSError:
                continue
            if size > _MAX_FILE_SIZE_BYTES or size == 0:
                continue

            language = registry.detect_language(filename)
            if language is None:
                continue

            results.append(
                DiscoveredFile(
                    path=file_path,
                    language=language,
                    size=size,
                )
            )

    results.sort(key=lambda f: f.path)
    logger.info(
        "Discovered %d source files in %s", len(results), repo_path
    )
    return results


def get_file_content(file_path: str) -> str:
    """Read and return the text content of *file_path*.

    Tries multiple encodings to handle files written on different
    platforms.  Returns an empty string if the file cannot be read
    or decoded.
    """
    for encoding in _ENCODINGS:
        try:
            with open(file_path, "r", encoding=encoding) as fh:
                return fh.read()
        except UnicodeDecodeError:
            continue
        except OSError:
            logger.warning("Could not read file: %s", file_path)
            return ""

    logger.warning(
        "Could not decode file with any supported encoding: %s", file_path
    )
    return ""
