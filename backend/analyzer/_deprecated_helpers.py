"""Deprecated analysis helper functions.

This module contains legacy analysis utilities that were replaced by the
new plugin-based analyzer architecture.  No other module imports from here,
so the entire file should be detected as **dead code** during self-analysis.

**TEST TRAP A**: The self-analysis pipeline should flag every function in
this file as dead code with high confidence (>= 0.90).
"""

from __future__ import annotations

import re
from typing import Any


def legacy_parse_imports(source: str) -> list[dict[str, Any]]:
    """Parse Python import statements using naive regex.

    Replaced by ``python_analyzer.py`` which uses the ``ast`` module for
    accurate import extraction.  This function is deliberately left unused
    to serve as a dead-code detection test.
    """
    pattern = r"^(?:from\s+([\w.]+)\s+)?import\s+(.+)$"
    results: list[dict[str, Any]] = []
    for line in source.splitlines():
        match = re.match(pattern, line.strip())
        if match:
            module = match.group(1) or ""
            names = [n.strip() for n in match.group(2).split(",")]
            results.append({"module": module, "names": names})
    return results


def old_feature_scorer(nodes: list[dict[str, Any]]) -> dict[str, float]:
    """Score functions by heuristic importance for feature grouping.

    This was the prototype scoring algorithm before the BFS-based feature
    grouper was implemented.  It is no longer referenced anywhere.
    """
    scores: dict[str, float] = {}
    for node in nodes:
        name = node.get("name", "")
        score = 0.0
        # Heuristic: longer functions are "more important"
        line_count = node.get("line_end", 0) - node.get("line_start", 0)
        score += min(line_count / 50.0, 1.0)
        # Heuristic: functions with docstrings are more intentional
        if node.get("docstring"):
            score += 0.3
        # Heuristic: entry points score higher
        if node.get("is_entry_point"):
            score += 0.5
        scores[name] = round(score, 3)
    return scores


def compute_cyclomatic_complexity(source: str) -> int:
    """Estimate cyclomatic complexity via keyword counting.

    A rough approximation that counts branching keywords.  Never integrated
    into the main pipeline -- purely a leftover experiment.
    """
    keywords = ["if", "elif", "else", "for", "while", "except", "with", "and", "or"]
    complexity = 1
    for line in source.splitlines():
        stripped = line.strip()
        for kw in keywords:
            if stripped.startswith(kw + " ") or stripped.startswith(kw + ":"):
                complexity += 1
    return complexity
