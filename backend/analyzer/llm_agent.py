"""LLM Agent for code analysis - discovers features and maps code flows.

This module replaces the AST-first analysis pipeline with an LLM-first approach.
The agent reads code, identifies features, and maps execution flows, using AST
only for verification (line numbers, source code extraction).
"""

from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any

from backend.llm.client import LLMClient
from backend.llm.prompts import (
    discover_features_prompt,
    discover_features_system_prompt,
    map_feature_flow_prompt,
    map_feature_flow_system_prompt,
)
from backend.utils.file_discovery import get_file_content

logger = logging.getLogger(__name__)

# Max lines to include per file in the codebase summary (for token management)
_MAX_SUMMARY_LINES = 50
# Max total characters for codebase summary sent to LLM
_MAX_SUMMARY_CHARS = 80_000
# Max characters per file for flow mapping
_MAX_FILE_CHARS = 15_000


@dataclass
class FlowStep:
    """A single step in a feature's execution flow."""
    order: int
    file: str
    function: str
    description: str
    calls_next: list[str] = field(default_factory=list)
    # Populated by AST verification
    line_start: int | None = None
    line_end: int | None = None
    source_code: str | None = None
    node_id: str | None = None  # Set during DB persistence


@dataclass
class DiscoveredFeature:
    """A feature discovered by the LLM agent."""
    name: str
    description: str
    files: list[str]
    entry_points: list[str]
    category: str = "core"
    # Populated by flow mapping
    flow_summary: str | None = None
    flow_steps: list[FlowStep] = field(default_factory=list)


@dataclass
class AgentAnalysisResult:
    """Complete result from the LLM agent analysis."""
    features: list[DiscoveredFeature]
    file_tree: list[str]


class CodeAnalysisAgent:
    """LLM Agent that reads code, discovers features, and maps execution flows."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def analyze(
        self,
        repo_path: str,
        progress_callback: Any = None,
    ) -> AgentAnalysisResult:
        """Run the complete LLM agent analysis pipeline.

        Steps:
        1. Build codebase summary (AST + file reading)
        2. LLM discovers features from summary
        3. LLM maps execution flow for each feature
        4. AST verifies function existence and extracts line numbers
        """
        repo_path = os.path.abspath(repo_path)

        # Step 1: Build codebase summary
        if progress_callback:
            progress_callback("Building codebase summary", 5.0)
        summary, file_tree, file_sigs = self._build_codebase_summary(repo_path)

        # Step 2: LLM discovers features
        if progress_callback:
            progress_callback("LLM discovering features", 15.0)
        features = await self._discover_features(summary)
        logger.info("LLM discovered %d features", len(features))

        # Step 3: Map execution flow for each feature
        total_features = len(features)
        for i, feature in enumerate(features):
            pct = 30.0 + (i / max(total_features, 1)) * 50.0
            if progress_callback:
                progress_callback(f"Mapping flow: {feature.name}", pct)
            await self._map_feature_flow(feature, repo_path)

        # Step 4: AST verification
        if progress_callback:
            progress_callback("Verifying with AST", 85.0)
        self._verify_with_ast(features, repo_path, file_sigs)

        if progress_callback:
            progress_callback("Analysis complete", 100.0)

        return AgentAnalysisResult(features=features, file_tree=file_tree)

    # ------------------------------------------------------------------
    # Step 1: Build codebase summary
    # ------------------------------------------------------------------

    def _build_codebase_summary(
        self, repo_path: str,
    ) -> tuple[str, list[str], dict[str, dict[str, tuple[int, int]]]]:
        """Build a compact summary of the codebase for LLM consumption.

        Returns:
            (summary_text, file_tree, file_signatures)
            file_signatures maps file_path -> {func_name: (line_start, line_end)}
        """
        file_tree: list[str] = []
        file_summaries: list[str] = []
        file_sigs: dict[str, dict[str, tuple[int, int]]] = {}
        total_chars = 0

        # Walk the repo and collect file info
        skip_dirs = {
            "node_modules", ".git", "__pycache__", ".venv", "venv",
            ".env", "dist", "build", ".next", ".tox", ".mypy_cache",
            ".pytest_cache", ".ruff_cache", "egg-info", ".eggs",
            "coverage", ".idea", ".vscode", ".output",
        }

        for dirpath, dirnames, filenames in os.walk(repo_path, topdown=True):
            dirnames[:] = [
                d for d in dirnames
                if d not in skip_dirs and not d.startswith(".")
            ]

            for filename in filenames:
                if not self._is_source_file(filename):
                    continue

                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, repo_path)
                file_tree.append(rel_path)

                # Read file content
                content = get_file_content(full_path)
                if not content:
                    continue

                # Extract signatures using AST (for Python files)
                sigs: dict[str, tuple[int, int]] = {}
                if filename.endswith(".py"):
                    sigs = self._extract_python_signatures(content)
                    file_sigs[rel_path] = sigs

                # Build compact summary for this file
                file_summary = self._summarize_file(rel_path, content, sigs)
                if total_chars + len(file_summary) > _MAX_SUMMARY_CHARS:
                    # Only include signature info for remaining files
                    file_summary = self._minimal_file_summary(rel_path, sigs)

                file_summaries.append(file_summary)
                total_chars += len(file_summary)

        # Build the full summary
        tree_text = "\n".join(f"  {f}" for f in sorted(file_tree))
        summary = (
            f"File tree ({len(file_tree)} source files):\n{tree_text}\n\n"
            f"File details:\n{''.join(file_summaries)}"
        )

        return summary, file_tree, file_sigs

    @staticmethod
    def _is_source_file(filename: str) -> bool:
        """Check if a file is a source code file worth analyzing."""
        source_exts = {
            ".py", ".js", ".ts", ".tsx", ".jsx",
            ".java", ".go", ".rs", ".rb", ".php",
            ".c", ".cpp", ".h", ".hpp", ".cs",
        }
        _, ext = os.path.splitext(filename)
        return ext.lower() in source_exts

    @staticmethod
    def _extract_python_signatures(content: str) -> dict[str, tuple[int, int]]:
        """Extract function/class signatures and their line ranges from Python code."""
        sigs: dict[str, tuple[int, int]] = {}
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return sigs

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end_line = node.end_lineno or node.lineno
                sigs[node.name] = (node.lineno, end_line)
            elif isinstance(node, ast.ClassDef):
                end_line = node.end_lineno or node.lineno
                sigs[node.name] = (node.lineno, end_line)
                # Also extract methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_end = item.end_lineno or item.lineno
                        sigs[f"{node.name}.{item.name}"] = (item.lineno, method_end)

        return sigs

    @staticmethod
    def _summarize_file(
        rel_path: str,
        content: str,
        sigs: dict[str, tuple[int, int]],
    ) -> str:
        """Create a compact summary of a file for LLM consumption."""
        lines = content.splitlines()

        # Include imports and first N lines
        head = "\n".join(lines[:min(_MAX_SUMMARY_LINES, len(lines))])

        # List function/class signatures
        sig_list = ""
        if sigs:
            sig_items = [f"    {name} (L{start}-{end})" for name, (start, end) in sorted(sigs.items(), key=lambda x: x[1][0])]
            sig_list = "\n".join(sig_items)

        return (
            f"\n=== {rel_path} ({len(lines)} lines) ===\n"
            f"{head}\n"
            f"{'Signatures:' + chr(10) + sig_list if sig_list else ''}\n"
        )

    @staticmethod
    def _minimal_file_summary(
        rel_path: str,
        sigs: dict[str, tuple[int, int]],
    ) -> str:
        """Minimal summary when we're running out of token budget."""
        if not sigs:
            return f"\n=== {rel_path} ===\n(signatures not extracted)\n"
        return f"\n=== {rel_path} ===\nFunctions: {', '.join(sorted(sigs.keys()))}\n"

    # ------------------------------------------------------------------
    # Step 2: LLM discovers features
    # ------------------------------------------------------------------

    async def _discover_features(self, summary: str) -> list[DiscoveredFeature]:
        """Ask the LLM to identify features from the codebase summary."""
        result = await self._llm.complete_json(
            discover_features_prompt(summary),
            system=discover_features_system_prompt(),
            temperature=0.2,
            max_tokens=4000,
        )

        features: list[DiscoveredFeature] = []
        for feat_data in result.get("features", []):
            category = feat_data.get("category", "core")
            # Skip infrastructure-only features
            if category in ("infrastructure", "utility"):
                continue

            features.append(DiscoveredFeature(
                name=feat_data.get("name", "Unknown Feature"),
                description=feat_data.get("description", ""),
                files=feat_data.get("files", []),
                entry_points=feat_data.get("entry_points", []),
                category=category,
            ))

        if not features:
            logger.warning("LLM returned no features, using fallback")
            features.append(DiscoveredFeature(
                name="Main Application",
                description="Primary application logic",
                files=[],
                entry_points=[],
            ))

        return features

    # ------------------------------------------------------------------
    # Step 3: Map execution flow for each feature
    # ------------------------------------------------------------------

    async def _map_feature_flow(
        self,
        feature: DiscoveredFeature,
        repo_path: str,
    ) -> None:
        """Ask the LLM to map the execution flow for a feature."""
        # Collect file contents for this feature
        file_contents: dict[str, str] = {}
        for rel_path in feature.files:
            full_path = os.path.join(repo_path, rel_path)
            content = get_file_content(full_path)
            if content:
                # Truncate very long files
                if len(content) > _MAX_FILE_CHARS:
                    content = content[:_MAX_FILE_CHARS] + "\n... (truncated)"
                file_contents[rel_path] = content

        if not file_contents:
            logger.warning("No file contents found for feature: %s", feature.name)
            return

        result = await self._llm.complete_json(
            map_feature_flow_prompt(
                feature_name=feature.name,
                feature_description=feature.description,
                file_contents=file_contents,
            ),
            system=map_feature_flow_system_prompt(),
            temperature=0.1,
            max_tokens=4000,
        )

        feature.flow_summary = result.get("flow_summary", "")

        for step_data in result.get("flow_steps", []):
            feature.flow_steps.append(FlowStep(
                order=step_data.get("order", 0),
                file=step_data.get("file", ""),
                function=step_data.get("function", ""),
                description=step_data.get("description", ""),
                calls_next=step_data.get("calls_next", []),
            ))

    # ------------------------------------------------------------------
    # Step 4: AST verification
    # ------------------------------------------------------------------

    def _verify_with_ast(
        self,
        features: list[DiscoveredFeature],
        repo_path: str,
        file_sigs: dict[str, dict[str, tuple[int, int]]],
    ) -> None:
        """Verify LLM results against actual code using AST information."""
        for feature in features:
            for step in feature.flow_steps:
                file_path = step.file
                func_name = step.function

                # Try to find the function in our signatures
                sigs = file_sigs.get(file_path, {})
                if func_name in sigs:
                    step.line_start, step.line_end = sigs[func_name]
                    # Extract source code
                    full_path = os.path.join(repo_path, file_path)
                    content = get_file_content(full_path)
                    if content:
                        lines = content.splitlines()
                        step.source_code = "\n".join(
                            lines[step.line_start - 1:step.line_end]
                        )
                else:
                    # Try fuzzy matching
                    all_funcs = list(sigs.keys())
                    matches = get_close_matches(func_name, all_funcs, n=1, cutoff=0.6)
                    if matches:
                        matched = matches[0]
                        logger.info(
                            "Fuzzy matched %s -> %s in %s",
                            func_name, matched, file_path,
                        )
                        step.function = matched
                        step.line_start, step.line_end = sigs[matched]
                        full_path = os.path.join(repo_path, file_path)
                        content = get_file_content(full_path)
                        if content:
                            lines = content.splitlines()
                            step.source_code = "\n".join(
                                lines[step.line_start - 1:step.line_end]
                            )
                    else:
                        logger.warning(
                            "Could not verify function %s in %s",
                            func_name, file_path,
                        )
