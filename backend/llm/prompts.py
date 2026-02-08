"""Prompt templates for LLM-assisted code analysis stages.

Each function accepts structured context parameters and returns a
fully-formatted prompt string ready for ``LLMClient.complete()`` or
``LLMClient.complete_json()``.
"""

from __future__ import annotations


def describe_function_prompt(
    *,
    function_name: str,
    source_code: str,
    language: str,
    file_path: str,
) -> str:
    """Generate a 1-line natural-language description of a function.

    Intended for ``LLMClient.complete()`` -- returns plain text.
    """
    return (
        f"You are a senior software engineer reading {language} source code.\n"
        f"\n"
        f"File: {file_path}\n"
        f"Function: {function_name}\n"
        f"\n"
        f"```{language}\n"
        f"{source_code}\n"
        f"```\n"
        f"\n"
        f"Write exactly ONE concise sentence (max 120 characters) that describes "
        f"what this function does. Focus on its purpose and side effects, not "
        f"implementation details. Do not include the function name in the description."
    )


def name_feature_prompt(
    *,
    entry_points: list[dict[str, str]],
) -> str:
    """Suggest a meaningful feature name for a group of entry points.

    *entry_points* is a list of dicts each having at minimum ``name`` and
    optionally ``route`` (URL path) and ``method`` (HTTP method).

    Intended for ``LLMClient.complete_json()`` -- expects JSON response
    with keys ``name`` (str) and ``description`` (str).
    """
    entries_text = "\n".join(
        f"- {ep.get('method', 'N/A'):>6} {ep.get('route', 'N/A'):<40} -> {ep['name']}"
        for ep in entry_points
    )
    return (
        f"You are naming a feature in a web application based on its entry points.\n"
        f"\n"
        f"Entry points:\n"
        f"{entries_text}\n"
        f"\n"
        f"Respond with a JSON object containing:\n"
        f'  "name": a short, descriptive feature name in snake_case (2-4 words),\n'
        f'  "description": a one-sentence summary of what this feature does.\n'
        f"\n"
        f"Example: "
        f'{{"name": "user_authentication", "description": "Handles user login, logout, and session management."}}'
    )


def analyze_dynamic_calls_prompt(
    *,
    source_code: str,
    language: str,
    file_path: str,
    function_name: str,
    known_symbols: list[str] | None = None,
) -> str:
    """Identify hidden call targets in code with dynamic dispatch.

    Covers patterns like ``getattr()``, reflection, string-based routing,
    plugin loading, and similar indirection.

    Intended for ``LLMClient.complete_json()`` -- expects JSON response
    with key ``calls`` (list of dicts with ``target`` and ``confidence``).
    """
    symbols_section = ""
    if known_symbols:
        symbols_text = "\n".join(f"  - {s}" for s in known_symbols[:50])
        symbols_section = (
            f"\nKnown symbols in the project:\n{symbols_text}\n"
        )

    return (
        f"You are an expert {language} developer analyzing code for dynamic dispatch.\n"
        f"\n"
        f"File: {file_path}\n"
        f"Function: {function_name}\n"
        f"{symbols_section}\n"
        f"```{language}\n"
        f"{source_code}\n"
        f"```\n"
        f"\n"
        f"Identify any function/method calls that are made through dynamic dispatch, "
        f"reflection, string-based lookup, plugin loading, or other indirection that "
        f"a static analyzer would miss.\n"
        f"\n"
        f"Respond with a JSON object:\n"
        f'{{"calls": [\n'
        f'  {{"target": "fully.qualified.name", "confidence": 0.8, "mechanism": "getattr"}}\n'
        f"]}}\n"
        f"\n"
        f"Rules:\n"
        f"- confidence is a float between 0.0 and 1.0\n"
        f"- Only include calls where confidence >= 0.5\n"
        f'- If no dynamic calls are found, return {{"calls": []}}'
    )


def classify_unmatched_code_prompt(
    *,
    source_code: str,
    language: str,
    file_path: str,
    surrounding_functions: list[str] | None = None,
) -> str:
    """Explain why a code block is unmatched and suggest classification.

    Used when code fragments cannot be attributed to any known function
    or feature during the analysis pipeline.

    Intended for ``LLMClient.complete_json()`` -- expects JSON response
    with keys ``reason``, ``classification``, and ``suggested_owner``.
    """
    context_section = ""
    if surrounding_functions:
        funcs_text = ", ".join(surrounding_functions[:10])
        context_section = f"\nNearby functions: {funcs_text}\n"

    return (
        f"You are analyzing {language} code that could not be matched to any "
        f"known function or feature during static analysis.\n"
        f"\n"
        f"File: {file_path}\n"
        f"{context_section}\n"
        f"Unmatched code:\n"
        f"```{language}\n"
        f"{source_code}\n"
        f"```\n"
        f"\n"
        f"Respond with a JSON object:\n"
        f'{{"reason": "why this code is unmatched (1 sentence)",\n'
        f' "classification": one of ["module_init", "global_config", "decorator", '
        f'"type_definition", "constant", "dead_code", "generated", "other"],\n'
        f' "suggested_owner": "fully.qualified.function that likely owns this code, or null"}}'
    )


def verify_feature_grouping_prompt(
    *,
    features: list[dict[str, object]],
) -> str:
    """Verify whether proposed feature groupings make semantic sense.

    *features* is a list of dicts each having ``name`` (str),
    ``entry_points`` (list[str]), and ``functions`` (list[str]).

    Intended for ``LLMClient.complete_json()`` -- expects JSON response
    with key ``features`` (list of dicts with ``name``, ``valid`` bool,
    ``reason``, and optional ``suggested_split``).
    """
    features_text = ""
    for feat in features:
        ep_list = ", ".join(str(e) for e in feat.get("entry_points", [])[:5])
        fn_list = ", ".join(str(f) for f in feat.get("functions", [])[:10])
        features_text += (
            f"\nFeature: {feat['name']}\n"
            f"  Entry points: {ep_list}\n"
            f"  Functions: {fn_list}\n"
        )

    return (
        f"You are reviewing how a codebase has been grouped into features.\n"
        f"\n"
        f"Proposed groupings:\n"
        f"{features_text}\n"
        f"\n"
        f"For each feature, determine if the grouping is semantically coherent. "
        f"A good grouping has functions that clearly support the same user-facing "
        f"capability.\n"
        f"\n"
        f"Respond with a JSON object:\n"
        f'{{"features": [\n'
        f"  {{\n"
        f'    "name": "feature_name",\n'
        f'    "valid": true,\n'
        f'    "reason": "All functions relate to user authentication flow.",\n'
        f'    "suggested_split": null\n'
        f"  }},\n"
        f"  {{\n"
        f'    "name": "mixed_feature",\n'
        f'    "valid": false,\n'
        f'    "reason": "Combines payment processing with email notifications.",\n'
        f'    "suggested_split": ["payment_processing", "email_notifications"]\n'
        f"  }}\n"
        f"]}}"
    )


def discover_features_system_prompt() -> str:
    """System prompt for feature discovery."""
    return (
        "You are a senior software engineer analyzing a codebase. "
        "Your task is to identify the main user-facing features and capabilities "
        "of the application by reading its code structure and content. "
        "Focus on distinct functional areas that a developer would recognize as "
        "separate features. Do NOT include infrastructure, configuration, or "
        "utility code as features - only user-facing or API-facing capabilities."
    )


def discover_features_prompt(codebase_summary: str) -> str:
    """Prompt to discover features from a codebase summary."""
    return (
        f"Analyze this codebase and identify its main features/capabilities.\n"
        f"\n"
        f"Codebase summary:\n"
        f"```\n"
        f"{codebase_summary}\n"
        f"```\n"
        f"\n"
        f"Respond with a JSON object:\n"
        f'{{"features": [\n'
        f"  {{\n"
        f'    "name": "Human-readable feature name (2-5 words)",\n'
        f'    "description": "One sentence describing what this feature does",\n'
        f'    "files": ["path/to/file1.py", "path/to/file2.py"],\n'
        f'    "entry_points": ["file.py:function_name"],\n'
        f'    "category": "core|api|utility|infrastructure"\n'
        f"  }}\n"
        f"]}}\n"
        f"\n"
        f"Rules:\n"
        f"- Identify 3-10 features depending on codebase size\n"
        f"- Each feature should have clear boundaries\n"
        f"- Files can belong to multiple features\n"
        f"- entry_points are the main function(s) that start the feature's execution\n"
        f"- Only include features with category 'core' or 'api', skip pure infrastructure"
    )


def map_feature_flow_system_prompt() -> str:
    """System prompt for feature flow mapping."""
    return (
        "You are a senior software engineer mapping the execution flow of a feature. "
        "Trace how the code executes step by step, from entry point to final result. "
        "Focus on the main happy path. Include function calls, data transformations, "
        "and important side effects."
    )


def map_feature_flow_prompt(
    *,
    feature_name: str,
    feature_description: str,
    file_contents: dict[str, str],
) -> str:
    """Prompt to map the execution flow of a specific feature."""
    files_text = ""
    for path, content in file_contents.items():
        files_text += f"\n--- {path} ---\n{content}\n"

    return (
        f"Map the code execution flow for this feature:\n"
        f"Feature: {feature_name}\n"
        f"Description: {feature_description}\n"
        f"\n"
        f"Source files:\n"
        f"{files_text}\n"
        f"\n"
        f"Respond with a JSON object:\n"
        f'{{"flow_summary": "2-3 sentence overview of the complete flow",\n'
        f' "flow_steps": [\n'
        f"  {{\n"
        f'    "order": 1,\n'
        f'    "file": "relative/path/to/file.py",\n'
        f'    "function": "function_name",\n'
        f'    "description": "What this step does (1-2 sentences)",\n'
        f'    "calls_next": ["relative/path/to/file.py:next_function"]\n'
        f"  }}\n"
        f"]}}\n"
        f"\n"
        f"Rules:\n"
        f"- Order steps by execution sequence (1, 2, 3...)\n"
        f"- Each step is ONE function/method call\n"
        f"- calls_next lists the functions called by this step\n"
        f"- Include 3-15 steps covering the main execution path\n"
        f"- Use exact file paths and function names from the source code"
    )
