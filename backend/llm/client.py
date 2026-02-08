"""Unified async LLM client backed by litellm.

Provides a singleton ``LLMClient`` that abstracts over multiple providers
(Claude, OpenAI, Ollama, OpenRouter) with structured JSON completion
support and graceful error handling.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

import litellm

from backend.llm.config import LLMConfig, get_llm_config

logger = logging.getLogger(__name__)

# Suppress litellm's verbose default logging; we handle our own.
litellm.suppress_debug_info = True

_MAX_JSON_RETRIES = 2


class LLMClient:
    """Async LLM completion client with provider-agnostic interface.

    Uses litellm under the hood so callers never need to know which
    provider is active.  Instantiate via ``LLMClient.get_instance()``
    to reuse a single configured client across the application.
    """

    _instance: LLMClient | None = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, config: LLMConfig | None = None) -> None:
        self._config = config or get_llm_config()

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls, config: LLMConfig | None = None) -> LLMClient:
        """Return the shared ``LLMClient`` singleton.

        On first call the instance is created using the supplied *config*
        (or ``get_llm_config()`` if none is provided).  Subsequent calls
        ignore the *config* argument and return the existing instance.
        """
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking for thread safety.
                if cls._instance is None:
                    cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Discard the cached singleton (useful for testing)."""
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str | None:
        """Request a plain-text completion from the LLM.

        Returns the assistant's text content on success, or ``None`` if
        the request fails for any reason (network error, rate limit,
        invalid response, etc.).
        """
        messages = self._build_messages(prompt, system)
        try:
            response = await litellm.acompletion(
                model=self._config.litellm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=self._config.api_key or None,
                api_base=self._config.base_url,
            )
            return self._extract_content(response)
        except Exception:
            logger.exception(
                "LLM completion failed (model=%s)", self._config.litellm_model
            )
            return None

    async def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        """Request a JSON-structured completion from the LLM.

        Automatically retries up to ``_MAX_JSON_RETRIES`` times when the
        response cannot be parsed as valid JSON.  Returns an empty dict
        on persistent failure.
        """
        json_system = self._build_json_system(system)

        for attempt in range(1, _MAX_JSON_RETRIES + 1):
            text = await self.complete(
                prompt,
                system=json_system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if text is None:
                return {}

            parsed = self._try_parse_json(text)
            if parsed is not None:
                return parsed

            logger.warning(
                "JSON parse attempt %d/%d failed; retrying",
                attempt,
                _MAX_JSON_RETRIES,
            )

        logger.error("Failed to parse JSON response after %d attempts", _MAX_JSON_RETRIES)
        return {}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_messages(
        self, prompt: str, system: str | None
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _build_json_system(system: str | None) -> str:
        json_instruction = (
            "You MUST respond with valid JSON only. "
            "Do not include markdown fences, commentary, or any text outside the JSON object."
        )
        if system:
            return f"{system}\n\n{json_instruction}"
        return json_instruction

    @staticmethod
    def _extract_content(response: Any) -> str | None:
        """Safely pull the text content from a litellm response."""
        try:
            return response.choices[0].message.content  # type: ignore[union-attr]
        except (AttributeError, IndexError, TypeError):
            logger.error("Unexpected response structure: %s", type(response))
            return None

    @staticmethod
    def _try_parse_json(text: str) -> dict[str, Any] | None:
        """Attempt to parse *text* as JSON, with light cleanup for common LLM habits."""
        cleaned = text.strip()

        # Strip markdown code fences if the model wraps its response.
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Remove opening fence (```json or ```)
            lines = lines[1:]
            # Remove closing fence if present
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                return result
            logger.warning("JSON response is not a dict: %s", type(result).__name__)
            return None
        except json.JSONDecodeError:
            return None
