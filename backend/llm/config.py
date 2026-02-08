"""LLM provider configuration for the Code Flow Visualizer.

Reads LLM settings from the application's central ``Settings`` instance
(which itself loads from environment variables / ``.env``), and exposes
an immutable ``LLMConfig`` dataclass consumed by ``LLMClient``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from backend.config import get_settings

logger = logging.getLogger(__name__)

# Provider-specific base URL defaults applied when Settings.LLM_BASE_URL is empty.
_DEFAULT_BASE_URLS: dict[str, str] = {
    "ollama": "http://localhost:11434",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

SUPPORTED_PROVIDERS = frozenset({"claude", "openai", "ollama", "openrouter"})


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Immutable configuration for the LLM provider.

    Attributes:
        provider: LLM provider identifier (claude, openai, ollama, openrouter).
        model: Model name as expected by the provider.
        api_key: API key for authenticated providers. Empty string for local providers.
        base_url: Provider API base URL. None lets litellm use its own default.
    """

    provider: str
    model: str
    api_key: str
    base_url: str | None

    def __post_init__(self) -> None:
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM provider: {self.provider!r}. "
                f"Supported: {', '.join(sorted(SUPPORTED_PROVIDERS))}"
            )

    @property
    def litellm_model(self) -> str:
        """Return the model string formatted for litellm's routing conventions.

        litellm expects provider-prefixed model names for certain backends.
        For example, OpenRouter models should be passed as-is (they already
        contain a slash like 'anthropic/claude-sonnet'), while Claude models
        need no prefix because litellm recognises them natively.
        """
        if self.provider == "ollama":
            return f"ollama/{self.model}"
        if self.provider == "openrouter":
            # OpenRouter models already carry the org prefix (e.g.
            # "anthropic/claude-sonnet-4-20250514"), so pass through.
            return f"openrouter/{self.model}"
        # Claude and OpenAI models are recognized directly by litellm.
        return self.model

    @property
    def requires_api_key(self) -> bool:
        """Return True if the provider requires an API key."""
        return self.provider not in {"ollama"}


def get_llm_config() -> LLMConfig:
    """Build an ``LLMConfig`` from the application's ``Settings``.

    Delegates to ``backend.config.get_settings()`` so that all
    configuration is centralized in one place (environment variables
    and ``.env`` are read only once by pydantic-settings).
    """
    settings = get_settings()
    provider = settings.LLM_PROVIDER.value  # LLMProvider enum -> str

    base_url: str | None = settings.LLM_BASE_URL
    if not base_url:
        base_url = _DEFAULT_BASE_URLS.get(provider)

    config = LLMConfig(
        provider=provider,
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=base_url,
    )

    if config.requires_api_key and not config.api_key:
        logger.warning(
            "LLM_API_KEY is empty for provider=%r. API calls will likely fail.",
            provider,
        )

    logger.info(
        "LLM config loaded: provider=%s model=%s base_url=%s",
        config.provider,
        config.model,
        config.base_url or "(litellm default)",
    )

    return config
