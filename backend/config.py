"""Application configuration loaded from environment variables and .env file."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM provider backends."""

    CLAUDE = "claude"
    OPENAI = "openai"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    GEMINI = "gemini"


class Settings(BaseSettings):
    """Application settings with environment variable binding.

    Values are loaded from a ``.env`` file at the project root and can be
    overridden by actual environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Server -----------------------------------------------------------
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # --- Storage ----------------------------------------------------------
    DATA_DIR: Path = Path("./data")

    # --- LLM --------------------------------------------------------------
    LLM_PROVIDER: LLMProvider = LLMProvider.OPENROUTER
    LLM_MODEL: str = "anthropic/claude-sonnet-4-5-20250929"
    LLM_API_KEY: str = ""
    LLM_BASE_URL: Optional[str] = None

    # --- Derived paths (computed after validation) ------------------------
    REPOS_DIR: Optional[Path] = None
    DB_URL: Optional[str] = None

    @model_validator(mode="after")
    def _derive_paths(self) -> Settings:
        """Compute REPOS_DIR and DB_URL from DATA_DIR when not set explicitly."""
        if self.REPOS_DIR is None:
            self.REPOS_DIR = self.DATA_DIR / "repos"
        if self.DB_URL is None:
            db_path = self.DATA_DIR / "db.sqlite"
            self.DB_URL = f"sqlite+aiosqlite:///{db_path}"
        return self

    @field_validator("DATA_DIR", mode="before")
    @classmethod
    def _coerce_data_dir(cls, value: object) -> Path:
        """Ensure DATA_DIR is always a resolved ``Path``."""
        return Path(str(value)).resolve()

    def ensure_directories(self) -> None:
        """Create required data directories if they do not exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        if self.REPOS_DIR is not None:
            self.REPOS_DIR.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Return a cached application ``Settings`` instance.

    Uses ``lru_cache`` semantics via module-level singleton so the ``.env``
    file is read only once per process.
    """
    return _settings


_settings = Settings()
