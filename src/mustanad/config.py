"""Application configuration.

All settings come from the environment (or a local ``.env`` file). Required settings are
validated at import of :func:`get_settings`, so a misconfigured deployment fails loudly at
startup instead of degrading silently at request time.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ProviderName = Literal["extractive", "openai", "ollama"]

#: A tuple-of-strings setting that is supplied as a plain comma-separated environment value.
#:
#: ``pydantic-settings`` classifies any sequence-typed field as "complex" and tries to
#: **JSON-decode** the raw environment string before field validators run. That makes
#: ``MUSTANAD_API_KEYS=my-key`` a hard startup failure ("error parsing value for field
#: api_keys"), because ``my-key`` is not valid JSON. ``NoDecode`` suppresses that step so the
#: raw string reaches :meth:`Settings._split_csv`, which splits it on commas.
#:
#: This was found by running the server, not by the unit tests -- which passed real tuples to
#: the constructor and so never exercised the environment-parsing path. There is now a test
#: that sets these variables through the environment.
CommaSeparated = Annotated[tuple[str, ...], NoDecode]


class Settings(BaseSettings):
    """Validated runtime configuration.

    Every field has a safe default that works offline, so ``mustanad`` is runnable
    immediately after install. The only settings that *must* be supplied are the ones that
    a chosen non-default provider needs -- and that requirement is enforced below.
    """

    model_config = SettingsConfigDict(
        env_prefix="MUSTANAD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- storage -----------------------------------------------------------------
    db_path: Path = Field(
        default=Path("data/mustanad.sqlite3"),
        description="SQLite file holding documents, chunks and the inverted index.",
    )

    # --- retrieval ---------------------------------------------------------------
    chunk_tokens: int = Field(default=180, ge=40, le=1200)
    chunk_overlap_tokens: int = Field(default=40, ge=0, le=600)
    candidate_pool: int = Field(
        default=50, ge=5, le=500, description="Chunks pulled from BM25 before reranking."
    )
    default_top_k: int = Field(default=5, ge=1, le=50)
    bm25_k1: float = Field(default=1.5, gt=0, le=5)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    min_coverage: float = Field(
        default=0.34,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of the question's distinct terms that the best passage "
        "must contain before the system is willing to answer at all. Below this it reports "
        "that it found no supporting passage, which is the honest answer.",
    )

    # --- ingestion limits --------------------------------------------------------
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, ge=1024)
    allowed_suffixes: CommaSeparated = (".pdf", ".docx", ".md", ".markdown", ".txt")

    # --- answer provider ---------------------------------------------------------
    provider: ProviderName = Field(
        default="extractive",
        description="'extractive' needs no API key and no network. 'openai' targets any "
        "OpenAI-compatible chat-completions endpoint. 'ollama' targets a local Ollama server.",
    )
    llm_model: str = Field(default="gpt-4o-mini")
    llm_base_url: str = Field(default="https://api.openai.com/v1")
    llm_api_key: str | None = Field(default=None, repr=False)
    ollama_base_url: str = Field(default="http://127.0.0.1:11434")

    # --- cost / safety controls for paid providers -------------------------------
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    llm_max_output_tokens: int = Field(default=500, ge=32, le=4096)
    llm_max_context_chars: int = Field(
        default=8000, ge=500, le=200_000, description="Hard ceiling on prompt context size."
    )
    llm_daily_call_budget: int = Field(
        default=200, ge=0, description="Process-local cap on paid provider calls. 0 disables calls."
    )

    # --- HTTP API ----------------------------------------------------------------
    auth_required: bool = Field(default=True)
    api_keys: CommaSeparated = Field(
        default=(),
        repr=False,
        description="Comma-separated API keys accepted in the X-API-Key header.",
    )
    rate_limit_per_minute: int = Field(default=60, ge=1, le=10_000)
    cors_allow_origins: CommaSeparated = ()

    # --- logging -----------------------------------------------------------------
    log_level: str = Field(default="INFO")

    @field_validator("api_keys", "allowed_suffixes", "cors_allow_origins", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Accept both a comma-separated string (from env) and a real sequence (from tests)."""
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {value!r}")
        return upper

    @model_validator(mode="after")
    def _check_coherence(self) -> Settings:
        if self.chunk_overlap_tokens >= self.chunk_tokens:
            raise ValueError(
                "chunk_overlap_tokens must be smaller than chunk_tokens, otherwise chunking "
                f"cannot advance (got overlap={self.chunk_overlap_tokens}, "
                f"size={self.chunk_tokens})"
            )
        if self.auth_required and not self.api_keys:
            raise ValueError(
                "MUSTANAD_AUTH_REQUIRED is true but MUSTANAD_API_KEYS is empty. Set at least "
                "one API key, or set MUSTANAD_AUTH_REQUIRED=false to run without auth "
                "(only ever do that on a trusted local machine)."
            )
        if self.provider == "openai" and not self.llm_api_key:
            raise ValueError(
                "MUSTANAD_PROVIDER=openai requires MUSTANAD_LLM_API_KEY. Use "
                "MUSTANAD_PROVIDER=extractive to run with no API key at all."
            )
        if self.candidate_pool < self.default_top_k:
            raise ValueError("candidate_pool must be >= default_top_k")
        return self

    @property
    def api_key_hashes(self) -> frozenset[str]:
        """SHA-256 of each configured key.

        Keys are compared as hashes so a timing-safe comparison operates on fixed-length
        values and the plaintext is not retained in a searchable structure.
        """
        return frozenset(hashlib.sha256(key.encode("utf-8")).hexdigest() for key in self.api_keys)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, constructed once.

    Raises:
        pydantic.ValidationError: if the environment is invalid or incomplete.
    """
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings. Used by tests that manipulate the environment."""
    get_settings.cache_clear()
