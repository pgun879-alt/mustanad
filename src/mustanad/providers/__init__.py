"""Answer providers and the factory that picks one from configuration."""

from __future__ import annotations

from ..config import Settings
from .base import (
    AnswerProvider,
    AnswerRequest,
    AnswerResult,
    BudgetExceededError,
    ProviderError,
    build_context_block,
)
from .extractive import ExtractiveProvider
from .remote import CallBudget, OllamaProvider, OpenAICompatibleProvider

__all__ = [
    "AnswerProvider",
    "AnswerRequest",
    "AnswerResult",
    "BudgetExceededError",
    "CallBudget",
    "ExtractiveProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "ProviderError",
    "build_context_block",
    "build_provider",
]


def build_provider(settings: Settings) -> AnswerProvider:
    """Instantiate the provider named by ``settings.provider``.

    Configuration validity is already guaranteed by :class:`~mustanad.config.Settings` -- for
    example, ``provider="openai"`` without an API key fails at settings construction, not here.
    """
    if settings.provider == "extractive":
        return ExtractiveProvider()
    if settings.provider == "ollama":
        return OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
        )
    if settings.provider == "openai":
        if not settings.llm_api_key:  # pragma: no cover - guarded by Settings validation
            raise ProviderError("openai provider selected without an API key")
        return OpenAICompatibleProvider(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
            daily_call_budget=settings.llm_daily_call_budget,
        )
    raise ProviderError(f"unknown provider {settings.provider!r}")  # pragma: no cover
