"""Tests for configuration validation.

The point of these: a misconfigured deployment must fail loudly at startup, never silently
serve traffic in an unintended state (auth off, provider unusable, chunking that cannot
advance).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mustanad.config import Settings


def _base(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "db_path": Path("data/test.sqlite3"),
        "auth_required": False,
        "provider": "extractive",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def test_defaults_are_offline_and_usable() -> None:
    settings = _base()
    assert settings.provider == "extractive"
    assert settings.llm_api_key is None


def test_auth_required_without_keys_is_rejected() -> None:
    """The most dangerous misconfiguration: intending auth, getting none."""
    with pytest.raises(ValidationError, match="MUSTANAD_API_KEYS is empty"):
        _base(auth_required=True, api_keys=())


def test_auth_required_with_keys_is_accepted() -> None:
    settings = _base(auth_required=True, api_keys=("key-one", "key-two"))
    assert len(settings.api_key_hashes) == 2


def test_openai_provider_without_a_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="requires MUSTANAD_LLM_API_KEY"):
        _base(provider="openai", llm_api_key=None)


def test_openai_provider_with_a_key_is_accepted() -> None:
    assert _base(provider="openai", llm_api_key="sk-test").provider == "openai"


def test_ollama_provider_needs_no_key() -> None:
    assert _base(provider="ollama").provider == "ollama"


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _base(provider="magic")


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    """Equal values would make the chunker unable to advance."""
    with pytest.raises(ValidationError, match="chunk_overlap_tokens must be smaller"):
        _base(chunk_tokens=100, chunk_overlap_tokens=100)


def test_candidate_pool_must_not_be_smaller_than_top_k() -> None:
    with pytest.raises(ValidationError, match="candidate_pool must be >= default_top_k"):
        _base(candidate_pool=5, default_top_k=10)


def test_comma_separated_api_keys_from_the_environment_are_split() -> None:
    settings = _base(auth_required=True, api_keys="alpha, beta , gamma")
    assert len(settings.api_key_hashes) == 3


def test_blank_entries_in_a_csv_list_are_dropped() -> None:
    assert _base(auth_required=True, api_keys="alpha,,  ,beta").api_keys == ("alpha", "beta")


def test_log_level_is_normalised_and_validated() -> None:
    assert _base(log_level="debug").log_level == "DEBUG"
    with pytest.raises(ValidationError, match="log_level must be one of"):
        _base(log_level="chatty")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunk_tokens", 5),  # below the floor
        ("bm25_b", 1.5),  # above 1.0
        ("bm25_k1", 0),  # must be > 0
        ("rate_limit_per_minute", 0),
        ("llm_timeout_seconds", 0),
        ("min_coverage", 1.5),
        ("max_upload_bytes", 10),
    ],
)
def test_out_of_range_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        _base(**{field: value})


def test_api_key_hashes_are_not_the_raw_keys() -> None:
    settings = _base(auth_required=True, api_keys=("super-secret",))
    (digest,) = settings.api_key_hashes
    assert "super-secret" not in digest
    assert len(digest) == 64


def test_api_key_is_not_in_the_repr() -> None:
    """A settings object ending up in a log line must not leak credentials."""
    settings = _base(auth_required=True, api_keys=("super-secret",), llm_api_key="sk-secret")
    text = repr(settings)
    assert "super-secret" not in text
    assert "sk-secret" not in text


def test_environment_variables_are_read_with_the_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MUSTANAD_AUTH_REQUIRED", "false")
    monkeypatch.setenv("MUSTANAD_CHUNK_TOKENS", "250")
    monkeypatch.setenv("MUSTANAD_LOG_LEVEL", "warning")
    monkeypatch.setenv("MUSTANAD_DB_PATH", str(tmp_path / "env.sqlite3"))
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.chunk_tokens == 250
    assert settings.log_level == "WARNING"


def test_list_settings_parse_from_plain_environment_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for a startup crash the unit tests originally missed.

    ``pydantic-settings`` treats any sequence-typed field as "complex" and JSON-decodes the raw
    environment value *before* field validators run, so ``MUSTANAD_API_KEYS=demo-key`` raised
    ``SettingsError: error parsing value for field "api_keys"`` and the server refused to
    start. The earlier tests all passed real tuples straight to the constructor, so none of
    them touched the environment path -- only running the binary found it. The fields are now
    annotated with ``NoDecode``.
    """
    monkeypatch.setenv("MUSTANAD_AUTH_REQUIRED", "true")
    monkeypatch.setenv("MUSTANAD_API_KEYS", "demo-key-123")
    monkeypatch.setenv("MUSTANAD_CORS_ALLOW_ORIGINS", "http://localhost:3000,https://example.com")
    monkeypatch.setenv("MUSTANAD_ALLOWED_SUFFIXES", ".md,.txt")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.api_keys == ("demo-key-123",)
    assert settings.cors_allow_origins == ("http://localhost:3000", "https://example.com")
    assert settings.allowed_suffixes == (".md", ".txt")


def test_multiple_api_keys_parse_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUSTANAD_AUTH_REQUIRED", "true")
    monkeypatch.setenv("MUSTANAD_API_KEYS", "key-one, key-two ,key-three")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.api_keys == ("key-one", "key-two", "key-three")
    assert len(settings.api_key_hashes) == 3
