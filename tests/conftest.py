"""Shared fixtures.

Every fixture uses the offline extractive provider and a temporary database, so the suite
never touches the network, never needs an API key, and leaves nothing behind.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from mustanad.config import Settings
from mustanad.index.store import DocumentStore
from mustanad.service import MustanadService

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture
def samples_dir() -> Path:
    """Path to the bundled sample corpus."""
    return SAMPLES


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Offline settings backed by a throwaway database."""
    return Settings(
        db_path=tmp_path / "test.sqlite3",
        provider="extractive",
        auth_required=False,
        api_keys=(),
        chunk_tokens=120,
        chunk_overlap_tokens=30,
    )


@pytest.fixture
def store(settings: Settings) -> Iterator[DocumentStore]:
    with DocumentStore(settings.db_path) as opened:
        yield opened


@pytest.fixture
def service(settings: Settings, store: DocumentStore) -> Iterator[MustanadService]:
    built = MustanadService(settings, store)
    yield built
    built.provider.close()


@pytest.fixture
def loaded_service(service: MustanadService) -> MustanadService:
    """A service with the whole bundled sample corpus indexed."""
    results = service.ingest_directory(SAMPLES)
    assert len(results) == 4, f"expected 4 sample documents, indexed {len(results)}"
    return service
