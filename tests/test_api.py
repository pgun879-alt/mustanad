"""HTTP API tests, driven through FastAPI's TestClient against a temporary database."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mustanad.api import create_app
from mustanad.config import Settings

API_KEY = "test-key-alpha"


@pytest.fixture
def open_client(tmp_path: Path) -> Iterator[TestClient]:
    """A client with authentication disabled, for testing behaviour rather than auth."""
    settings = Settings(
        db_path=tmp_path / "open.sqlite3",
        provider="extractive",
        auth_required=False,
        api_keys=(),
    )
    with TestClient(create_app(settings)) as client:
        yield client


@pytest.fixture
def secure_client(tmp_path: Path) -> Iterator[TestClient]:
    """A client with authentication and a low rate limit enabled."""
    settings = Settings(
        db_path=tmp_path / "secure.sqlite3",
        provider="extractive",
        auth_required=True,
        api_keys=(API_KEY,),
        rate_limit_per_minute=5,
    )
    with TestClient(create_app(settings)) as client:
        yield client


def _seed(client: TestClient, headers: dict[str, str] | None = None) -> None:
    response = client.post(
        "/v1/ingest/text",
        json={
            "title": "Employee Handbook",
            "text": (
                "Working hours are 40 per week. Every full-time employee accrues 22 working "
                "days of paid annual leave per year. The per-diem for domestic travel is "
                "4,500 DZD per night."
            ),
        },
        headers=headers or {},
    )
    assert response.status_code == 201, response.text


# ------------------------------------------------------------------------- health


def test_healthz_is_public_and_reports_the_provider(open_client: TestClient) -> None:
    response = open_client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["provider"] == "extractive"


def test_readyz_reports_503_while_the_corpus_is_empty(open_client: TestClient) -> None:
    """An empty index is a live service that cannot do its job yet."""
    response = open_client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["status"] == "empty-corpus"


def test_readyz_reports_200_once_documents_exist(open_client: TestClient) -> None:
    _seed(open_client)
    response = open_client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["chunks"] >= 1


def test_every_response_carries_a_request_id(open_client: TestClient) -> None:
    assert open_client.get("/healthz").headers["X-Request-ID"]


def test_a_supplied_request_id_is_echoed(open_client: TestClient) -> None:
    response = open_client.get("/healthz", headers={"X-Request-ID": "trace-me-123"})
    assert response.headers["X-Request-ID"] == "trace-me-123"


# --------------------------------------------------------------------------- auth


def test_protected_routes_require_a_key(secure_client: TestClient) -> None:
    assert secure_client.post("/v1/ask", json={"question": "anything"}).status_code == 401
    assert secure_client.get("/v1/documents").status_code == 401
    assert secure_client.delete("/v1/documents/1").status_code == 401
    assert (
        secure_client.post("/v1/ingest/text", json={"title": "t", "text": "x"}).status_code == 401
    )


def test_an_invalid_key_is_rejected(secure_client: TestClient) -> None:
    response = secure_client.post(
        "/v1/ask", json={"question": "anything"}, headers={"X-API-Key": "wrong"}
    )
    assert response.status_code == 401


def test_a_valid_key_is_accepted(secure_client: TestClient) -> None:
    headers = {"X-API-Key": API_KEY}
    _seed(secure_client, headers)
    response = secure_client.post(
        "/v1/ask", json={"question": "How many days of annual leave?"}, headers=headers
    )
    assert response.status_code == 200


def test_health_endpoints_stay_public_when_auth_is_on(secure_client: TestClient) -> None:
    assert secure_client.get("/healthz").status_code == 200


def test_the_rate_limit_returns_429_with_retry_after(secure_client: TestClient) -> None:
    headers = {"X-API-Key": API_KEY}
    _seed(secure_client, headers)  # consumes one of the five allowed requests
    statuses = [
        secure_client.post("/v1/ask", json={"question": "leave"}, headers=headers).status_code
        for _ in range(8)
    ]
    assert 429 in statuses
    limited = secure_client.post("/v1/ask", json={"question": "leave"}, headers=headers)
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1


# ---------------------------------------------------------------------------- ask


def test_ask_returns_an_answer_with_citations(open_client: TestClient) -> None:
    _seed(open_client)
    body = open_client.post("/v1/ask", json={"question": "How many days of annual leave?"}).json()
    assert body["grounded"] is True
    assert "22 working days" in body["answer"]
    assert body["citations"]
    assert body["citations"][0]["document_title"] == "Employee Handbook"
    assert 0.0 <= body["confidence"] <= 1.0


def test_ask_declines_an_unanswerable_question_rather_than_guessing(
    open_client: TestClient,
) -> None:
    _seed(open_client)
    body = open_client.post("/v1/ask", json={"question": "What is the capital of Japan?"}).json()
    assert body["grounded"] is False
    assert "insufficient_evidence" in body["warnings"]


def test_ask_works_in_arabic(open_client: TestClient) -> None:
    open_client.post(
        "/v1/ingest/text",
        json={"title": "لائحة", "text": "الإجازة السنوية اثنان وعشرون يوم عمل مدفوعة الأجر."},
    )
    body = open_client.post("/v1/ask", json={"question": "ما هي الإجازة السنوية؟"}).json()
    assert body["grounded"] is True
    assert body["citations"]


def test_ask_validates_its_input(open_client: TestClient) -> None:
    assert open_client.post("/v1/ask", json={"question": "x"}).status_code == 422
    assert open_client.post("/v1/ask", json={}).status_code == 422
    assert open_client.post("/v1/ask", json={"question": "a" * 1001}).status_code == 422
    assert open_client.post("/v1/ask", json={"question": "ok?", "top_k": 99}).status_code == 422


def test_top_k_limits_the_citation_count(open_client: TestClient) -> None:
    _seed(open_client)
    body = open_client.post("/v1/ask", json={"question": "leave", "top_k": 1}).json()
    assert len(body["citations"]) <= 1


# ------------------------------------------------------------------------- ingest


def test_ingest_text_then_list_documents(open_client: TestClient) -> None:
    _seed(open_client)
    documents = open_client.get("/v1/documents").json()
    assert len(documents) == 1
    assert documents[0]["title"] == "Employee Handbook"
    assert documents[0]["chunk_count"] >= 1


def test_re_ingesting_the_same_text_reports_already_indexed(open_client: TestClient) -> None:
    _seed(open_client)
    _seed(open_client)
    assert len(open_client.get("/v1/documents").json()) == 1


def test_ingest_rejects_text_with_nothing_indexable(open_client: TestClient) -> None:
    response = open_client.post("/v1/ingest/text", json={"title": "Junk", "text": "### ---"})
    assert response.status_code == 422


def test_ingest_upload_accepts_markdown(open_client: TestClient) -> None:
    response = open_client.post(
        "/v1/ingest/upload",
        files={
            "file": (
                "refund-policy.md",
                b"# Refunds\n\nRefunds are issued within seven working days.",
                "text/markdown",
            )
        },
    )
    assert response.status_code == 201, response.text
    assert response.json()["title"] == "refund-policy"


def test_ingest_upload_rejects_a_disallowed_extension(open_client: TestClient) -> None:
    response = open_client.post(
        "/v1/ingest/upload",
        files={"file": ("payload.exe", b"MZ\x90\x00", "application/octet-stream")},
    )
    assert response.status_code == 415


def test_ingest_upload_rejects_an_empty_file(open_client: TestClient) -> None:
    response = open_client.post("/v1/ingest/upload", files={"file": ("empty.md", b"", "text/md")})
    assert response.status_code == 422


def test_ingest_upload_enforces_the_size_limit(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "small.sqlite3",
        provider="extractive",
        auth_required=False,
        max_upload_bytes=1024,
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/ingest/upload", files={"file": ("big.md", b"x" * 4096, "text/markdown")}
        )
        assert response.status_code == 413


def test_ingest_upload_cannot_be_used_to_traverse_the_filesystem(
    open_client: TestClient,
) -> None:
    """A crafted filename must not influence where anything is written or read.

    The server generates the temporary path itself and uses the client's filename only as a
    display title, so the traversal segments end up as inert text.
    """
    response = open_client.post(
        "/v1/ingest/upload",
        files={
            "file": (
                "../../../../etc/passwd.md",
                b"Refunds are issued within seven working days.",
                "text/markdown",
            )
        },
    )
    assert response.status_code == 201
    documents = open_client.get("/v1/documents").json()
    assert "/etc/passwd" not in documents[0]["source"]
    assert documents[0]["source"].endswith(".md")


def test_ingest_text_enforces_the_size_limit(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "small.sqlite3",
        provider="extractive",
        auth_required=False,
        max_upload_bytes=1024,
    )
    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/ingest/text", json={"title": "Big", "text": "word " * 1000})
        assert response.status_code == 413


# ---------------------------------------------------------------------- documents


def test_delete_document_removes_it(open_client: TestClient) -> None:
    _seed(open_client)
    document_id = open_client.get("/v1/documents").json()[0]["id"]
    assert open_client.delete(f"/v1/documents/{document_id}").status_code == 204
    assert open_client.get("/v1/documents").json() == []


def test_delete_missing_document_is_404(open_client: TestClient) -> None:
    assert open_client.delete("/v1/documents/9999").status_code == 404


def test_deleted_content_stops_appearing_in_answers(open_client: TestClient) -> None:
    """Deletion must clear the index, not just the document listing."""
    _seed(open_client)
    before = open_client.post("/v1/ask", json={"question": "annual leave days"}).json()
    assert before["grounded"] is True
    document_id = open_client.get("/v1/documents").json()[0]["id"]
    open_client.delete(f"/v1/documents/{document_id}")
    after = open_client.post("/v1/ask", json={"question": "annual leave days"}).json()
    assert after["grounded"] is False
    assert after["citations"] == []


# --------------------------------------------------------------------- demo page


def test_demo_page_is_served(open_client: TestClient) -> None:
    response = open_client.get("/")
    assert response.status_code == 200
    assert "mustanad" in response.text


def test_openapi_schema_is_generated(open_client: TestClient) -> None:
    schema = open_client.get("/openapi.json").json()
    assert "/v1/ask" in schema["paths"]
    assert "/v1/ingest/upload" in schema["paths"]
