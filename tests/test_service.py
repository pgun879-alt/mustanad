"""Tests for ingestion and the service layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from mustanad.loaders import (
    DocumentReadError,
    UnsupportedDocumentError,
    iter_documents,
    load_document,
)
from mustanad.service import IngestionError, MustanadService


def test_ingest_text(service: MustanadService) -> None:
    result = service.ingest_text("Annual leave is 22 days per year.", source="test", title="Policy")
    assert result.already_indexed is False
    assert result.chunk_count == 1
    assert result.document.language == "en"


def test_ingest_detects_arabic(service: MustanadService) -> None:
    result = service.ingest_text("الإجازة السنوية مدفوعة.", source="test", title="لائحة")
    assert result.document.language == "ar"


def test_re_ingesting_identical_content_is_a_no_op(service: MustanadService) -> None:
    """Re-running the demo or a sync job must not duplicate the corpus."""
    first = service.ingest_text("Some policy text here.", source="a", title="A")
    second = service.ingest_text("Some policy text here.", source="b", title="B")
    assert second.already_indexed is True
    assert second.document.id == first.document.id
    assert len(service.list_documents()) == 1


def test_ingest_rejects_empty_text(service: MustanadService) -> None:
    with pytest.raises(IngestionError, match="empty text"):
        service.ingest_text("   ", source="a", title="A")


def test_ingest_rejects_text_with_no_indexable_tokens(service: MustanadService) -> None:
    with pytest.raises(IngestionError, match="no indexable text"):
        service.ingest_text("### ... ---  ///", source="a", title="Symbols")


def test_ingest_missing_file(service: MustanadService, tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="is not a file"):
        service.ingest_path(tmp_path / "nope.md")


def test_ingest_unsupported_file_type(service: MustanadService, tmp_path: Path) -> None:
    binary = tmp_path / "image.png"
    binary.write_bytes(b"\x89PNG\r\n")
    with pytest.raises(IngestionError, match="unsupported file type"):
        service.ingest_path(binary)


def test_ingest_directory_indexes_the_whole_sample_corpus(
    service: MustanadService, samples_dir: Path
) -> None:
    results = service.ingest_directory(samples_dir)
    titles = {result.document.title for result in results}
    assert titles == {
        "employee handbook",
        "support sla",
        "leave policy ar",
        "refund policy ar",
    }
    assert {result.document.language for result in results} == {"en", "ar"}


def test_ingest_directory_skips_a_bad_file_and_continues(
    service: MustanadService, tmp_path: Path
) -> None:
    """One malformed document must not abort a bulk import."""
    (tmp_path / "good.md").write_text("Valid policy content about refunds.", encoding="utf-8")
    (tmp_path / "empty.md").write_text("   ", encoding="utf-8")
    (tmp_path / "symbols.txt").write_text("###", encoding="utf-8")
    results = service.ingest_directory(tmp_path)
    assert [result.document.title for result in results] == ["good"]


def test_ingest_directory_of_unsupported_files_returns_nothing(
    service: MustanadService, tmp_path: Path
) -> None:
    (tmp_path / "a.png").write_bytes(b"x")
    assert service.ingest_directory(tmp_path) == []


def test_title_is_derived_from_the_filename(service: MustanadService, tmp_path: Path) -> None:
    path = tmp_path / "refund_policy-2026.md"
    path.write_text("Refunds are processed within seven days.", encoding="utf-8")
    result = service.ingest_path(path)
    assert result.document.title == "refund policy 2026"


def test_explicit_title_overrides_the_filename(service: MustanadService, tmp_path: Path) -> None:
    path = tmp_path / "x.md"
    path.write_text("Refunds are processed within seven days.", encoding="utf-8")
    result = service.ingest_path(path, title="Refund Policy")
    assert result.document.title == "Refund Policy"


def test_ask_rejects_an_empty_question(loaded_service: MustanadService) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        loaded_service.ask("   ")


def test_excerpt_is_truncated_and_whitespace_normalised(loaded_service: MustanadService) -> None:
    result = loaded_service.ask("annual leave")
    for citation in result.citations:
        assert "\n" not in citation.excerpt
        assert len(citation.excerpt) <= 401  # 400 chars plus the ellipsis


def test_delete_missing_document_returns_false(service: MustanadService) -> None:
    assert service.delete_document(999) is False


# ------------------------------------------------------------------------- loaders


def test_load_markdown_returns_one_unpaged_page(samples_dir: Path) -> None:
    pages = load_document(samples_dir / "en" / "employee-handbook.md")
    assert len(pages) == 1
    assert pages[0][0] is None
    assert "annual leave" in pages[0][1].lower()


def test_load_unsupported_suffix_raises(tmp_path: Path) -> None:
    """A file that exists but has no loader is a different error from a missing file."""
    path = tmp_path / "spreadsheet.xlsx"
    path.write_bytes(b"PK\x03\x04")
    with pytest.raises(UnsupportedDocumentError, match="unsupported file type"):
        load_document(path)


def test_load_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_document(Path("/nonexistent/file.md"))


def test_iter_documents_is_sorted_and_filtered(tmp_path: Path) -> None:
    for name in ["b.md", "a.md", "c.png", "d.txt"]:
        (tmp_path / name).write_text("x", encoding="utf-8")
    found = iter_documents(tmp_path, (".md", ".txt"))
    assert [path.name for path in found] == ["a.md", "b.md", "d.txt"]


def test_iter_documents_skips_hidden_directories(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.md").write_text("secret", encoding="utf-8")
    (tmp_path / "real.md").write_text("content", encoding="utf-8")
    assert [path.name for path in iter_documents(tmp_path, (".md",))] == ["real.md"]


def test_iter_documents_accepts_a_single_file(tmp_path: Path) -> None:
    path = tmp_path / "one.md"
    path.write_text("x", encoding="utf-8")
    assert iter_documents(path, (".md",)) == [path]
    assert iter_documents(path, (".pdf",)) == []


# ------------------------------------------------------- untrusted archive handling


def test_a_docx_decompression_bomb_is_refused_before_expansion(tmp_path: Path) -> None:
    """A DOCX is a ZIP, so the HTTP upload cap bounds only its *compressed* size.

    A file of a couple of hundred kilobytes can legitimately declare hundreds of megabytes of
    contents. The ZIP central directory records those sizes, so the archive is inspected and
    rejected before anything is decompressed.
    """
    import zipfile

    bomb = tmp_path / "bomb.docx"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", b"\0" * (200 * 1024 * 1024))

    assert bomb.stat().st_size < 5 * 1024 * 1024, "the bomb must be small on disk"
    with pytest.raises(DocumentReadError, match=r"decompression bomb|refusing to expand"):
        load_document(bomb)


def test_a_docx_that_is_not_a_zip_is_reported_clearly(tmp_path: Path) -> None:
    path = tmp_path / "broken.docx"
    path.write_bytes(b"this is not a zip archive at all")
    with pytest.raises(DocumentReadError, match="bad ZIP container"):
        load_document(path)


def test_an_ordinary_docx_is_not_refused_by_the_guard(tmp_path: Path) -> None:
    """The guard must not reject real documents; normal prose compresses nowhere near the limit."""
    import docx

    path = tmp_path / "ordinary.docx"
    document = docx.Document()
    for _ in range(200):
        document.add_paragraph("Annual leave is 22 working days per calendar year. ")
    document.save(str(path))

    pages = load_document(path)
    assert pages
    assert "Annual leave" in pages[0][1]
