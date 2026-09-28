"""Application service layer: ingestion and question answering.

The HTTP API and the CLI are both thin shells over this class. Keeping the orchestration here
means the demo, the tests and the API exercise exactly the same code path -- there is no
"works in the CLI but not the API" gap to fall into.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .index.store import ChunkRecord, DocumentRecord, DocumentStore, content_hash
from .loaders import DocumentReadError, UnsupportedDocumentError, iter_documents, load_document
from .providers import AnswerProvider, AnswerRequest, build_provider
from .retrieval import RetrievedPassage, Retriever
from .text.chunk import Chunk, chunk_text
from .text.normalize import is_probably_arabic

logger = logging.getLogger(__name__)


class IngestionError(RuntimeError):
    """A document could not be ingested. The message is safe to return to a caller."""


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Outcome of ingesting one document."""

    document: DocumentRecord
    chunk_count: int
    already_indexed: bool


@dataclass(frozen=True, slots=True)
class Citation:
    """A source reference attached to an answer."""

    index: int
    document_id: int
    document_title: str
    source: str
    page: int | None
    label: str
    excerpt: str
    score: float


@dataclass(frozen=True, slots=True)
class AskResult:
    """A full answer payload: the text, its provenance, and how it was produced."""

    question: str
    answer: str
    citations: tuple[Citation, ...]
    provider: str
    model: str | None
    grounded: bool
    confidence: float
    warnings: tuple[str, ...]


#: How much of a passage is echoed back as a human-readable excerpt.
EXCERPT_CHARS = 400


class MustanadService:
    """Ingest documents and answer questions about them."""

    def __init__(
        self,
        settings: Settings,
        store: DocumentStore,
        provider: AnswerProvider | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.provider = provider or build_provider(settings)
        self.retriever = Retriever(
            store,
            candidate_pool=settings.candidate_pool,
            k1=settings.bm25_k1,
            b=settings.bm25_b,
        )

    # -- ingestion ---------------------------------------------------------------

    def ingest_path(self, path: Path, *, title: str | None = None) -> IngestionResult:
        """Read, chunk and index a single file.

        Re-ingesting a byte-identical document is a no-op rather than an error, so re-running
        the demo or a sync job does not duplicate the corpus.

        Raises:
            IngestionError: if the file is missing, unsupported, unreadable, or has no text.
        """
        try:
            pages = load_document(path)
        except FileNotFoundError as exc:
            raise IngestionError(str(exc)) from exc
        except UnsupportedDocumentError as exc:
            raise IngestionError(str(exc)) from exc
        except DocumentReadError as exc:
            raise IngestionError(str(exc)) from exc

        return self._ingest_pages(
            pages=pages,
            source=str(path),
            title=title or path.stem.replace("_", " ").replace("-", " ").strip() or path.name,
        )

    def ingest_text(self, text: str, *, source: str, title: str) -> IngestionResult:
        """Index a raw string. Used for uploads held in memory and for tests."""
        if not text.strip():
            raise IngestionError("refusing to ingest empty text")
        return self._ingest_pages(pages=[(None, text)], source=source, title=title)

    def _ingest_pages(
        self, *, pages: list[tuple[int | None, str]], source: str, title: str
    ) -> IngestionResult:
        combined = "\n\n".join(text for _, text in pages)
        fingerprint = content_hash(combined)

        existing = self.store.find_by_hash(fingerprint)
        if existing is not None:
            logger.info("document %r already indexed as id=%d", title, existing.id)
            return IngestionResult(
                document=existing, chunk_count=existing.n_chunks, already_indexed=True
            )

        chunks: list[Chunk] = []
        for page_number, page_text in pages:
            chunks.extend(
                chunk_text(
                    page_text,
                    target_tokens=self.settings.chunk_tokens,
                    overlap_tokens=self.settings.chunk_overlap_tokens,
                    page=page_number,
                    start_ordinal=len(chunks),
                )
            )

        if not chunks:
            raise IngestionError(
                f"{title!r} produced no indexable text after normalisation; it may contain "
                "only images, symbols or whitespace"
            )

        document = self.store.add_document(
            source=source,
            title=title,
            sha256=fingerprint,
            language="ar" if is_probably_arabic(combined) else "en",
            chunks=chunks,
        )
        logger.info("indexed %r as id=%d with %d chunks", title, document.id, len(chunks))
        return IngestionResult(document=document, chunk_count=len(chunks), already_indexed=False)

    def ingest_directory(self, root: Path) -> list[IngestionResult]:
        """Ingest every supported file under ``root``, skipping and logging failures.

        One malformed file must not abort a 200-document import, so per-file errors are logged
        and the rest continue.
        """
        results: list[IngestionResult] = []
        for path in iter_documents(root, self.settings.allowed_suffixes):
            try:
                results.append(self.ingest_path(path))
            except IngestionError as exc:
                logger.warning("skipping %s: %s", path, exc)
        return results

    # -- querying ----------------------------------------------------------------

    def ask(self, question: str, *, top_k: int | None = None) -> AskResult:
        """Retrieve supporting passages and compose a cited answer.

        When the best passage covers less than ``settings.min_coverage`` of the question's
        terms, no answer is composed and ``grounded`` is ``False``. That is deliberate: an
        answer built on evidence this weak is worse than admitting the gap.
        """
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        limit = top_k or self.settings.default_top_k
        passages = self.retriever.search(question, top_k=limit)
        confidence = passages[0].score if passages else 0.0
        best_coverage = passages[0].coverage if passages else 0.0

        if not passages or best_coverage < self.settings.min_coverage:
            logger.info(
                "declining to answer: %d passages, best coverage %.2f < %.2f",
                len(passages),
                best_coverage,
                self.settings.min_coverage,
            )
            return AskResult(
                question=question,
                answer=_insufficient_evidence_message(question),
                citations=self._citations(passages),
                provider=self.provider.name,
                model=None,
                grounded=False,
                confidence=confidence,
                warnings=("insufficient_evidence",),
            )

        result = self.provider.answer(
            AnswerRequest(
                question=question,
                passages=tuple(passages),
                max_output_tokens=self.settings.llm_max_output_tokens,
                max_context_chars=self.settings.llm_max_context_chars,
            )
        )
        return AskResult(
            question=question,
            answer=result.answer,
            citations=self._citations(passages),
            provider=result.provider,
            model=result.model,
            grounded=result.grounded,
            confidence=confidence,
            warnings=result.warnings,
        )

    def _citations(self, passages: list[RetrievedPassage]) -> tuple[Citation, ...]:
        return tuple(
            Citation(
                index=index,
                document_id=passage.chunk.document_id,
                document_title=passage.chunk.document_title,
                source=passage.chunk.document_source,
                page=passage.chunk.page,
                label=passage.citation_label,
                excerpt=_excerpt(passage.chunk),
                score=round(passage.score, 4),
            )
            for index, passage in enumerate(passages, start=1)
        )

    # -- corpus management -------------------------------------------------------

    def list_documents(self) -> list[DocumentRecord]:
        return self.store.list_documents()

    def delete_document(self, document_id: int) -> bool:
        return self.store.delete_document(document_id)

    def close(self) -> None:
        self.provider.close()
        self.store.close()


def _excerpt(chunk: ChunkRecord) -> str:
    text = " ".join(chunk.text.split())
    if len(text) <= EXCERPT_CHARS:
        return text
    return text[:EXCERPT_CHARS].rstrip() + "…"


def _insufficient_evidence_message(question: str) -> str:
    if is_probably_arabic(question):
        return (
            "لم أجد في المستندات المُفهرسة نصًّا يدعم الإجابة على هذا السؤال بثقة كافية. "
            "المقاطع الأقرب مرفقة للمراجعة."
        )
    return (
        "I could not find a passage in the indexed documents that supports a confident "
        "answer to this question. The closest passages are attached for review."
    )
