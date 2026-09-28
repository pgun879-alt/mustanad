"""Two-stage retrieval: BM25 recall, then a proximity-aware rerank.

Why two stages
--------------
BM25 is a bag-of-words model: it cannot tell "the *annual* leave *balance*" from a chunk that
mentions "annual" in one paragraph and "balance" in another. That distinction is exactly what
decides whether a quoted passage answers the user's question.

So stage one uses the inverted index to pull a generous candidate pool cheaply, and stage two
re-reads *only those candidates* and scores three additional signals:

``coverage``
    Fraction of the question's distinct terms present in the chunk. This is the honest
    relevance signal and the one used to decide whether to answer at all.
``proximity``
    How tightly the matched terms cluster, from the smallest window containing the most
    distinct matches. Ratio of the ideal window to the observed one, so 1.0 means adjacent.
``phrase``
    1.0 when the question's token sequence appears verbatim in the chunk.

Recomputing tokens for ~50 candidates costs microseconds, which is why token *positions* are
deliberately not stored in the index -- it keeps the database small and the schema simple.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

from .index.bm25 import score_query
from .index.store import ChunkRecord, DocumentStore
from .text.normalize import tokenize

logger = logging.getLogger(__name__)

# Rerank weights. They sum to 1.0 so a final score is comparable across queries and can be
# reasoned about as "how much of the question does this passage actually cover".
WEIGHT_COVERAGE: Final = 0.50
WEIGHT_BM25: Final = 0.30
WEIGHT_PROXIMITY: Final = 0.20
PHRASE_BONUS: Final = 0.10


@dataclass(frozen=True, slots=True)
class RetrievedPassage:
    """One reranked passage, ready to be cited."""

    chunk: ChunkRecord
    score: float
    bm25_score: float
    coverage: float
    proximity: float
    phrase_match: bool
    matched_terms: frozenset[str]

    @property
    def citation_label(self) -> str:
        """Human-readable source reference, e.g. ``Leave Policy, p. 3`` or ``Handbook #4``."""
        if self.chunk.page is not None:
            return f"{self.chunk.document_title}, p. {self.chunk.page}"
        return f"{self.chunk.document_title} #{self.chunk.ordinal + 1}"


def _smallest_window(tokens: list[str], wanted: set[str]) -> tuple[int, int]:
    """Return ``(distinct_matched, window_size)`` for the tightest cluster of ``wanted``.

    Uses a sliding window that expands to include a new distinct term and contracts from the
    left while still holding every term it has seen, which is the standard linear-time
    minimum-window approach.
    """
    positions = [(index, token) for index, token in enumerate(tokens) if token in wanted]
    if not positions:
        return 0, 0
    if len(positions) == 1:
        return 1, 1

    best_distinct = 1
    best_window = 1
    counts: dict[str, int] = {}
    left = 0
    for right in range(len(positions)):
        token = positions[right][1]
        counts[token] = counts.get(token, 0) + 1
        while True:
            left_token = positions[left][1]
            if counts[left_token] > 1:
                counts[left_token] -= 1
                left += 1
            else:
                break
        distinct = len(counts)
        window = positions[right][0] - positions[left][0] + 1
        if distinct > best_distinct or (distinct == best_distinct and window < best_window):
            best_distinct = distinct
            best_window = window
    return best_distinct, best_window


def _contains_phrase(tokens: list[str], phrase: list[str]) -> bool:
    """True when ``phrase`` appears as a contiguous run in ``tokens``."""
    if not phrase or len(phrase) > len(tokens):
        return False
    first = phrase[0]
    span = len(phrase)
    for index, token in enumerate(tokens):
        if token == first and tokens[index : index + span] == phrase:
            return True
    return False


def rerank_score(
    *,
    bm25_relative: float,
    coverage: float,
    proximity: float,
    phrase_match: bool,
) -> float:
    """Blend the four signals into a single score in ``[0.0, 1.0]``."""
    score = (
        WEIGHT_COVERAGE * coverage + WEIGHT_BM25 * bm25_relative + WEIGHT_PROXIMITY * proximity
    )
    if phrase_match:
        score += PHRASE_BONUS
    return min(score, 1.0)


class Retriever:
    """Runs the two-stage search against a :class:`DocumentStore`."""

    def __init__(
        self,
        store: DocumentStore,
        *,
        candidate_pool: int = 50,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.store = store
        self.candidate_pool = candidate_pool
        self.k1 = k1
        self.b = b

    def search(self, question: str, *, top_k: int = 5) -> list[RetrievedPassage]:
        """Return the ``top_k`` best passages for ``question``, best first.

        An empty list means either an empty corpus or a question whose terms appear nowhere;
        callers must treat that as "no evidence" rather than retrying with a lower bar.
        """
        query_terms = tokenize(question)
        if not query_terms:
            logger.debug("question contained no indexable terms after normalisation")
            return []

        distinct_terms = list(dict.fromkeys(query_terms))
        stats = self.store.collection_stats()
        if stats.chunk_count == 0:
            return []

        postings = self.store.postings_for(distinct_terms)
        candidate_ids = {chunk_id for entries in postings.values() for chunk_id, _ in entries}
        if not candidate_ids:
            return []

        scored = score_query(
            query_terms=distinct_terms,
            postings=postings,
            document_frequencies=self.store.document_frequencies(distinct_terms),
            chunk_lengths=self.store.chunk_lengths(sorted(candidate_ids)),
            chunk_count=stats.chunk_count,
            average_length=stats.average_length,
            k1=self.k1,
            b=self.b,
        )
        if not scored:
            return []

        pool = scored[: self.candidate_pool]
        records = self.store.get_chunks([item.chunk_id for item in pool])
        # Relative to the best candidate: BM25 is unbounded, so an absolute threshold on it
        # would be meaningless across corpora of different sizes.
        best_bm25 = max(item.score for item in pool) or 1.0
        wanted = set(distinct_terms)
        phrase = query_terms

        passages: list[RetrievedPassage] = []
        for item in pool:
            record = records.get(item.chunk_id)
            if record is None:  # pragma: no cover - only on concurrent deletion
                continue
            chunk_tokens = tokenize(record.text)
            distinct_matched, window = _smallest_window(chunk_tokens, wanted)
            coverage = distinct_matched / len(wanted)
            if distinct_matched >= 2 and window > 0:
                proximity = min(distinct_matched / window, 1.0)
            elif distinct_matched == 1:
                # A single matched term has no meaningful spread; score it neutrally rather
                # than rewarding or punishing it.
                proximity = 0.5
            else:
                proximity = 0.0
            phrase_match = _contains_phrase(chunk_tokens, phrase)
            passages.append(
                RetrievedPassage(
                    chunk=record,
                    score=rerank_score(
                        bm25_relative=item.score / best_bm25,
                        coverage=coverage,
                        proximity=proximity,
                        phrase_match=phrase_match,
                    ),
                    bm25_score=item.score,
                    coverage=coverage,
                    proximity=proximity,
                    phrase_match=phrase_match,
                    matched_terms=item.matched_terms,
                )
            )

        passages.sort(key=lambda passage: (-passage.score, passage.chunk.id))
        return passages[:top_k]
