"""BM25 ranking, implemented directly rather than pulled from a library.

The scoring function is the Okapi BM25 variant used by Lucene:

    score(d, q) = SUM over t in q of  idf(t) * ( tf(t,d) * (k1 + 1) )
                                     / ( tf(t,d) + k1 * (1 - b + b * |d| / avgdl) )

    idf(t) = ln( 1 + (N - df(t) + 0.5) / (df(t) + 0.5) )

Two properties of that ``idf`` form matter in practice:

* it is **always positive**, so a term appearing in more than half the corpus cannot drag a
  chunk's score below zero the way the textbook ``ln(N/df)`` form can;
* it degrades gracefully on a tiny corpus, which is the normal case here -- a buyer's policy
  handbook is a few hundred chunks, not a web crawl.

``b`` controls length normalisation (0 = ignore chunk length, 1 = fully normalise) and ``k1``
controls how fast term-frequency saturates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk id with its BM25 score and the query terms it actually matched."""

    chunk_id: int
    score: float
    matched_terms: frozenset[str]


def inverse_document_frequency(document_frequency: int, chunk_count: int) -> float:
    """BM25 IDF for a term seen in ``document_frequency`` of ``chunk_count`` chunks."""
    if chunk_count <= 0:
        return 0.0
    numerator = chunk_count - document_frequency + 0.5
    denominator = document_frequency + 0.5
    return math.log(1.0 + numerator / denominator)


def score_query(
    *,
    query_terms: Sequence[str],
    postings: Mapping[str, Sequence[tuple[int, int]]],
    document_frequencies: Mapping[str, int],
    chunk_lengths: Mapping[int, int],
    chunk_count: int,
    average_length: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[ScoredChunk]:
    """Score every chunk that contains at least one query term.

    Args:
        query_terms: Normalised query tokens. Repeats are collapsed -- BM25 as implemented
            here treats the query as a set, which is standard for short questions.
        postings: ``{term: [(chunk_id, term_frequency), ...]}``.
        document_frequencies: ``{term: number_of_chunks_containing_term}``.
        chunk_lengths: ``{chunk_id: length_in_tokens}``.
        chunk_count: Total chunks in the corpus (``N``).
        average_length: Mean chunk length in tokens (``avgdl``).
        k1: Term-frequency saturation parameter.
        b: Length-normalisation parameter.

    Returns:
        Chunks sorted by descending score, ties broken by ascending chunk id so results are
        deterministic across runs.
    """
    if chunk_count <= 0 or average_length <= 0:
        return []

    accumulated: dict[int, float] = {}
    matched: dict[int, set[str]] = {}

    for term in dict.fromkeys(query_terms):
        term_postings = postings.get(term)
        if not term_postings:
            continue
        idf = inverse_document_frequency(document_frequencies.get(term, 0), chunk_count)
        if idf <= 0.0:
            continue
        for chunk_id, term_frequency in term_postings:
            length = chunk_lengths.get(chunk_id)
            if length is None or length <= 0:
                continue
            normalisation = k1 * (1.0 - b + b * (length / average_length))
            contribution = idf * (term_frequency * (k1 + 1.0)) / (term_frequency + normalisation)
            accumulated[chunk_id] = accumulated.get(chunk_id, 0.0) + contribution
            matched.setdefault(chunk_id, set()).add(term)

    results = [
        ScoredChunk(chunk_id=chunk_id, score=score, matched_terms=frozenset(matched[chunk_id]))
        for chunk_id, score in accumulated.items()
    ]
    results.sort(key=lambda item: (-item.score, item.chunk_id))
    return results
