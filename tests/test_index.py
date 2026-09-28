"""Tests for the SQLite store and the BM25 scorer."""

from __future__ import annotations

import pytest

from mustanad.index.bm25 import inverse_document_frequency, score_query
from mustanad.index.store import DocumentStore, content_hash
from mustanad.text.chunk import chunk_text
from mustanad.text.normalize import tokenize


def _chunks(text: str) -> list:
    return chunk_text(text, target_tokens=60, overlap_tokens=10)


def test_add_and_list_document(store: DocumentStore) -> None:
    chunks = _chunks("Annual leave is 22 days per year. Sick leave is 15 days.")
    document = store.add_document(
        source="memory",
        title="Handbook",
        sha256=content_hash("a"),
        language="en",
        chunks=chunks,
    )
    assert document.id > 0
    assert document.n_chunks == len(chunks)
    assert [record.title for record in store.list_documents()] == ["Handbook"]


def test_duplicate_hash_is_rejected(store: DocumentStore) -> None:
    chunks = _chunks("Some indexable content here.")
    store.add_document(
        source="a", title="A", sha256=content_hash("same"), language="en", chunks=chunks
    )
    with pytest.raises(ValueError, match="already indexed"):
        store.add_document(
            source="b", title="B", sha256=content_hash("same"), language="en", chunks=chunks
        )


def test_document_with_no_chunks_is_rejected(store: DocumentStore) -> None:
    with pytest.raises(ValueError, match="no indexable chunks"):
        store.add_document(
            source="a", title="A", sha256=content_hash("x"), language="en", chunks=[]
        )


def test_delete_cascades_to_chunks_and_postings(store: DocumentStore) -> None:
    """Without `PRAGMA foreign_keys = ON`, deleting a document orphans its index entries.

    Orphaned postings are worse than a leak: they keep matching queries and return passages
    the operator believes they deleted.
    """
    document = store.add_document(
        source="a",
        title="A",
        sha256=content_hash("x"),
        language="en",
        chunks=_chunks("Confidential salary information appears here."),
    )
    assert store.collection_stats().chunk_count > 0
    assert store.document_frequencies(["confidential"])["confidential"] > 0

    assert store.delete_document(document.id) is True

    assert store.collection_stats().chunk_count == 0
    assert store.document_frequencies(["confidential"]) == {}
    assert store.list_documents() == []


def test_delete_missing_document_returns_false(store: DocumentStore) -> None:
    assert store.delete_document(4242) is False


def test_collection_stats_are_derived_and_stay_correct_after_delete(store: DocumentStore) -> None:
    first = store.add_document(
        source="a",
        title="A",
        sha256=content_hash("1"),
        language="en",
        chunks=_chunks("Alpha beta gamma delta."),
    )
    store.add_document(
        source="b",
        title="B",
        sha256=content_hash("2"),
        language="en",
        chunks=_chunks("Epsilon zeta eta theta."),
    )
    before = store.collection_stats()
    assert before.chunk_count == 2
    assert before.average_length > 0

    store.delete_document(first.id)
    after = store.collection_stats()
    assert after.chunk_count == 1
    assert after.total_tokens < before.total_tokens


def test_empty_collection_has_zero_average_length(store: DocumentStore) -> None:
    stats = store.collection_stats()
    assert stats.chunk_count == 0
    assert stats.average_length == 0.0


def test_postings_and_lengths_round_trip(store: DocumentStore) -> None:
    store.add_document(
        source="a",
        title="A",
        sha256=content_hash("1"),
        language="en",
        chunks=_chunks("reimbursement reimbursement receipt"),
    )
    postings = store.postings_for(["reimbursement", "receipt", "absent"])
    assert "absent" not in postings
    ((chunk_id, term_frequency),) = postings["reimbursement"]
    assert term_frequency == 2, "term frequency must count repeats within a chunk"
    assert store.chunk_lengths([chunk_id])[chunk_id] == 3


def test_empty_term_queries_do_not_hit_the_database(store: DocumentStore) -> None:
    assert store.postings_for([]) == {}
    assert store.document_frequencies([]) == {}
    assert store.chunk_lengths([]) == {}
    assert store.get_chunks([]) == {}


def test_arabic_content_is_indexed_in_normalised_stemmed_form(store: DocumentStore) -> None:
    """Index terms are the output of `tokenize`, so queries must be tokenised the same way."""
    store.add_document(
        source="a",
        title="لائحة",
        sha256=content_hash("ar"),
        language="ar",
        chunks=_chunks("الإجازة السنوية اثنان وعشرون يومًا."),
    )
    # The stored term for "الإجازة" is its light stem, reachable from any inflected spelling.
    for spelling in ["الإجازة", "الاجازة", "إجازات", "اجازه"]:
        (term,) = tokenize(spelling)
        assert store.document_frequencies([term]).get(term) == 1, spelling


def test_a_query_term_absent_from_the_corpus_has_no_frequency(store: DocumentStore) -> None:
    store.add_document(
        source="a",
        title="A",
        sha256=content_hash("ar2"),
        language="ar",
        chunks=_chunks("الإجازة السنوية مدفوعة."),
    )
    assert store.document_frequencies(["كسكس"]) == {}


# --------------------------------------------------------------------------- BM25


def test_idf_is_always_positive() -> None:
    """The Lucene IDF form never goes negative, even for a term in every chunk."""
    assert inverse_document_frequency(100, 100) > 0
    assert inverse_document_frequency(1, 100) > inverse_document_frequency(50, 100)


def test_idf_of_empty_collection_is_zero() -> None:
    assert inverse_document_frequency(0, 0) == 0.0


def test_rare_terms_outrank_common_ones() -> None:
    results = score_query(
        query_terms=["common", "rare"],
        postings={"common": [(1, 1)], "rare": [(2, 1)]},
        document_frequencies={"common": 90, "rare": 1},
        chunk_lengths={1: 10, 2: 10},
        chunk_count=100,
        average_length=10.0,
    )
    assert [item.chunk_id for item in results] == [2, 1]


def test_term_frequency_saturates() -> None:
    """Doubling term frequency must not double the score -- that is what k1 is for."""

    def score_for(term_frequency: int) -> float:
        return score_query(
            query_terms=["x"],
            postings={"x": [(1, term_frequency)]},
            document_frequencies={"x": 1},
            chunk_lengths={1: 100},
            chunk_count=10,
            average_length=100.0,
        )[0].score

    low, high = score_for(2), score_for(20)
    assert high > low
    assert high < low * 4, "term frequency should saturate, not scale linearly"


def test_shorter_chunks_score_higher_for_the_same_term_frequency() -> None:
    results = score_query(
        query_terms=["x"],
        postings={"x": [(1, 2), (2, 2)]},
        document_frequencies={"x": 2},
        chunk_lengths={1: 20, 2: 200},
        chunk_count=10,
        average_length=100.0,
    )
    assert [item.chunk_id for item in results] == [1, 2]


def test_matched_terms_are_reported() -> None:
    (result,) = score_query(
        query_terms=["a", "b", "missing"],
        postings={"a": [(1, 1)], "b": [(1, 1)]},
        document_frequencies={"a": 1, "b": 1},
        chunk_lengths={1: 10},
        chunk_count=10,
        average_length=10.0,
    )
    assert result.matched_terms == frozenset({"a", "b"})


def test_empty_collection_scores_nothing() -> None:
    assert (
        score_query(
            query_terms=["x"],
            postings={"x": [(1, 1)]},
            document_frequencies={"x": 1},
            chunk_lengths={1: 10},
            chunk_count=0,
            average_length=0.0,
        )
        == []
    )


def test_ties_break_deterministically_by_chunk_id() -> None:
    results = score_query(
        query_terms=["x"],
        postings={"x": [(9, 1), (3, 1), (5, 1)]},
        document_frequencies={"x": 3},
        chunk_lengths={3: 10, 5: 10, 9: 10},
        chunk_count=10,
        average_length=10.0,
    )
    assert [item.chunk_id for item in results] == [3, 5, 9]
