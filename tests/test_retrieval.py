"""Retrieval tests, including a labelled end-to-end evaluation set.

The evaluation set below is the honest measurement of this system's quality. It runs against
the bundled sample corpus with the offline provider, so it is reproducible by anyone who clones
the repository -- no API key, no network, no fixtures that flatter the result.

Known failures are listed in :data:`KNOWN_FAILURES` with the reason, and are asserted to
*still fail*. If a change fixes one, this suite fails and forces the list to be updated. That
is deliberate: a portfolio that hides its failure cases is not demonstrating engineering.
"""

from __future__ import annotations

import pytest

from mustanad.retrieval import (
    PHRASE_BONUS,
    _contains_phrase,
    _smallest_window,
    rerank_score,
)
from mustanad.service import MustanadService

# (question, expected document title, substring that must appear in the answer)
EVALUATION_SET: list[tuple[str, str, str]] = [
    # --- English, factual lookups -------------------------------------------------
    ("How many days of annual leave do I get?", "employee handbook", "22 working days"),
    ("What is the per-diem for domestic travel?", "employee handbook", "4,500 DZD"),
    ("How long is the notice period for a team lead?", "employee handbook", "60 days"),
    ("When do carried-over leave days expire?", "employee handbook", "31 March"),
    ("Can I be reimbursed for a taxi without a receipt?", "employee handbook", "1,200"),
    ("What is the first response target for a critical issue?", "support sla", "1 hour"),
    ("How much service credit if uptime drops below 99%?", "support sla", "25%"),
    ("How long are tickets retained after closing?", "support sla", "24 months"),
    ("How do I escalate a missed response target?", "support sla", "ESCALATE"),
    # --- Arabic, factual lookups --------------------------------------------------
    ("ما هو بدل السفر الداخلي؟", "leave policy ar", "أربعة آلاف"),
    ("ما مدة إجازة الأمومة؟", "leave policy ar", "أربعة عشر أسبوع"),
    ("ما هي مدة الاسترجاع؟", "refund policy ar", "أربعة عشر يوم"),
    ("متى يكون الشحن مجانيا؟", "refund policy ar", "ثمانية آلاف"),
    ("ما مدة الضمان على الأجهزة الإلكترونية؟", "refund policy ar", "اثنا عشر شهر"),
    ("هل يمكن إرجاع بطاقة رقمية؟", "refund policy ar", "البطاقات الرقمية"),
    ("ما هي تكلفة الشحن العكسي؟", "refund policy ar", "ستمئة"),
]

#: Questions this system is currently known to get wrong, with the reason.
KNOWN_FAILURES: list[tuple[str, str]] = [
    (
        "كم عدد أيام الإجازة السنوية؟",
        "The question uses the broken plural أيام while the document says يوم. A "
        "dictionary-free light stemmer cannot connect them, so a neighbouring sentence about "
        "leave-request notice periods wins the sentence-selection step. The correct document "
        "is still retrieved -- only the quoted sentence is wrong.",
    ),
]

#: Questions with no answer anywhere in the corpus. The system must decline, not guess.
OUT_OF_SCOPE = [
    "What is the capital of Japan?",
    "كيف أطبخ الكسكس؟",
    "What is our policy on interplanetary shipping?",
]


@pytest.mark.parametrize(("question", "document", "expected"), EVALUATION_SET)
def test_evaluation_set(
    loaded_service: MustanadService, question: str, document: str, expected: str
) -> None:
    result = loaded_service.ask(question, top_k=3)
    assert result.grounded, f"declined to answer {question!r}"
    assert result.citations, "a grounded answer must carry citations"
    assert result.citations[0].document_title == document, (
        f"{question!r} retrieved {result.citations[0].document_title!r}"
    )
    assert expected in result.answer, f"{question!r} answered: {result.answer[:200]!r}"


@pytest.mark.parametrize(("question", "reason"), KNOWN_FAILURES)
def test_known_failures_still_fail(
    loaded_service: MustanadService, question: str, reason: str
) -> None:
    """Pin the documented gaps. Fixing one should break this test and update the docs."""
    result = loaded_service.ask(question, top_k=3)
    assert "وعشرين" not in result.answer, (
        f"{question!r} now succeeds -- remove it from KNOWN_FAILURES and update the README. "
        f"Recorded reason was: {reason}"
    )
    # Even when the quoted sentence is wrong, the right document must still be found.
    assert result.citations
    assert result.citations[0].document_title == "leave policy ar"


@pytest.mark.parametrize("question", OUT_OF_SCOPE)
def test_out_of_scope_questions_are_declined(
    loaded_service: MustanadService, question: str
) -> None:
    """The single most important behaviour: admit the gap instead of inventing an answer."""
    result = loaded_service.ask(question)
    assert result.grounded is False
    assert "insufficient_evidence" in result.warnings
    assert result.confidence < loaded_service.settings.min_coverage + 0.5


def test_arabic_question_is_declined_in_arabic(loaded_service: MustanadService) -> None:
    result = loaded_service.ask("كيف أطبخ الكسكس؟")
    assert "لم أجد" in result.answer


def test_english_question_is_declined_in_english(loaded_service: MustanadService) -> None:
    result = loaded_service.ask("What is the capital of Japan?")
    assert "could not find" in result.answer


def test_empty_corpus_returns_no_passages(service: MustanadService) -> None:
    result = service.ask("anything at all")
    assert result.grounded is False
    assert result.citations == ()


def test_top_k_limits_the_number_of_citations(loaded_service: MustanadService) -> None:
    assert len(loaded_service.ask("leave", top_k=1).citations) == 1
    assert len(loaded_service.ask("leave", top_k=3).citations) <= 3


def test_results_are_deterministic(loaded_service: MustanadService) -> None:
    """Same question, same corpus, same answer -- every time."""
    first = loaded_service.ask("What is the per-diem for domestic travel?")
    second = loaded_service.ask("What is the per-diem for domestic travel?")
    assert first.answer == second.answer
    assert [c.label for c in first.citations] == [c.label for c in second.citations]
    assert first.confidence == second.confidence


def test_citations_point_at_a_real_document(loaded_service: MustanadService) -> None:
    result = loaded_service.ask("How many days of annual leave do I get?")
    titles = {document.title for document in loaded_service.list_documents()}
    for citation in result.citations:
        assert citation.document_title in titles
        assert citation.excerpt
        assert 0.0 <= citation.score <= 1.0


def test_deleting_a_document_removes_it_from_answers(loaded_service: MustanadService) -> None:
    before = loaded_service.ask("What is the per-diem for domestic travel?")
    assert before.grounded
    target = next(
        document
        for document in loaded_service.list_documents()
        if document.title == "employee handbook"
    )
    assert loaded_service.delete_document(target.id)
    after = loaded_service.ask("What is the per-diem for domestic travel?")
    assert all(c.document_title != "employee handbook" for c in after.citations)


# --------------------------------------------------------------- rerank internals


def test_smallest_window_finds_the_tightest_cluster() -> None:
    tokens = ["a", "x", "y", "z", "b", "a", "b"]
    distinct, window = _smallest_window(tokens, {"a", "b"})
    assert distinct == 2
    assert window == 2  # the adjacent "a b" at the end, not the spread-out pair


def test_smallest_window_with_no_match() -> None:
    assert _smallest_window(["a", "b"], {"z"}) == (0, 0)


def test_smallest_window_with_one_match() -> None:
    assert _smallest_window(["a", "b", "c"], {"b"}) == (1, 1)


def test_smallest_window_prefers_more_distinct_terms_over_a_tighter_window() -> None:
    tokens = ["a", "a", "x", "x", "a", "b", "c"]
    distinct, window = _smallest_window(tokens, {"a", "b", "c"})
    assert distinct == 3
    assert window == 3


def test_contains_phrase() -> None:
    tokens = ["the", "annual", "leave", "balance"]
    assert _contains_phrase(tokens, ["annual", "leave"])
    assert not _contains_phrase(tokens, ["leave", "annual"])
    assert not _contains_phrase(tokens, ["annual", "holiday"])
    assert not _contains_phrase(tokens, [])
    assert not _contains_phrase(["short"], ["too", "long", "a", "phrase"])


def test_rerank_score_is_bounded_and_monotonic_in_coverage() -> None:
    low = rerank_score(bm25_relative=1.0, coverage=0.2, proximity=0.5, phrase_match=False)
    high = rerank_score(bm25_relative=1.0, coverage=1.0, proximity=0.5, phrase_match=False)
    assert 0.0 <= low < high <= 1.0


def test_phrase_match_adds_a_bonus_without_exceeding_one() -> None:
    without = rerank_score(bm25_relative=0.5, coverage=0.5, proximity=0.5, phrase_match=False)
    with_bonus = rerank_score(bm25_relative=0.5, coverage=0.5, proximity=0.5, phrase_match=True)
    assert with_bonus == pytest.approx(without + PHRASE_BONUS)
    assert rerank_score(bm25_relative=1.0, coverage=1.0, proximity=1.0, phrase_match=True) == 1.0


def test_perfect_match_scores_one() -> None:
    assert rerank_score(bm25_relative=1.0, coverage=1.0, proximity=1.0, phrase_match=False) == 1.0
