"""Tests for chunking: sizes, overlap, sentence integrity, and termination."""

from __future__ import annotations

import pytest

from mustanad.text.chunk import chunk_text
from mustanad.text.normalize import tokenize


def _sentences(count: int, words_each: int = 10) -> str:
    return " ".join(
        " ".join(f"word{index}x{position}" for position in range(words_each)) + "."
        for index in range(count)
    )


def test_short_text_becomes_one_chunk() -> None:
    chunks = chunk_text("A short policy sentence.", target_tokens=100, overlap_tokens=10)
    assert len(chunks) == 1
    assert chunks[0].ordinal == 0
    assert chunks[0].text == "A short policy sentence."


def test_empty_and_symbol_only_text_produce_no_chunks() -> None:
    assert chunk_text("", target_tokens=50, overlap_tokens=5) == []
    assert chunk_text("...   ###   ", target_tokens=50, overlap_tokens=5) == []


def test_long_text_is_split_into_multiple_ordered_chunks() -> None:
    chunks = chunk_text(_sentences(20), target_tokens=40, overlap_tokens=10)
    assert len(chunks) > 1
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_chunks_respect_the_target_size_within_one_sentence() -> None:
    """A chunk may overshoot by the final sentence, but never by more than that."""
    target = 40
    chunks = chunk_text(_sentences(30, words_each=8), target_tokens=target, overlap_tokens=8)
    # Every sentence here is 8 tokens, so a chunk can reach target + 8 at worst.
    assert all(chunk.token_count <= target + 8 for chunk in chunks)


def test_overlap_repeats_content_between_neighbours() -> None:
    chunks = chunk_text(_sentences(12), target_tokens=40, overlap_tokens=12)
    assert len(chunks) >= 2
    first = set(chunks[0].tokens)
    second = set(chunks[1].tokens)
    assert first & second, "consecutive chunks should share the carried-over sentence"


def test_zero_overlap_produces_disjoint_chunks() -> None:
    chunks = chunk_text(_sentences(12), target_tokens=40, overlap_tokens=0)
    assert len(chunks) >= 2
    assert not set(chunks[0].tokens) & set(chunks[1].tokens)


def test_sentences_are_not_cut_mid_way_when_they_fit() -> None:
    """Readable quotations are the point of citations, so boundaries land between sentences."""
    text = "First sentence here. Second sentence here. Third sentence here."
    chunks = chunk_text(text, target_tokens=6, overlap_tokens=0)
    for chunk in chunks:
        assert chunk.text.endswith("."), chunk.text


def test_a_single_over_long_sentence_is_hard_split_and_terminates() -> None:
    """One 500-word sentence with no full stop must still chunk, and must not loop."""
    monster = " ".join(f"token{index}" for index in range(500))
    chunks = chunk_text(monster, target_tokens=50, overlap_tokens=10)
    assert len(chunks) >= 8
    assert sum(chunk.token_count for chunk in chunks) >= 500


def test_page_number_is_attached_to_every_chunk() -> None:
    chunks = chunk_text(_sentences(6), target_tokens=30, overlap_tokens=5, page=7)
    assert chunks
    assert {chunk.page for chunk in chunks} == {7}


def test_start_ordinal_continues_numbering_across_pages() -> None:
    chunks = chunk_text(_sentences(6), target_tokens=30, overlap_tokens=5, start_ordinal=4)
    assert chunks[0].ordinal == 4


def test_tokens_are_normalised_not_raw() -> None:
    chunks = chunk_text("الإجازة السنوية مدفوعة.", target_tokens=50, overlap_tokens=5)
    assert chunks[0].tokens == tuple(tokenize("الإجازة السنوية مدفوعة"))
    # ...while the displayed text stays verbatim, diacritics and all.
    assert chunks[0].text == "الإجازة السنوية مدفوعة."


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="target_tokens must be positive"):
        chunk_text("text", target_tokens=0, overlap_tokens=0)
    with pytest.raises(ValueError, match="overlap_tokens must be smaller"):
        chunk_text("text", target_tokens=10, overlap_tokens=10)
