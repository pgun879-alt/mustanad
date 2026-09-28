"""Tests for Arabic/English normalisation -- the layer most naive pipelines get wrong."""

from __future__ import annotations

import pytest

from mustanad.text.normalize import (
    is_probably_arabic,
    normalize,
    split_sentences,
    tokenize,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Diacritics are stripped: the same word written vocalised and unvocalised must match.
        ("مُسْتَنَدٌ", "مستند"),
        ("كِتَاب", "كتاب"),
        # All four alef forms fold together.
        ("أحمد", "احمد"),
        ("إجازة", "اجازه"),
        ("آخر", "اخر"),
        # Teh marbuta folds to heh, alef maksura to yeh.
        ("سياسة", "سياسه"),
        ("على", "علي"),
        # Tatweel (kashida) stretching is decoration, not identity.
        ("الـــكتاب", "الكتاب"),
        # Hamza carriers fold to their base letter.
        ("مسؤول", "مسوول"),
        ("شيئ", "شيي"),
        # Latin text is case-folded.
        ("Annual Leave", "annual leave"),
        ("", ""),
    ],
)
def test_normalize_folds_equivalent_forms(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_normalize_preserves_digits_in_both_numeral_systems() -> None:
    """Regression guard: a careless tashkeel range deletes Arabic-Indic digits.

    U+0660..U+0669 sit immediately after the combining-mark block, so the obvious-looking
    range U+0653..U+0670 silently removes every Arabic numeral. Numbers are exactly what
    policy questions ask about, so losing them would be silent and severe.
    """
    assert normalize("رقم ٢٠٢٥") == "رقم 2025"
    assert normalize("٤٥٠٠ دينار") == "4500 دينار"
    assert normalize("Invoice 2025") == "invoice 2025"
    assert "22" in normalize("اثنان وعشرون ٢٢ يومًا")


def test_normalize_resolves_presentation_ligatures() -> None:
    """NFKC turns the lam-alef presentation form into its two base letters."""
    assert normalize("ﻻ") == "لا"


def test_normalize_collapses_whitespace() -> None:
    assert normalize("  two \n\t  words  ") == "two words"


def test_vocalised_and_plain_text_produce_identical_tokens() -> None:
    """The practical payoff: a vocalised document is searchable with a plain-text query."""
    assert tokenize("الإجازةُ السَّنويّة") == tokenize("الاجازة السنوية")


def test_tokenize_drops_stopwords_by_default() -> None:
    assert tokenize("The invoice from the supplier") == ["invoice", "supplier"]
    assert "من" not in tokenize("ما هي مدة الإجازة من فضلك")


def test_tokenize_can_keep_stopwords_for_positional_work() -> None:
    kept = tokenize("the annual leave", drop_stopwords=False)
    assert kept == ["the", "annual", "leave"]


def test_tokenize_splits_on_punctuation_and_symbols() -> None:
    assert tokenize("leave-policy_v2 (final)!") == ["leave", "policy", "v2", "final"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("مرحبا بالعالم", True),
        ("hello world", False),
        ("mustanad مستند", True),  # mixed text counts as Arabic above the threshold
        ("12345", False),
        ("", False),
    ],
)
def test_is_probably_arabic(text: str, expected: bool) -> None:
    assert is_probably_arabic(text) is expected


def test_split_sentences_handles_both_scripts() -> None:
    sentences = split_sentences("First one. ثانيا؟ Third! ورابعا؛ خامسا")
    assert sentences == ["First one.", "ثانيا؟", "Third!", "ورابعا؛", "خامسا"]


def test_split_sentences_drops_punctuation_only_fragments() -> None:
    assert split_sentences("...  ???  ") == []


def test_split_sentences_treats_blank_lines_as_boundaries() -> None:
    assert split_sentences("Heading\n\nBody text") == ["Heading", "Body text"]
