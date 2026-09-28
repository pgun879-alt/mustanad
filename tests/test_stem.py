"""Tests for the light stemmer, including the property that makes it correct at all."""

from __future__ import annotations

import pytest

from mustanad.text.normalize import tokenize
from mustanad.text.stem import MIN_STEM_LENGTH, light_stem


@pytest.mark.parametrize(
    ("first", "second"),
    [
        # Definite article and inflection: the forms a buyer's document and a buyer's
        # question actually use.
        ("المنتجات", "المنتج"),
        ("الاجازه", "اجازات"),
        ("السنويه", "سنوي"),
        ("يوما", "يوم"),
        ("شهرا", "شهر"),
        ("اداري", "اداره"),
        ("بيانات", "بيان"),
        ("مجانيا", "مجاني"),
        ("بالشحن", "الشحن"),
        # English plurals.
        ("days", "day"),
        ("tickets", "ticket"),
        ("policies", "policy"),
    ],
)
def test_related_forms_share_a_stem(first: str, second: str) -> None:
    assert light_stem(first) == light_stem(second), (
        f"{first} -> {light_stem(first)} but {second} -> {light_stem(second)}"
    )


@pytest.mark.parametrize(
    "word",
    ["مجانيا", "الاجازه", "السنويه", "يوما", "المنتجات", "days", "business", "tickets", "4500"],
)
def test_stemming_is_idempotent(word: str) -> None:
    """Stemming twice must equal stemming once.

    This is the property that a single-pass suffix stripper violates, and violating it breaks
    retrieval in a way that is invisible from either side alone: ``مجانيًا`` normalises to
    ``مجانيا`` and loses its alef to become ``مجاني``, while the bare ``مجاني`` loses its yeh
    to become ``مجان``. Two spellings of one word, two index terms, zero matches.
    """
    once = light_stem(word)
    assert light_stem(once) == once


def test_stem_never_goes_below_the_minimum_length() -> None:
    for word in ["علي", "الي", "له", "في", "يد", "ا"]:
        assert len(light_stem(word)) >= min(len(word), MIN_STEM_LENGTH)


def test_digits_and_alphanumerics_are_never_stemmed() -> None:
    assert light_stem("4500") == "4500"
    assert light_stem("2025") == "2025"
    assert light_stem("v2") == "v2"
    assert light_stem("99.5") == "99.5"


def test_english_double_s_words_keep_their_s() -> None:
    for word in ["business", "process", "access", "status", "analysis"]:
        assert light_stem(word) == word


def test_short_words_are_left_alone() -> None:
    assert light_stem("is") == "is"
    assert light_stem("as") == "as"


def test_unrelated_words_do_not_collide() -> None:
    """A root extractor would merge these; a light stemmer must not."""
    assert light_stem("كتاب") != light_stem("مكتب")
    assert light_stem("شحن") != light_stem("سحن")
    assert light_stem("invoice") != light_stem("invite")


def test_tokenize_applies_stemming_by_default_and_can_skip_it() -> None:
    assert tokenize("الإجازة السنوية") == ["اجاز", "سنو"]
    assert tokenize("الإجازة السنوية", stem=False) == ["الاجازه", "السنويه"]


def test_known_limitation_broken_plurals_do_not_match() -> None:
    """Documented gap: Arabic irregular plurals need a dictionary this stemmer does not have.

    ``يوم`` (day) and ``أيام`` (days) are a broken plural -- the plural is formed by changing
    the internal vowel pattern, not by adding a suffix -- so no affix-stripping stemmer can
    connect them. This test exists so the limitation stays visible and documented rather than
    being discovered later by a user, and so that adding a dictionary later has a target.
    """
    assert light_stem("يوم") != light_stem("ايام")
    assert light_stem("كتاب") != light_stem("كتب")
