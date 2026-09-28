"""Arabic- and English-aware text normalisation and tokenisation.

Why this module exists
----------------------
Lexical retrieval matches *strings*. Arabic writes the same word many ways: with or without
diacritics, with any of four alef forms, with tatweel stretching, with teh-marbuta where a
heh was meant, and with Arabic-Indic digits. A retriever that skips normalisation will fail
to match ``مُسْتَنَد`` against ``مستند`` -- which is the single most common reason naive RAG
pipelines perform badly on Arabic.

The transformations are intentionally *lossy and aggressive*: the normalised form is only
ever used for indexing and matching, never for display. Original text is always preserved
alongside it so quotes shown to the user are verbatim.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

from .stem import light_stem

# Combining marks that carry vocalisation but not identity: U+064B..U+065F (fathatan
# through the Quranic combining marks, including shadda), U+0670 (superscript alef), and
# U+06D6..U+06ED (Quranic annotation signs).
#
# These are written as explicit escapes rather than as literal characters on purpose: the
# obvious-looking literal range "\u0653-\u0670" silently swallows the Arabic-Indic digits
# at U+0660..U+0669, which would delete every number from Arabic text.
_TASHKEEL: Final = re.compile("[\u064b-\u065f\u0670\u06d6-\u06ed]")
_TATWEEL: Final = "ـ"

# Letter folding. Each group collapses to one canonical form.
_LETTER_FOLDING: Final[dict[str, str]] = {
    "آ": "ا",  # alef with madda       آ -> ا
    "أ": "ا",  # alef with hamza above أ -> ا
    "إ": "ا",  # alef with hamza below إ -> ا
    "ٱ": "ا",  # alef wasla           ٱ -> ا
    "ى": "ي",  # alef maksura         ى -> ي
    "ئ": "ي",  # yeh with hamza       ئ -> ي
    "ؤ": "و",  # waw with hamza       ؤ -> و
    "ة": "ه",  # teh marbuta          ة -> ه
    "ک": "ك",  # keheh (Persian)      ک -> ك
    "گ": "ك",  # gaf                  گ -> ك
    "ی": "ي",  # farsi yeh            ی -> ي
}

# Arabic-Indic (U+0660..) and Eastern Arabic-Indic (U+06F0..) digits to ASCII.
_DIGIT_FOLDING: Final[dict[str, str]] = {
    **{chr(0x0660 + i): str(i) for i in range(10)},
    **{chr(0x06F0 + i): str(i) for i in range(10)},
}

_TRANSLATION_TABLE: Final = str.maketrans({**_LETTER_FOLDING, **_DIGIT_FOLDING, _TATWEEL: None})

# A token is a run of letters (any script) or digits. Everything else -- punctuation,
# underscores, symbols -- separates tokens.
_TOKEN_RE: Final = re.compile(r"[^\W_]+", re.UNICODE)

_WHITESPACE_RE: Final = re.compile(r"\s+")

# Deliberately short, high-frequency-only stop lists. Aggressive stop-word removal hurts
# phrase queries, so these cover only words that carry no retrieval signal at all.
ARABIC_STOPWORDS: Final[frozenset[str]] = frozenset(
    """
    من الى على عن في مع هذا هذه ذلك تلك التي الذي الذين ما لا لم لن ان انه انها كان كانت
    يكون تكون قد و او ثم حتى كل بعض غير بين عند لدى هو هي هم هن نحن انا انت اي ايضا بعد
    قبل هناك هنالك مثل لكن بل اذا لو كما حيث سوف س له لها لهم به بها بهم
    """.split()
)

ENGLISH_STOPWORDS: Final[frozenset[str]] = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for from by with
    without about as is are was were be been being do does did doing have has had having it
    its i you he she they we not no nor so such can could will would shall should may might
    must there here what which who whom whose when where why how all any both each more most
    other some only own same too very s t just don now
    """.split()
)

STOPWORDS: Final[frozenset[str]] = ARABIC_STOPWORDS | ENGLISH_STOPWORDS

_ARABIC_BLOCK: Final = re.compile(r"[؀-ۿݐ-ݿ]")


def normalize(text: str) -> str:
    """Return the canonical matching form of ``text``.

    Steps, in order: Unicode NFKC (unifies presentation forms and ligatures), diacritic
    removal, tatweel removal, letter and digit folding, ASCII case folding, whitespace
    collapse.

    >>> normalize("مُسْتَنَدٌ")
    'مستند'
    >>> normalize("الإجازة")
    'الاجازه'
    >>> normalize("Invoice  #٤٢")
    'invoice #42'
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text)
    folded = _TASHKEEL.sub("", folded)
    folded = folded.translate(_TRANSLATION_TABLE)
    folded = folded.lower()
    return _WHITESPACE_RE.sub(" ", folded).strip()


def tokenize(text: str, *, drop_stopwords: bool = True, stem: bool = True) -> list[str]:
    """Normalise ``text`` and split it into index/query matching tokens.

    This is *the* tokenisation entry point: indexing, querying, proximity scoring and the
    extractive answerer all call it. That is deliberate -- if indexing and querying disagreed
    about tokenisation, retrieval would fail in ways no unit test on either side would catch.

    Args:
        text: Raw input in any script.
        drop_stopwords: Remove high-frequency function words. Disable when token *positions*
            in the original sequence matter.
        stem: Apply :func:`mustanad.text.stem.light_stem` so that inflected forms of the same
            word match. Disable to inspect surface forms.

    >>> tokenize("The invoice from the suppliers")
    ['invoice', 'supplier']
    >>> tokenize("ما هي مدة الإجازة السنوية؟")
    ['مده', 'اجاز', 'سنو']
    >>> tokenize("ما هي مدة الإجازة السنوية؟", stem=False)
    ['مده', 'الاجازه', 'السنويه']
    """
    tokens = _TOKEN_RE.findall(normalize(text))
    if drop_stopwords:
        # Stop words are matched on the *unstemmed* form, because the stop lists are written
        # as real words and stemming them would make the membership test unpredictable.
        tokens = [token for token in tokens if token not in STOPWORDS]
    if stem:
        return [light_stem(token) for token in tokens]
    return tokens


def is_probably_arabic(text: str, *, threshold: float = 0.2) -> bool:
    """True when at least ``threshold`` of the letters are in an Arabic Unicode block.

    Used only to pick a sentence splitter and to tag documents; nothing depends on it being
    exactly right.
    """
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False
    arabic = sum(1 for character in letters if _ARABIC_BLOCK.match(character))
    return arabic / len(letters) >= threshold


# Sentence terminators for both scripts, including the Arabic question mark (U+061F),
# Arabic comma-free full stop usage, and the Arabic semicolon (U+061B).
_SENTENCE_SPLIT_RE: Final = re.compile(r"(?<=[.!?؟؛۔])\s+|\n{2,}")


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into display-ready sentences, preserving the original characters.

    This is a pragmatic regex splitter, not a linguistic model: it errs toward longer
    fragments rather than cutting mid-clause, because fragments are shown to users as
    quotations.
    """
    pieces = (piece.strip() for piece in _SENTENCE_SPLIT_RE.split(text) if piece and piece.strip())
    return [piece for piece in pieces if any(character.isalnum() for character in piece)]
