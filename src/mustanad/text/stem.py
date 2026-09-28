"""Light stemming for Arabic and English.

Why a stemmer is needed at all
------------------------------
Normalisation alone (see :mod:`.normalize`) makes ``الإجازة`` and ``الاجازة`` match, but it
cannot make ``مجانيًا`` match ``مجاني`` or ``المنتجات`` match ``المنتج``. Arabic attaches the
definite article, conjunctions and prepositions directly to the word and inflects heavily by
suffix, so a purely surface-level lexical index misses a large share of genuine matches. This
was not a theoretical concern here: measuring the bundled evaluation set showed real questions
failing for exactly that reason.

What this is, and what it is not
--------------------------------
This is a **light stemmer** in the Larkey/Darwish tradition: strip a known affix, keep a
minimum stem length, do not attempt morphological analysis. It is intentionally *not*:

* a root extractor -- ``استخراج`` is not reduced to ``خرج``, because root-level conflation
  merges unrelated words and hurts precision more than it helps recall;
* a lemmatiser -- it has no dictionary, so **broken (irregular) plurals such as**
  ``يوم``/``أيام`` **remain unmatched**. That limitation is real, is documented in the README,
  and is asserted by a test so it cannot be quietly forgotten.

Stemming is applied identically at index time and at query time, which is the only property
that actually matters for correctness.
"""

from __future__ import annotations

import re
from typing import Final

#: Never produce a stem shorter than this; below it, affix stripping destroys meaning and
#: starts merging unrelated words.
MIN_STEM_LENGTH: Final = 3

# Longest first: "وال" must be tried before "و", or the article would survive.
_ARABIC_PREFIXES: Final[tuple[str, ...]] = (
    "وال",
    "فال",
    "بال",
    "كال",
    "لل",
    "ال",
)

_ARABIC_SUFFIXES: Final[tuple[str, ...]] = (
    "اتها",
    "اتهم",
    "تها",
    "تهم",
    "نها",
    "كما",
    "هما",
    "ات",
    "ون",
    "ين",
    "ان",
    "وا",
    "هم",
    "هن",
    "ها",
    "كم",
    "كن",
    "نا",
    "يه",
    "ه",
    "ي",
    "ا",
)

_ARABIC_LETTER: Final = re.compile(r"[؀-ۿ]")

_ENGLISH_KEEP_TRAILING_S: Final = frozenset({"ss", "us", "is"})


def _is_arabic_token(token: str) -> bool:
    return bool(_ARABIC_LETTER.search(token))


#: Upper bound on suffix-stripping passes. The MIN_STEM_LENGTH guard already guarantees
#: termination; this makes that guarantee independent of the suffix table's contents.
_MAX_SUFFIX_PASSES: Final = 3


def _stem_arabic(token: str) -> str:
    """Strip one prefix, then suffixes repeatedly, honouring :data:`MIN_STEM_LENGTH`.

    Suffix stripping **must** iterate to reach a fixed point. Stripping only once is not
    idempotent, and non-idempotent stemming silently breaks retrieval: ``مجانيًا`` would lose
    its final alef to become ``مجاني``, while the bare form ``مجاني`` would lose its final yeh
    to become ``مجان`` -- two spellings of one word landing on two different index terms,
    matching neither each other nor a query. Iterating sends both to ``مجان``.
    """
    stem = token
    for prefix in _ARABIC_PREFIXES:
        if stem.startswith(prefix) and len(stem) - len(prefix) >= MIN_STEM_LENGTH:
            stem = stem[len(prefix) :]
            break
    for _ in range(_MAX_SUFFIX_PASSES):
        for suffix in _ARABIC_SUFFIXES:
            if stem.endswith(suffix) and len(stem) - len(suffix) >= MIN_STEM_LENGTH:
                stem = stem[: -len(suffix)]
                break
        else:
            break
    return stem


def _stem_english(token: str) -> str:
    """A deliberately tiny plural rule. No Porter stemmer, no dependency, no surprises."""
    if len(token) <= MIN_STEM_LENGTH or not token.endswith("s"):
        return token
    if token[-2:] in _ENGLISH_KEEP_TRAILING_S:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    return token[:-1]


def light_stem(token: str) -> str:
    """Reduce ``token`` to a light stem.

    The token is expected to be already normalised by
    :func:`mustanad.text.normalize.normalize`. Digits and mixed alphanumerics pass through
    untouched, because ``4500`` and ``v2`` must never be stemmed.

    >>> light_stem("مجانيا")
    'مجاني'
    >>> light_stem("المنتجات")
    'منتج'
    >>> light_stem("days")
    'day'
    >>> light_stem("business")
    'business'
    >>> light_stem("4500")
    '4500'
    """
    if not token or token.isdigit():
        return token
    if _is_arabic_token(token):
        return _stem_arabic(token)
    if token.isalpha():
        return _stem_english(token)
    return token
