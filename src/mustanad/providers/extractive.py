"""The default provider: deterministic extractive answering, no model and no network.

This exists for three reasons, in order of importance:

1. **The product must work with no API key.** A buyer evaluating this on their own laptop, or
   a reviewer reading the repository, gets a working system immediately.
2. **Tests must be deterministic.** Every retrieval and answer-shape test in this repository
   runs against this provider, so the suite never depends on a paid endpoint or on a
   generative model's mood.
3. **It is genuinely useful.** For policy and manual lookups -- "how many days of annual
   leave?" -- the sentence containing the answer *is* the answer, and quoting it verbatim is
   more trustworthy than paraphrasing it.

What it does **not** do is synthesise across passages, resolve pronouns, or rewrite for
fluency. It selects and quotes. The README says so plainly, and the API marks the answer as
extractive so a caller can never mistake a quotation for a generated summary.
"""

from __future__ import annotations

from ..text.normalize import is_probably_arabic, split_sentences, tokenize
from .base import AnswerProvider, AnswerRequest, AnswerResult

#: Sentences quoted in one answer, at most.
MAX_QUOTED_SENTENCES = 3
#: A candidate sentence must share at least this fraction of the question's terms.
MIN_SENTENCE_OVERLAP = 0.2


class ExtractiveProvider(AnswerProvider):
    """Selects the sentences that best overlap the question and quotes them with citations."""

    name = "extractive"

    def answer(self, request: AnswerRequest) -> AnswerResult:
        question_terms = set(tokenize(request.question))
        if not request.passages:
            return AnswerResult(
                answer=_no_evidence_message(request.question),
                provider=self.name,
                grounded=False,
            )

        candidates: list[tuple[float, int, int, str]] = []
        for passage_index, passage in enumerate(request.passages, start=1):
            for sentence_index, sentence in enumerate(split_sentences(passage.chunk.text)):
                sentence_terms = set(tokenize(sentence))
                if not sentence_terms or not question_terms:
                    continue
                overlap = len(question_terms & sentence_terms) / len(question_terms)
                if overlap < MIN_SENTENCE_OVERLAP:
                    continue
                # Rank by question overlap first, then prefer earlier passages (better
                # retrieval rank) and earlier sentences (usually the topic sentence).
                candidates.append((overlap, -passage_index, -sentence_index, sentence))

        if not candidates:
            # Retrieval found the terms somewhere in the chunk, but no single sentence carries
            # enough of the question. Quoting the best chunk's opening is more useful than
            # silence, and the low confidence is reported honestly.
            best = request.passages[0]
            opening = " ".join(split_sentences(best.chunk.text)[:2]) or best.chunk.text[:400]
            return AnswerResult(
                answer=(
                    "I did not find a sentence that directly answers this. The closest "
                    f"passage says: [1] {opening}"
                ),
                cited_indices=(1,),
                provider=self.name,
                grounded=True,
                warnings=("low_confidence_no_matching_sentence",),
            )

        candidates.sort(reverse=True)
        chosen = candidates[:MAX_QUOTED_SENTENCES]

        parts: list[str] = []
        cited: list[int] = []
        seen: set[str] = set()
        for _, negative_passage_index, _, sentence in chosen:
            passage_index = -negative_passage_index
            normalised = " ".join(tokenize(sentence))
            if normalised in seen:
                continue
            seen.add(normalised)
            parts.append(f"[{passage_index}] {sentence.strip()}")
            if passage_index not in cited:
                cited.append(passage_index)

        return AnswerResult(
            answer=" ".join(parts),
            cited_indices=tuple(sorted(cited)),
            provider=self.name,
            grounded=True,
        )


def _no_evidence_message(question: str) -> str:
    """Bilingual "not found" message, matched to the language of the question."""
    if is_probably_arabic(question):
        return "لم أجد في المستندات المُفهرسة ما يجيب على هذا السؤال."
    return "I could not find anything in the indexed documents that answers this question."
