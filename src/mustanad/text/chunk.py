"""Splitting documents into retrievable chunks.

Chunk boundaries are chosen at *sentence* granularity but sized in *tokens*. Cutting
mid-sentence is what makes retrieved quotations unreadable, and readable quotations are the
whole point of a cited answer -- so a sentence is never split unless it is on its own longer
than the target chunk size.
"""

from __future__ import annotations

from dataclasses import dataclass

from .normalize import split_sentences, tokenize


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable passage.

    Attributes:
        ordinal: Zero-based position within its document, used for stable citation labels.
        text: Verbatim original text, shown to the user.
        tokens: Normalised tokens used for indexing and scoring.
        page: 1-based source page when the loader knew it (PDFs), else ``None``.
    """

    ordinal: int
    text: str
    tokens: tuple[str, ...]
    page: int | None = None

    @property
    def token_count(self) -> int:
        return len(self.tokens)


def _hard_split(sentence: str, size: int) -> list[str]:
    """Break a single over-long sentence on whitespace into ``size``-word pieces."""
    words = sentence.split()
    if not words:
        return []
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]


def chunk_text(
    text: str,
    *,
    target_tokens: int,
    overlap_tokens: int,
    page: int | None = None,
    start_ordinal: int = 0,
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks of roughly ``target_tokens`` tokens.

    Overlap is applied by carrying whole trailing sentences from the previous chunk into the
    next one, so a fact that straddles a boundary is fully present in at least one chunk.

    Args:
        text: Raw text of one page or one document.
        target_tokens: Desired chunk size, counted in normalised tokens.
        overlap_tokens: Approximate number of tokens to repeat from the previous chunk.
        page: Source page number to attach to every chunk produced here.
        start_ordinal: Ordinal to assign the first chunk.

    Returns:
        Chunks in document order. Empty if ``text`` holds no indexable tokens.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be smaller than target_tokens")

    sentences: list[str] = []
    for sentence in split_sentences(text):
        if len(tokenize(sentence)) > target_tokens:
            sentences.extend(_hard_split(sentence, target_tokens))
        else:
            sentences.append(sentence)

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_tokens = 0
    ordinal = start_ordinal

    def flush() -> list[str]:
        """Emit the buffered sentences as a chunk; return the sentences to carry over."""
        nonlocal ordinal
        body = " ".join(buffer).strip()
        tokens = tuple(tokenize(body))
        if tokens:
            chunks.append(Chunk(ordinal=ordinal, text=body, tokens=tokens, page=page))
            ordinal += 1
        if overlap_tokens == 0:
            return []
        carried: list[str] = []
        carried_tokens = 0
        for sentence in reversed(buffer):
            count = len(tokenize(sentence))
            if carried_tokens + count > overlap_tokens and carried:
                break
            carried.insert(0, sentence)
            carried_tokens += count
        # Never carry the entire buffer: that would loop forever on a single long sentence.
        if len(carried) == len(buffer):
            carried = carried[1:]
        return carried

    for sentence in sentences:
        count = len(tokenize(sentence))
        if buffer and buffer_tokens + count > target_tokens:
            buffer = flush()
            buffer_tokens = sum(len(tokenize(item)) for item in buffer)
        buffer.append(sentence)
        buffer_tokens += count

    if buffer:
        flush()

    return chunks
