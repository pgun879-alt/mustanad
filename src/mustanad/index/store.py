"""SQLite persistence for documents, chunks and the inverted index.

Design notes
------------
* **One file, no server.** The whole index is a single SQLite file, which is what makes the
  demo a one-liner and the deployment a file copy.
* **Every statement is parameterised.** No SQL is ever built by string concatenation with
  caller data; the only interpolation is the fixed-length ``?`` placeholder run used for
  ``IN`` clauses, whose length comes from ``len()`` of an internal list.
* **Collection statistics are derived, not cached.** ``collection_stats`` aggregates on read.
  That is slightly slower than maintaining counters, but it cannot drift out of sync after a
  delete -- and a wrong average document length silently corrupts every BM25 score.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from ..text.chunk import Chunk

SCHEMA_VERSION: Final = 1

_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT    NOT NULL,
    title       TEXT    NOT NULL,
    sha256      TEXT    NOT NULL UNIQUE,
    language    TEXT    NOT NULL,
    n_chunks    INTEGER NOT NULL,
    n_tokens    INTEGER NOT NULL,
    ingested_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    page        INTEGER,
    text        TEXT    NOT NULL,
    n_tokens    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

CREATE TABLE IF NOT EXISTS postings (
    term     TEXT    NOT NULL,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    tf       INTEGER NOT NULL,
    PRIMARY KEY (term, chunk_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_postings_chunk ON postings(chunk_id);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """A stored source document."""

    id: int
    source: str
    title: str
    sha256: str
    language: str
    n_chunks: int
    n_tokens: int
    ingested_at: str


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """A stored chunk, joined with its document's display fields."""

    id: int
    document_id: int
    document_title: str
    document_source: str
    ordinal: int
    page: int | None
    text: str
    n_tokens: int


@dataclass(frozen=True, slots=True)
class CollectionStats:
    """Corpus-level statistics required by BM25."""

    chunk_count: int
    total_tokens: int

    @property
    def average_length(self) -> float:
        """Mean chunk length in tokens; 0.0 for an empty corpus."""
        if self.chunk_count == 0:
            return 0.0
        return self.total_tokens / self.chunk_count


def content_hash(text: str) -> str:
    """Stable content fingerprint used to detect re-ingestion of an unchanged document."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DocumentStore:
    """Owns the SQLite connection and all index reads and writes."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        if str(db_path) != ":memory:":
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(db_path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        # ON DELETE CASCADE is off by default in SQLite and must be enabled per connection,
        # otherwise deleting a document silently orphans its chunks and postings.
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = NORMAL")
        self._migrate()

    # -- lifecycle ---------------------------------------------------------------

    def _migrate(self) -> None:
        with self._transaction() as connection:
            connection.executescript(_SCHEMA)
            connection.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(SCHEMA_VERSION),),
            )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block in one transaction, rolling back on any exception."""
        try:
            with self._connection:
                yield self._connection
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"database error: {exc}") from exc

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> DocumentStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- writes ------------------------------------------------------------------

    def find_by_hash(self, sha256: str) -> DocumentRecord | None:
        row = self._connection.execute(
            "SELECT * FROM documents WHERE sha256 = ?", (sha256,)
        ).fetchone()
        return _to_document(row) if row else None

    def add_document(
        self,
        *,
        source: str,
        title: str,
        sha256: str,
        language: str,
        chunks: Sequence[Chunk],
    ) -> DocumentRecord:
        """Insert a document with its chunks and postings in a single transaction.

        Raises:
            ValueError: if ``chunks`` is empty or a document with this hash already exists.
        """
        if not chunks:
            raise ValueError("refusing to store a document with no indexable chunks")
        if self.find_by_hash(sha256) is not None:
            raise ValueError(f"document with hash {sha256[:12]} is already indexed")

        total_tokens = sum(chunk.token_count for chunk in chunks)
        ingested_at = datetime.now(UTC).isoformat(timespec="seconds")

        with self._transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO documents(source, title, sha256, language, n_chunks, n_tokens,"
                " ingested_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
                (source, title, sha256, language, len(chunks), total_tokens, ingested_at),
            )
            document_id = int(cursor.lastrowid or 0)

            for chunk in chunks:
                chunk_cursor = connection.execute(
                    "INSERT INTO chunks(document_id, ordinal, page, text, n_tokens)"
                    " VALUES(?, ?, ?, ?, ?)",
                    (document_id, chunk.ordinal, chunk.page, chunk.text, chunk.token_count),
                )
                chunk_id = int(chunk_cursor.lastrowid or 0)
                frequencies: dict[str, int] = {}
                for token in chunk.tokens:
                    frequencies[token] = frequencies.get(token, 0) + 1
                connection.executemany(
                    "INSERT INTO postings(term, chunk_id, tf) VALUES(?, ?, ?)",
                    [(term, chunk_id, tf) for term, tf in frequencies.items()],
                )

        stored = self.find_by_hash(sha256)
        assert stored is not None  # just inserted in a committed transaction
        return stored

    def delete_document(self, document_id: int) -> bool:
        """Delete a document and, by cascade, its chunks and postings."""
        with self._transaction() as connection:
            cursor = connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return cursor.rowcount > 0

    # -- reads -------------------------------------------------------------------

    def list_documents(self) -> list[DocumentRecord]:
        rows = self._connection.execute("SELECT * FROM documents ORDER BY id").fetchall()
        return [_to_document(row) for row in rows]

    def collection_stats(self) -> CollectionStats:
        row = self._connection.execute(
            "SELECT COUNT(*) AS chunk_count, COALESCE(SUM(n_tokens), 0) AS total FROM chunks"
        ).fetchone()
        return CollectionStats(chunk_count=int(row["chunk_count"]), total_tokens=int(row["total"]))

    def document_frequencies(self, terms: Sequence[str]) -> dict[str, int]:
        """Return, for each supplied term, the number of chunks containing it."""
        if not terms:
            return {}
        in_clause = _in_clause(len(terms))
        query = (
            f"SELECT term, COUNT(*) AS df FROM postings WHERE term IN ({in_clause})"  # noqa: S608 - placeholders only; see _in_clause
            " GROUP BY term"
        )
        rows = self._connection.execute(query, tuple(terms)).fetchall()
        return {str(row["term"]): int(row["df"]) for row in rows}

    def postings_for(self, terms: Sequence[str]) -> dict[str, list[tuple[int, int]]]:
        """Return ``{term: [(chunk_id, term_frequency), ...]}`` for the supplied terms."""
        if not terms:
            return {}
        in_clause = _in_clause(len(terms))
        query = f"SELECT term, chunk_id, tf FROM postings WHERE term IN ({in_clause})"  # noqa: S608 - placeholders only; see _in_clause
        rows = self._connection.execute(query, tuple(terms)).fetchall()
        result: dict[str, list[tuple[int, int]]] = {}
        for row in rows:
            result.setdefault(str(row["term"]), []).append((int(row["chunk_id"]), int(row["tf"])))
        return result

    def chunk_lengths(self, chunk_ids: Sequence[int]) -> dict[int, int]:
        if not chunk_ids:
            return {}
        in_clause = _in_clause(len(chunk_ids))
        query = f"SELECT id, n_tokens FROM chunks WHERE id IN ({in_clause})"  # noqa: S608 - placeholders only; see _in_clause
        rows = self._connection.execute(query, tuple(chunk_ids)).fetchall()
        return {int(row["id"]): int(row["n_tokens"]) for row in rows}

    def get_chunks(self, chunk_ids: Sequence[int]) -> dict[int, ChunkRecord]:
        """Fetch chunks with their document context, keyed by chunk id."""
        if not chunk_ids:
            return {}
        in_clause = _in_clause(len(chunk_ids))
        query = (
            "SELECT c.id, c.document_id, c.ordinal, c.page, c.text, c.n_tokens,"  # noqa: S608 - placeholders only; see _in_clause
            " d.title AS document_title, d.source AS document_source"
            " FROM chunks c JOIN documents d ON d.id = c.document_id"
            f" WHERE c.id IN ({in_clause})"
        )
        rows = self._connection.execute(query, tuple(chunk_ids)).fetchall()
        return {
            int(row["id"]): ChunkRecord(
                id=int(row["id"]),
                document_id=int(row["document_id"]),
                document_title=str(row["document_title"]),
                document_source=str(row["document_source"]),
                ordinal=int(row["ordinal"]),
                page=int(row["page"]) if row["page"] is not None else None,
                text=str(row["text"]),
                n_tokens=int(row["n_tokens"]),
            )
            for row in rows
        }


def _in_clause(count: int) -> str:
    """Return a placeholder run like ``?,?,?`` for a variable-length ``IN`` clause.

    SQLite cannot bind a list to a single parameter, so the *placeholder run* has to be
    interpolated into the SQL text. It is generated purely from ``count`` -- an ``int`` taken
    from ``len()`` of an internal list -- so no caller-supplied value ever reaches the query
    string. Every actual value is still bound as a parameter.

    This is the only interpolation anywhere in this module, which is why it lives in one
    audited function instead of being repeated at four call sites.
    """
    if count <= 0:
        raise ValueError("an IN clause needs at least one placeholder")
    return ",".join("?" * count)


def _to_document(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        id=int(row["id"]),
        source=str(row["source"]),
        title=str(row["title"]),
        sha256=str(row["sha256"]),
        language=str(row["language"]),
        n_chunks=int(row["n_chunks"]),
        n_tokens=int(row["n_tokens"]),
        ingested_at=str(row["ingested_at"]),
    )
