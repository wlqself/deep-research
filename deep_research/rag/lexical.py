"""SQLite FTS5 lexical retrieval used by the production RAG pipeline."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path
from threading import RLock


_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_./:@+][A-Za-z0-9]+)*")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")

_CJK_QUERY_STOPWORDS = {
    "什么", "为何", "为什么", "如何", "怎么", "怎样", "这篇", "本文",
    "总体", "结论", "作用", "问题", "分别", "哪些", "多少", "是否", "使用",
    "进行", "需要", "可以", "以及", "其中", "关于", "当前", "这个", "这份",
    "一个", "一种", "实验",
}
_LATIN_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "does", "for", "how", "is", "of", "on", "or",
    "the", "this", "what", "why",
}


def build_lexical_query(query: str) -> str:
    """Keep searchable entities while removing common question wording."""
    if not query.strip():
        raise ValueError("query must not be empty")

    terms: list[str] = []
    for match in _LATIN_TOKEN_RE.finditer(query):
        token = match.group(0).lower()
        if token not in _LATIN_QUERY_STOPWORDS:
            terms.append(token)
        terms.extend(
            part
            for part in re.findall(r"[A-Za-z0-9]+", token)
            if part not in _LATIN_QUERY_STOPWORDS
        )

    for match in _CJK_RE.finditer(query):
        sequence = match.group(0)
        for stopword in sorted(_CJK_QUERY_STOPWORDS, key=len, reverse=True):
            sequence = sequence.replace(stopword, " ")
        for segment in sequence.split():
            if len(segment) == 1:
                terms.append(segment)
                continue
            terms.extend(segment[index:index + 2] for index in range(len(segment) - 1))

    unique_terms = list(dict.fromkeys(terms))
    return " ".join(unique_terms) or query.strip()


def _lexical_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _LATIN_TOKEN_RE.finditer(text):
        token = match.group(0).lower()
        tokens.append(token)
        tokens.extend(
            part.lower()
            for part in re.findall(r"[A-Za-z0-9]+", token)
            if part.lower() != token
        )

    for match in _CJK_RE.finditer(text):
        sequence = match.group(0)
        if len(sequence) == 1:
            tokens.append(sequence)
        else:
            tokens.extend(sequence[index:index + 2] for index in range(len(sequence) - 1))
    return tokens


def _tokenized_text(text: str) -> str:
    return " ".join(_lexical_tokens(text))


def _match_expression(query: str) -> str:
    tokens = _lexical_tokens(query)
    if not tokens:
        raise ValueError("query must contain searchable terms")
    return " OR ".join(
        '"' + token.replace('"', '""') + '"'
        for token in dict.fromkeys(tokens)
    )


class SqliteBm25Index:
    """Persistent FTS5 index with serialized access to one SQLite connection."""

    _VALID_STATUSES = frozenset({"pending", "indexed"})

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._closed = False
        self.connection = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self._initialize()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("BM25 index is closed")

    def _initialize(self) -> None:
        with self._lock:
            self._ensure_open()
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS bm25_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    status TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS bm25_chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    document_id UNINDEXED,
                    title,
                    body,
                    tokenize = 'unicode61 remove_diacritics 2'
                );
                """
            )
            self.connection.commit()

    @staticmethod
    def _validate_rows(
        rows: list[Mapping[str, object]],
    ) -> list[tuple[str, str, str, str, str]]:
        validated: list[tuple[str, str, str, str, str]] = []
        for chunk in rows:
            chunk_id = chunk.get("_id")
            text = chunk.get("text")
            metadata = chunk.get("metadata")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError("chunk must contain a non-empty '_id'")
            if not isinstance(text, str) or not text:
                raise ValueError(f"chunk {chunk_id!r} has invalid text")
            if not isinstance(metadata, Mapping):
                raise ValueError(f"chunk {chunk_id!r} has invalid metadata")
            document_id = metadata.get("document_id")
            title = chunk.get("title") or metadata.get("title") or document_id
            status = metadata.get("status", "pending")
            if not isinstance(document_id, str) or not document_id:
                raise ValueError(f"chunk {chunk_id!r} has invalid document_id")
            if not isinstance(title, str) or not title:
                raise ValueError(f"chunk {chunk_id!r} has invalid title")
            if not isinstance(status, str) or status not in SqliteBm25Index._VALID_STATUSES:
                raise ValueError(f"chunk {chunk_id!r} has invalid status")
            validated.append((chunk_id, document_id, title, text, status))
        return validated

    def _upsert_rows_locked(
        self,
        rows: list[tuple[str, str, str, str, str]],
    ) -> None:
        for chunk_id, document_id, title, text, status in rows:
            self.connection.execute(
                "DELETE FROM bm25_chunks_fts WHERE chunk_id = ?",
                (chunk_id,),
            )
            self.connection.execute(
                """
                INSERT INTO bm25_chunks (
                    chunk_id, document_id, title, text, status
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    document_id = excluded.document_id,
                    title = excluded.title,
                    text = excluded.text,
                    status = excluded.status
                """,
                (chunk_id, document_id, title, text, status),
            )
            self.connection.execute(
                """
                INSERT INTO bm25_chunks_fts (
                    chunk_id, document_id, title, body
                ) VALUES (?, ?, ?, ?)
                """,
                (chunk_id, document_id, _tokenized_text(title), _tokenized_text(text)),
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self.connection.close()
            self._closed = True

    def clear(self) -> None:
        """Remove all rows atomically."""
        with self._lock:
            self._ensure_open()
            try:
                self.connection.execute("BEGIN")
                self.connection.execute("DELETE FROM bm25_chunks_fts")
                self.connection.execute("DELETE FROM bm25_chunks")
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise

    def replace_chunks(self, chunks: Iterable[Mapping[str, object]]) -> int:
        """Atomically replace the entire derived index during startup rebuild."""
        rows = self._validate_rows(list(chunks))
        with self._lock:
            self._ensure_open()
            try:
                self.connection.execute("BEGIN")
                self.connection.execute("DELETE FROM bm25_chunks_fts")
                self.connection.execute("DELETE FROM bm25_chunks")
                self._upsert_rows_locked(rows)
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        return len(rows)

    def upsert_chunks(self, chunks: Iterable[Mapping[str, object]]) -> int:
        rows = self._validate_rows(list(chunks))
        if not rows:
            return 0
        with self._lock:
            self._ensure_open()
            try:
                self.connection.execute("BEGIN")
                self._upsert_rows_locked(rows)
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        return len(rows)

    def update_document_status(self, document_id: str, status: str) -> int:
        normalized_document_id = document_id.strip()
        if not normalized_document_id:
            raise ValueError("document_id must not be empty")
        if status not in self._VALID_STATUSES:
            raise ValueError("status must be pending or indexed")
        with self._lock:
            self._ensure_open()
            try:
                cursor = self.connection.execute(
                    "UPDATE bm25_chunks SET status = ? WHERE document_id = ?",
                    (status, normalized_document_id),
                )
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        return max(cursor.rowcount, 0)

    def delete_document(self, document_id: str) -> int:
        normalized_document_id = document_id.strip()
        if not normalized_document_id:
            raise ValueError("document_id must not be empty")
        with self._lock:
            self._ensure_open()
            try:
                self.connection.execute("BEGIN")
                chunk_ids = self.connection.execute(
                    "SELECT chunk_id FROM bm25_chunks WHERE document_id = ?",
                    (normalized_document_id,),
                ).fetchall()
                self.connection.execute(
                    "DELETE FROM bm25_chunks_fts WHERE document_id = ?",
                    (normalized_document_id,),
                )
                cursor = self.connection.execute(
                    "DELETE FROM bm25_chunks WHERE document_id = ?",
                    (normalized_document_id,),
                )
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        return max(cursor.rowcount, len(chunk_ids))

    def search(
        self,
        query: str,
        *,
        top_k: int,
        document_id: str | None = None,
    ) -> list[tuple[str, float]]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        expression = _match_expression(query)
        normalized_document_id = None
        if document_id is not None:
            normalized_document_id = document_id.strip()
            if not normalized_document_id:
                raise ValueError("document_id must not be empty")

        sql = """
            SELECT
                f.chunk_id,
                bm25(bm25_chunks_fts, 1.0, 1.0, 5.0, 1.0) AS score
            FROM bm25_chunks_fts AS f
            JOIN bm25_chunks AS c ON c.chunk_id = f.chunk_id
            WHERE bm25_chunks_fts MATCH ?
              AND c.status = 'indexed'
        """
        parameters: list[object] = [expression]
        if normalized_document_id is not None:
            sql += " AND f.document_id = ?"
            parameters.append(normalized_document_id)
        sql += " ORDER BY score ASC, f.chunk_id ASC LIMIT ?"
        parameters.append(top_k)

        with self._lock:
            self._ensure_open()
            rows = self.connection.execute(sql, parameters).fetchall()
        return [(str(row["chunk_id"]), float(row["score"])) for row in rows]

    def search_ids(
        self,
        query: str,
        *,
        top_k: int,
        document_id: str | None = None,
    ) -> list[str]:
        return [
            chunk_id
            for chunk_id, _score in self.search(
                query,
                top_k=top_k,
                document_id=document_id,
            )
        ]
