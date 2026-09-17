"""Retrieval Layer — BM25 Index.

Wraps rank_bm25.BM25Okapi with build/search interface.
Index is in-memory; rebuilt on startup from Qdrant payloads or provided chunks.
"""
from __future__ import annotations

import logging
from typing import Sequence

from rank_bm25 import BM25Okapi

from incidentrag.core.models import Chunk

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


class BM25Index:
    """In-memory BM25 index over a Chunk corpus."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._index: BM25Okapi | None = None

    @property
    def size(self) -> int:
        return len(self._chunks)

    def build(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        corpus = [_tokenize(c.header_breadcrumb + " " + c.content) for c in self._chunks]
        self._index = BM25Okapi(corpus)
        logger.info("BM25 index built — %d chunks", len(self._chunks))

    def search(self, query: str, k: int = 20) -> list[tuple[Chunk, float]]:
        """Returns (chunk, raw_bm25_score) pairs, sorted descending."""
        if not self._index or not self._chunks:
            return []
        tokens = _tokenize(query)
        scores = self._index.get_scores(tokens)
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:k]
        return [(self._chunks[i], float(s)) for i, s in ranked if s > 0]

    def add(self, chunk: Chunk) -> None:
        """Incrementally add a single chunk (rebuilds index — use for small updates)."""
        self._chunks.append(chunk)
        self.build(self._chunks)
