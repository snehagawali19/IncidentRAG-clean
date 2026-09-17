"""Retrieval Layer — Cross-Encoder Reranker.

Uses sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2.
Model is loaded lazily (singleton) to avoid repeated disk I/O.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # Optional in lean API images.
    CrossEncoder = None  # type: ignore[assignment, misc]

from incidentrag.core.models import RetrievalResult
from incidentrag.core.settings import settings

logger = logging.getLogger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_encoder: Any | None = None


def _get_encoder() -> Any:
    global _encoder
    if CrossEncoder is None:
        return None
    if _encoder is None:
        logger.info("Loading cross-encoder: %s", _MODEL_NAME)
        _encoder = CrossEncoder(_MODEL_NAME)
    return _encoder


class Reranker:
    """Wraps the cross-encoder with async support via thread executor."""

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Scores each candidate against the query.
        Sets rerank_score and final_score; sorts descending by final_score.
        """
        if not candidates:
            return candidates

        encoder = _get_encoder()
        if encoder is None:
            logger.info("Cross-encoder is unavailable; retaining hybrid retrieval ranking")
            ranked = sorted(
                candidates, key=lambda item: item.scores.final_score, reverse=True
            )
            return ranked[:top_k] if top_k else ranked

        pairs = [(query, rc.chunk.content[:512]) for rc in candidates]

        # Run CPU-bound inference in thread pool
        scores: list[float] = await asyncio.to_thread(
            encoder.predict, pairs
        )

        for rc, score in zip(candidates, scores, strict=True):
            rc.scores.reranker_score = float(score)
            rc.scores.final_score = float(score)

        relevant = [
            item
            for item in candidates
            if item.scores.reranker_score is not None
            and item.scores.reranker_score >= settings.reranker_min_score
        ]
        if not relevant:
            logger.warning(
                "All %d reranker scores were below %.2f; keeping the hybrid ranking",
                len(candidates),
                settings.reranker_min_score,
            )
            relevant = candidates
        reranked = sorted(relevant, key=lambda x: x.scores.final_score, reverse=True)
        return reranked[:top_k] if top_k else reranked
