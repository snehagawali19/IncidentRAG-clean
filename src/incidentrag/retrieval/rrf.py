"""Retrieval Layer — Reciprocal Rank Fusion + Recency Prior.

RRF formula:  score = Σ 1 / (k + rank_i),  k = settings.rrf_k (default 60)
Recency prior: multiplier = 0.5 ** (age_days / recency_half_life_days)

References (do not import — reimplemented fresh):
  - references/hybrid-rag/src/retrieval.py  (RRF formula)
  - references/redevops-rag               (recency prior)
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

from incidentrag.core.models import Chunk
from incidentrag.core.settings import settings


# ── Intermediate fusion state ─────────────────────────────────────────────────


@dataclasses.dataclass
class _FusedEntry:
    """
    Mutable intermediate state used during RRF fusion.

    Attributes are accumulated across BM25 and dense ranked lists before
    being converted to a canonical RetrievalResult in the HybridRetriever.
    """

    chunk: Chunk
    rrf_score: float = 0.0
    bm25_score: float = 0.0
    dense_score: float = 0.0
    recency_multiplier: float = 1.0


# ── Public functions ──────────────────────────────────────────────────────────


def rrf_score(rank: int, k: int | None = None) -> float:
    """
    Compute the RRF contribution for a single rank position.

    Formula: 1 / (k + rank)  — SPEC §8.1, k defaults to settings.rrf_k (60).
    """
    effective_k = k if k is not None else settings.rrf_k
    return 1.0 / (effective_k + rank)


def recency_multiplier(updated_at_iso: str | None) -> float:
    """
    Return a [0, 1] multiplicative weight based on document age.

    Formula: 0.5 ** (age_days / half_life)
      - Fresh doc (age=0) → 1.0
      - Doc aged half_life days → 0.5
      - Doc aged 2×half_life days → 0.25
    """
    if not updated_at_iso:
        return 1.0
    try:
        updated = datetime.fromisoformat(updated_at_iso)
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        age_days = max(0, (datetime.now(tz=timezone.utc) - updated).days)
        return 0.5 ** (age_days / settings.recency_half_life_days)
    except (ValueError, TypeError):
        return 1.0


def fuse_rrf(
    bm25_results: list[tuple[Chunk, float]],
    dense_results: list[tuple[Chunk, float]],
    k: int | None = None,
) -> dict[str, _FusedEntry]:
    """
    Fuse two ranked lists via Reciprocal Rank Fusion.

    Args:
        bm25_results:  Ranked list of (Chunk, raw_score) from the BM25 index.
        dense_results: Ranked list of (Chunk, raw_score) from the dense index.
        k:             RRF constant.  Defaults to settings.rrf_k (60).

    Returns:
        Dict keyed by chunk_id, values are _FusedEntry with accumulated scores.
        bm25_score / dense_score reflect the original raw scores (0.0 if absent).
    """
    merged: dict[str, _FusedEntry] = {}

    for rank, (chunk, score) in enumerate(bm25_results):
        entry = merged.setdefault(chunk.chunk_id, _FusedEntry(chunk=chunk))
        entry.bm25_score = score
        entry.rrf_score += rrf_score(rank, k)

    for rank, (chunk, score) in enumerate(dense_results):
        entry = merged.setdefault(chunk.chunk_id, _FusedEntry(chunk=chunk))
        # Prefer the chunk object from dense if the BM25 slot was empty
        if not entry.chunk.content and chunk.content:
            entry.chunk = chunk
        entry.dense_score = score
        entry.rrf_score += rrf_score(rank, k)

    return merged


def apply_recency_and_boost(merged: dict[str, _FusedEntry]) -> None:
    """
    Mutate each _FusedEntry in-place, applying the recency and feedback boost:

        rrf_score *= recency_multiplier * chunk.boost_score

    recency_multiplier is derived from chunk.runbook_last_updated.
    boost_score is the per-chunk feedback multiplier (default 1.0, boosted
    by the feedback loop when a chunk was used in a resolved incident).
    """
    for entry in merged.values():
        rm = recency_multiplier(entry.chunk.runbook_last_updated.isoformat())
        entry.recency_multiplier = rm
        entry.rrf_score *= rm * entry.chunk.boost_score
