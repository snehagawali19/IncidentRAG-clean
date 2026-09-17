"""Tests for Layer 3 — RRF Formula and Retrieval Pipeline.

Verifies:
- RRF score formula: 1 / (k + rank)
- Recency multiplier formula: 0.5 ** (age_days / 90)
- Fusion merges correctly across ranked lists
- Recency and boost are applied multiplicatively

All types imported from core/models.py only.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from incidentrag.core.models import Chunk, ChunkType, Severity
from incidentrag.retrieval.rrf import apply_recency_and_boost, fuse_rrf, recency_multiplier, rrf_score


# ── Chunk factory ─────────────────────────────────────────────────────────────


def _make_chunk(
    chunk_id: str,
    *,
    boost_score: float = 1.0,
    days_old: int = 0,
) -> Chunk:
    """Build a minimal valid Chunk for RRF testing."""
    now = datetime.now(tz=timezone.utc)
    content = f"Content for {chunk_id}"
    return Chunk(
        chunk_id=chunk_id,
        runbook_id="rb-test",
        chunk_type=ChunkType.PROSE,
        content=content,
        header_path=["Test Runbook"],
        header_breadcrumb="# Test Runbook",
        position=0,
        token_count=max(1, len(content) // 4),
        content_sha256=hashlib.sha256(content.encode()).hexdigest()[:16],
        service="test-service",
        severity_applicable=[],
        runbook_last_updated=now - timedelta(days=days_old),
        boost_score=boost_score,
    )


# ── RRF score formula ─────────────────────────────────────────────────────────


def test_rrf_score_rank_0() -> None:
    """rank=0 → 1/(60+0) = 0.01667"""
    assert abs(rrf_score(0, k=60) - 1 / 60) < 1e-9


def test_rrf_score_rank_59() -> None:
    """rank=59 → 1/(60+59) = 1/119"""
    assert abs(rrf_score(59, k=60) - 1 / 119) < 1e-9


def test_rrf_score_higher_rank_lower_score() -> None:
    assert rrf_score(0) > rrf_score(1) > rrf_score(10) > rrf_score(100)


# ── Recency multiplier ────────────────────────────────────────────────────────


def test_recency_fresh_document() -> None:
    """Age=0 → multiplier≈1.0"""
    now = datetime.now(tz=timezone.utc).isoformat()
    assert abs(recency_multiplier(now) - 1.0) < 0.01


def test_recency_90_day_old_document() -> None:
    """Age=90 → 0.5 ** (90/90) = 0.5"""
    old = (datetime.now(tz=timezone.utc) - timedelta(days=90)).isoformat()
    assert abs(recency_multiplier(old) - 0.5) < 0.01


def test_recency_180_day_old_document() -> None:
    """Age=180 → 0.5 ** (180/90) = 0.25"""
    old = (datetime.now(tz=timezone.utc) - timedelta(days=180)).isoformat()
    assert abs(recency_multiplier(old) - 0.25) < 0.01


def test_recency_none_returns_1() -> None:
    assert recency_multiplier(None) == 1.0


def test_recency_invalid_iso_returns_1() -> None:
    assert recency_multiplier("not-a-date") == 1.0


# ── RRF Fusion ────────────────────────────────────────────────────────────────


def test_fuse_rrf_combines_scores() -> None:
    c1 = _make_chunk("c1")
    c2 = _make_chunk("c2")
    c3 = _make_chunk("c3")

    bm25 = [(c1, 0.9), (c2, 0.5), (c3, 0.1)]
    dense = [(c2, 0.95), (c1, 0.7), (c3, 0.3)]

    merged = fuse_rrf(bm25, dense)

    # c1 appears at rank 0 in BM25 + rank 1 in dense
    assert "c1" in merged
    expected_c1 = rrf_score(0) + rrf_score(1)
    assert abs(merged["c1"].rrf_score - expected_c1) < 1e-9

    # c2 appears at rank 1 in BM25 + rank 0 in dense
    expected_c2 = rrf_score(1) + rrf_score(0)
    assert abs(merged["c2"].rrf_score - expected_c2) < 1e-9

    # c3 worst in both lists — lowest combined score
    assert merged["c3"].rrf_score < merged["c1"].rrf_score
    assert merged["c3"].rrf_score < merged["c2"].rrf_score


def test_fuse_rrf_chunk_only_in_one_list() -> None:
    """Chunk appears only in dense — must still appear in merged with bm25_score=0."""
    c1 = _make_chunk("c1")
    c_only_dense = _make_chunk("c_only")

    bm25 = [(c1, 0.8)]
    dense = [(c1, 0.9), (c_only_dense, 0.4)]

    merged = fuse_rrf(bm25, dense)
    assert "c_only" in merged
    assert merged["c_only"].bm25_score == 0.0


def test_apply_recency_and_boost() -> None:
    """rrf_score *= recency_multiplier(90 days ≈ 0.5) * boost_score(1.5)"""
    c = _make_chunk("c1", boost_score=1.5, days_old=90)

    bm25 = [(c, 0.5)]
    dense = [(c, 0.5)]
    merged = fuse_rrf(bm25, dense)

    # Before applying recency: rrf_score = rrf(rank=0) + rrf(rank=0)
    raw_rrf = rrf_score(0) + rrf_score(0)

    apply_recency_and_boost(merged)
    result = merged["c1"]

    # recency_multiplier(90 days) ≈ 0.5; boost = 1.5
    expected = raw_rrf * 0.5 * 1.5
    assert abs(result.rrf_score - expected) < 1e-4
    assert abs(result.recency_multiplier - 0.5) < 0.01
