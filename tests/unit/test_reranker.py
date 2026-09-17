"""Tests for the cross-encoder Reranker."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from incidentrag.retrieval.reranker import Reranker
from tests.unit.conftest import make_chunk, make_retrieval_result

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_encoder(monkeypatch):
    """Patch _get_encoder so no model is downloaded during tests."""
    mock = MagicMock()
    # Default: return scores for 2 candidates (adjusted per test)
    mock.predict.return_value = [0.8, 0.5]
    monkeypatch.setattr("incidentrag.retrieval.reranker._get_encoder", lambda: mock)
    return mock


def _make_two_candidates():
    """Return two RetrievalResult objects for reranking tests."""
    c1 = make_chunk("c1", content="First chunk content")
    c2 = make_chunk("c2", content="Second chunk content")
    return [
        make_retrieval_result(c1, final_score=0.5),
        make_retrieval_result(c2, final_score=0.3),
    ]


# ── Tests ─────────────────────────────────────────────────────────────────────

async def test_rerank_empty_candidates_returns_empty():
    """rerank() with no candidates should return empty list without touching the model."""
    reranker = Reranker()
    result = await reranker.rerank("query", [])
    assert result == []


async def test_rerank_sets_reranker_score_on_scores(mock_encoder):
    """After rerank(), each result's scores.reranker_score should be set."""
    mock_encoder.predict.return_value = [0.9, 0.6]
    reranker = Reranker()
    candidates = _make_two_candidates()

    results = await reranker.rerank("OOMKilled query", candidates)

    for r in results:
        assert r.scores.reranker_score is not None


async def test_rerank_sets_final_score_on_scores(mock_encoder):
    """After rerank(), each result's scores.final_score should be set."""
    mock_encoder.predict.return_value = [0.9, 0.6]
    reranker = Reranker()
    candidates = _make_two_candidates()

    results = await reranker.rerank("OOMKilled query", candidates)

    for r in results:
        assert r.scores.final_score is not None


async def test_rerank_sorts_by_final_score_descending(mock_encoder):
    """Results should be sorted so highest final_score comes first."""
    # c1 gets score 0.9, c2 gets score 0.3
    mock_encoder.predict.return_value = [0.9, 0.3]
    reranker = Reranker()
    candidates = _make_two_candidates()

    results = await reranker.rerank("query", candidates)

    assert results[0].scores.final_score >= results[1].scores.final_score


async def test_rerank_respects_top_k(mock_encoder):
    """With top_k=1, only one result should be returned."""
    mock_encoder.predict.return_value = [0.8, 0.5]
    reranker = Reranker()
    candidates = _make_two_candidates()

    results = await reranker.rerank("query", candidates, top_k=1)

    assert len(results) == 1


async def test_rerank_no_top_k_returns_all(mock_encoder):
    """Without top_k, all candidates are returned."""
    mock_encoder.predict.return_value = [0.8, 0.5]
    reranker = Reranker()
    candidates = _make_two_candidates()

    results = await reranker.rerank("query", candidates)

    assert len(results) == 2


async def test_rerank_drops_candidates_below_relevance_threshold(mock_encoder):
    mock_encoder.predict.return_value = [0.8, -0.4]
    results = await Reranker().rerank("OOMKilled query", _make_two_candidates())

    assert [item.chunk.chunk_id for item in results] == ["c1"]


async def test_rerank_keeps_hybrid_ranking_when_all_scores_are_below_threshold(
    mock_encoder,
) -> None:
    mock_encoder.predict.return_value = [-0.2, -0.8]
    results = await Reranker().rerank("TLS repository timeout", _make_two_candidates())

    assert [item.chunk.chunk_id for item in results] == ["c1", "c2"]
    assert results[0].scores.final_score >= results[1].scores.final_score
