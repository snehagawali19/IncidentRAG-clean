"""Tests for HybridRetriever."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from incidentrag.core.models import RetrievalOutput
from incidentrag.retrieval.hybrid import (
    HybridRetriever,
    _fused_to_result,
    _service_matches,
)
from incidentrag.retrieval.rrf import _FusedEntry
from tests.unit.conftest import make_chunk, make_retrieval_result

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_chunk():
    return make_chunk()


@pytest.fixture
def mock_bm25(sample_chunk):
    bm25 = MagicMock()
    bm25.search = MagicMock(return_value=[(sample_chunk, 5.0)])
    return bm25


@pytest.fixture
def mock_dense(sample_chunk):
    dense = AsyncMock()
    dense.search = AsyncMock(return_value=[(sample_chunk, 0.85)])
    return dense


@pytest.fixture
def mock_reranker(sample_chunk):
    reranker = AsyncMock()
    result = make_retrieval_result(sample_chunk, final_score=0.9)
    reranker.rerank = AsyncMock(return_value=[result])
    return reranker


@pytest.fixture
def retriever(mock_bm25, mock_dense, mock_reranker) -> HybridRetriever:
    return HybridRetriever(bm25=mock_bm25, dense=mock_dense, reranker=mock_reranker)


# ── Tests ─────────────────────────────────────────────────────────────────────

async def test_retrieve_returns_retrieval_output(retriever):
    """retrieve() should return a RetrievalOutput instance."""
    output = await retriever.retrieve(
        alert_id="alert-001",
        queries=["payment OOMKilled"],
    )
    assert isinstance(output, RetrievalOutput)


async def test_retrieve_calls_bm25_and_dense(retriever, mock_bm25, mock_dense):
    """retrieve() should call both BM25 and Dense search."""
    await retriever.retrieve(alert_id="alert-001", queries=["payment OOMKilled"])
    mock_bm25.search.assert_called()
    mock_dense.search.assert_called()


async def test_retrieve_with_hyde_calls_dense_extra(retriever, mock_dense):
    """When hyde_document is provided, dense.search is called an extra time."""
    await retriever.retrieve(
        alert_id="alert-001",
        queries=["payment OOMKilled"],
        hyde_document="The payment service is OOMKilled due to heap exhaustion.",
    )
    # Once for the fanout query, once for the HyDE document
    assert mock_dense.search.call_count >= 2


async def test_retrieve_empty_queries_returns_empty(mock_bm25, mock_dense, mock_reranker):
    """Empty queries list should produce a valid (possibly empty) output."""
    mock_bm25.search = MagicMock(return_value=[])
    mock_dense.search = AsyncMock(return_value=[])
    mock_reranker.rerank = AsyncMock(return_value=[])

    retriever = HybridRetriever(bm25=mock_bm25, dense=mock_dense, reranker=mock_reranker)
    output = await retriever.retrieve(alert_id="a1", queries=[])
    assert isinstance(output, RetrievalOutput)
    assert output.results == []


def test_fused_to_result_bm25_only():
    """_fused_to_result with bm25_score>0 and dense_score=0 → retrieved_via=['bm25']."""
    chunk = make_chunk()
    entry = _FusedEntry(chunk=chunk, rrf_score=0.02, bm25_score=3.5, dense_score=0.0)
    result = _fused_to_result(entry, "test query")
    assert result.retrieved_via == ["bm25"]
    assert result.scores.bm25_score == 3.5
    assert result.scores.dense_similarity is None


def test_fused_to_result_dense_only():
    """_fused_to_result with dense_score>0 and bm25_score=0 → retrieved_via=['dense']."""
    chunk = make_chunk()
    entry = _FusedEntry(chunk=chunk, rrf_score=0.015, bm25_score=0.0, dense_score=0.78)
    result = _fused_to_result(entry, "another query")
    assert result.retrieved_via == ["dense"]
    assert result.scores.dense_similarity == 0.78
    assert result.scores.bm25_score is None


async def test_retrieve_result_count(retriever, mock_reranker):
    """Results list length should be <= top_k."""
    output = await retriever.retrieve(
        alert_id="a1",
        queries=["query"],
        top_k=5,
    )
    assert len(output.results) <= 5


def test_service_matching_rejects_unrelated_runbook():
    assert _service_matches("argocd-repo-server", "argocd-repo-server") is True
    assert _service_matches("argocd-repo-server", "database-proxy") is False
    assert _service_matches("argo-cd", "argocd-repo-server") is True


async def test_retrieve_filters_unrelated_service_before_reranking(
    mock_bm25, mock_dense, mock_reranker
):
    mock_reranker.rerank.side_effect = (
        lambda query, candidates, top_k=None: candidates
    )
    output = await HybridRetriever(
        bm25=mock_bm25, dense=mock_dense, reranker=mock_reranker
    ).retrieve(
        alert_id="a1",
        queries=["repository sync stale revision"],
        service="argocd-repo-server",
    )

    assert output.results == []
    assert mock_reranker.rerank.await_args.args[1] == []


async def test_retrieve_query_id_matches_alert_id(retriever):
    """output.query_id should equal the alert_id passed in."""
    output = await retriever.retrieve(
        alert_id="my-alert-999",
        queries=["some query"],
    )
    assert output.query_id == "my-alert-999"
