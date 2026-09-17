"""Tests for DenseIndex (Qdrant-backed dense retrieval)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from incidentrag.core.models import Chunk, ChunkType
from incidentrag.retrieval.dense_index import DenseIndex, _qdrant_id
from tests.unit.conftest import make_chunk


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_embed_response(n_texts: int, dim: int = 4) -> MagicMock:
    """Return a mock OpenAI embedding response."""
    resp = MagicMock()
    resp.data = [MagicMock(embedding=[0.1] * dim) for _ in range(n_texts)]
    return resp


def _mock_collection_list(names: list[str]) -> MagicMock:
    """Return a mock get_collections() response."""
    resp = MagicMock()
    collections = []
    for n in names:
        col = MagicMock()
        col.name = n  # Must set as attribute, not constructor kwarg (which is the mock's repr name)
        collections.append(col)
    resp.collections = collections
    return resp


def _mock_search_result(chunk_id: str, score: float = 0.9) -> MagicMock:
    """Return a minimal Qdrant ScoredPoint mock."""
    pt = MagicMock()
    pt.id = str(uuid.uuid4())
    pt.score = score
    pt.payload = {
        "chunk_id": chunk_id,
        "runbook_id": "rb-test",
        "chunk_type": "prose",
        "content": "test content for chunk",
        "header_path": ["# Test"],
        "header_breadcrumb": "# Test",
        "position": 0,
        "token_count": 10,
        "content_sha256": "abc123",
        "service": "svc",
        "boost_score": 1.0,
        "runbook_last_updated": datetime.now(timezone.utc).isoformat(),
    }
    return pt


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_qdrant():
    with patch("incidentrag.retrieval.dense_index.AsyncQdrantClient") as mock_cls:
        instance = AsyncMock()
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def mock_openai():
    with patch("incidentrag.retrieval.dense_index.AsyncOpenAI") as mock_cls:
        instance = AsyncMock()
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def dense_index(mock_qdrant, mock_openai) -> DenseIndex:
    return DenseIndex()


# ── Tests ─────────────────────────────────────────────────────────────────────

async def test_embed_calls_openai(dense_index, mock_openai):
    """embed() should call OpenAI embeddings and return a list of vectors."""
    mock_openai.embeddings.create = AsyncMock(
        return_value=_mock_embed_response(2, dim=4)
    )
    result = await dense_index.embed(["hello", "world"])
    mock_openai.embeddings.create.assert_called_once()
    assert len(result) == 2
    assert len(result[0]) == 4


async def test_upsert_calls_qdrant(dense_index, mock_qdrant, mock_openai):
    """upsert([chunk]) should call qdrant.upsert once."""
    chunk = make_chunk()
    mock_qdrant.get_collections = AsyncMock(
        return_value=_mock_collection_list(["incidentrag_chunks"])
    )
    mock_openai.embeddings.create = AsyncMock(
        return_value=_mock_embed_response(1, dim=4)
    )
    mock_qdrant.upsert = AsyncMock()

    await dense_index.upsert([chunk])
    mock_qdrant.upsert.assert_called_once()


async def test_upsert_empty_is_noop(dense_index, mock_qdrant, mock_openai):
    """upsert([]) should not call qdrant.upsert at all."""
    mock_qdrant.get_collections = AsyncMock(
        return_value=_mock_collection_list(["incidentrag_chunks"])
    )
    mock_qdrant.upsert = AsyncMock()
    mock_openai.embeddings.create = AsyncMock(
        return_value=_mock_embed_response(0)
    )

    await dense_index.upsert([])
    mock_qdrant.upsert.assert_not_called()


async def test_ensure_collection_creates_when_missing(dense_index, mock_qdrant):
    """ensure_collection() should call create_collection when collection is absent."""
    mock_qdrant.get_collections = AsyncMock(
        return_value=_mock_collection_list([])  # empty
    )
    mock_qdrant.create_collection = AsyncMock()

    await dense_index.ensure_collection()
    mock_qdrant.create_collection.assert_called_once()


async def test_ensure_collection_skips_when_exists(dense_index, mock_qdrant):
    """ensure_collection() should NOT call create_collection when collection exists."""
    mock_qdrant.get_collections = AsyncMock(
        return_value=_mock_collection_list(["incidentrag_chunks"])
    )
    mock_qdrant.create_collection = AsyncMock()

    await dense_index.ensure_collection()
    mock_qdrant.create_collection.assert_not_called()


async def test_search_returns_chunks(dense_index, mock_qdrant, mock_openai):
    """search() should return Chunk objects parsed from Qdrant payload."""
    mock_openai.embeddings.create = AsyncMock(
        return_value=_mock_embed_response(1, dim=4)
    )
    mock_response = MagicMock()
    mock_response.points = [_mock_search_result("chunk-xyz", score=0.88)]
    mock_qdrant.query_points = AsyncMock(
        return_value=mock_response
    )

    results = await dense_index.search("test query", k=5)
    assert len(results) == 1
    chunk, score = results[0]
    assert chunk.chunk_id == "chunk-xyz"
    assert abs(score - 0.88) < 1e-6


async def test_search_falls_back_for_legacy_qdrant(
    dense_index,
    mock_qdrant,
    mock_openai,
):
    """A pre-query_points Qdrant server should use the legacy search route."""
    mock_openai.embeddings.create = AsyncMock(
        return_value=_mock_embed_response(1, dim=4)
    )
    mock_qdrant.query_points = AsyncMock(
        side_effect=UnexpectedResponse(404, "Not Found", b"", httpx.Headers())
    )
    legacy_point = _mock_search_result("chunk-legacy", score=0.77)
    dense_index._legacy_search = AsyncMock(return_value=[legacy_point])

    results = await dense_index.search("legacy query", k=5)

    dense_index._legacy_search.assert_awaited_once()
    assert results[0][0].chunk_id == "chunk-legacy"
    assert results[0][1] == 0.77


def test_qdrant_id_deterministic():
    """_qdrant_id() should always produce the same UUID for the same chunk_id."""
    cid = "chunk-abc-123"
    result1 = _qdrant_id(cid)
    result2 = _qdrant_id(cid)
    assert result1 == result2
    # It should be a valid UUID string
    uuid.UUID(result1)


async def test_upsert_chunks_alias(dense_index, mock_qdrant, mock_openai):
    """upsert_chunks() should delegate to upsert()."""
    chunk = make_chunk()
    mock_qdrant.get_collections = AsyncMock(
        return_value=_mock_collection_list(["incidentrag_chunks"])
    )
    mock_openai.embeddings.create = AsyncMock(
        return_value=_mock_embed_response(1, dim=4)
    )
    mock_qdrant.upsert = AsyncMock()

    await dense_index.upsert_chunks([chunk])
    mock_qdrant.upsert.assert_called_once()
