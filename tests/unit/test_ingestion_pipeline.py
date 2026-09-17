"""Tests for IngestionPipeline (Phase 2)."""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call

from incidentrag.ingestion.pipeline import IngestionPipeline, IndexerProtocol

# Real runbook files included in the repo
RUNBOOKS_DIR = Path(__file__).parents[2] / "data" / "runbooks"
SAMPLE_RUNBOOK = RUNBOOKS_DIR / "payment_service_memory.md"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_indexer() -> AsyncMock:
    indexer = AsyncMock(spec=IndexerProtocol)
    indexer.upsert_chunks = AsyncMock()
    indexer.delete_chunks = AsyncMock()
    return indexer


# ── Tests ─────────────────────────────────────────────────────────────────────

async def test_ingest_file_success():
    """Ingest a real runbook file — result should be successful with chunks."""
    pipeline = IngestionPipeline()
    result = await pipeline.ingest_file(str(SAMPLE_RUNBOOK))
    assert result.success, f"Expected success but got error: {result.error}"
    assert len(result.chunks) > 0


async def test_ingest_file_missing_raises_error_result():
    """Non-existent path should return a failed result, not raise."""
    pipeline = IngestionPipeline()
    result = await pipeline.ingest_file("/nonexistent/path/runbook.md")
    assert result.success is False
    assert result.error is not None


async def test_ingest_directory_processes_all_files():
    """Ingest data/runbooks/ — all 3 files should produce successful results."""
    pipeline = IngestionPipeline()
    results = await pipeline.ingest_directory(str(RUNBOOKS_DIR))
    assert len(results) == 4
    assert all(r.success for r in results), [r.error for r in results if not r.success]


async def test_ingest_directory_total_chunks():
    """Total chunks from all runbooks in data/runbooks/ should be >= 40."""
    pipeline = IngestionPipeline()
    results = await pipeline.ingest_directory(str(RUNBOOKS_DIR))
    total = sum(len(r.chunks) for r in results)
    assert total >= 40, f"Expected >= 40 chunks total, got {total}"


async def test_ingest_directory_missing_dir_raises():
    """Missing directory should raise FileNotFoundError."""
    pipeline = IngestionPipeline()
    with pytest.raises(FileNotFoundError):
        await pipeline.ingest_directory("/nonexistent/directory")


async def test_ingest_with_mock_indexer(mock_indexer):
    """When indexer is provided, upsert_chunks should be called."""
    pipeline = IngestionPipeline(indexer=mock_indexer)
    result = await pipeline.ingest_file(str(SAMPLE_RUNBOOK))
    assert result.success
    mock_indexer.upsert_chunks.assert_called_once()


async def test_on_result_callback_called():
    """on_result callback should be called once per file."""
    pipeline = IngestionPipeline()
    callback = MagicMock()
    results = await pipeline.ingest_directory(str(RUNBOOKS_DIR), on_result=callback)
    assert callback.call_count == 4


async def test_register_existing_chunks():
    """register_existing_chunks should update the existing_chunks property."""
    pipeline = IngestionPipeline()
    assert len(pipeline.existing_chunks) == 0

    # Ingest one file to get some chunks
    result = await pipeline.ingest_file(str(SAMPLE_RUNBOOK))
    pipeline.register_existing_chunks(result.chunks)
    assert len(pipeline.existing_chunks) == len(result.chunks)


async def test_indexed_flag_false_without_indexer():
    """Without an indexer, result.indexed should be False."""
    pipeline = IngestionPipeline()
    result = await pipeline.ingest_file(str(SAMPLE_RUNBOOK))
    assert result.indexed is False


async def test_indexed_flag_true_with_indexer(mock_indexer):
    """With an indexer, result.indexed should be True."""
    pipeline = IngestionPipeline(indexer=mock_indexer)
    result = await pipeline.ingest_file(str(SAMPLE_RUNBOOK))
    assert result.indexed is True
