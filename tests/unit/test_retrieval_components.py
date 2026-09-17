"""Tests for Layer 3 — Retrieval Components (BM25)."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from incidentrag.core.models import Chunk, ChunkType
from incidentrag.retrieval.bm25_index import BM25Index


def _chunk(chunk_id: str, content: str, runbook_id: str = "rb1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        runbook_id=runbook_id,
        chunk_type=ChunkType.PROSE,
        content=content,
        header_path=["Runbook"],
        header_breadcrumb="# Runbook",
        position=0,
        token_count=len(content.split()),
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        service="payment-service",
        runbook_last_updated=datetime.now(timezone.utc),
    )


def _corpus() -> list[Chunk]:
    return [
        _chunk("c1", "JVM heap memory OOMKilled container restart"),
        _chunk("c2", "p99 latency timeout slow response time"),
        _chunk("c3", "CPU throttling iowait saturation load"),
        _chunk("c4", "database connection pool exhaustion pgbouncer max_connections"),
        _chunk("c5", "CrashLoopBackOff pod restart backoff exit code"),
    ]


class TestBM25Index:
    def test_build_and_size(self):
        idx = BM25Index()
        idx.build(_corpus())
        assert idx.size == 5

    def test_search_returns_relevant_results(self):
        idx = BM25Index()
        idx.build(_corpus())
        results = idx.search("OOMKilled memory heap", k=3)
        assert len(results) > 0
        top_chunk, top_score = results[0]
        assert top_chunk.chunk_id == "c1"
        assert top_score > 0

    def test_search_latency_query(self):
        idx = BM25Index()
        idx.build(_corpus())
        results = idx.search("latency timeout p99", k=3)
        assert results[0][0].chunk_id == "c2"

    def test_search_no_results_for_gibberish(self):
        idx = BM25Index()
        idx.build(_corpus())
        results = idx.search("xyzzy foobar bazzle", k=3)
        assert len(results) == 0

    def test_search_empty_index(self):
        idx = BM25Index()
        assert idx.search("anything", k=5) == []

    def test_search_respects_k(self):
        idx = BM25Index()
        idx.build(_corpus())
        results = idx.search("pod restart", k=2)
        assert len(results) <= 2

    def test_add_chunk_incremental(self):
        idx = BM25Index()
        idx.build(_corpus())
        assert idx.size == 5
        idx.add(_chunk("c6", "certificate TLS expiry x509 renewal"))
        assert idx.size == 6
        results = idx.search("TLS certificate expiry", k=1)
        assert results[0][0].chunk_id == "c6"
