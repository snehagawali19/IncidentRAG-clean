"""Retrieval Layer — HybridRetriever."""
from __future__ import annotations

import asyncio
import logging
import time

from incidentrag.core.models import RetrievalOutput, RetrievalResult, RetrievalScores
from incidentrag.core.settings import settings
from incidentrag.retrieval.bm25_index import BM25Index
from incidentrag.retrieval.dense_index import DenseIndex
from incidentrag.retrieval.reranker import Reranker
from incidentrag.retrieval.rrf import _FusedEntry, apply_recency_and_boost, fuse_rrf

logger = logging.getLogger(__name__)


def _service_matches(expected: str, actual: str) -> bool:
    """Prevent unrelated service runbooks from being presented as evidence."""
    expected = expected.casefold().strip().replace("_", "-")
    actual = actual.casefold().strip().replace("_", "-")
    if not expected or expected in {"unknown", "unknown-service"}:
        return True
    if expected in {"argo-cd", "argocd"}:
        return actual.startswith("argocd-") or actual in {"argo-cd", "argocd"}
    return expected == actual


def _fused_to_result(entry: _FusedEntry, primary_query: str) -> RetrievalResult:
    """Convert _FusedEntry to RetrievalResult for reranking."""
    retrieved_via: list = []
    if entry.bm25_score > 0:
        retrieved_via.append("bm25")
    if entry.dense_score > 0:
        retrieved_via.append("dense")
    if not retrieved_via:
        retrieved_via = ["bm25"]
    return RetrievalResult(
        chunk=entry.chunk,
        scores=RetrievalScores(
            dense_similarity=entry.dense_score if entry.dense_score > 0 else None,
            bm25_score=entry.bm25_score if entry.bm25_score > 0 else None,
            rrf_score=entry.rrf_score,
            reranker_score=None,
            recency_boost=entry.recency_multiplier,
            feedback_boost=entry.chunk.boost_score,
            graph_hops_from_query=None,
            final_score=entry.rrf_score * entry.recency_multiplier * entry.chunk.boost_score,
        ),
        retrieved_via=retrieved_via,
        query_that_matched=primary_query,
    )


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        reranker: Reranker | None = None,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker or Reranker()

    async def _bm25_all(self, queries: list[str], k: int) -> list[tuple]:
        tasks = [asyncio.to_thread(self.bm25.search, q, k) for q in queries]
        results = await asyncio.gather(*tasks)
        combined = []
        for r in results:
            combined.extend(r)
        return combined

    async def _dense_all(self, queries: list[str], k: int) -> list[tuple]:
        tasks = [self.dense.search(q, k) for q in queries]
        results = await asyncio.gather(*tasks)
        combined = []
        for r in results:
            combined.extend(r)
        return combined

    async def retrieve(
        self,
        alert_id: str,
        queries: list[str],
        hyde_document: str = "",
        top_k: int | None = None,
        service: str | None = None,
    ) -> RetrievalOutput:
        top_k = top_k or settings.retrieval_top_k
        k = settings.retrieval_top_k
        t0 = time.monotonic()

        bm25_task = self._bm25_all(queries, k)
        dense_task = self._dense_all(queries, k)

        if hyde_document:
            hyde_task = self.dense.search(hyde_document[:1000], k)
            bm25_results, dense_results, hyde_results = await asyncio.gather(
                bm25_task, dense_task, hyde_task
            )
            dense_results = dense_results + hyde_results
        else:
            bm25_results, dense_results = await asyncio.gather(bm25_task, dense_task)

        merged = fuse_rrf(bm25_results, dense_results)
        apply_recency_and_boost(merged)

        primary_query = queries[0] if queries else ""
        candidates_fused = sorted(merged.values(), key=lambda x: x.rrf_score, reverse=True)[:k]
        candidates = [_fused_to_result(e, primary_query) for e in candidates_fused]

        if service:
            candidates = [
                item for item in candidates
                if _service_matches(service, item.chunk.service)
            ]

        reranked = await self.reranker.rerank(primary_query, candidates, top_k=top_k)

        total_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "HybridRetriever: %d queries | BM25=%d Dense=%d → top %d in %.0fms",
            len(queries), len(bm25_results), len(dense_results), len(reranked), total_ms,
        )

        return RetrievalOutput(
            query_id=alert_id,
            total_candidates_pre_rerank=len(merged),
            reranker_used=True,
            results=reranked,
            latency_ms=total_ms,
        )
