"""Retrieval Layer — Dense Index (Qdrant)."""
from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import httpx
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

from incidentrag.core.ai_provider import embedding_model_name, openai_client_kwargs
from incidentrag.core.models import Chunk, ChunkType
from incidentrag.core.settings import settings

logger = logging.getLogger(__name__)


def _chunk_from_payload(point_id: object, payload: dict[str, Any] | None) -> Chunk:
    p = payload or {}
    ru = p.get("runbook_last_updated")
    return Chunk(
        chunk_id=p.get("chunk_id", str(point_id)),
        runbook_id=p.get("runbook_id", "unknown"),
        chunk_type=ChunkType(p.get("chunk_type", "prose")),
        content=p.get("content", ""),
        header_path=p.get("header_path", []),
        header_breadcrumb=p.get("header_breadcrumb", ""),
        position=p.get("position", 0),
        token_count=p.get("token_count", 0),
        content_sha256=p.get("content_sha256", ""),
        service=p.get("service") or "unknown-service",
        boost_score=p.get("boost_score", 1.0),
        runbook_last_updated=(
            datetime.fromisoformat(ru) if ru else datetime.now(UTC)
        ),
    )


def _qdrant_id(chunk_id: str) -> str:
    """Deterministic UUID from chunk_id for Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class DenseIndex:
    def __init__(self) -> None:
        self._qdrant = AsyncQdrantClient(url=settings.qdrant_url)
        self._openai = AsyncOpenAI(**openai_client_kwargs())
        self._embedding_model = embedding_model_name()

    async def ensure_collection(self) -> None:
        collections = await self._qdrant.get_collections()
        names = [c.name for c in collections.collections]
        if settings.qdrant_collection_name not in names:
            await self._qdrant.create_collection(
                collection_name=settings.qdrant_collection_name,
                vectors_config=VectorParams(
                    size=settings.qdrant_embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection: %s", settings.qdrant_collection_name)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._openai.embeddings.create(
            model=self._embedding_model,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    async def upsert_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Implements IndexerProtocol."""
        await self.upsert(chunks)

    async def upsert(self, chunks: Sequence[Chunk]) -> None:
        await self.ensure_collection()
        if not chunks:
            return
        for i in range(0, len(chunks), 100):
            batch = list(chunks[i : i + 100])
            embeddings = await self.embed([c.content for c in batch])
            points = [
                PointStruct(
                    id=_qdrant_id(c.chunk_id),
                    vector=emb,
                    payload={
                        "chunk_id": c.chunk_id,
                        "runbook_id": c.runbook_id,
                        "chunk_type": c.chunk_type.value,
                        "content": c.content,
                        "header_path": c.header_path,
                        "header_breadcrumb": c.header_breadcrumb,
                        "position": c.position,
                        "token_count": c.token_count,
                        "content_sha256": c.content_sha256,
                        "service": c.service,
                        "boost_score": c.boost_score,
                        "runbook_last_updated": c.runbook_last_updated.isoformat(),
                    },
                )
                for c, emb in zip(batch, embeddings, strict=True)
            ]
            await self._qdrant.upsert(
                collection_name=settings.qdrant_collection_name,
                points=points,
            )

    async def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Implements IndexerProtocol."""
        await self._qdrant.delete(
            collection_name=settings.qdrant_collection_name,
            points_selector=PointIdsList(points=[_qdrant_id(cid) for cid in chunk_ids]),
        )

    async def search(self, query: str, k: int = 20) -> list[tuple[Chunk, float]]:
        vector = (await self.embed([query]))[0]
        return await self.search_by_embedding(vector, k)

    async def search_by_embedding(
        self,
        embedding: list[float],
        k: int = 20,
    ) -> list[tuple[Chunk, float]]:
        try:
            response = await self._qdrant.query_points(
                collection_name=settings.qdrant_collection_name,
                query=embedding,
                limit=k,
                with_payload=True,
            )
            points = response.points
        except UnexpectedResponse as exc:
            if exc.status_code != 404:
                raise
            points = await self._legacy_search(embedding, k)
        chunks: list[tuple[Chunk, float]] = []
        for r in points:
            try:
                chunks.append((_chunk_from_payload(r.id, r.payload), float(r.score)))
            except Exception as exc:
                logger.warning("Skipping malformed Qdrant result: %s", exc)
        return chunks

    async def list_chunks(self, limit: int = 2000) -> list[Chunk]:
        """Load stored chunks so BM25 can be rebuilt after process restart."""
        try:
            records, _offset = await self._qdrant.scroll(
                collection_name=settings.qdrant_collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.warning("Could not load Qdrant payloads for BM25: %s", exc)
            return []
        chunks: list[Chunk] = []
        for record in records or []:
            try:
                chunks.append(_chunk_from_payload(record.id, record.payload))
            except Exception as exc:
                logger.warning("Skipping malformed Qdrant payload: %s", exc)
        return chunks

    async def _legacy_search(self, embedding: list[float], k: int) -> list[Any]:
        """Use Qdrant's legacy search route for pre-query_points servers."""
        headers = {"api-key": settings.qdrant_api_key} if settings.qdrant_api_key else {}
        path = f"/collections/{settings.qdrant_collection_name}/points/search"
        async with httpx.AsyncClient(
            base_url=settings.qdrant_url,
            headers=headers,
            timeout=10.0,
        ) as client:
            response = await client.post(
                path,
                json={"vector": embedding, "limit": k, "with_payload": True},
            )
            response.raise_for_status()
        return [SimpleNamespace(**point) for point in response.json().get("result", [])]

    async def get_boost_score(self, chunk_id: str) -> float | None:
        """Implements _ChunkStoreProtocol for feedback loop."""
        results = await self._qdrant.retrieve(
            collection_name=settings.qdrant_collection_name,
            ids=[_qdrant_id(chunk_id)],
            with_payload=True,
        )
        if not results:
            return None
        return (results[0].payload or {}).get("boost_score")

    async def set_boost_score(self, chunk_id: str, boost_score: float) -> None:
        """Implements _ChunkStoreProtocol for feedback loop."""
        await self._qdrant.set_payload(
            collection_name=settings.qdrant_collection_name,
            payload={"boost_score": boost_score},
            points=[_qdrant_id(chunk_id)],
        )
