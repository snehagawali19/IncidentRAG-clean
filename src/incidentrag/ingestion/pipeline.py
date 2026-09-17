"""
IngestionPipeline — orchestrates the full runbook ingestion flow.

Pipeline stages:
    1. **load**    — ``MarkdownLoader`` reads a ``.md`` file and returns
                     ``(RunbookMetadata, list[Chunk])``.
    2. **chunk**   — Already performed by the loader (via ``SemanticMarkdownChunker``).
    3. **monitor** — ``IngestionMonitor`` scans the newly loaded chunks
                     against the existing index for dead/stale signals.
    4. **index**   — An optional ``IndexerProtocol`` implementation stores
                     chunks in the vector index (Qdrant in Phase 4).

The pipeline is async throughout so it can run inside FastAPI request handlers
or CLI commands without blocking the event loop.

Do NOT import from references/ — fresh implementation.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from ..core.models import Chunk, RunbookMetadata
from .loaders.markdown_loader import MarkdownLoader
from .monitor import IngestionMonitor

logger = logging.getLogger(__name__)


# ── Indexer protocol ──────────────────────────────────────────────────────


@runtime_checkable
class IndexerProtocol(Protocol):
    """
    Minimal interface for a vector/keyword index backend.

    In Phase 4 this will be satisfied by the Qdrant-backed ``DenseIndex``
    and the BM25 ``BM25Index``.  For Phase 2 unit tests a simple
    in-memory mock can be used.
    """

    async def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """Upsert ``chunks`` into the index (idempotent by ``chunk_id``)."""
        ...

    async def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Delete chunks by ID (used for dead-chunk pruning)."""
        ...


# ── Ingestion result dataclass ────────────────────────────────────────────


class IngestionResult:
    """
    Immutable result object returned by ``IngestionPipeline.ingest_file``.
    """

    __slots__ = (
        "path",
        "metadata",
        "chunks",
        "dead_chunks",
        "stale_runbook_ids",
        "indexed",
        "error",
    )

    def __init__(
        self,
        *,
        path: str,
        metadata: RunbookMetadata | None = None,
        chunks: list[Chunk] | None = None,
        dead_chunks: list[Chunk] | None = None,
        stale_runbook_ids: set[str] | None = None,
        indexed: bool = False,
        error: Exception | None = None,
    ) -> None:
        self.path = path
        self.metadata = metadata
        self.chunks = chunks or []
        self.dead_chunks = dead_chunks or []
        self.stale_runbook_ids = stale_runbook_ids or set()
        self.indexed = indexed
        self.error = error

    @property
    def success(self) -> bool:
        return self.error is None

    def __repr__(self) -> str:
        return (
            f"IngestionResult(path={self.path!r}, "
            f"chunks={len(self.chunks)}, "
            f"dead={len(self.dead_chunks)}, "
            f"stale={len(self.stale_runbook_ids)}, "
            f"indexed={self.indexed}, "
            f"error={self.error!r})"
        )


# ── IngestionPipeline ─────────────────────────────────────────────────────


class IngestionPipeline:
    """
    Orchestrates the full load → chunk → monitor → index cycle.

    Typical usage::

        pipeline = IngestionPipeline(indexer=my_qdrant_indexer)
        results  = await pipeline.ingest_directory("data/runbooks/")

    Omit ``indexer`` during Phase 2 / unit tests — chunks are returned but
    not persisted.

    Args:
        loader:         ``MarkdownLoader`` instance (created with defaults if
                        omitted).
        monitor:        ``IngestionMonitor`` instance (created with defaults if
                        omitted).
        indexer:        Optional indexer conforming to ``IndexerProtocol``.
                        When ``None``, the index stage is skipped.
        existing_chunks: Chunks already in the index — used by the monitor to
                         detect stale/dead entries.  Pass an empty list for a
                         fresh index.
        concurrency:    Maximum number of files processed concurrently when
                        using ``ingest_directory``.
    """

    def __init__(
        self,
        loader: MarkdownLoader | None = None,
        monitor: IngestionMonitor | None = None,
        indexer: IndexerProtocol | None = None,
        existing_chunks: list[Chunk] | None = None,
        concurrency: int = 4,
    ) -> None:
        self._loader = loader or MarkdownLoader()
        self._monitor = monitor or IngestionMonitor()
        self._indexer = indexer
        self._existing_chunks: list[Chunk] = existing_chunks or []
        self._concurrency = max(1, concurrency)

    # ── Public API ────────────────────────────────────────────────────────

    async def ingest_file(self, path: str) -> IngestionResult:
        """
        Ingest a single Markdown runbook.

        Stage 1 — load + chunk
            ``MarkdownLoader.load()`` reads the file, parses front matter,
            and uses the embedded chunker to produce ``list[Chunk]``.

        Stage 2 — monitor
            The new chunks are appended to ``existing_chunks`` and passed to
            ``IngestionMonitor`` so stale/dead signals are computed in context
            of the whole index.

        Stage 3 — index
            If an ``IndexerProtocol`` implementation is configured, all new
            chunks are upserted.  Errors during indexing are caught and
            surfaced in ``IngestionResult.error`` without aborting monitoring.

        Returns:
            ``IngestionResult`` containing metadata, chunks, and health signals.
        """
        logger.info("Ingesting %s", path)
        try:
            # ── Stage 1: load + chunk ─────────────────────────────────────
            metadata, chunks = self._loader.load(path)
            logger.debug(
                "Loaded %s: %d chunks from runbook '%s'",
                path,
                len(chunks),
                metadata.runbook_id,
            )

            # ── Stage 2: monitor ──────────────────────────────────────────
            combined = self._existing_chunks + chunks
            dead_chunks = await self._monitor.find_dead_chunks(combined)
            stale_ids = await self._monitor.find_stale_runbooks(combined)

            if dead_chunks:
                logger.warning(
                    "%d dead chunks detected after ingesting %s",
                    len(dead_chunks),
                    path,
                )
            if stale_ids:
                logger.warning("Stale runbooks: %s", stale_ids)

            # ── Stage 3: index ────────────────────────────────────────────
            indexed = False
            if self._indexer is not None:
                await self._indexer.upsert_chunks(chunks)
                # Update our local view of existing chunks
                self._existing_chunks.extend(chunks)
                indexed = True
                logger.info("Indexed %d chunks for %s", len(chunks), metadata.runbook_id)

            return IngestionResult(
                path=path,
                metadata=metadata,
                chunks=chunks,
                dead_chunks=dead_chunks,
                stale_runbook_ids=stale_ids,
                indexed=indexed,
            )

        except Exception as exc:
            logger.error("Failed to ingest %s: %s", path, exc, exc_info=True)
            return IngestionResult(path=path, error=exc)

    async def ingest_directory(
        self,
        directory: str,
        glob_pattern: str = "**/*.md",
        on_result: Callable[[IngestionResult], None] | None = None,
    ) -> list[IngestionResult]:
        """
        Ingest all Markdown files under ``directory`` matching ``glob_pattern``.

        Files are processed with bounded concurrency (``self._concurrency``).
        Each result is optionally delivered to ``on_result`` as it completes.

        Args:
            directory:    Root directory to scan.
            glob_pattern: Glob relative to ``directory`` (default ``**/*.md``).
            on_result:    Optional callback invoked with each ``IngestionResult``.

        Returns:
            List of ``IngestionResult`` objects, one per file.
        """
        root = Path(directory)
        if not root.exists():
            raise FileNotFoundError(f"Ingestion directory not found: {directory}")

        paths = sorted(root.glob(glob_pattern))
        if not paths:
            logger.warning("No Markdown files found in %s matching %s", directory, glob_pattern)
            return []

        logger.info("Found %d Markdown files in %s", len(paths), directory)

        semaphore = asyncio.Semaphore(self._concurrency)
        results: list[IngestionResult] = []

        async def _bounded_ingest(p: Path) -> IngestionResult:
            async with semaphore:
                result = await self.ingest_file(str(p))
                if on_result:
                    on_result(result)
                return result

        tasks = [_bounded_ingest(p) for p in paths]
        results = list(await asyncio.gather(*tasks))

        total = len(results)
        ok = sum(1 for r in results if r.success)
        logger.info(
            "Ingestion complete: %d/%d files succeeded", ok, total
        )
        return results

    # ── Accessors ─────────────────────────────────────────────────────────

    @property
    def existing_chunks(self) -> list[Chunk]:
        """Read-only view of chunks known to the pipeline."""
        return list(self._existing_chunks)

    def register_existing_chunks(self, chunks: list[Chunk]) -> None:
        """
        Register chunks already present in the index so the monitor can
        evaluate them alongside newly ingested chunks.
        """
        self._existing_chunks.extend(chunks)
