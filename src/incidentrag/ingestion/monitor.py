"""
Ingestion Monitor — detects dead chunks and stale runbooks.

This module provides ongoing health checks for the knowledge base:

  - **Dead chunks**: chunks that have never been retrieved, or whose last
    retrieval timestamp is older than ``dead_threshold_days`` days.
  - **Stale runbooks**: runbooks whose ``last_updated_at`` is older than
    ``stale_threshold_days`` days.

Both methods are async so they can run inside the async request path or be
scheduled as background tasks without blocking the event loop.

From SPEC.md §6.3 — no imports from references/.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..core.models import Chunk


class IngestionMonitor:
    """
    Continuous health monitoring for the ingested knowledge base.

    Typical usage::

        monitor = IngestionMonitor(dead_threshold_days=30, stale_threshold_days=90)

        dead   = await monitor.find_dead_chunks(all_chunks)
        stale  = await monitor.find_stale_runbooks(all_chunks)

        if dead:
            logger.warning("Dead chunks detected: %d", len(dead))
        if stale:
            logger.warning("Stale runbooks: %s", stale)
    """

    def __init__(
        self,
        dead_threshold_days: int = 30,
        stale_threshold_days: int = 90,
    ) -> None:
        """
        Args:
            dead_threshold_days:   A chunk not retrieved within this many days
                                   (or never retrieved) is considered dead.
            stale_threshold_days:  A runbook not updated within this many days
                                   is considered stale.
        """
        if dead_threshold_days <= 0:
            raise ValueError("dead_threshold_days must be positive")
        if stale_threshold_days <= 0:
            raise ValueError("stale_threshold_days must be positive")

        self.dead_threshold = timedelta(days=dead_threshold_days)
        self.stale_threshold = timedelta(days=stale_threshold_days)

    # ── Public async API ──────────────────────────────────────────────────

    async def find_dead_chunks(self, all_chunks: list[Chunk]) -> list[Chunk]:
        """
        Return chunks that are dead — either never retrieved, or not
        retrieved within ``dead_threshold_days``.

        A chunk is considered dead if:
          * ``retrieval_count == 0``  (never surfaced in any query), OR
          * ``last_retrieved_at < cutoff``  (retrieved but not recently)

        Note: chunks with ``retrieval_count > 0`` but ``last_retrieved_at``
        set to ``None`` are treated as alive (data integrity guard).

        Args:
            all_chunks: Full list of ``Chunk`` objects currently in the index.

        Returns:
            Subset of ``all_chunks`` that qualify as dead.
        """
        cutoff = datetime.now(timezone.utc) - self.dead_threshold
        dead: list[Chunk] = []

        for chunk in all_chunks:
            never_retrieved = chunk.retrieval_count == 0
            stale_retrieval = (
                chunk.last_retrieved_at is not None
                and _ensure_aware(chunk.last_retrieved_at) < cutoff
            )
            if never_retrieved or stale_retrieval:
                dead.append(chunk)

        return dead

    async def find_stale_runbooks(self, all_chunks: list[Chunk]) -> set[str]:
        """
        Return the set of ``runbook_id`` values whose source runbook has not
        been updated within ``stale_threshold_days``.

        Args:
            all_chunks: Full list of ``Chunk`` objects currently in the index.

        Returns:
            Set of stale ``runbook_id`` strings.
        """
        cutoff = datetime.now(timezone.utc) - self.stale_threshold
        stale: set[str] = set()

        for chunk in all_chunks:
            if _ensure_aware(chunk.runbook_last_updated) < cutoff:
                stale.add(chunk.runbook_id)

        return stale

    # ── Summary helper ────────────────────────────────────────────────────

    async def summary(self, all_chunks: list[Chunk]) -> dict[str, object]:
        """
        Return a combined health summary dict.  Useful for health-check
        endpoints and observability dashboards.

        Returns a dict with keys:
            ``total_chunks``, ``dead_chunks``, ``stale_runbooks``,
            ``dead_chunk_ids``, ``stale_runbook_ids``
        """
        dead = await self.find_dead_chunks(all_chunks)
        stale = await self.find_stale_runbooks(all_chunks)

        return {
            "total_chunks": len(all_chunks),
            "dead_chunks": len(dead),
            "stale_runbooks": len(stale),
            "dead_chunk_ids": [c.chunk_id for c in dead],
            "stale_runbook_ids": sorted(stale),
        }


# ── Internal helpers ──────────────────────────────────────────────────────


def _ensure_aware(dt: datetime) -> datetime:
    """Return ``dt`` with UTC timezone attached if it is naive."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
