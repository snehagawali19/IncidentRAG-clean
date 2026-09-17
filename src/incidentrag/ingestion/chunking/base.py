"""
Abstract base class for all IncidentRAG document chunkers.

Concrete implementations MUST satisfy these invariants:
  1. Code fences (``` ... ```) are never split across chunk boundaries.
  2. Markdown tables are never split across chunk boundaries.
  3. Numbered step groups are never split across chunk boundaries.
  4. Every emitted Chunk has a non-empty ``header_path`` whose first element
     equals the document title (H1 injection from structchunk pattern).
  5. ``content_sha256`` is deterministic — identical input always yields the
     same hash (no UUIDs or timestamps baked into the hash preimage).

Do NOT import from references/ — this is a fresh implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...core.models import Chunk, RunbookMetadata


class BaseChunker(ABC):
    """
    Minimal abstract interface every chunker must implement.

    Downstream code should type-hint against ``BaseChunker`` (or the
    ``ChunkerProtocol`` in ``core.protocols``) rather than a concrete class
    so that alternative strategies (fixed-size, sliding-window, etc.) can be
    swapped in for benchmarking without touching calling code.
    """

    # ── Public API ────────────────────────────────────────────────────────

    @abstractmethod
    def chunk(self, text: str, metadata: RunbookMetadata) -> list[Chunk]:
        """
        Chunk a raw Markdown runbook into semantically coherent ``Chunk``
        objects.

        Args:
            text:     Full Markdown source of the runbook (UTF-8 string).
            metadata: Parsed ``RunbookMetadata`` for this document.

        Returns:
            Ordered list of ``Chunk`` objects that collectively cover the
            entire document.  Position values are 0-indexed and contiguous.
        """
        ...

    # ── Shared utilities available to all subclasses ──────────────────────

    @staticmethod
    def _approx_tokens(text: str) -> int:
        """
        Rough token count approximation: 1 token ≈ 4 characters.

        This avoids a hard dependency on ``tiktoken`` in the chunking hot-path.
        Callers that need exact counts should post-process with their tokeniser
        of choice.
        """
        return max(1, len(text) // 4)
