"""
Semantic Markdown chunker that preserves runbook structure.

Rules (from chunkana + structchunk patterns — reimplemented fresh):
  * NEVER splits inside a fenced code block (``` ... ```)
  * NEVER splits inside a Markdown table
  * NEVER splits inside a numbered step group
  * Every chunk content is prefixed with its full header breadcrumb so the
    embedding model sees full section context
  * H1 (document title) is injected into every chunk's header_path so no
    chunk is contextually orphaned

References (read-only, do NOT import):
  - references/rag-cookbooks  (chunk metadata pattern)
  - chunkana PyPI README       (code-fence protection)
  - structchunk PyPI README    (H1 injection, breadcrumbs)
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator

from ...core.models import Chunk, ChunkType, RunbookMetadata
from .base import BaseChunker

# ── Compiled regex patterns ───────────────────────────────────────────────

# Matches opening or closing triple-backtick fence (optionally with language tag)
_CODE_FENCE = re.compile(r"^```")

# Matches a Markdown ATX header: `## Some Heading`
_HEADER = re.compile(r"^(#{1,6})\s+(.+)$")

# Matches the first line of a numbered list item
_NUMBERED_STEP = re.compile(r"^\s*\d+\.\s+")

# Matches a Markdown table row (starts and ends with |)
_TABLE_ROW = re.compile(r"^\|.*\|$")

# Matches warning/caution/note callouts
_WARNING_BLOCK = re.compile(
    r"^(⚠️|Warning:|CAUTION:|NOTE:|> \*\*Warning|> \*\*Note|> \*\*Caution)",
    re.IGNORECASE,
)

# ── Block type sentinel used internally ──────────────────────────────────

_BLOCK_HEADER = "HEADER_SENTINEL"


class SemanticMarkdownChunker(BaseChunker):
    """
    Structure-aware chunker for incident runbooks written in Markdown.

    Chunking pipeline:
        1. ``_parse_blocks`` — scan the raw text line-by-line and produce
           atomic blocks.  Code fences, tables, and step groups are consumed
           as single indivisible blocks.
        2. ``_build_chunks`` — walk the block sequence maintaining a header
           stack.  Accumulate blocks until the token budget is reached, then
           flush a ``Chunk`` with the full header breadcrumb prepended.
        3. H1 injection — ``_make_chunk`` ensures the document title is
           always the first element of ``header_path`` (structchunk pattern).

    Token budget arithmetic uses a 4-char-per-token approximation to avoid
    a hard ``tiktoken`` dependency in the ingestion hot-path.
    """

    def __init__(
        self,
        target_tokens: int = 500,
        max_tokens: int = 800,
        min_tokens: int = 50,
        overlap_blocks: int = 1,
    ) -> None:
        """
        Args:
            target_tokens:  Soft cap — flush a chunk when accumulated content
                            reaches this many estimated tokens.
            max_tokens:     Hard cap — a single indivisible block that exceeds
                            this is still kept whole (never split mid-block).
            min_tokens:     Chunks below this size are merged with the next
                            chunk rather than emitted alone (not yet enforced —
                            placeholder for future merge pass).
            overlap_blocks: Number of trailing blocks to carry forward into
                            the next chunk for context continuity.
        """
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_blocks = overlap_blocks

    # ── Public interface ──────────────────────────────────────────────────

    def chunk(self, text: str, metadata: RunbookMetadata) -> list[Chunk]:
        """
        Chunk ``text`` into a list of ``Chunk`` objects.

        The returned list is ordered by position (0-indexed, contiguous).
        Running this method twice on the same ``(text, metadata)`` pair
        produces identical ``content_sha256`` values.
        """
        blocks = list(self._parse_blocks(text))
        return list(self._build_chunks(blocks, metadata))

    # ── Block parser ──────────────────────────────────────────────────────

    def _parse_blocks(self, text: str) -> Iterator[dict]:  # type: ignore[type-arg]
        """
        Scan ``text`` line-by-line and yield atomic block dicts.

        Block dict keys:
            type       — ``ChunkType`` member (or ``_BLOCK_HEADER`` sentinel)
            content    — full text of the block
            start_line — 0-indexed line number of the first line
            level      — heading level (only present for HEADER blocks)

        Guarantees:
            - A ``CODE_BLOCK`` block spans from the opening ``` to the
              closing ``` inclusive — it is never split.
            - A ``TABLE`` block spans all consecutive table rows — never split.
            - A ``STEP`` block spans all consecutive numbered-list items and
              their continuation lines — never split.
        """
        lines = text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # ── Fenced code block ────────────────────────────────────────
            if _CODE_FENCE.match(line):
                start = i
                i += 1
                # Consume until the matching closing fence
                while i < len(lines) and not _CODE_FENCE.match(lines[i]):
                    i += 1
                i += 1  # include closing fence line
                yield {
                    "type": ChunkType.CODE_BLOCK,
                    "content": "\n".join(lines[start:i]),
                    "start_line": start,
                }
                continue

            # ── ATX heading ──────────────────────────────────────────────
            m = _HEADER.match(line)
            if m:
                yield {
                    "type": _BLOCK_HEADER,
                    "level": len(m.group(1)),
                    "content": m.group(2).strip(),
                    "start_line": i,
                }
                i += 1
                continue

            # ── Numbered step group ──────────────────────────────────────
            if _NUMBERED_STEP.match(line):
                start = i
                i += 1
                # Consume continuation lines: indented lines or blank lines
                # that are followed by another numbered item or indented line
                while i < len(lines):
                    cur = lines[i]
                    # Blank line — peek ahead to see if the list continues
                    if cur.strip() == "":
                        if i + 1 < len(lines) and (
                            _NUMBERED_STEP.match(lines[i + 1])
                            or (lines[i + 1].startswith("   ") and lines[i + 1].strip())
                        ):
                            i += 1
                            continue
                        else:
                            break  # blank line ends the list
                    elif _NUMBERED_STEP.match(cur) or (
                        cur.startswith("   ") and cur.strip()
                    ):
                        i += 1
                        continue
                    else:
                        break
                yield {
                    "type": ChunkType.STEP,
                    "content": "\n".join(lines[start:i]).strip(),
                    "start_line": start,
                }
                continue

            # ── Markdown table ────────────────────────────────────────────
            if _TABLE_ROW.match(line):
                start = i
                while i < len(lines) and _TABLE_ROW.match(lines[i]):
                    i += 1
                yield {
                    "type": ChunkType.TABLE,
                    "content": "\n".join(lines[start:i]),
                    "start_line": start,
                }
                continue

            # ── Warning / callout block ───────────────────────────────────
            if _WARNING_BLOCK.match(line):
                # Consume the entire blockquote (lines starting with >)
                start = i
                while i < len(lines) and (lines[i].startswith(">") or _WARNING_BLOCK.match(lines[i])):
                    i += 1
                yield {
                    "type": ChunkType.WARNING,
                    "content": "\n".join(lines[start:i]),
                    "start_line": start,
                }
                continue

            # ── Prose paragraph ───────────────────────────────────────────
            if line.strip():
                start = i
                while i < len(lines) and lines[i].strip():
                    i += 1
                yield {
                    "type": ChunkType.PROSE,
                    "content": "\n".join(lines[start:i]),
                    "start_line": start,
                }
                continue

            # ── Blank line ────────────────────────────────────────────────
            i += 1

    # ── Chunk builder ─────────────────────────────────────────────────────

    def _build_chunks(
        self,
        blocks: list[dict],  # type: ignore[type-arg]
        metadata: RunbookMetadata,
    ) -> Iterator[Chunk]:
        """
        Walk parsed blocks and accumulate them into ``Chunk`` objects.

        Header blocks update the header stack and trigger a flush of the
        current accumulator rather than being added to content.  Non-header
        blocks accumulate until the token budget is reached.

        Overlap: after flushing, the last ``self.overlap_blocks`` content
        blocks are carried forward so embedding context is not lost at
        chunk boundaries.
        """
        # Header stack: list of (level: int, text: str)
        header_stack: list[tuple[int, str]] = []
        current_blocks: list[dict] = []  # type: ignore[type-arg]
        chunk_position = 0

        for block in blocks:
            if block["type"] == _BLOCK_HEADER:
                # Flush accumulated content before updating the header stack
                if current_blocks:
                    yield self._make_chunk(
                        content_blocks=current_blocks,
                        header_stack=header_stack,
                        metadata=metadata,
                        position=chunk_position,
                    )
                    chunk_position += 1
                    # Carry overlap blocks forward for context continuity
                    current_blocks = current_blocks[-self.overlap_blocks :] if self.overlap_blocks else []

                # Update header stack: pop equal or deeper levels
                while header_stack and header_stack[-1][0] >= block["level"]:
                    header_stack.pop()
                header_stack.append((block["level"], block["content"]))
                continue

            # Regular content block
            current_blocks.append(block)

            # Check token budget (soft cap — never split a block already added)
            accumulated = "\n".join(b["content"] for b in current_blocks)
            if self._approx_tokens(accumulated) >= self.target_tokens:
                yield self._make_chunk(
                    content_blocks=current_blocks,
                    header_stack=header_stack,
                    metadata=metadata,
                    position=chunk_position,
                )
                chunk_position += 1
                # Carry overlap blocks forward
                current_blocks = current_blocks[-self.overlap_blocks :] if self.overlap_blocks else []

        # Final chunk — flush whatever remains
        if current_blocks:
            yield self._make_chunk(
                content_blocks=current_blocks,
                header_stack=header_stack,
                metadata=metadata,
                position=chunk_position,
            )

    # ── Chunk factory ─────────────────────────────────────────────────────

    def _make_chunk(
        self,
        content_blocks: list[dict],  # type: ignore[type-arg]
        header_stack: list[tuple[int, str]],
        metadata: RunbookMetadata,
        position: int,
    ) -> Chunk:
        """
        Construct a ``Chunk`` from accumulated blocks and the current header
        stack, applying H1 injection and breadcrumb prepending.

        H1 injection (structchunk pattern): ensures the document title is
        always ``header_path[0]`` so no chunk is contextually orphaned.

        Breadcrumb prepending: the full breadcrumb string is prepended to the
        chunk content so the embedding model sees full section context.
        """
        # Build header_path from stack (text only)
        header_path = [h[1] for h in header_stack]

        # H1 injection — always ensure the document title is first
        if not header_path or header_path[0] != metadata.title:
            header_path = [metadata.title, *header_path]

        # Breadcrumb: "# Title > ## Section > ### Sub-section"
        # Index-based # depth mirrors structchunk convention
        breadcrumb = " > ".join(
            f"{'#' * (idx + 1)} {title}" for idx, title in enumerate(header_path)
        )

        # Join block content and prepend breadcrumb
        raw_content = "\n\n".join(b["content"] for b in content_blocks)
        full_content = f"{breadcrumb}\n\n{raw_content}"

        # Determine dominant chunk type: CODE_BLOCK > TABLE > STEP > WARNING > PROSE
        type_priority: dict[ChunkType | str, int] = {
            ChunkType.CODE_BLOCK: 5,
            ChunkType.TABLE: 4,
            ChunkType.STEP: 3,
            ChunkType.WARNING: 2,
            ChunkType.PROSE: 1,
        }
        chunk_type = max(
            (b["type"] for b in content_blocks),
            key=lambda t: type_priority.get(t, 0),
            default=ChunkType.PROSE,
        )
        if not isinstance(chunk_type, ChunkType):
            chunk_type = ChunkType.PROSE

        # SHA-256 is computed on the full_content only (deterministic)
        sha = hashlib.sha256(full_content.encode("utf-8")).hexdigest()[:16]

        return Chunk(
            runbook_id=metadata.runbook_id,
            chunk_type=chunk_type,
            content=full_content,
            header_path=header_path,
            header_breadcrumb=breadcrumb,
            position=position,
            token_count=self._approx_tokens(full_content),
            content_sha256=sha,
            service=metadata.service,
            severity_applicable=metadata.severity_applicable,
            runbook_last_updated=metadata.last_updated_at,
        )
