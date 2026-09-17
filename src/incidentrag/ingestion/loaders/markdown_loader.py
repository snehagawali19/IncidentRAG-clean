"""
Markdown runbook loader.

Reads a ``.md`` file, extracts ``RunbookMetadata`` from YAML front matter
(or infers it from the file content), and returns the raw text ready for
chunking.

Front matter format (YAML block between ``---`` fences at file top):

    ---
    runbook_id: rb-001
    title: Payment Service Memory Issues
    service: payment-service
    environment: [prod, staging]
    severity_applicable: [sev1, sev2]
    author: platform-team
    created_at: 2024-01-15T00:00:00Z
    last_updated_at: 2024-09-10T00:00:00Z
    tags: [memory, oom, payment]
    related_runbook_ids: []
    source_uri: file://data/runbooks/payment_service_memory.md
    ---

If no front matter is present, ``MarkdownLoader`` infers as much as it can
from the file path and the H1 heading.

Do NOT import from references/ — this is a fresh implementation.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...core.models import Chunk, RunbookMetadata, Severity
from ..chunking.semantic_chunker import SemanticMarkdownChunker

# ── YAML-like front matter helpers ───────────────────────────────────────

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def _parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """
    Extract YAML front matter and return ``(data_dict, body_without_front_matter)``.

    Uses a minimal line-by-line parser so we avoid pulling in ``PyYAML`` as a
    hard requirement.  Supported value types:
      - strings
      - bracket-lists: ``[a, b, c]``
      - ISO-8601 datetimes
      - integers
    """
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text

    raw = m.group(1)
    body = text[m.end() :]
    data: dict[str, Any] = {}

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()

        # Bracket list: [a, b, c]
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1]
            items = [v.strip().strip("\"'") for v in inner.split(",") if v.strip()]
            data[key] = items
        else:
            # Try integer
            try:
                data[key] = int(value)
            except ValueError:
                # Try ISO datetime
                try:
                    data[key] = datetime.fromisoformat(
                        value.rstrip("Z").replace("Z", "+00:00")
                    )
                except ValueError:
                    data[key] = value.strip("\"'")

    return data, body


def _infer_title(text: str, path: Path) -> str:
    """Return the H1 heading text, or derive a title from the filename."""
    m = _H1_RE.search(text)
    if m:
        return m.group(1).strip()
    return path.stem.replace("_", " ").replace("-", " ").title()


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ── MarkdownLoader ────────────────────────────────────────────────────────


class MarkdownLoader:
    """
    Loads a Markdown runbook from disk, extracts metadata, and chunks it.

    Usage::

        loader = MarkdownLoader()
        metadata, chunks = loader.load("data/runbooks/payment_service_memory.md")

    The chunker used defaults to ``SemanticMarkdownChunker`` with stock
    settings.  Pass a custom ``chunker`` instance to override.
    """

    def __init__(
        self,
        chunker: SemanticMarkdownChunker | None = None,
        default_author: str = "unknown",
        default_service: str = "unknown-service",
        default_environment: list[str] | None = None,
    ) -> None:
        self._chunker = chunker or SemanticMarkdownChunker()
        self._default_author = default_author
        self._default_service = default_service
        self._default_environment = default_environment or ["prod"]

    # ── Public API ────────────────────────────────────────────────────────

    def load(self, path: str) -> tuple[RunbookMetadata, list[Chunk]]:
        """
        Load a Markdown file and return ``(metadata, chunks)``.

        Args:
            path: Absolute or relative path to the ``.md`` file.

        Returns:
            A tuple of the parsed ``RunbookMetadata`` and the list of
            ``Chunk`` objects produced by the chunker.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError:        If the file is empty after stripping front matter.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Runbook not found: {path}")

        raw_text = file_path.read_text(encoding="utf-8")
        if not raw_text.strip():
            raise ValueError(f"Runbook file is empty: {path}")

        front_matter, body = _parse_front_matter(raw_text)
        metadata = self._build_metadata(front_matter, body, file_path)
        chunks = self._chunker.chunk(body, metadata)

        return metadata, chunks

    # ── Metadata builder ──────────────────────────────────────────────────

    def _build_metadata(
        self,
        fm: dict[str, Any],
        body: str,
        path: Path,
    ) -> RunbookMetadata:
        """
        Build ``RunbookMetadata`` from front matter, falling back to inferred
        values where front matter is absent.
        """
        now = _now_utc()
        title = fm.get("title") or _infer_title(body, path)

        # Severity list: normalise strings to Severity enum members
        raw_severities: list[str] = fm.get("severity_applicable", [])
        severities: list[Severity] = []
        for s in raw_severities:
            try:
                severities.append(Severity(s.lower()))
            except ValueError:
                pass  # ignore unknown severity strings

        # Fallback to all severities if none specified
        if not severities:
            severities = list(Severity)

        # Environment list
        raw_env = fm.get("environment", self._default_environment)
        environments: list[str] = (
            raw_env if isinstance(raw_env, list) else [str(raw_env)]
        )

        # Timestamps — accept datetime objects or ISO strings
        def _to_dt(val: Any, fallback: datetime) -> datetime:
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                try:
                    return datetime.fromisoformat(
                        val.rstrip("Z").replace("Z", "+00:00")
                    )
                except ValueError:
                    pass
            return fallback

        created_at = _to_dt(fm.get("created_at"), now)
        last_updated_at = _to_dt(fm.get("last_updated_at"), now)

        return RunbookMetadata(
            runbook_id=str(fm.get("runbook_id", f"rb-{uuid.uuid4().hex[:8]}")),
            title=title,
            service=str(fm.get("service", self._default_service)),
            environment=environments,
            severity_applicable=severities,
            author=str(fm.get("author", self._default_author)),
            created_at=created_at,
            last_updated_at=last_updated_at,
            version=int(fm.get("version", 1)),
            tags=list(fm.get("tags", [])),
            related_runbook_ids=list(fm.get("related_runbook_ids", [])),
            source_uri=str(fm.get("source_uri", f"file://{path.as_posix()}")),
        )
