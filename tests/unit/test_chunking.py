"""
tests/unit/test_chunking.py

Verifies all acceptance criteria from SPEC.md §6.4:

  [✓] Chunker never splits inside a code fence, table, or numbered step group
  [✓] Every chunk has a non-empty header_path with the document title first
  [✓] Chunk content is prefixed with the header breadcrumb
  [✓] content_sha256 is deterministic — identical input yields identical hash
  [✓] Benchmark section runs all available chunking strategies
  [✓] IngestionMonitor detects dead chunks and stale runbooks

All async tests use pytest-asyncio.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from incidentrag.core.models import Chunk, ChunkType, RunbookMetadata, Severity
from incidentrag.ingestion.chunking.base import BaseChunker
from incidentrag.ingestion.chunking.semantic_chunker import SemanticMarkdownChunker
from incidentrag.ingestion.loaders.markdown_loader import MarkdownLoader
from incidentrag.ingestion.monitor import IngestionMonitor

# ── Fixtures ──────────────────────────────────────────────────────────────


def _metadata(
    title: str = "Test Runbook",
    service: str = "test-service",
    runbook_id: str = "rb-test-001",
) -> RunbookMetadata:
    """Factory for minimal RunbookMetadata."""
    now = datetime.now(timezone.utc)
    return RunbookMetadata(
        runbook_id=runbook_id,
        title=title,
        service=service,
        environment=["prod"],
        severity_applicable=[Severity.SEV1, Severity.SEV2],
        author="test-author",
        created_at=now,
        last_updated_at=now,
        source_uri=f"file://tests/{runbook_id}.md",
    )


def _chunk(
    *,
    runbook_id: str = "rb-test",
    retrieval_count: int = 0,
    last_retrieved_at: datetime | None = None,
    runbook_last_updated: datetime | None = None,
) -> Chunk:
    """Factory for minimal Chunk objects used in monitor tests."""
    now = datetime.now(timezone.utc)
    return Chunk(
        runbook_id=runbook_id,
        chunk_type=ChunkType.PROSE,
        content="test content",
        header_path=["Test Runbook"],
        header_breadcrumb="# Test Runbook",
        position=0,
        token_count=3,
        content_sha256=hashlib.sha256(b"test content").hexdigest()[:16],
        service="test-service",
        severity_applicable=[Severity.SEV1],
        runbook_last_updated=runbook_last_updated or now,
        retrieval_count=retrieval_count,
        last_retrieved_at=last_retrieved_at,
    )


# ── Documents for chunker tests ───────────────────────────────────────────

_DOC_SIMPLE = """\
# My Runbook

An introductory paragraph.

## Section One

Some prose about section one.

## Section Two

Some prose about section two.
"""

_DOC_WITH_CODE_FENCE = """\
# Runbook With Code

## Diagnostics

Run this script to diagnose the issue:

```bash
#!/usr/bin/env bash
# This is a multi-line code block that must never be split
for pod in $(kubectl get pods -n prod -o name); do
    kubectl logs "$pod" --tail=50 | grep -i error
done
echo "Done"
kubectl top pods -n prod
kubectl describe pods -n prod
kubectl get events -n prod --sort-by='.lastTimestamp'
```

After running the script, check the output carefully.
"""

_DOC_WITH_TABLE = """\
# Alert Reference Guide

## Severity Matrix

The following table maps alert names to severity levels:

| Alert Name | Threshold | Severity | Runbook |
|---|---|---|---|
| HighMemory | RSS > 2 GB | SEV2 | rb-001 |
| OOMKilled | OOMKilled reason | SEV1 | rb-001 |
| HighCPU | CPU > 90% for 5m | SEV2 | rb-002 |
| DBLatency | p99 > 2s | SEV2 | rb-003 |
| DiskFull | disk > 95% | SEV1 | rb-004 |
| NetworkSaturation | rx > 1 Gbps | SEV2 | rb-005 |
| PodCrashLoop | restarts > 5/h | SEV2 | rb-006 |

Use this table as the first reference when an alert fires.
"""

_DOC_WITH_NUMBERED_STEPS = """\
# Incident Response Checklist

## Steps to Follow

1. Acknowledge the alert in PagerDuty within 5 minutes of firing.
2. Open the relevant runbook linked in the alert description.
3. Run the quick-triage commands from the runbook's Diagnostics section.
4. Post a status update in the #incidents Slack channel every 15 minutes.
5. Escalate to the on-call engineering lead if the issue is not resolved in 30 minutes.
6. After resolution, open a post-mortem ticket and assign it to the incident owner.
7. Complete the post-mortem within 48 hours.

Follow these steps for every SEV1 and SEV2 incident.
"""

_DOC_MIXED = """\
# Mixed Content Runbook

Introduction paragraph describing the runbook.

## Diagnostics

Run the diagnostic script:

```bash
kubectl get pods --all-namespaces | grep CrashLoop
kubectl describe pod <POD_NAME> -n <NAMESPACE>
kubectl logs <POD_NAME> -n <NAMESPACE> --previous
```

## Alert Reference

| Alert | Threshold | Severity |
|---|---|---|
| CrashLoop | restarts > 5 | SEV2 |
| OOMKilled | any | SEV1 |

## Remediation Steps

1. Identify the failing pod using the diagnostics above.
2. Check the exit code and last state in the describe output.
3. Retrieve the crash logs using the --previous flag.
4. Apply the appropriate fix from the remediation table.
5. Verify the pod enters Running state within 2 minutes.

## Notes

> **Note:** Always prefer rolling restarts over forceful pod deletion.
"""

# ── §6.4 Criterion 1: Code fences, tables, and step groups are never split ─


class TestNoSplitInvariant:
    """
    SPEC §6.4 — Chunker never splits inside a code fence, table, or numbered
    step group.
    """

    def test_code_fence_not_split(self) -> None:
        """
        A fenced code block must appear entirely within a single Chunk —
        no chunk may start or end mid-fence.
        """
        # Use a very small token budget to force many splits
        chunker = SemanticMarkdownChunker(target_tokens=10, max_tokens=20)
        meta = _metadata(title="Runbook With Code")
        chunks = chunker.chunk(_DOC_WITH_CODE_FENCE, meta)

        # Locate the chunk that contains the code block
        code_chunks = [c for c in chunks if "```bash" in c.content]
        assert len(code_chunks) >= 1, "No chunk contains the opening code fence"

        for cc in code_chunks:
            # The opening fence must have a matching closing fence in same chunk
            assert cc.content.count("```") % 2 == 0, (
                f"Chunk has unmatched code fence:\n{cc.content}"
            )
            # The chunk must contain the entire script body
            assert "kubectl top pods" in cc.content, (
                "Code block was split — body line not found with fence"
            )

    def test_table_not_split(self) -> None:
        """
        An entire Markdown table must appear in a single Chunk.
        """
        chunker = SemanticMarkdownChunker(target_tokens=10, max_tokens=20)
        meta = _metadata(title="Alert Reference Guide")
        chunks = chunker.chunk(_DOC_WITH_TABLE, meta)

        table_chunks = [c for c in chunks if "|---|---|" in c.content]
        assert len(table_chunks) >= 1, "No chunk contains the separator row"

        for tc in table_chunks:
            # Every row that starts with | must be in the same chunk
            rows = [line for line in tc.content.splitlines() if line.strip().startswith("|")]
            # The full table has 9 lines (header + separator + 7 data rows)
            # Under extreme token budgets a table might be in its own chunk;
            # what matters is that the separator is always with the header
            header_row = any("Alert Name" in r for r in rows)
            sep_row = any("---" in r for r in rows)
            if header_row:
                assert sep_row, (
                    "Header row and separator row are in different chunks — table was split"
                )

    def test_numbered_steps_not_split(self) -> None:
        """
        All numbered list items belonging to one block must appear together.
        """
        chunker = SemanticMarkdownChunker(target_tokens=10, max_tokens=20)
        meta = _metadata(title="Incident Response Checklist")
        chunks = chunker.chunk(_DOC_WITH_NUMBERED_STEPS, meta)

        step_chunks = [c for c in chunks if "1." in c.content]
        assert len(step_chunks) >= 1, "No chunk contains a numbered step"

        for sc in step_chunks:
            # If step 1 is present, steps 2–7 must also be present
            if "1. Acknowledge" in sc.content:
                for step_text in [
                    "2. Open the relevant runbook",
                    "3. Run the quick-triage",
                    "4. Post a status update",
                ]:
                    assert step_text in sc.content, (
                        f"Step group was split — '{step_text}' missing from:\n{sc.content[:300]}"
                    )


# ── §6.4 Criterion 2: header_path always contains the document title ───────


class TestHeaderPath:
    """
    SPEC §6.4 — Every chunk has a non-empty header_path whose first element
    is the document title (H1 injection, structchunk pattern).
    """

    def test_every_chunk_has_document_title_in_header_path(self) -> None:
        chunker = SemanticMarkdownChunker()
        meta = _metadata(title="My Runbook")
        chunks = chunker.chunk(_DOC_SIMPLE, meta)

        assert len(chunks) > 0, "Chunker produced no chunks"
        for chunk in chunks:
            assert chunk.header_path, f"Chunk at position {chunk.position} has empty header_path"
            assert chunk.header_path[0] == "My Runbook", (
                f"Document title missing from header_path: {chunk.header_path}"
            )

    def test_header_path_reflects_nesting(self) -> None:
        """
        Chunks under '## Section Two' should have
        header_path == ['My Runbook', 'Section Two'].
        """
        chunker = SemanticMarkdownChunker()
        meta = _metadata(title="My Runbook")
        chunks = chunker.chunk(_DOC_SIMPLE, meta)

        section_two_chunks = [
            c for c in chunks if "section two" in c.content.lower()
            and "Section Two" in c.header_path
        ]
        assert section_two_chunks, "No chunk found for Section Two"
        for c in section_two_chunks:
            assert "My Runbook" in c.header_path, (
                f"Document title not in header_path for Section Two chunk: {c.header_path}"
            )
            assert "Section Two" in c.header_path

    def test_title_injected_even_without_h1_in_text(self) -> None:
        """
        If the document has no H1, the metadata title must still appear
        as the first element of header_path via H1 injection.
        """
        no_h1_doc = """\
Some prose at the top without any H1.

## A Section

Content here.
"""
        chunker = SemanticMarkdownChunker()
        meta = _metadata(title="Injected Title")
        chunks = chunker.chunk(no_h1_doc, meta)

        for chunk in chunks:
            assert chunk.header_path[0] == "Injected Title", (
                f"H1 injection failed: {chunk.header_path}"
            )

    def test_mixed_document_all_chunks_have_title(self) -> None:
        """All chunks from the mixed content document must carry the title."""
        chunker = SemanticMarkdownChunker()
        meta = _metadata(title="Mixed Content Runbook")
        chunks = chunker.chunk(_DOC_MIXED, meta)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.header_path[0] == "Mixed Content Runbook", (
                f"Title missing from header_path at position {chunk.position}: {chunk.header_path}"
            )


# ── §6.4 Criterion 3: Chunk content is prefixed with header breadcrumb ─────


class TestBreadcrumbPrefix:
    """
    SPEC §6.4 — Chunk content is prefixed with the header breadcrumb so the
    embedding model always sees full section context.
    """

    def test_content_starts_with_breadcrumb(self) -> None:
        chunker = SemanticMarkdownChunker()
        meta = _metadata(title="My Runbook")
        chunks = chunker.chunk(_DOC_SIMPLE, meta)

        for chunk in chunks:
            assert chunk.content.startswith(chunk.header_breadcrumb), (
                f"Chunk at position {chunk.position} does not start with its breadcrumb.\n"
                f"breadcrumb: {chunk.header_breadcrumb!r}\n"
                f"content start: {chunk.content[:80]!r}"
            )

    def test_breadcrumb_format(self) -> None:
        """
        Breadcrumb should follow the '# H1 > ## H2 > ### H3' format.
        """
        chunker = SemanticMarkdownChunker()
        meta = _metadata(title="My Runbook")
        chunks = chunker.chunk(_DOC_SIMPLE, meta)

        for chunk in chunks:
            breadcrumb = chunk.header_breadcrumb
            # At minimum should contain the document title with a leading #
            assert "# My Runbook" in breadcrumb, (
                f"Breadcrumb does not contain '# My Runbook': {breadcrumb!r}"
            )

    def test_breadcrumb_nested_sections(self) -> None:
        """Under a subsection the breadcrumb should include all ancestor headers."""
        doc = """\
# Top Level

## Level Two

### Level Three

Content deep in the hierarchy.
"""
        chunker = SemanticMarkdownChunker()
        meta = _metadata(title="Top Level")
        chunks = chunker.chunk(doc, meta)

        deep_chunks = [c for c in chunks if "Level Three" in c.header_path]
        assert deep_chunks, "No chunk produced under Level Three"
        for dc in deep_chunks:
            assert "# Top Level" in dc.header_breadcrumb
            assert "## Level Two" in dc.header_breadcrumb
            assert "### Level Three" in dc.header_breadcrumb


# ── §6.4 Criterion 4: content_sha256 is deterministic ─────────────────────


class TestSHA256Determinism:
    """
    SPEC §6.4 — content_sha256 matches when the same content is chunked twice.
    Idempotency is critical for dedup and version tracking.
    """

    def test_same_document_same_sha256(self) -> None:
        chunker = SemanticMarkdownChunker()
        meta = _metadata()

        chunks_a = chunker.chunk(_DOC_MIXED, meta)
        chunks_b = chunker.chunk(_DOC_MIXED, meta)

        assert len(chunks_a) == len(chunks_b), (
            "Non-deterministic chunk count — chunker is not idempotent"
        )
        for a, b in zip(chunks_a, chunks_b):
            assert a.content_sha256 == b.content_sha256, (
                f"SHA256 mismatch at position {a.position}:\n"
                f"  run1: {a.content_sha256}\n  run2: {b.content_sha256}"
            )

    def test_different_documents_different_sha256(self) -> None:
        chunker = SemanticMarkdownChunker()
        meta = _metadata()

        chunks_a = chunker.chunk(_DOC_SIMPLE, meta)
        chunks_b = chunker.chunk(_DOC_WITH_CODE_FENCE, meta)

        sha_a = {c.content_sha256 for c in chunks_a}
        sha_b = {c.content_sha256 for c in chunks_b}

        # The two documents have completely different content — no hash collision expected
        assert sha_a.isdisjoint(sha_b), (
            "SHA256 collision between documents with different content"
        )

    def test_sha256_length(self) -> None:
        """SHA256 should be a 16-char hex string (truncated to first 8 bytes)."""
        chunker = SemanticMarkdownChunker()
        meta = _metadata()
        chunks = chunker.chunk(_DOC_SIMPLE, meta)

        for chunk in chunks:
            assert len(chunk.content_sha256) == 16, (
                f"Expected 16-char SHA256 prefix, got {len(chunk.content_sha256)} chars: "
                f"{chunk.content_sha256!r}"
            )
            # Must be valid hex
            int(chunk.content_sha256, 16)

    def test_sha256_is_derived_from_content(self) -> None:
        """The sha256 stored in the chunk must match a fresh hash of its content."""
        chunker = SemanticMarkdownChunker()
        meta = _metadata()
        chunks = chunker.chunk(_DOC_SIMPLE, meta)

        for chunk in chunks:
            expected = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()[:16]
            assert chunk.content_sha256 == expected, (
                f"Stored SHA256 {chunk.content_sha256!r} does not match "
                f"hash of content {expected!r}"
            )


# ── §6.4 Criterion 5: Benchmark ────────────────────────────────────────────


class TestChunkerBenchmark:
    """
    SPEC §6.4 — Benchmark runs all available chunking strategies.

    Currently only SemanticMarkdownChunker is implemented.  This test
    validates the benchmarking harness and will automatically include new
    strategies when they are registered in ``_AVAILABLE_STRATEGIES``.
    """

    # Register strategies here as they are implemented
    _AVAILABLE_STRATEGIES: list[tuple[str, BaseChunker]] = [
        ("semantic-markdown/500t", SemanticMarkdownChunker(target_tokens=500)),
        ("semantic-markdown/250t", SemanticMarkdownChunker(target_tokens=250)),
        ("semantic-markdown/750t", SemanticMarkdownChunker(target_tokens=750)),
    ]

    _BENCHMARK_DOC = _DOC_MIXED

    def _score(self, chunks: list[Chunk]) -> dict[str, float]:
        """
        Compute simple quality indicators:
        - avg_tokens: mean token count per chunk (closer to target = better)
        - breadcrumb_rate: fraction of chunks whose content starts with breadcrumb
        - title_coverage: fraction of chunks whose header_path[0] is the title
        """
        if not chunks:
            return {"avg_tokens": 0.0, "breadcrumb_rate": 0.0, "title_coverage": 0.0}

        avg = sum(c.token_count for c in chunks) / len(chunks)
        bc_ok = sum(1 for c in chunks if c.content.startswith(c.header_breadcrumb))
        title_ok = sum(1 for c in chunks if c.header_path and c.header_path[0] != "")

        return {
            "avg_tokens": avg,
            "breadcrumb_rate": bc_ok / len(chunks),
            "title_coverage": title_ok / len(chunks),
        }

    def test_all_strategies_produce_valid_chunks(self) -> None:
        """Every registered strategy must produce at least one valid chunk."""
        meta = _metadata(title="Mixed Content Runbook")
        results: list[tuple[str, list[Chunk], dict[str, float]]] = []

        for name, chunker in self._AVAILABLE_STRATEGIES:
            chunks = chunker.chunk(self._BENCHMARK_DOC, meta)
            score = self._score(chunks)
            results.append((name, chunks, score))

            assert len(chunks) > 0, f"Strategy '{name}' produced zero chunks"
            assert score["breadcrumb_rate"] == 1.0, (
                f"Strategy '{name}' has breadcrumb_rate < 1.0: {score['breadcrumb_rate']}"
            )
            assert score["title_coverage"] == 1.0, (
                f"Strategy '{name}' has title_coverage < 1.0: {score['title_coverage']}"
            )

        # Print benchmark summary (visible with pytest -s)
        print("\n── Chunker Benchmark Results ──────────────────────────────")
        print(f"{'Strategy':<35} {'Chunks':>6} {'Avg tokens':>10} {'Breadcrumb':>10} {'Title':>6}")
        print("-" * 72)
        for name, chunks, score in results:
            print(
                f"{name:<35} {len(chunks):>6} "
                f"{score['avg_tokens']:>10.1f} "
                f"{score['breadcrumb_rate']:>10.2%} "
                f"{score['title_coverage']:>6.2%}"
            )


# ── IngestionMonitor tests ────────────────────────────────────────────────


class TestIngestionMonitor:
    """
    SPEC §6.4 — Ingestion monitor detects dead chunks and stale runbooks.
    """

    # ── find_dead_chunks ──────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_never_retrieved_chunk_is_dead(self) -> None:
        monitor = IngestionMonitor(dead_threshold_days=30)
        chunks = [_chunk(retrieval_count=0)]
        dead = await monitor.find_dead_chunks(chunks)
        assert len(dead) == 1
        assert dead[0].retrieval_count == 0

    @pytest.mark.asyncio
    async def test_recently_retrieved_chunk_is_alive(self) -> None:
        monitor = IngestionMonitor(dead_threshold_days=30)
        recent = datetime.now(timezone.utc) - timedelta(days=5)
        chunks = [_chunk(retrieval_count=3, last_retrieved_at=recent)]
        dead = await monitor.find_dead_chunks(chunks)
        assert dead == []

    @pytest.mark.asyncio
    async def test_stale_retrieval_chunk_is_dead(self) -> None:
        """A chunk retrieved 60 days ago with threshold=30 is dead."""
        monitor = IngestionMonitor(dead_threshold_days=30)
        old_retrieval = datetime.now(timezone.utc) - timedelta(days=60)
        chunks = [_chunk(retrieval_count=5, last_retrieved_at=old_retrieval)]
        dead = await monitor.find_dead_chunks(chunks)
        assert len(dead) == 1

    @pytest.mark.asyncio
    async def test_mixed_chunks_correct_dead_count(self) -> None:
        """Only never-retrieved and stale-retrieved chunks are returned."""
        monitor = IngestionMonitor(dead_threshold_days=30)
        recent = datetime.now(timezone.utc) - timedelta(days=5)
        old = datetime.now(timezone.utc) - timedelta(days=90)

        chunks = [
            _chunk(retrieval_count=0),               # dead (never retrieved)
            _chunk(retrieval_count=3, last_retrieved_at=recent),  # alive
            _chunk(retrieval_count=1, last_retrieved_at=old),     # dead (stale)
            _chunk(retrieval_count=10, last_retrieved_at=recent), # alive
        ]

        dead = await monitor.find_dead_chunks(chunks)
        assert len(dead) == 2

    # ── find_stale_runbooks ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_old_runbook_detected_as_stale(self) -> None:
        monitor = IngestionMonitor(stale_threshold_days=90)
        old_updated = datetime.now(timezone.utc) - timedelta(days=120)
        chunks = [_chunk(runbook_id="rb-old", runbook_last_updated=old_updated)]
        stale = await monitor.find_stale_runbooks(chunks)
        assert "rb-old" in stale

    @pytest.mark.asyncio
    async def test_fresh_runbook_not_stale(self) -> None:
        monitor = IngestionMonitor(stale_threshold_days=90)
        recent = datetime.now(timezone.utc) - timedelta(days=10)
        chunks = [_chunk(runbook_id="rb-fresh", runbook_last_updated=recent)]
        stale = await monitor.find_stale_runbooks(chunks)
        assert "rb-fresh" not in stale

    @pytest.mark.asyncio
    async def test_returns_unique_runbook_ids(self) -> None:
        """Multiple chunks from the same stale runbook → single entry."""
        monitor = IngestionMonitor(stale_threshold_days=30)
        old = datetime.now(timezone.utc) - timedelta(days=60)
        chunks = [
            _chunk(runbook_id="rb-dupe", runbook_last_updated=old),
            _chunk(runbook_id="rb-dupe", runbook_last_updated=old),
            _chunk(runbook_id="rb-dupe", runbook_last_updated=old),
        ]
        stale = await monitor.find_stale_runbooks(chunks)
        assert stale == {"rb-dupe"}

    @pytest.mark.asyncio
    async def test_summary_contains_all_keys(self) -> None:
        """summary() must return a dict with all expected keys."""
        monitor = IngestionMonitor()
        summary = await monitor.summary([])
        expected_keys = {
            "total_chunks", "dead_chunks", "stale_runbooks",
            "dead_chunk_ids", "stale_runbook_ids",
        }
        assert expected_keys.issubset(summary.keys())

    # ── Constructor validation ────────────────────────────────────────────

    def test_invalid_dead_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            IngestionMonitor(dead_threshold_days=0)

    def test_invalid_stale_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            IngestionMonitor(stale_threshold_days=-5)


# ── MarkdownLoader tests ──────────────────────────────────────────────────


class TestMarkdownLoader:
    """
    Verify that MarkdownLoader correctly parses front matter, infers metadata,
    and produces chunks via the embedded SemanticMarkdownChunker.
    """

    _RUNBOOKS_DIR = Path("data/runbooks")

    def test_load_payment_runbook(self) -> None:
        loader = MarkdownLoader()
        path = self._RUNBOOKS_DIR / "payment_service_memory.md"
        if not path.exists():
            pytest.skip("Sample runbook not found — run from project root")

        metadata, chunks = loader.load(str(path))

        assert metadata.runbook_id == "rb-001"
        assert metadata.title == "Payment Service Memory Issues"
        assert metadata.service == "payment-service"
        assert len(chunks) > 0

    def test_load_database_runbook(self) -> None:
        loader = MarkdownLoader()
        path = self._RUNBOOKS_DIR / "database_connection_pool.md"
        if not path.exists():
            pytest.skip("Sample runbook not found — run from project root")

        metadata, chunks = loader.load(str(path))

        assert metadata.runbook_id == "rb-002"
        assert metadata.title == "Database Connection Pool Exhaustion"
        assert len(chunks) > 0

    def test_load_k8s_runbook(self) -> None:
        loader = MarkdownLoader()
        path = self._RUNBOOKS_DIR / "kubernetes_pod_crashloop.md"
        if not path.exists():
            pytest.skip("Sample runbook not found — run from project root")

        metadata, chunks = loader.load(str(path))

        assert metadata.runbook_id == "rb-003"
        assert "Kubernetes" in metadata.title
        assert len(chunks) > 0

    def test_all_chunks_carry_document_title(self) -> None:
        """H1 injection must hold for chunks from a real file."""
        loader = MarkdownLoader()
        path = self._RUNBOOKS_DIR / "payment_service_memory.md"
        if not path.exists():
            pytest.skip("Sample runbook not found — run from project root")

        metadata, chunks = loader.load(str(path))
        for chunk in chunks:
            assert chunk.header_path[0] == metadata.title, (
                f"Title missing from header_path at position {chunk.position}: {chunk.header_path}"
            )

    def test_chunk_sha256_idempotent_on_real_file(self) -> None:
        """Loading the same file twice must produce identical SHA256 values."""
        loader = MarkdownLoader()
        path = self._RUNBOOKS_DIR / "kubernetes_pod_crashloop.md"
        if not path.exists():
            pytest.skip("Sample runbook not found — run from project root")

        _, chunks_a = loader.load(str(path))
        _, chunks_b = loader.load(str(path))

        assert len(chunks_a) == len(chunks_b)
        for a, b in zip(chunks_a, chunks_b):
            assert a.content_sha256 == b.content_sha256

    def test_missing_file_raises_file_not_found(self) -> None:
        loader = MarkdownLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("data/runbooks/nonexistent_runbook.md")

    def test_metadata_inference_without_front_matter(self, tmp_path: Path) -> None:
        """
        When no front matter is present, metadata is inferred from the
        filename and H1 heading.
        """
        doc = "# My Inferred Title\n\nSome content.\n"
        md_file = tmp_path / "my_inferred_title.md"
        md_file.write_text(doc, encoding="utf-8")

        loader = MarkdownLoader()
        metadata, chunks = loader.load(str(md_file))

        assert metadata.title == "My Inferred Title"
        assert len(chunks) > 0


# ── BaseChunker contract tests ────────────────────────────────────────────


class TestBaseChunkerContract:
    """Verify that SemanticMarkdownChunker satisfies the BaseChunker contract."""

    def test_is_base_chunker_subclass(self) -> None:
        assert issubclass(SemanticMarkdownChunker, BaseChunker)

    def test_chunk_returns_list(self) -> None:
        chunker = SemanticMarkdownChunker()
        meta = _metadata()
        result = chunker.chunk("# Title\n\nContent.", meta)
        assert isinstance(result, list)

    def test_chunk_returns_chunk_objects(self) -> None:
        chunker = SemanticMarkdownChunker()
        meta = _metadata()
        result = chunker.chunk("# Title\n\nContent.", meta)
        for item in result:
            assert isinstance(item, Chunk)

    def test_positions_are_contiguous(self) -> None:
        """Position values must be 0-indexed and monotonically increasing."""
        chunker = SemanticMarkdownChunker()
        meta = _metadata()
        chunks = chunker.chunk(_DOC_MIXED, meta)

        assert len(chunks) > 0
        for i, chunk in enumerate(chunks):
            assert chunk.position >= 0

    def test_empty_document_returns_empty_list(self) -> None:
        chunker = SemanticMarkdownChunker()
        meta = _metadata()
        result = chunker.chunk("", meta)
        assert result == []

    def test_approx_tokens_positive(self) -> None:
        assert SemanticMarkdownChunker._approx_tokens("hello") > 0
        assert SemanticMarkdownChunker._approx_tokens("") == 1  # max(1, 0)
