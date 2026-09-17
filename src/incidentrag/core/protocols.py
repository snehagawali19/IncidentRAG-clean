"""
Protocol definitions for IncidentRAG.
Every implementation must conform to these structural interfaces.
Use these protocols for dependency injection and type-safe mocking in tests.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    Chunk,
    Claim,
    Evidence,
    ExternalIssue,
    IncidentAssessment,
    IssueFilters,
    Query,
    RawAlert,
    RetrievalResult,
)


@runtime_checkable
class IncidentSource(Protocol):
    """Read-only, manually invoked provider of external incident candidates."""

    async def fetch_candidates(self, filters: IssueFilters) -> list[ExternalIssue]:
        """Return a bounded, ranked set of sanitized candidates."""
        ...

    async def fetch_issue(self, issue_number: int) -> ExternalIssue:
        """Fetch one sanitized issue without mutating the source system."""
        ...

    def normalize(self, issue: ExternalIssue) -> RawAlert:
        """Map an external issue into the canonical IncidentRAG alert model."""
        ...


@runtime_checkable
class RetrieverProtocol(Protocol):
    """
    Hybrid retriever: fuses dense, BM25, and graph-traversal results.
    """

    async def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]:
        """Return the top-k scored RetrievalResult objects for a query."""
        ...


@runtime_checkable
class RerankerProtocol(Protocol):
    """
    Cross-encoder reranker that re-scores candidate chunks against the query.
    """

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Re-score and return top_k results ordered by reranker score."""
        ...


@runtime_checkable
class ChunkerProtocol(Protocol):
    """
    Document chunker that preserves runbook structure.
    Must never split inside fenced code blocks, tables, or numbered steps.
    """

    def chunk(self, document_text: str, metadata: dict) -> list[Chunk]:  # type: ignore[type-arg]
        """Chunk a runbook document into semantically coherent Chunk objects."""
        ...


@runtime_checkable
class GroundingVerifierProtocol(Protocol):
    """
    Grounding verifier: checks whether a Claim is substantiated by its Evidence.
    Every LLM-generated claim must pass through this before being surfaced.
    """

    async def verify(
        self,
        claim: Claim,
        evidence: list[Evidence],
    ) -> tuple[bool, str]:
        """
        Verify that the claim is grounded in the provided evidence.

        Returns:
            (is_grounded, reasoning) — bool flag and human-readable explanation.
        """
        ...


@runtime_checkable
class GraphClientProtocol(Protocol):
    """
    Async Neo4j graph client for service-dependency queries.
    """

    async def blast_radius(self, service: str, hops: int) -> list[str]:
        """Return service names reachable within `hops` from `service`."""
        ...

    async def related_runbooks(self, runbook_id: str, hops: int) -> list[str]:
        """Return runbook_ids related to `runbook_id` within `hops`."""
        ...


@runtime_checkable
class GeneratorProtocol(Protocol):
    """
    Structured LLM generator that produces grounded IncidentAssessment objects.
    """

    async def generate(
        self,
        query: Query,
        retrieval_results: list[RetrievalResult],
    ) -> IncidentAssessment:
        """Generate a fully structured, grounded incident assessment."""
        ...
