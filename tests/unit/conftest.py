"""Shared fixtures using the REAL Phase 1 models."""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

import pytest

from incidentrag.core.models import (
    BlastRadiusResult,
    Chunk,
    ChunkType,
    Claim,
    Evidence,
    ExtractedEntities,
    IncidentAssessment,
    IncidentCategory,
    Query,
    RawAlert,
    RemediationAction,
    RetrievalResult,
    RetrievalScores,
    RiskLevel,
    ServiceNode,
    Severity,
)


# ── Chunks ────────────────────────────────────────────────────────────────

def make_chunk(
    chunk_id: str = "c1",
    content: str = "Sample runbook content about OOMKilled containers and memory limits.",
    runbook_id: str = "rb-payment-memory",
    service: str = "payment-service",
    header_breadcrumb: str = "# Payment Service > ## Memory Issues",
    chunk_type: ChunkType = ChunkType.PROSE,
    position: int = 0,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        runbook_id=runbook_id,
        chunk_type=chunk_type,
        content=content,
        header_path=header_breadcrumb.replace("#", "").split(">"),
        header_breadcrumb=header_breadcrumb,
        position=position,
        token_count=len(content.split()),
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        service=service,
        severity_applicable=[Severity.SEV1, Severity.SEV2],
        runbook_last_updated=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        make_chunk("c1", "OOMKilled containers should be diagnosed via kubectl describe pod"),
        make_chunk("c2", "Increase memory limits with kubectl set resources deploy/payment"),
        make_chunk("c3", "JVM heap size configured via JAVA_OPTS environment variable"),
    ]


# ── Alerts & Queries ──────────────────────────────────────────────────────

def make_raw_alert(
    alert_id: str = "alert-001",
    alert_text: str = "payment-service OOMKilled in production namespace",
    service: str = "payment-service",
    env: str = "production",
    severity: Severity = Severity.SEV1,
) -> RawAlert:
    return RawAlert(
        alert_id=alert_id,
        source="pagerduty",
        received_at=datetime.now(timezone.utc),
        raw_payload={"raw": "payload"},
        service_name=service,
        environment=env,
        severity=severity,
        alert_text=alert_text,
        metric_name="container_memory_usage_bytes",
        metric_value=536870912.0,  # 512 MiB
        threshold=536870912.0,
    )


def make_query(alert: RawAlert | None = None, category: IncidentCategory = IncidentCategory.PERFORMANCE_DEGRADATION) -> Query:
    alert = alert or make_raw_alert()
    return Query(
        query_id="query-001",
        original_alert=alert,
        entities=ExtractedEntities(
            service=alert.service_name,
            environment=alert.environment,
            metric=alert.metric_name,
            metric_value=alert.metric_value,
            threshold=alert.threshold,
            error_signature="OOMKilled",
        ),
        category=category,
        fanout_queries=[
            "payment-service OOMKilled remediation",
            "JVM heap memory configuration",
            "kubectl set resources memory limit",
            "container memory limit increase",
            "past OOMKilled incidents payment",
        ],
        hypothetical_document="When payment-service is OOMKilled, restart the deployment...",
    )


@pytest.fixture
def sample_alert() -> RawAlert:
    return make_raw_alert()


@pytest.fixture
def sample_query(sample_alert) -> Query:
    return make_query(sample_alert)


# ── Retrieval ─────────────────────────────────────────────────────────────

def make_retrieval_scores(final_score: float = 0.85) -> RetrievalScores:
    return RetrievalScores(
        dense_similarity=0.82,
        bm25_score=12.5,
        rrf_score=0.033,
        reranker_score=final_score,
        recency_boost=1.0,
        feedback_boost=1.0,
        graph_hops_from_query=0,
        final_score=final_score,
    )


def make_retrieval_result(chunk: Chunk, final_score: float = 0.85, matched_query: str = "OOMKilled memory") -> RetrievalResult:
    return RetrievalResult(
        chunk=chunk,
        scores=make_retrieval_scores(final_score),
        retrieved_via=["dense", "bm25"],
        query_that_matched=matched_query,
    )


@pytest.fixture
def sample_retrieval_results(sample_chunks) -> list[RetrievalResult]:
    return [
        make_retrieval_result(c, final_score=0.9 - i * 0.1)
        for i, c in enumerate(sample_chunks)
    ]


# ── Graph ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_blast_radius() -> BlastRadiusResult:
    return BlastRadiusResult(
        origin_service="payment-service",
        hop_depth=2,
        affected_services=[
            ServiceNode(service_name="billing-service", tier="api", environment="production", criticality="high"),
            ServiceNode(service_name="notification-service", tier="core", environment="production", criticality="medium"),
        ],
        critical_path_services=["billing-service"],
    )


# ── Evidence & Claims ─────────────────────────────────────────────────────

def make_evidence(chunk: Chunk | None = None, relevance: str = "direct") -> Evidence:
    chunk = chunk or make_chunk()
    return Evidence(
        chunk_id=chunk.chunk_id,
        runbook_id=chunk.runbook_id,
        header_breadcrumb=chunk.header_breadcrumb,
        excerpt=chunk.content[:200],
        relevance=relevance,
    )


def make_claim(statement: str = "JVM heap exceeded container memory limit", confidence: float = 0.85) -> Claim:
    return Claim(
        statement=statement,
        evidence=[make_evidence()],
        confidence=confidence,
    )


# ── Assessment ────────────────────────────────────────────────────────────

def make_remediation_action(
    description: str = "Increase memory limit to 1Gi",
    command: str | None = "kubectl set resources deploy/payment-service -n pay --limits=memory=1Gi",
    risk_level: RiskLevel = RiskLevel.MEDIUM,
    action_type: str = "kubectl",
) -> RemediationAction:
    return RemediationAction(
        description=description,
        command=command,
        action_type=action_type,  # type: ignore[arg-type]
        risk_level=risk_level,
        evidence=[make_evidence()],
        reversible=True,
        estimated_duration_seconds=30,
        expected_impact="Container will be recreated with 1Gi memory limit",
        rollback_command="kubectl set resources deploy/payment-service -n pay --limits=memory=512Mi",
    )


def make_assessment(
    query: Query | None = None,
    incident_id: str = "inc-001",
    actions: list[RemediationAction] | None = None,
) -> IncidentAssessment:
    query = query or make_query()
    return IncidentAssessment(
        incident_id=incident_id,
        query_id=query.query_id,
        root_cause=make_claim(),
        contributing_factors=[],
        overall_confidence=0.80,
        evidence_chunks_used=["c1", "c2"],
        proposed_actions=actions if actions is not None else [make_remediation_action()],
        diagnostic_actions=[],
        escalate_to_human=False,
        llm_model="gpt-4o",
        total_tokens=1500,
        cost_usd=0.02,
    )


@pytest.fixture
def sample_assessment(sample_query) -> IncidentAssessment:
    return make_assessment(sample_query)
