"""
Canonical Pydantic v2 data models for IncidentRAG.
No module should define its own version of these types — always import from here.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════


class IncidentCategory(str, Enum):
    PERFORMANCE_DEGRADATION = "performance_degradation"
    COMPLETE_OUTAGE = "complete_outage"
    SECURITY_EVENT = "security_event"
    DEPENDENCY_FAILURE = "dependency_failure"
    CONFIGURATION_DRIFT = "configuration_drift"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    SEV1 = "sev1"  # Full outage
    SEV2 = "sev2"  # Major degradation
    SEV3 = "sev3"  # Minor issue
    SEV4 = "sev4"  # Notification only


class RiskLevel(str, Enum):
    LOW = "low"          # Safe to auto-execute
    MEDIUM = "medium"    # Requires approval
    HIGH = "high"        # Requires senior approval
    DANGEROUS = "dangerous"  # Should not be suggested


class ChunkType(str, Enum):
    HEADING = "heading"
    STEP = "step"
    CODE_BLOCK = "code_block"
    WARNING = "warning"
    PROSE = "prose"
    TABLE = "table"


# ═══════════════════════════════════════════════════════════════════════════
# INGESTION LAYER
# ═══════════════════════════════════════════════════════════════════════════


class RunbookMetadata(BaseModel):
    runbook_id: str
    title: str
    service: str                          # e.g. "payment-service"
    environment: list[str]                # ["prod", "staging"]
    severity_applicable: list[Severity]
    author: str
    created_at: datetime
    last_updated_at: datetime
    version: int = 1
    tags: list[str] = Field(default_factory=list)
    related_runbook_ids: list[str] = Field(default_factory=list)
    source_uri: str                       # notion://, confluence://, file://


class Chunk(BaseModel):
    """
    The atomic unit of the vector index.
    A single semantically-coherent section of a runbook.
    """

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    runbook_id: str
    chunk_type: ChunkType

    content: str                          # The actual text
    header_path: list[str]               # From structchunk: ["Payment Service", "Memory Issues", "Remediation Steps"]
    header_breadcrumb: str               # "# Payment Service > ## Memory Issues > ### Remediation Steps"

    position: int                         # 0-indexed position in runbook
    token_count: int

    # Retrieval metadata
    boost_score: float = 1.0             # From feedback loop: chunks used in resolutions get boosted
    retrieval_count: int = 0
    last_retrieved_at: datetime | None = None

    # Embedding metadata (from embspec pattern)
    embedding_model: str = "text-embedding-3-small"
    embedding_version: str = "v1"
    content_sha256: str                  # For dedup + version tracking

    # Full RunbookMetadata denormalized for filter performance
    service: str
    severity_applicable: list[Severity] = Field(default_factory=list)
    runbook_last_updated: datetime


# ═══════════════════════════════════════════════════════════════════════════
# ALERT + QUERY LAYER
# ═══════════════════════════════════════════════════════════════════════════


class RawAlert(BaseModel):
    """
    Alert as received from PagerDuty/Datadog/Prometheus.
    """

    alert_id: str
    source: Literal[
        "pagerduty", "datadog", "prometheus", "grafana", "opsgenie", "github"
    ]
    received_at: datetime
    raw_payload: dict[str, Any]

    # Extracted core fields
    service_name: str | None = None
    environment: str | None = None
    severity: Severity | None = None
    alert_text: str
    metric_name: str | None = None
    metric_value: float | None = None
    threshold: float | None = None


class IssueFilters(BaseModel):
    """User-selected filters for a manual external-issue fetch."""

    keyword: str | None = Field(default=None, max_length=200)
    component: str | None = Field(default=None, max_length=100)
    severity: str | None = Field(default=None, max_length=100)
    priority: str | None = Field(default=None, max_length=100)
    regression: bool = False
    updated_within_days: int = Field(default=90, ge=1, le=3650)
    limit: int = Field(default=2, ge=1, le=2)


class GitHubRateLimit(BaseModel):
    """Safe GitHub rate-limit metadata; never includes credentials."""

    remaining: int | None = None
    reset_at: datetime | None = None


class ExternalIssue(BaseModel):
    """Sanitized, bounded representation of an untrusted public issue."""

    repository: str
    number: int
    state: Literal["open", "closed"] = "open"
    title: str
    body: str
    labels: list[str] = Field(default_factory=list)
    author: str | None = None
    created_at: datetime
    updated_at: datetime
    comments_count: int = 0
    html_url: str
    api_url: str
    suitability_score: float = Field(default=0.0, ge=0.0, le=100.0)
    selection_explanation: list[str] = Field(default_factory=list)
    recent_comments: list[str] = Field(default_factory=list)


class GitHubLabel(BaseModel):
    """Safe subset of repository label metadata."""

    name: str
    color: str = ""
    description: str = ""


class ExtractedEntities(BaseModel):
    """
    Output of the entity extractor (Layer 2).
    All fields are optional (default None) — the extractor populates what it can.
    """

    service: str | None = None
    environment: str | None = None
    metric: str | None = None
    metric_value: float | None = None
    threshold: float | None = None
    host_identifier: str | None = None
    error_signature: str | None = None          # e.g. "OOMKilled"
    correlated_services: list[str] = Field(default_factory=list)


class Query(BaseModel):
    """
    Post-understanding query object. This is what gets passed to retrieval.
    """

    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_alert: RawAlert
    entities: ExtractedEntities
    category: IncidentCategory

    # Query fanout — 5 rewrites
    fanout_queries: list[str]

    # HyDE
    hypothetical_document: str
    hypothetical_embedding: list[float] | None = None

    # Filters derived from entities
    service_filter: str | None = None
    environment_filter: str | None = None


# ═══════════════════════════════════════════════════════════════════════════
# RETRIEVAL LAYER
# ═══════════════════════════════════════════════════════════════════════════


class RetrievalScores(BaseModel):
    """
    Per-chunk score breakdown for explainability.
    This is what makes IncidentRAG's retrieval explainable vs. Aurora's opaque.
    """

    dense_similarity: float | None       # Cosine similarity, 0-1
    bm25_score: float | None             # Raw BM25 score
    rrf_score: float                     # After Reciprocal Rank Fusion
    reranker_score: float | None         # Cross-encoder score
    recency_boost: float                 # 0.5 ** (age_days / 90)
    feedback_boost: float                # From chunk.boost_score
    graph_hops_from_query: int | None    # 0 = direct, 1+ = via graph traversal
    final_score: float                   # Composite

    def explanation(self) -> str:
        """Human-readable score breakdown for the debugger UI."""
        parts: list[str] = []
        if self.dense_similarity is not None:
            parts.append(f"dense={self.dense_similarity:.3f}")
        if self.bm25_score is not None:
            parts.append(f"bm25={self.bm25_score:.2f}")
        parts.append(f"rrf={self.rrf_score:.3f}")
        if self.reranker_score is not None:
            parts.append(f"rerank={self.reranker_score:.3f}")
        parts.append(f"recency={self.recency_boost:.2f}")
        parts.append(f"feedback={self.feedback_boost:.2f}")
        if self.graph_hops_from_query is not None:
            parts.append(f"hops={self.graph_hops_from_query}")
        return " | ".join(parts) + f" → final={self.final_score:.3f}"


class RetrievalResult(BaseModel):
    """
    A single retrieved chunk with all its scoring context.
    """

    chunk: Chunk
    scores: RetrievalScores
    retrieved_via: list[Literal["dense", "bm25", "graph", "hyde"]]
    query_that_matched: str              # Which fanout query surfaced this


class RetrievalOutput(BaseModel):
    """
    Full output of the retrieval layer for one query.
    """

    query_id: str
    total_candidates_pre_rerank: int
    reranker_used: bool
    results: list[RetrievalResult]       # Post-rerank, top-k
    latency_ms: float


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH LAYER
# ═══════════════════════════════════════════════════════════════════════════


class ServiceNode(BaseModel):
    service_name: str
    tier: Literal["frontend", "api", "core", "infrastructure", "external"]
    environment: str
    owner_team: str | None = None
    criticality: Literal["critical", "high", "medium", "low"]


class ServiceEdge(BaseModel):
    from_service: str
    to_service: str
    relationship: Literal[
        "depends_on", "calls", "reads_from", "writes_to", "publishes_to", "subscribes_to"
    ]
    is_critical_path: bool = False


class BlastRadiusResult(BaseModel):
    origin_service: str
    hop_depth: int
    affected_services: list[ServiceNode]
    critical_path_services: list[str]


# ═══════════════════════════════════════════════════════════════════════════
# GENERATION LAYER (STRUCTURED OUTPUT)
# ═══════════════════════════════════════════════════════════════════════════


class Evidence(BaseModel):
    """
    From houndex pattern. A single piece of source-backed evidence.
    """

    chunk_id: str
    runbook_id: str
    header_breadcrumb: str
    excerpt: str                         # The exact text supporting the claim
    relevance: Literal["direct", "indirect", "background"]


class Claim(BaseModel):
    """
    From houndex pattern. A single factual statement made by the LLM.
    Every claim must have at least one Evidence.
    """

    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    statement: str
    evidence: list[Evidence]
    confidence: float                    # 0.0 - 1.0
    verified: bool = False               # Set by grounding verifier
    verification_reasoning: str | None = None


class RemediationAction(BaseModel):
    """
    A single proposed action, with risk classification and evidence.
    """

    action_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str                     # Human-readable
    command: str | None = None           # Executable command, if applicable
    action_type: Literal["kubectl", "aws_cli", "shell", "http_request", "manual"]
    risk_level: RiskLevel
    evidence: list[Evidence]
    reversible: bool
    estimated_duration_seconds: int
    expected_impact: str
    rollback_command: str | None = None


class IncidentAssessment(BaseModel):
    """
    The full structured output from the LLM reasoning step.
    This is what replaces Aurora's natural-language output.
    """

    incident_id: str
    query_id: str

    # Root cause
    root_cause: Claim
    contributing_factors: list[Claim] = Field(default_factory=list)

    # Confidence and evidence
    overall_confidence: float
    evidence_chunks_used: list[str]      # chunk_ids

    # Actions
    proposed_actions: list[RemediationAction]
    diagnostic_actions: list[RemediationAction] = Field(default_factory=list)

    # Escalation
    escalate_to_human: bool
    escalation_reason: str | None = None

    # Historical pattern matching
    matches_historical_incident: str | None = None  # incident_id
    pattern_confidence: float | None = None

    # What would increase confidence
    additional_info_needed: list[str] = Field(default_factory=list)

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    llm_model: str
    total_tokens: int
    cost_usd: float


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION LAYER
# ═══════════════════════════════════════════════════════════════════════════


class RAGASMetrics(BaseModel):
    """
    RAGAS four canonical metrics + custom IncidentRAG additions.
    """

    faithfulness: float                  # LLM claims grounded in context
    answer_relevancy: float              # Answer addresses the question
    context_precision: float             # Retrieved chunks are relevant
    context_recall: float                # All relevant chunks retrieved

    # Custom IncidentRAG metrics
    rca_accuracy: float                  # Did we identify the actual root cause?
    remediation_safety_score: float      # Did we avoid dangerous suggestions?
    grounding_verification_rate: float   # % of claims that passed grounding


class EvaluationCase(BaseModel):
    """
    A single ground-truth case in the eval set.
    """

    case_id: str
    incident_source: str                 # e.g. "cloudflare-2024-03-outage"
    input_alerts: list[RawAlert]
    ground_truth_root_cause: str
    ground_truth_relevant_runbooks: list[str]
    ground_truth_correct_actions: list[str]
    dangerous_actions_to_avoid: list[str]


# ═══════════════════════════════════════════════════════════════════════════
# OBSERVABILITY LAYER
# ═══════════════════════════════════════════════════════════════════════════


class DriftMeasurement(BaseModel):
    """
    Per-class embedding drift snapshot.
    """

    measurement_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    class_label: str                     # e.g. "payment-service-memory-alerts"
    mean_similarity: float               # Query→known-good chunks
    std_similarity: float
    noise_floor_max: float               # Max sim to unrelated chunks
    signal_gap: float                    # mean_similarity - noise_floor_max
    z_score_vs_baseline: float
    is_drift_detected: bool              # z_score > 3


class IndexManifest(BaseModel):
    """
    From embspec pattern. Enforces query encoder matches index encoder.
    """

    manifest_id: str
    embedding_model: str
    embedding_dimension: int
    embedding_version: str
    chunker_version: str
    total_chunks: int
    total_runbooks: int
    created_at: datetime
    last_reindex_at: datetime
