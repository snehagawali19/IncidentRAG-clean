"""IncidentRAG — Main Orchestration Pipeline.

Wires all 10 layers into a single async pipeline:
  L1 — Ingestion (pre-run, not part of live path)
  L2 — Query Understanding
  L3 — Hybrid Retrieval
  L4 — Graph Traversal
  L5 — Context Construction
  L6 — Structured Output + Grounding
  L7 — Evaluation (async, runs in background)
  L8 — Observability traces wrap each step
  L9 — Approval Gate (returns requests, does not block)
  L10— Feedback (called on resolution, not inline)
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from incidentrag.approval.gate import ApprovalGate, ApprovalRequest
from incidentrag.core.exceptions import RetrievalError
from incidentrag.core.models import IncidentAssessment, Query, RawAlert
from incidentrag.execution.sandbox import SandboxExecutor
from incidentrag.feedback.loop import ChunkScoreUpdater, EvalSetExpander
from incidentrag.generation.context_builder import ContextBuilder
from incidentrag.generation.reasoner import GroundingVerifier, StructuredReasoner
from incidentrag.graph.layer import GraphLayer
from incidentrag.observability.tracer import CostTracker, trace_step
from incidentrag.query.pipeline import QueryUnderstandingPipeline
from incidentrag.retrieval.bm25_index import BM25Index
from incidentrag.retrieval.dense_index import DenseIndex
from incidentrag.retrieval.hybrid import HybridRetriever
from incidentrag.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    assessment: IncidentAssessment
    approval_requests: list[ApprovalRequest] = field(default_factory=list)
    cost_summary: dict = field(default_factory=dict)


class IncidentRAGPipeline:
    """Full 10-layer pipeline for a single incident."""

    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        graph: GraphLayer,
        dry_run: bool = False,
    ) -> None:
        self.query = QueryUnderstandingPipeline()
        self.retriever = HybridRetriever(bm25, dense, Reranker())
        self.graph = graph
        self.context_builder = ContextBuilder()
        self.reasoner = StructuredReasoner()
        self.verifier = GroundingVerifier()
        self.approval = ApprovalGate()
        self.executor = SandboxExecutor(dry_run=dry_run)
        self.score_updater = ChunkScoreUpdater(dense)
        self.eval_expander = EvalSetExpander()

    async def process(self, alert: RawAlert) -> PipelineResult:
        cost = CostTracker()

        async with trace_step("pipeline.full", {"alert_id": alert.alert_id}):

            # ── L2: Query Understanding ───────────────────────────────────
            async with trace_step("l2.query_understanding"):
                entities, category, queries, hyde_doc = await self.query.process(alert)

            # ── L3: Hybrid Retrieval ──────────────────────────────────────
            async with trace_step("l3.retrieval"):
                retrieval_output = await self.retriever.retrieve(
                    alert_id=alert.alert_id,
                    queries=queries,
                    hyde_document=hyde_doc,
                    service=alert.service_name or entities.service,
                )
                if not retrieval_output.results:
                    raise RetrievalError(
                        "No relevant runbook evidence matched the incident service"
                    )

            # ── L4: Graph Traversal ───────────────────────────────────────
            blast_radius = None
            past_incidents = []
            if entities.service:
                async with trace_step("l4.graph"):
                    try:
                        blast_radius = await self.graph.blast_radius.analyze(entities.service)
                        past_incidents = await self.graph.memory.get_similar(entities.service)
                    except Exception as exc:
                        logger.warning("Graph layer skipped: %s", exc)

            # ── L5: Context Construction ──────────────────────────────────
            async with trace_step("l5.context_build"):
                query_obj = Query(
                    query_id=str(uuid.uuid4()),
                    original_alert=alert,
                    entities=entities,
                    category=category,
                    fanout_queries=queries,
                    hypothetical_document=hyde_doc,
                )
                context, token_usage = self.context_builder.build(
                    query=query_obj,
                    retrieval_results=retrieval_output.results,
                    blast_radius=blast_radius,
                    past_incidents=None,  # PastIncident != IncidentAssessment
                )

            # ── L6: Structured Output + Grounding ─────────────────────────
            async with trace_step("l6.reasoning"):
                assessment = await self.reasoner.reason(
                    query=query_obj,
                    context=context,
                    retrieval_results=retrieval_output.results,
                )
                assessment = await self.verifier.verify_assessment(assessment)

            # ── L9: Approval Gate ─────────────────────────────────────────
            async with trace_step("l9.approval"):
                approval_requests = self.approval.process(assessment)

            # ── Auto-execute LOW risk ─────────────────────────────────────
            for req in approval_requests:
                if req.status == "auto_executed" and req.action.command:
                    action_id = str(id(req.action))
                    result = await self.executor.execute(
                        action_id, req.action.command, req.status
                    )
                    if not result.success:
                        logger.error("Auto-execution failed: %s", result.stderr)

        return PipelineResult(
            assessment=assessment,
            approval_requests=approval_requests,
            cost_summary=cost.summary(),
        )

    async def on_resolved(
        self,
        assessment: IncidentAssessment,
        alert: RawAlert | None = None,
        ground_truth: str | None = None,
    ) -> None:
        """Call after human confirms resolution — triggers L10 feedback."""
        async with trace_step("l10.feedback"):
            await self.score_updater.boost_used_chunks(assessment)
            if alert is not None:
                self.eval_expander.expand(
                    assessment,
                    original_alert=alert,
                    ground_truth_root_cause=ground_truth,
                )
