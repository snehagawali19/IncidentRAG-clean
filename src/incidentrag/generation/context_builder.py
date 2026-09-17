"""
Phase 5 — Context Construction.

Assembles the LLM prompt context from:
  - RawAlert          (alert_text, service_name, environment, severity)
  - Query             (fanout_queries, entities, category)
  - RetrievalResult[] (top-k chunks with scoring context)
  - BlastRadiusResult (origin_service, affected_services, critical_path_services)
  - Past IncidentAssessments (via matches_historical_incident)

Token-budgeted per section. Sanitizes retrieved content against prompt injection.
"""
from __future__ import annotations

import logging
import re

import tiktoken

from incidentrag.core.models import (
    BlastRadiusResult,
    IncidentAssessment,
    Query,
    RetrievalResult,
)
from incidentrag.core.settings import get_settings

logger = logging.getLogger(__name__)

_ENCODER = tiktoken.get_encoding("cl100k_base")

# Per-section token budgets
BUDGET_ALERT = 200
BUDGET_RUNBOOKS = 2000
BUDGET_GRAPH = 500
BUDGET_HISTORY = 500

# Prompt-injection patterns
_INJECTION_RE = [
    re.compile(r"ignore (all )?previous instructions", re.I),
    re.compile(r"you are now", re.I),
    re.compile(r"<\|im_start\|>", re.I),
    re.compile(r"disregard (your|the) (previous|above)", re.I),
    re.compile(r"new system prompt", re.I),
]


def _tok(text: str) -> int:
    return len(_ENCODER.encode(text))


def _truncate(text: str, max_tokens: int) -> str:
    tokens = _ENCODER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _ENCODER.decode(tokens[:max_tokens]) + "\n[truncated]"


class ContextSanitizer:
    """Detects and redacts prompt-injection patterns in retrieved content."""

    def sanitize(self, text: str) -> tuple[str, bool]:
        flagged = False
        for pattern in _INJECTION_RE:
            if pattern.search(text):
                flagged = True
                text = pattern.sub("[REDACTED]", text)
        return text, flagged


class ContextBuilder:
    """Assembles a structured LLM prompt context."""

    def __init__(self) -> None:
        self._sanitizer = ContextSanitizer()

    def build(
        self,
        query: Query,
        retrieval_results: list[RetrievalResult],
        blast_radius: BlastRadiusResult | None = None,
        past_incidents: list[IncidentAssessment] | None = None,
    ) -> tuple[str, dict[str, int]]:
        """
        Returns (context_string, per_section_token_usage).
        """
        past_incidents = past_incidents or []
        sections: list[str] = []
        usage: dict[str, int] = {}
        redactions = 0

        # ── Alert ─────────────────────────────────────────────────────────
        alert_txt = _truncate(self._fmt_alert(query), BUDGET_ALERT)
        sections.append(f"## INCIDENT ALERT\n{alert_txt}")
        usage["alert"] = _tok(alert_txt)

        # ── Retrieved Runbooks ────────────────────────────────────────────
        parts: list[str] = []
        rb_tokens = 0
        for i, rr in enumerate(retrieval_results, 1):
            clean, flagged = self._sanitizer.sanitize(rr.chunk.content)
            if flagged:
                redactions += 1
                logger.warning("Injection pattern in chunk %s", rr.chunk.chunk_id)
            entry = (
                f"### [{i}] chunk_id={rr.chunk.chunk_id} | runbook={rr.chunk.runbook_id}\n"
                f"*{rr.chunk.header_breadcrumb}*  (final_score={rr.scores.final_score:.3f})\n\n"
                f"{clean}\n"
            )
            et = _tok(entry)
            if rb_tokens + et > BUDGET_RUNBOOKS:
                break
            parts.append(entry)
            rb_tokens += et
        sections.append("## RETRIEVED RUNBOOKS\n" + "\n---\n".join(parts))
        usage["runbooks"] = rb_tokens

        # ── Graph ─────────────────────────────────────────────────────────
        graph_txt = _truncate(self._fmt_graph(blast_radius), BUDGET_GRAPH)
        sections.append(f"## INFRASTRUCTURE GRAPH\n{graph_txt}")
        usage["graph"] = _tok(graph_txt)

        # ── History ───────────────────────────────────────────────────────
        hist_txt = _truncate(self._fmt_history(past_incidents), BUDGET_HISTORY)
        sections.append(f"## PAST INCIDENTS\n{hist_txt}")
        usage["history"] = _tok(hist_txt)

        if redactions:
            sections.append(
                f"\n⚠️ SYSTEM: {redactions} retrieved chunk(s) contained "
                "suspicious patterns and were partially redacted."
            )

        context = "\n\n".join(sections)
        usage["total"] = _tok(context)
        return context, usage

    # ── Formatters ────────────────────────────────────────────────────────

    def _fmt_alert(self, query: Query) -> str:
        alert = query.original_alert
        entities = query.entities
        sev = alert.severity.value if alert.severity else "unknown"

        lines = [
            f"**Source:** {alert.source}",
            f"**Received at:** {alert.received_at.isoformat()}",
            f"**Severity:** {sev}",
            f"**Service:** {alert.service_name or entities.service or 'unknown'}",
            f"**Environment:** {alert.environment or entities.environment or 'unknown'}",
            f"**Category:** {query.category.value}",
            "",
            f"**Alert text:** {alert.alert_text[:400]}",
        ]
        if alert.metric_name:
            lines.append(
                f"**Metric:** {alert.metric_name} = {alert.metric_value} "
                f"(threshold: {alert.threshold})"
            )
        if entities.error_signature:
            lines.append(f"**Error signature:** {entities.error_signature}")
        if entities.correlated_services:
            lines.append(
                f"**Correlated services:** {', '.join(entities.correlated_services)}"
            )
        return "\n".join(lines)

    def _fmt_graph(self, br: BlastRadiusResult | None) -> str:
        if not br:
            return "No dependency graph data available."
        affected_names = [s.service_name for s in br.affected_services]
        lines = [
            f"**Origin service:** {br.origin_service}",
            f"**Hop depth analyzed:** {br.hop_depth}",
            f"**Affected services ({len(affected_names)}):** {', '.join(affected_names[:20])}",
            f"**Critical path:** {', '.join(br.critical_path_services[:10])}",
        ]
        # Add tier info for affected services
        if br.affected_services:
            tier_summary = {}
            for s in br.affected_services:
                tier_summary.setdefault(s.tier, []).append(s.service_name)
            lines.append("")
            lines.append("**By tier:**")
            for tier, svcs in tier_summary.items():
                lines.append(f"  - {tier}: {', '.join(svcs[:5])}")
        return "\n".join(lines)

    def _fmt_history(self, incidents: list[IncidentAssessment]) -> str:
        if not incidents:
            return "No historical incident data available."
        parts = []
        for inc in incidents[:3]:
            parts.append(
                f"- **{inc.incident_id}** "
                f"(confidence: {inc.overall_confidence:.0%})\n"
                f"  Root cause: {inc.root_cause.statement[:200]}"
            )
        return "\n".join(parts)
