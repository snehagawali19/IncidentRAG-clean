"""
Phase 6 — Structured Reasoner + Grounding Verifier.

StructuredReasoner:
  - Uses the reasoning LLM (via OpenRouter or Anthropic direct)
  - Generates a fully-typed IncidentAssessment with root_cause + contributing_factors as Claims
  - Every Claim MUST have >=1 Evidence citation mapped back to a Chunk

GroundingVerifier (conforms to GroundingVerifierProtocol):
  - Uses a judge LLM to verify each claim against its cited excerpts
  - Sets claim.verified and claim.verification_reasoning
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from openai import AsyncOpenAI

from incidentrag.core.ai_provider import openai_client_kwargs, reasoning_model_name
from incidentrag.core.exceptions import GenerationError
from incidentrag.core.models import (
    Claim,
    Evidence,
    IncidentAssessment,
    Query,
    RemediationAction,
    RetrievalResult,
    RiskLevel,
)
from incidentrag.core.settings import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are IncidentRAG, an expert SRE performing structured root cause analysis.

Return a JSON object with EXACTLY this shape:

{
  "root_cause": {
    "statement": "concise root cause description",
    "evidence_indices": [1, 3],
    "confidence": 0.80
  },
  "contributing_factors": [
    {"statement": "...", "evidence_indices": [2], "confidence": 0.60}
  ],
  "overall_confidence": 0.75,
  "proposed_actions": [
    {
      "description": "human-readable action",
      "command": "kubectl ...",
      "action_type": "kubectl|aws_cli|shell|http_request|manual",
      "risk_level": "low|medium|high|dangerous",
      "evidence_indices": [1],
      "reversible": true,
      "estimated_duration_seconds": 60,
      "expected_impact": "what will change",
      "rollback_command": "kubectl ..."
    }
  ],
  "diagnostic_actions": [ ...same shape as proposed_actions... ],
  "escalate_to_human": false,
  "escalation_reason": null,
  "additional_info_needed": ["logs from service X"]
}

RULES:
- Every claim (root_cause and every contributing_factor) MUST cite >=1 runbook by [N] index.
- Confidences: 0.0-1.0. Be conservative (0.5-0.8 unless evidence is overwhelming).
- Risk levels are lowercase: "low", "medium", "high", "dangerous".
- Action types are lowercase: "kubectl", "aws_cli", "shell", "http_request", "manual".
- DANGEROUS actions MUST include escalate_to_human=true and escalation_reason.
- Provide rollback_command whenever the action is reversible.
"""


def _make_evidence(index: int, chunk_map: dict[int, RetrievalResult]) -> Evidence | None:
    rr = chunk_map.get(index)
    if not rr:
        return None
    return Evidence(
        chunk_id=rr.chunk.chunk_id,
        runbook_id=rr.chunk.runbook_id,
        header_breadcrumb=rr.chunk.header_breadcrumb,
        excerpt=rr.chunk.content[:200],
        relevance="direct",
    )


def _build_evidence_list(
    indices: list[int],
    chunk_map: dict[int, RetrievalResult],
) -> list[Evidence]:
    result: list[Evidence] = []
    for idx in indices:
        ev = _make_evidence(idx, chunk_map)
        if ev is not None:
            result.append(ev)
    return result


def _parse_risk(raw: str) -> RiskLevel:
    try:
        return RiskLevel(raw.lower())
    except (ValueError, AttributeError):
        return RiskLevel.MEDIUM


def _parse_action_type(raw: str) -> str:
    allowed = {"kubectl", "aws_cli", "shell", "http_request", "manual"}
    val = (raw or "").lower()
    return val if val in allowed else "manual"


class StructuredReasoner:
    """Generates a fully-structured IncidentAssessment from context + chunks."""

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        self._client = client or AsyncOpenAI(**openai_client_kwargs())
        self._model = reasoning_model_name()

    async def reason(
        self,
        query: Query,
        context: str,
        retrieval_results: list[RetrievalResult],
        incident_id: str | None = None,
    ) -> IncidentAssessment:
        chunk_map = {i + 1: rr for i, rr in enumerate(retrieval_results)}
        user_msg = f"{context}\n\n---\nProduce the JSON assessment for the alert above."

        # Provider errors must remain errors. Turning them into an empty assessment
        # creates a convincing-looking 200 response with unrelated fallback evidence.
        resp = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.0,
            max_tokens=settings.reasoning_max_output_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        try:
            raw = json.loads(resp.choices[0].message.content or "")
        except (json.JSONDecodeError, TypeError) as exc:
            raise GenerationError("Reasoning model returned malformed JSON") from exc
        if not isinstance(raw, dict):
            raise GenerationError("Reasoning model returned an invalid assessment")

        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        # ── Root cause (required) ──────────────────────────────────────────
        rc_raw = raw.get("root_cause") or {}
        if not isinstance(rc_raw, dict) or not str(rc_raw.get("statement") or "").strip():
            raise GenerationError("Reasoning model omitted the root-cause statement")
        root_evidence = _build_evidence_list(
            rc_raw.get("evidence_indices", []), chunk_map
        )
        if not root_evidence:
            raise GenerationError("Reasoning model did not cite retrieved evidence")
        root_cause = Claim(
            statement=str(rc_raw["statement"]).strip(),
            evidence=root_evidence,
            confidence=float(rc_raw.get("confidence", 0.0)),
        )

        # ── Contributing factors ───────────────────────────────────────────
        contributing = []
        for cf in raw.get("contributing_factors", []):
            evidence = _build_evidence_list(cf.get("evidence_indices", []), chunk_map)
            if not evidence:
                continue  # skip factors without evidence
            contributing.append(
                Claim(
                    statement=cf.get("statement", ""),
                    evidence=evidence,
                    confidence=float(cf.get("confidence", 0.5)),
                )
            )

        # ── Actions ────────────────────────────────────────────────────────
        def _build_actions(raw_list: list[dict[str, Any]]) -> list[RemediationAction]:
            out: list[RemediationAction] = []
            for a in raw_list:
                ev = _build_evidence_list(a.get("evidence_indices", []), chunk_map)
                if not ev:
                    logger.warning("Discarding action without valid evidence citation")
                    continue

                out.append(
                    RemediationAction(
                        description=a.get("description", ""),
                        command=a.get("command"),
                        action_type=_parse_action_type(a.get("action_type", "manual")),
                        risk_level=_parse_risk(a.get("risk_level", "medium")),
                        evidence=ev,
                        reversible=bool(a.get("reversible", False)),
                        estimated_duration_seconds=int(a.get("estimated_duration_seconds", 60)),
                        expected_impact=a.get("expected_impact", ""),
                        rollback_command=a.get("rollback_command"),
                    )
                )
            return out

        proposed = _build_actions(raw.get("proposed_actions", []))
        diagnostic = _build_actions(raw.get("diagnostic_actions", []))

        # Auto-escalate if any DANGEROUS action is proposed
        has_dangerous = any(a.risk_level == RiskLevel.DANGEROUS for a in proposed)
        escalate = bool(raw.get("escalate_to_human")) or has_dangerous
        escalation_reason = raw.get("escalation_reason")
        if has_dangerous and not escalation_reason:
            escalation_reason = "Proposed action classified as DANGEROUS."

        # ── Cost estimate ──────────────────────────────────────────────────
        # gpt-4o-mini pricing (per 1K tok): $0.00015 prompt / $0.0006 completion
        cost = (prompt_tokens * 0.00015 + completion_tokens * 0.0006) / 1000

        evidence_chunks_used = list({
            ev.chunk_id
            for claim in [root_cause, *contributing]
            for ev in claim.evidence
        })

        return IncidentAssessment(
            incident_id=incident_id or str(uuid.uuid4()),
            query_id=query.query_id,
            root_cause=root_cause,
            contributing_factors=contributing,
            overall_confidence=float(raw.get("overall_confidence", root_cause.confidence)),
            evidence_chunks_used=evidence_chunks_used,
            proposed_actions=proposed,
            diagnostic_actions=diagnostic,
            escalate_to_human=escalate,
            escalation_reason=escalation_reason,
            additional_info_needed=raw.get("additional_info_needed", []),
            llm_model=self._model,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost,
        )


class GroundingVerifier:
    """
    Verifies each Claim against its cited Evidence excerpts.
    Conforms to GroundingVerifierProtocol.
    """

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        self._client = client or AsyncOpenAI(**openai_client_kwargs())
        self._model = reasoning_model_name()

    async def verify(
        self,
        claim: Claim,
        evidence: list[Evidence],
    ) -> tuple[bool, str]:
        """
        Verify that the claim is grounded in the provided evidence.
        Returns (is_grounded, reasoning).
        """
        if not evidence:
            return False, "No evidence provided."

        excerpts = "\n\n".join(
            f"[E{i+1}] ({ev.header_breadcrumb}): {ev.excerpt}"
            for i, ev in enumerate(evidence)
        )
        prompt = (
            f"Does the following evidence SUPPORT the claim?\n\n"
            f"CLAIM: {claim.statement}\n\n"
            f"EVIDENCE:\n{excerpts}\n\n"
            "Reply with a single line: 'YES: <one-sentence reason>' or "
            "'NO: <one-sentence reason>'."
        )
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                temperature=0.0,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Grounding verify failed: %s", exc)
            return False, f"Verification error: {exc}"

        is_grounded = answer.upper().startswith("YES")
        reasoning = answer.split(":", 1)[1].strip() if ":" in answer else answer
        return is_grounded, reasoning

    async def verify_assessment(self, assessment: IncidentAssessment) -> IncidentAssessment:
        """
        Convenience method: verify every Claim in an assessment and mutate in place.
        """
        claims = [assessment.root_cause, *assessment.contributing_factors]
        results = await asyncio.gather(
            *[self.verify(c, c.evidence) for c in claims],
            return_exceptions=True,
        )
        for claim, res in zip(claims, results, strict=True):
            if isinstance(res, Exception):
                claim.verified = False
                claim.verification_reasoning = f"Error: {res}"
            else:
                is_grounded, reasoning = res
                claim.verified = is_grounded
                claim.verification_reasoning = reasoning
        return assessment
