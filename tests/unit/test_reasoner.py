"""Tests for Phase 6 — Structured Reasoner + Grounding Verifier."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from incidentrag.core.exceptions import GenerationError
from incidentrag.core.models import Claim, RiskLevel
from incidentrag.generation.reasoner import GroundingVerifier, StructuredReasoner
from tests.unit.conftest import make_chunk, make_evidence, make_query, make_retrieval_result


def _mock_llm_response(content: str, prompt_tokens: int = 500, completion_tokens: int = 200):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


class TestStructuredReasoner:
    @pytest.mark.asyncio
    async def test_reason_returns_valid_assessment(self, sample_query, sample_retrieval_results):
        llm_output = json.dumps({
            "root_cause": {
                "statement": "JVM heap exceeded container memory limit of 512Mi",
                "evidence_indices": [1, 2],
                "confidence": 0.85,
            },
            "contributing_factors": [
                {
                    "statement": "No JVM heap size explicitly configured",
                    "evidence_indices": [3],
                    "confidence": 0.60,
                }
            ],
            "overall_confidence": 0.80,
            "proposed_actions": [
                {
                    "description": "Increase memory limit to 1Gi",
                    "command": "kubectl set resources deploy/payment -n pay --limits=memory=1Gi",
                    "action_type": "kubectl",
                    "risk_level": "medium",
                    "evidence_indices": [2],
                    "reversible": True,
                    "estimated_duration_seconds": 30,
                    "expected_impact": "Pod recreated with 1Gi memory",
                    "rollback_command": "kubectl set resources deploy/payment -n pay --limits=memory=512Mi",
                }
            ],
            "diagnostic_actions": [],
            "escalate_to_human": False,
            "escalation_reason": None,
            "additional_info_needed": [],
        })
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(return_value=_mock_llm_response(llm_output))

        reasoner = StructuredReasoner(client=client)
        assessment = await reasoner.reason(
            query=sample_query,
            context="test context",
            retrieval_results=sample_retrieval_results,
        )

        assert assessment.query_id == sample_query.query_id
        assert assessment.root_cause.statement.startswith("JVM heap exceeded")
        assert assessment.root_cause.confidence == 0.85
        assert len(assessment.root_cause.evidence) == 2
        assert len(assessment.contributing_factors) == 1
        assert len(assessment.proposed_actions) == 1
        assert assessment.proposed_actions[0].risk_level == RiskLevel.MEDIUM
        assert assessment.escalate_to_human is False
        assert assessment.total_tokens == 700

    @pytest.mark.asyncio
    async def test_reason_maps_evidence_to_chunks(self, sample_query, sample_retrieval_results):
        llm_output = json.dumps({
            "root_cause": {
                "statement": "test cause",
                "evidence_indices": [1],
                "confidence": 0.7,
            },
            "contributing_factors": [],
            "overall_confidence": 0.7,
            "proposed_actions": [],
            "diagnostic_actions": [],
            "escalate_to_human": False,
        })
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(return_value=_mock_llm_response(llm_output))

        reasoner = StructuredReasoner(client=client)
        assessment = await reasoner.reason(
            query=sample_query,
            context="ctx",
            retrieval_results=sample_retrieval_results,
        )

        expected_chunk_id = sample_retrieval_results[0].chunk.chunk_id
        assert assessment.root_cause.evidence[0].chunk_id == expected_chunk_id

    @pytest.mark.asyncio
    async def test_reason_handles_invalid_json(self, sample_query, sample_retrieval_results):
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response("not valid json {{{")
        )
        reasoner = StructuredReasoner(client=client)
        with pytest.raises(GenerationError, match="malformed JSON"):
            await reasoner.reason(
                query=sample_query,
                context="ctx",
                retrieval_results=sample_retrieval_results,
            )

    @pytest.mark.asyncio
    async def test_reason_rejects_root_cause_without_evidence(
        self, sample_query, sample_retrieval_results
    ):
        llm_output = json.dumps({
            "root_cause": {
                "statement": "A plausible but unsupported cause",
                "evidence_indices": [],
                "confidence": 0.7,
            },
            "contributing_factors": [],
            "overall_confidence": 0.7,
            "proposed_actions": [],
            "diagnostic_actions": [],
            "escalate_to_human": False,
        })
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(llm_output)
        )

        with pytest.raises(GenerationError, match="did not cite"):
            await StructuredReasoner(client=client).reason(
                query=sample_query,
                context="ctx",
                retrieval_results=sample_retrieval_results,
            )

    @pytest.mark.asyncio
    async def test_reason_auto_escalates_dangerous_action(self, sample_query, sample_retrieval_results):
        llm_output = json.dumps({
            "root_cause": {
                "statement": "corrupt state requires cleanup",
                "evidence_indices": [1],
                "confidence": 0.7,
            },
            "contributing_factors": [],
            "overall_confidence": 0.7,
            "proposed_actions": [
                {
                    "description": "Delete all payments state",
                    "command": "kubectl delete all -n payments",
                    "action_type": "kubectl",
                    "risk_level": "dangerous",
                    "evidence_indices": [1],
                    "reversible": False,
                    "estimated_duration_seconds": 60,
                    "expected_impact": "All payment pods destroyed",
                }
            ],
            "diagnostic_actions": [],
            "escalate_to_human": False,  # LLM said no — reasoner should override
        })
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(return_value=_mock_llm_response(llm_output))

        reasoner = StructuredReasoner(client=client)
        assessment = await reasoner.reason(
            query=sample_query,
            context="ctx",
            retrieval_results=sample_retrieval_results,
        )

        assert assessment.escalate_to_human is True
        assert assessment.escalation_reason  # non-empty
        assert assessment.proposed_actions[0].risk_level == RiskLevel.DANGEROUS

    @pytest.mark.asyncio
    async def test_reason_computes_cost(self, sample_query, sample_retrieval_results):
        llm_output = json.dumps({
            "root_cause": {"statement": "test", "evidence_indices": [1], "confidence": 0.5},
            "contributing_factors": [],
            "overall_confidence": 0.5,
            "proposed_actions": [],
            "diagnostic_actions": [],
            "escalate_to_human": False,
        })
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(llm_output, prompt_tokens=1000, completion_tokens=500)
        )
        reasoner = StructuredReasoner(client=client)
        assessment = await reasoner.reason(
            query=sample_query,
            context="ctx",
            retrieval_results=sample_retrieval_results,
        )
        # gpt-4o-mini: 1000 * 0.00015 + 500 * 0.0006 = 0.45 / 1000 = 0.00045
        assert 0.0004 < assessment.cost_usd < 0.0005


class TestGroundingVerifier:
    @pytest.mark.asyncio
    async def test_verify_grounded_claim(self):
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response("YES: the evidence directly supports the claim")
        )
        verifier = GroundingVerifier(client=client)

        chunk = make_chunk(content="When JVM heap exceeds container memory limit, OOMKilled fires")
        evidence = [make_evidence(chunk, relevance="direct")]
        claim = Claim(statement="JVM heap exceeded memory limit", evidence=evidence, confidence=0.85)

        is_grounded, reasoning = await verifier.verify(claim, evidence)
        assert is_grounded is True
        assert "supports" in reasoning.lower() or reasoning

    @pytest.mark.asyncio
    async def test_verify_ungrounded_claim(self):
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response("NO: the evidence discusses a different topic")
        )
        verifier = GroundingVerifier(client=client)
        evidence = [make_evidence()]
        claim = Claim(statement="Network partition caused failure", evidence=evidence, confidence=0.5)

        is_grounded, _ = await verifier.verify(claim, evidence)
        assert is_grounded is False

    @pytest.mark.asyncio
    async def test_verify_no_evidence_returns_false(self):
        verifier = GroundingVerifier(client=AsyncMock())
        # Can't build a Claim with no evidence (Pydantic validation), so pass empty list to verify
        chunk_ev = [make_evidence()]
        claim = Claim(statement="test", evidence=chunk_ev, confidence=0.5)
        is_grounded, reasoning = await verifier.verify(claim, [])
        assert is_grounded is False
        assert "No evidence" in reasoning

    @pytest.mark.asyncio
    async def test_verify_assessment_mutates_claims(self, sample_assessment):
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response("YES: supported by evidence")
        )
        verifier = GroundingVerifier(client=client)

        result = await verifier.verify_assessment(sample_assessment)
        assert result.root_cause.verified is True
        assert result.root_cause.verification_reasoning
