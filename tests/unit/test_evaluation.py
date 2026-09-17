"""Tests for Phase 7 — Evaluation Runner + CI Gate."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from incidentrag.core.models import EvaluationCase, RAGASMetrics, RiskLevel
from incidentrag.evaluation.runner import (
    CIGate,
    DEFAULT_THRESHOLDS,
    EvaluationCaseBuilder,
    RAGASRunner,
)
from tests.unit.conftest import make_assessment, make_raw_alert, make_remediation_action


def _make_metrics(**overrides) -> RAGASMetrics:
    defaults = {
        "faithfulness": 0.90,
        "answer_relevancy": 0.85,
        "context_precision": 0.85,
        "context_recall": 0.80,
        "rca_accuracy": 0.75,
        "remediation_safety_score": 1.0,
        "grounding_verification_rate": 0.85,
    }
    defaults.update(overrides)
    return RAGASMetrics(**defaults)


class TestCIGate:
    def test_passes_when_all_above(self):
        CIGate().check(_make_metrics())  # should not raise

    def test_fails_low_faithfulness(self):
        with pytest.raises(RuntimeError, match="faithfulness"):
            CIGate().check(_make_metrics(faithfulness=0.10))

    def test_fails_low_context_precision(self):
        with pytest.raises(RuntimeError, match="context_precision"):
            CIGate().check(_make_metrics(context_precision=0.10))

    def test_fails_low_answer_relevancy(self):
        with pytest.raises(RuntimeError, match="answer_relevancy"):
            CIGate().check(_make_metrics(answer_relevancy=0.10))

    def test_fails_low_context_recall(self):
        with pytest.raises(RuntimeError, match="context_recall"):
            CIGate().check(_make_metrics(context_recall=0.10))

    def test_fails_low_rca_accuracy(self):
        with pytest.raises(RuntimeError, match="rca_accuracy"):
            CIGate().check(_make_metrics(rca_accuracy=0.10))

    def test_fails_low_safety(self):
        with pytest.raises(RuntimeError, match="remediation_safety_score"):
            CIGate().check(_make_metrics(remediation_safety_score=0.10))

    def test_fails_low_grounding(self):
        with pytest.raises(RuntimeError, match="grounding_verification_rate"):
            CIGate().check(_make_metrics(grounding_verification_rate=0.10))

    def test_reports_all_failures(self):
        with pytest.raises(RuntimeError) as exc_info:
            CIGate().check(_make_metrics(
                faithfulness=0.1,
                context_precision=0.1,
                answer_relevancy=0.1,
            ))
        msg = str(exc_info.value)
        assert "faithfulness" in msg
        assert "context_precision" in msg
        assert "answer_relevancy" in msg

    def test_custom_thresholds(self):
        gate = CIGate(thresholds={"faithfulness": 0.99})
        with pytest.raises(RuntimeError, match="faithfulness"):
            gate.check(_make_metrics(faithfulness=0.95))


class TestRAGASRunner:
    @pytest.mark.asyncio
    async def test_empty_pairs_returns_zeros(self):
        metrics = await RAGASRunner().run([])
        assert metrics.faithfulness == 0.0
        assert metrics.rca_accuracy == 0.0

    @pytest.mark.asyncio
    async def test_computes_rca_accuracy(self, sample_query):
        assessment = make_assessment(sample_query)
        # root_cause.statement in fixtures = "JVM heap exceeded container memory limit"
        case = EvaluationCase(
            case_id="case-1",
            incident_source="test",
            input_alerts=[make_raw_alert()],
            ground_truth_root_cause="JVM heap exceeded container memory limit",
            ground_truth_relevant_runbooks=["rb-payment-memory"],
            ground_truth_correct_actions=["kubectl set resources"],
            dangerous_actions_to_avoid=[],
        )
        pairs = [(case, assessment, assessment.root_cause.statement, ["some context"])]

        metrics = await RAGASRunner().run(pairs)
        assert metrics.rca_accuracy > 0.9  # near-perfect match

    @pytest.mark.asyncio
    async def test_safety_score_flags_dangerous(self, sample_query):
        dangerous_action = make_remediation_action(
            command="kubectl delete all -n payments",
            risk_level=RiskLevel.DANGEROUS,
        )
        assessment = make_assessment(sample_query, actions=[dangerous_action])
        case = EvaluationCase(
            case_id="case-1",
            incident_source="test",
            input_alerts=[make_raw_alert()],
            ground_truth_root_cause="some cause",
            ground_truth_relevant_runbooks=[],
            ground_truth_correct_actions=[],
            dangerous_actions_to_avoid=["kubectl delete all"],
        )
        pairs = [(case, assessment, "some answer", ["ctx"])]
        metrics = await RAGASRunner().run(pairs)
        assert metrics.remediation_safety_score < 0.5

    @pytest.mark.asyncio
    async def test_safety_score_full_when_safe(self, sample_query):
        safe_action = make_remediation_action(risk_level=RiskLevel.LOW)
        assessment = make_assessment(sample_query, actions=[safe_action])
        case = EvaluationCase(
            case_id="case-1",
            incident_source="test",
            input_alerts=[make_raw_alert()],
            ground_truth_root_cause="cause",
            ground_truth_relevant_runbooks=[],
            ground_truth_correct_actions=[],
            dangerous_actions_to_avoid=["rm -rf"],
        )
        pairs = [(case, assessment, "answer", ["ctx"])]
        metrics = await RAGASRunner().run(pairs)
        assert metrics.remediation_safety_score == 1.0


class TestEvaluationCaseBuilder:
    def test_append_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "eval.jsonl"
            builder = EvaluationCaseBuilder(path=path)

            case = EvaluationCase(
                case_id="test-1",
                incident_source="unit-test",
                input_alerts=[make_raw_alert()],
                ground_truth_root_cause="OOM due to heap misconfiguration",
                ground_truth_relevant_runbooks=["rb-1"],
                ground_truth_correct_actions=["kubectl set resources"],
                dangerous_actions_to_avoid=["kubectl delete"],
            )
            builder.append(case)

            loaded = builder.load()
            assert len(loaded) == 1
            assert loaded[0].case_id == "test-1"
            assert loaded[0].ground_truth_root_cause == "OOM due to heap misconfiguration"

    def test_load_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.jsonl"
            builder = EvaluationCaseBuilder(path=path)
            assert builder.load() == []
