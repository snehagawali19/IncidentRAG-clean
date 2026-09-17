"""Tests for Phase 10 — Feedback Loop."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from incidentrag.core.models import RiskLevel
from incidentrag.feedback.loop import (
    BOOST_CAP,
    BOOST_INCREMENT,
    ChunkScoreUpdater,
    EvalSetExpander,
    InMemoryChunkStore,
)
from tests.unit.conftest import make_assessment, make_query, make_raw_alert, make_remediation_action


class TestChunkScoreUpdater:
    @pytest.mark.asyncio
    async def test_boost_new_chunks_starts_at_one(self, sample_query):
        assessment = make_assessment(sample_query)
        # evidence_chunks_used = ["c1", "c2"] in fixture
        store = InMemoryChunkStore()

        updater = ChunkScoreUpdater(store)
        updated = await updater.boost_used_chunks(assessment)

        expected = 1.0 + BOOST_INCREMENT
        assert updated["c1"] == pytest.approx(expected)
        assert updated["c2"] == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_boost_increments_existing_score(self, sample_query):
        assessment = make_assessment(sample_query)
        store = InMemoryChunkStore(initial={"c1": 1.5})

        updater = ChunkScoreUpdater(store)
        updated = await updater.boost_used_chunks(assessment)

        assert updated["c1"] == pytest.approx(1.5 + BOOST_INCREMENT)

    @pytest.mark.asyncio
    async def test_boost_respects_cap(self, sample_query):
        assessment = make_assessment(sample_query)
        # Start just below cap
        store = InMemoryChunkStore(initial={"c1": BOOST_CAP - 0.01})

        updater = ChunkScoreUpdater(store)
        updated = await updater.boost_used_chunks(assessment)

        assert updated["c1"] == BOOST_CAP  # cannot exceed cap

    @pytest.mark.asyncio
    async def test_boost_skips_none_placeholder(self, sample_query):
        assessment = make_assessment(sample_query)
        assessment.evidence_chunks_used = ["none", "c1"]
        store = InMemoryChunkStore()

        updater = ChunkScoreUpdater(store)
        updated = await updater.boost_used_chunks(assessment)

        assert "none" not in updated
        assert "c1" in updated

    @pytest.mark.asyncio
    async def test_boost_no_chunks_used(self, sample_query):
        assessment = make_assessment(sample_query)
        assessment.evidence_chunks_used = []
        store = InMemoryChunkStore()

        updater = ChunkScoreUpdater(store)
        updated = await updater.boost_used_chunks(assessment)
        assert updated == {}


class TestEvalSetExpander:
    def test_expand_creates_case(self, sample_query):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "eval.jsonl"
            expander = EvalSetExpander(eval_set_path=path)

            alert = make_raw_alert()
            assessment = make_assessment(sample_query)
            case = expander.expand(assessment, original_alert=alert)

            assert case.case_id.startswith("case-")
            assert case.incident_source.startswith("resolved-")
            assert len(case.input_alerts) == 1
            assert case.ground_truth_root_cause == assessment.root_cause.statement

    def test_expand_writes_to_disk(self, sample_query):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "eval.jsonl"
            expander = EvalSetExpander(eval_set_path=path)

            alert = make_raw_alert()
            expander.expand(make_assessment(sample_query), original_alert=alert)
            expander.expand(make_assessment(sample_query), original_alert=alert)

            assert path.exists()
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 2

    def test_expand_load_roundtrip(self, sample_query):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "eval.jsonl"
            expander = EvalSetExpander(eval_set_path=path)

            alert = make_raw_alert()
            original_case = expander.expand(
                make_assessment(sample_query),
                original_alert=alert,
                ground_truth_root_cause="Known root cause",
            )

            loaded = expander.load()
            assert len(loaded) == 1
            assert loaded[0].case_id == original_case.case_id
            assert loaded[0].ground_truth_root_cause == "Known root cause"

    def test_expand_captures_dangerous_actions_automatically(self, sample_query):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "eval.jsonl"
            expander = EvalSetExpander(eval_set_path=path)

            dangerous = make_remediation_action(
                command="kubectl delete all -n payments",
                risk_level=RiskLevel.DANGEROUS,
            )
            safe = make_remediation_action(risk_level=RiskLevel.LOW)
            assessment = make_assessment(sample_query, actions=[dangerous, safe])

            case = expander.expand(assessment, original_alert=make_raw_alert())
            assert "kubectl delete all -n payments" in case.dangerous_actions_to_avoid

    def test_load_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.jsonl"
            expander = EvalSetExpander(eval_set_path=path)
            assert expander.load() == []
