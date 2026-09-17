"""Tests for Layer 8 — Observability (Cost Tracker, embed_assert)."""
from __future__ import annotations

import pytest

from incidentrag.observability.tracer import CostTracker


class TestCostTracker:
    def test_record_and_total(self):
        ct = CostTracker()
        ct.record("openai/gpt-4o-mini", prompt_tokens=1000, completion_tokens=500, step="query")
        total = ct.total_cost()
        assert total > 0
        # gpt-4o-mini: 1000 * 0.00015/1000 + 500 * 0.0006/1000 = 0.00015 + 0.0003 = 0.00045
        assert abs(total - 0.00045) < 0.0001

    def test_multiple_records_accumulate(self):
        ct = CostTracker()
        ct.record("openai/gpt-4o-mini", 1000, 500, step="entity")
        ct.record("anthropic/claude-sonnet-4-6", 2000, 1000, step="reasoning")
        total = ct.total_cost()
        assert total > 0.00045  # more than just the mini call

    def test_unknown_model_zero_cost(self):
        ct = CostTracker()
        ct.record("some/unknown-model", 1000, 1000, step="test")
        assert ct.total_cost() == 0.0

    def test_summary_structure(self):
        ct = CostTracker()
        ct.record("openai/gpt-4o-mini", 100, 50, step="test")
        summary = ct.summary()
        assert "records" in summary
        assert "total_prompt_tokens" in summary
        assert "total_completion_tokens" in summary
        assert "total_estimated_usd" in summary
        assert summary["total_prompt_tokens"] == 100
        assert summary["total_completion_tokens"] == 50

    def test_empty_tracker(self):
        ct = CostTracker()
        assert ct.total_cost() == 0.0
        assert ct.summary()["records"] == []

    def test_embedding_model_cost(self):
        ct = CostTracker()
        ct.record("text-embedding-3-small", prompt_tokens=5000, completion_tokens=0, step="embed")
        # 5000 * 0.00002 / 1000 = 0.0001
        assert abs(ct.total_cost() - 0.0001) < 0.00001
