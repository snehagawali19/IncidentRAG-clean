"""Tests for QueryUnderstandingPipeline (Layer 2)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from incidentrag.core.models import ExtractedEntities, IncidentCategory
from incidentrag.query.pipeline import QueryUnderstandingPipeline
from tests.unit.conftest import make_raw_alert


# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_ENTITIES = ExtractedEntities(
    service="payment-service",
    environment="production",
    metric="container_memory_usage_bytes",
    metric_value=512.0,
    threshold=512.0,
    error_signature="OOMKilled",
)
MOCK_CATEGORY = IncidentCategory.PERFORMANCE_DEGRADATION
MOCK_QUERIES = [
    "payment-service OOMKilled remediation",
    "JVM heap memory configuration",
    "kubectl set resources memory limit",
    "container memory limit increase",
    "past OOMKilled incidents payment",
]
MOCK_HYDE_DOC = "When payment-service is OOMKilled, increase the container memory limit..."


@pytest.fixture
def patched_pipeline():
    """Return a QueryUnderstandingPipeline with all sub-components mocked."""
    with patch("incidentrag.query.pipeline.AsyncOpenAI"):
        with patch("incidentrag.query.pipeline.EntityExtractor") as MockExtractor:
            with patch("incidentrag.query.pipeline.IncidentClassifier") as MockClassifier:
                with patch("incidentrag.query.pipeline.QueryFanoutGenerator") as MockFanout:
                    with patch("incidentrag.query.pipeline.HyDEGenerator") as MockHyDE:
                        extractor_inst = AsyncMock()
                        extractor_inst.extract = AsyncMock(return_value=MOCK_ENTITIES)
                        MockExtractor.return_value = extractor_inst

                        classifier_inst = AsyncMock()
                        classifier_inst.classify = AsyncMock(return_value=MOCK_CATEGORY)
                        MockClassifier.return_value = classifier_inst

                        fanout_inst = AsyncMock()
                        fanout_inst.generate = AsyncMock(return_value=MOCK_QUERIES)
                        MockFanout.return_value = fanout_inst

                        hyde_inst = AsyncMock()
                        hyde_inst.generate = AsyncMock(return_value=MOCK_HYDE_DOC)
                        MockHyDE.return_value = hyde_inst

                        pipeline = QueryUnderstandingPipeline()
                        yield pipeline, extractor_inst, classifier_inst, fanout_inst, hyde_inst


# ── Tests ─────────────────────────────────────────────────────────────────────

async def test_process_returns_four_tuple(patched_pipeline):
    """process() should return a 4-tuple."""
    pipeline, *_ = patched_pipeline
    alert = make_raw_alert()
    result = await pipeline.process(alert)
    assert len(result) == 4


async def test_process_calls_extractor(patched_pipeline):
    """process() should call EntityExtractor.extract with the alert."""
    pipeline, extractor, *_ = patched_pipeline
    alert = make_raw_alert()
    await pipeline.process(alert)
    extractor.extract.assert_called_once_with(alert)


async def test_process_calls_classifier(patched_pipeline):
    """process() should call IncidentClassifier.classify."""
    pipeline, extractor, classifier, *_ = patched_pipeline
    alert = make_raw_alert()
    await pipeline.process(alert)
    classifier.classify.assert_called_once()


async def test_process_calls_fanout(patched_pipeline):
    """process() should call QueryFanoutGenerator.generate."""
    pipeline, extractor, classifier, fanout, hyde = patched_pipeline
    alert = make_raw_alert()
    await pipeline.process(alert)
    fanout.generate.assert_called_once()


async def test_process_calls_hyde(patched_pipeline):
    """process() should call HyDEGenerator.generate."""
    pipeline, extractor, classifier, fanout, hyde = patched_pipeline
    alert = make_raw_alert()
    await pipeline.process(alert)
    hyde.generate.assert_called_once()


async def test_process_returns_5_queries(patched_pipeline):
    """process() should return exactly 5 fanout queries."""
    pipeline, *_ = patched_pipeline
    alert = make_raw_alert()
    entities, category, queries, hyde_doc = await pipeline.process(alert)
    assert len(queries) == 5


async def test_process_entities_from_extractor(patched_pipeline):
    """Returned entities should match the mock extractor's return value."""
    pipeline, extractor, *_ = patched_pipeline
    alert = make_raw_alert()
    entities, category, queries, hyde_doc = await pipeline.process(alert)
    assert entities == MOCK_ENTITIES


async def test_process_category_from_classifier(patched_pipeline):
    """Returned category should match the mock classifier's return value."""
    pipeline, extractor, classifier, *_ = patched_pipeline
    alert = make_raw_alert()
    entities, category, queries, hyde_doc = await pipeline.process(alert)
    assert category == MOCK_CATEGORY
