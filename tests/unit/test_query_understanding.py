"""Tests for Layer 2 — Query Understanding.

Tests:
- HyDE output is 300-500 words
- Fanout returns exactly 5 distinct queries
- EntityExtractor parses a memory/OOM alert
- Classifier returns correct IncidentCategory

All types imported exclusively from core/models.py (SPEC rule 2).
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from incidentrag.core.models import (
    ExtractedEntities,
    IncidentCategory,
    RawAlert,
    Severity,
)
from incidentrag.query.classifier import IncidentClassifier
from incidentrag.query.entity_extractor import EntityExtractor
from incidentrag.query.fanout import QueryFanoutGenerator
from incidentrag.query.hyde import HyDEGenerator


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def oom_alert() -> RawAlert:
    return RawAlert(
        alert_id="alert-001",
        source="prometheus",
        received_at=datetime.now(timezone.utc),
        raw_payload={},
        service_name="payment-service",
        environment="production",
        severity=Severity.SEV1,
        alert_text=(
            "payment-service OOMKilled in production. "
            "Container payment-service in pod payment-service-6d8f9b-xkp2t was OOMKilled. "
            "Memory limit: 512Mi, Memory usage: 511Mi. Node: ip-10-0-1-23. "
            "Namespace: payments. Restart count: 5."
        ),
    )


@pytest.fixture
def latency_alert() -> RawAlert:
    return RawAlert(
        alert_id="alert-002",
        source="datadog",
        received_at=datetime.now(timezone.utc),
        raw_payload={},
        service_name="checkout-api",
        severity=Severity.SEV2,
        alert_text=(
            "p99 latency exceeded 2s for checkout-api. "
            "checkout-api p99 latency is 2340ms, threshold is 1000ms. "
            "Affecting /v2/checkout endpoint."
        ),
    )


# ── EntityExtractor tests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_entity_extractor_oom(oom_alert: RawAlert) -> None:
    mock_response = MagicMock()
    mock_response.choices[0].message.content = (
        '{"service": "payment-service", "environment": "production", '
        '"metric": "memory_rss_bytes", "metric_value": 511.0, "threshold": 512.0, '
        '"host_identifier": "ip-10-0-1-23", "error_signature": "OOMKilled"}'
    )

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    extractor = EntityExtractor(client=mock_client)
    entities = await extractor.extract(oom_alert)

    assert entities.service == "payment-service"
    assert entities.environment == "production"
    assert entities.error_signature == "OOMKilled"
    assert entities.metric_value == 511.0
    assert entities.host_identifier == "ip-10-0-1-23"


# ── Classifier tests ──────────────────────────────────────────────────────────


def test_classifier_rules_oom(oom_alert: RawAlert) -> None:
    """OOMKilled alert must classify as PERFORMANCE_DEGRADATION."""
    clf = IncidentClassifier()
    result = clf.classify_rules(oom_alert)
    assert result == IncidentCategory.PERFORMANCE_DEGRADATION


def test_classifier_rules_latency(latency_alert: RawAlert) -> None:
    """High p99 latency alert must classify as PERFORMANCE_DEGRADATION."""
    clf = IncidentClassifier()
    result = clf.classify_rules(latency_alert)
    assert result == IncidentCategory.PERFORMANCE_DEGRADATION


def test_classifier_rules_crashloop() -> None:
    """CrashLoopBackOff must classify as COMPLETE_OUTAGE."""
    alert = RawAlert(
        alert_id="alert-003",
        source="prometheus",
        received_at=datetime.now(timezone.utc),
        raw_payload={},
        alert_text="pod in CrashLoopBackOff. api-gateway pod restarted 15 times, backoff limit reached",
    )
    clf = IncidentClassifier()
    assert clf.classify_rules(alert) == IncidentCategory.COMPLETE_OUTAGE


def test_classifier_rules_security_cert() -> None:
    """TLS certificate expiry must classify as SECURITY_EVENT."""
    alert = RawAlert(
        alert_id="alert-004",
        source="prometheus",
        received_at=datetime.now(timezone.utc),
        raw_payload={},
        alert_text="TLS certificate for payment-service.example.com expires in 3 days",
    )
    clf = IncidentClassifier()
    assert clf.classify_rules(alert) == IncidentCategory.SECURITY_EVENT


def test_classifier_rules_db_connection() -> None:
    """Connection pool exhaustion must classify as DEPENDENCY_FAILURE."""
    alert = RawAlert(
        alert_id="alert-005",
        source="datadog",
        received_at=datetime.now(timezone.utc),
        raw_payload={},
        alert_text="pgbouncer connection pool exhausted: too many connections to postgres",
    )
    clf = IncidentClassifier()
    assert clf.classify_rules(alert) == IncidentCategory.DEPENDENCY_FAILURE


# ── QueryFanout tests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fanout_returns_exactly_5(oom_alert: RawAlert) -> None:
    mock_response = MagicMock()
    mock_response.choices[0].message.content = (
        '{"queries": ['
        '"payment-service OOMKill memory limit remediation steps",'
        '"root cause JVM heap exhaustion payment-service",'
        '"kubectl describe pod OOMKilled namespace payments",'
        '"payment-service dependencies affected services memory",'
        '"past incidents payment-service memory oom resolution"'
        ']}'
    )

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    entities = ExtractedEntities(
        service="payment-service",
        environment="production",
        error_signature="OOMKilled",
    )
    gen = QueryFanoutGenerator(client=mock_client)
    queries = await gen.generate(oom_alert, entities)

    assert len(queries) == 5
    # All queries must be distinct
    assert len(set(queries)) == 5


@pytest.mark.asyncio
async def test_fanout_pads_to_5_on_short_response(oom_alert: RawAlert) -> None:
    """If LLM returns fewer than 5 queries, generator must pad to exactly 5."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"queries": ["only one query"]}'

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    entities = ExtractedEntities(error_signature="OOMKilled")
    gen = QueryFanoutGenerator(client=mock_client)
    queries = await gen.generate(oom_alert, entities)

    assert len(queries) == 5


# ── HyDE tests ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hyde_generates_300_to_500_words(oom_alert: RawAlert) -> None:
    # Simulate a realistic 400-word response
    hypothetical = " ".join(["word"] * 400)

    mock_response = MagicMock()
    mock_response.choices[0].message.content = hypothetical

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    entities = ExtractedEntities(
        service="payment-service",
        error_signature="OOMKilled",
        environment="production",
    )
    gen = HyDEGenerator(client=mock_client)
    result = await gen.generate(oom_alert, entities)

    word_count = len(result.split())
    assert 300 <= word_count <= 500, f"HyDE output was {word_count} words, expected 300-500"


@pytest.mark.asyncio
async def test_hyde_returns_fallback_on_error(oom_alert: RawAlert) -> None:
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))

    entities = ExtractedEntities(error_signature="OOMKilled")
    gen = HyDEGenerator(client=mock_client)
    result = await gen.generate(oom_alert, entities)

    # Must return a non-empty fallback string
    assert len(result) > 0
