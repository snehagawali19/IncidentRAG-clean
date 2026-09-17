"""
Shared pytest fixtures for IncidentRAG unit and integration tests.

All fixtures are async-compatible (pytest-asyncio in auto mode).
External service clients are replaced with async mocks so unit tests
run without any live infrastructure.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from incidentrag.core.models import (
    Chunk,
    ChunkType,
    RawAlert,
    RunbookMetadata,
    Severity,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_NOW = datetime(2026, 9, 10, 4, 0, 0, tzinfo=timezone.utc)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# LLM Client Mocks
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_anthropic_client() -> MagicMock:
    """
    Async mock for the Anthropic Python SDK client.

    The mock pre-configures `messages.create` as an AsyncMock that returns
    a minimal response envelope so callers can extract `.content[0].text`
    without hitting the real API.

    Usage::

        async def test_something(mock_anthropic_client):
            mock_anthropic_client.messages.create.return_value.content[0].text = (
                '{"key": "value"}'
            )
    """
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text='{"stub": true}')]
    response.usage = MagicMock(input_tokens=100, output_tokens=50)
    response.model = "claude-opus-4-5"
    client.messages.create = AsyncMock(return_value=response)
    return client


@pytest.fixture()
def mock_openai_client() -> MagicMock:
    """
    Async mock for the OpenAI Python SDK client.

    Pre-configures both `chat.completions.create` (for reasoning) and
    `embeddings.create` (for dense retrieval) as AsyncMocks.

    Usage::

        async def test_something(mock_openai_client):
            mock_openai_client.chat.completions.create.return_value \
                .choices[0].message.content = '{"stub": true}'

            mock_openai_client.embeddings.create.return_value \
                .data[0].embedding = [0.1] * 1536
    """
    client = MagicMock()

    # Chat completions
    chat_response = MagicMock()
    chat_response.choices = [MagicMock()]
    chat_response.choices[0].message.content = '{"stub": true}'
    chat_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    chat_response.model = "gpt-4o"
    client.chat.completions.create = AsyncMock(return_value=chat_response)

    # Embeddings
    embed_response = MagicMock()
    embed_response.data = [MagicMock(embedding=[0.1] * 1536)]
    embed_response.usage = MagicMock(total_tokens=10)
    client.embeddings.create = AsyncMock(return_value=embed_response)

    return client


# ─────────────────────────────────────────────────────────────────────────────
# Domain Object Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_raw_alert() -> RawAlert:
    """
    A realistic PagerDuty-style alert for a payment-service OOMKilled pod.

    This covers the most common test scenario in the eval set:
    a Kubernetes memory pressure event on a critical-path service.
    """
    return RawAlert(
        alert_id="alert-test-001",
        source="pagerduty",
        received_at=_NOW,
        raw_payload={
            "event_type": "trigger",
            "service_key": "payment-service-prod",
            "incident_key": "k8s/pod/OOMKilled",
            "details": {
                "pod": "payment-svc-7d9f6b-xkp2v",
                "namespace": "production",
                "reason": "OOMKilled",
                "container": "payment-api",
            },
        },
        service_name="payment-service",
        environment="production",
        severity=Severity.SEV1,
        alert_text=(
            "CRITICAL: pod payment-svc-7d9f6b-xkp2v in namespace production "
            "was OOMKilled. Container payment-api exceeded memory limit 512Mi."
        ),
        metric_name="container_memory_usage_bytes",
        metric_value=536_870_912.0,  # 512 MiB
        threshold=536_870_912.0,
    )


@pytest.fixture()
def sample_runbook_metadata() -> RunbookMetadata:
    """
    Metadata for the payment-service memory runbook used across multiple tests.
    """
    return RunbookMetadata(
        runbook_id="runbook-payment-memory-001",
        title="Payment Service — Memory Pressure & OOMKilled Resolution",
        service="payment-service",
        environment=["production", "staging"],
        severity_applicable=[Severity.SEV1, Severity.SEV2],
        author="platform-oncall@example.com",
        created_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
        last_updated_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        version=3,
        tags=["kubernetes", "memory", "oom", "payment"],
        related_runbook_ids=["runbook-k8s-node-pressure-001"],
        source_uri="file://data/runbooks/payment-service/memory-oomkilled.md",
    )


@pytest.fixture()
def sample_chunk(sample_runbook_metadata: RunbookMetadata) -> Chunk:
    """
    A single Chunk representing the remediation steps section of the payment
    service memory runbook.  This fixture is used to validate retrieval,
    reranking, grounding, and feedback loop logic.
    """
    content = (
        "### Step 1 — Identify the offending pod\n"
        "```bash\n"
        "kubectl get pods -n production --field-selector=status.phase=Failed\n"
        "kubectl describe pod <pod-name> -n production | grep -A5 OOMKilled\n"
        "```\n\n"
        "### Step 2 — Check current memory limits\n"
        "```bash\n"
        "kubectl get deployment payment-svc -n production -o jsonpath="
        "'{.spec.template.spec.containers[0].resources}'\n"
        "```\n\n"
        "### Step 3 — Temporarily increase memory limit (low risk)\n"
        "```bash\n"
        "kubectl set resources deployment/payment-svc \\\n"
        "  -n production \\\n"
        "  --limits=memory=1Gi --requests=memory=512Mi\n"
        "```\n"
        "> **Reversible:** yes — rollback with `kubectl rollout undo deployment/payment-svc`"
    )

    return Chunk(
        chunk_id="chunk-payment-memory-step-001",
        runbook_id=sample_runbook_metadata.runbook_id,
        chunk_type=ChunkType.STEP,
        content=content,
        header_path=[
            "Payment Service",
            "Memory Pressure & OOMKilled Resolution",
            "Remediation Steps",
        ],
        header_breadcrumb=(
            "# Payment Service > "
            "## Memory Pressure & OOMKilled Resolution > "
            "### Remediation Steps"
        ),
        position=3,
        token_count=187,
        boost_score=1.0,
        retrieval_count=0,
        last_retrieved_at=None,
        embedding_model="text-embedding-3-small",
        embedding_version="v1",
        content_sha256=_sha256(content),
        service=sample_runbook_metadata.service,
        severity_applicable=sample_runbook_metadata.severity_applicable,
        runbook_last_updated=sample_runbook_metadata.last_updated_at,
    )
