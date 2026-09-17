"""Operational acceptance tests for Phases 9-11."""

from __future__ import annotations

from unittest.mock import AsyncMock

from incidentrag.alerts.webhook import WebhookAlertSource
from incidentrag.execution.kubectl_executor import KubectlExecutor
from incidentrag.execution.metric_monitor import MetricMonitor
from incidentrag.execution.sandbox import SandboxExecutor
from incidentrag.feedback.postmortem_generator import PostmortemGenerator
from tests.unit.conftest import make_assessment, make_raw_alert


async def test_kubectl_executor_rejects_other_executables() -> None:
    executor = KubectlExecutor(SandboxExecutor(dry_run=True))
    result = await executor.execute("a1", "aws s3 ls", "approved")
    assert result.success is False
    assert "kubectl commands only" in result.stderr


async def test_kubectl_executor_preserves_approval_gate() -> None:
    executor = KubectlExecutor(SandboxExecutor(dry_run=True))
    result = await executor.execute("a1", "kubectl get pods", "pending")
    assert result.success is False
    assert "pending" in result.stderr


async def test_metric_monitor_is_bounded() -> None:
    probe = AsyncMock(side_effect=[(False, 10.0), (True, 1.0)])
    result = await MetricMonitor(interval_seconds=0, max_attempts=3).verify(probe)
    assert result.healthy is True
    assert result.attempts == 2


def test_postmortem_contains_reviewable_sections() -> None:
    text = PostmortemGenerator().generate(make_assessment(), make_raw_alert())
    assert "## Root cause" in text
    assert "## Evidence" in text
    assert "## Remediation" in text


def test_webhook_adapter_discards_raw_payload() -> None:
    alert = WebhookAlertSource().normalize(
        {
            "source": "prometheus",
            "message": "payment-service memory is above threshold",
            "secret": "must-not-be-retained",
        }
    )
    assert alert.source == "prometheus"
    assert alert.raw_payload == {}
