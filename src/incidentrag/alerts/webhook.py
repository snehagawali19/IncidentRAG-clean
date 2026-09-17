"""Generic webhook-to-RawAlert adapter."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from incidentrag.core.models import RawAlert, Severity


class WebhookAlertSource:
    """Normalize bounded webhook payloads without executing embedded content."""

    def normalize(self, payload: dict[str, Any]) -> RawAlert:
        text = str(payload.get("alert_text") or payload.get("message") or "")[:8_000]
        if not text.strip():
            raise ValueError("Webhook payload requires alert_text or message")
        raw_severity = str(payload.get("severity", "sev3")).lower()
        try:
            severity = Severity(raw_severity)
        except ValueError:
            severity = Severity.SEV3
        allowed_sources = {"pagerduty", "datadog", "prometheus", "grafana", "opsgenie"}
        source = str(payload.get("source", "prometheus")).lower()
        if source not in allowed_sources:
            source = "prometheus"
        return RawAlert(
            alert_id=str(payload.get("alert_id") or uuid4()),
            source=source,
            received_at=datetime.now(UTC),
            alert_text=text,
            severity=severity,
            service_name=str(payload.get("service_name") or "unknown")[:200],
            environment=str(payload.get("environment") or "unknown")[:100],
            raw_payload={},
        )
