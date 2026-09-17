"""Query Layer — Incident Classifier.

Fast rule-based classification with LLM fallback for unknown cases.
Maps alert text to one of the six canonical IncidentCategory values
defined in core/models.py.

Do NOT add new IncidentCategory values — always import from models.py.
"""
from __future__ import annotations

import logging

from openai import AsyncOpenAI

from incidentrag.core.ai_provider import openai_client_kwargs, utility_model_name
from incidentrag.core.models import ExtractedEntities, IncidentCategory, RawAlert

logger = logging.getLogger(__name__)

# ── Keyword map against the 6 canonical categories ───────────────────────────

_KEYWORD_MAP: dict[IncidentCategory, list[str]] = {
    IncidentCategory.PERFORMANCE_DEGRADATION: [
        # Latency
        "latency", "slow", "timeout", "p99", "p95", "p50", "response time", "duration",
        # Memory / OOM
        "oomkill", "oom", "out of memory", "memory limit", "evicted", "container killed",
        "heap", "gc pause", "jvm memory",
        # CPU
        "cpu", "throttl", "iowait", "load average", "cpu_usage",
        # Disk / IO
        "disk full", "iops", "disk saturation", "disk_usage",
    ],
    IncidentCategory.COMPLETE_OUTAGE: [
        "error rate", "5xx", "4xx", "exception", "error%", "http_errors",
        "crashloopbackoff", "crash", "restart", "backoff", "exit code",
        "deployment", "rollout", "imagepullbackoff", "pending", "unavailable",
        "down", "not ready", "offline", "service unavailable",
    ],
    IncidentCategory.SECURITY_EVENT: [
        "certificate", "cert", "tls", "ssl", "expir", "x509",
        "unauthorized", "forbidden", "injection", "attack", "breach", "vulnerability",
        "403", "401",
    ],
    IncidentCategory.DEPENDENCY_FAILURE: [
        "network", "packet loss", "dns", "connection refused", "tcp", "bandwidth",
        "connection pool", "too many connections", "pgbouncer", "max_connections",
        "connection timeout", "upstream", "downstream", "dependency failed",
    ],
    IncidentCategory.CONFIGURATION_DRIFT: [
        "configuration", "config", "drift", "misconfigured", "invalid setting",
        "wrong value", "secret", "env var", "environment variable", "missing key",
    ],
}


class IncidentClassifier:
    """
    Rule-based keyword classifier with an LLM fallback for UNKNOWN cases.

    Uses canonical IncidentCategory from models.py exclusively.
    """

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        self._client = client or AsyncOpenAI(**openai_client_kwargs())
        self._model = utility_model_name()

    def classify_rules(self, alert: RawAlert) -> IncidentCategory:
        """Score the alert text against keyword lists; return highest-scoring category."""
        text = alert.alert_text.lower()
        scores: dict[IncidentCategory, int] = {}
        for category, keywords in _KEYWORD_MAP.items():
            score = sum(1 for kw in keywords if kw in text)
            if score:
                scores[category] = score
        if not scores:
            return IncidentCategory.UNKNOWN
        return max(scores, key=lambda c: scores[c])

    async def classify(
        self,
        alert: RawAlert,
        entities: ExtractedEntities,
    ) -> IncidentCategory:
        """
        Classify an alert.  Uses rule-based matching first; falls back to an
        LLM call when no keyword matches.
        """
        # Fast rule-based path
        category = self.classify_rules(alert)
        if category != IncidentCategory.UNKNOWN:
            return category

        # LLM fallback for ambiguous cases
        try:
            categories = [c.value for c in IncidentCategory if c != IncidentCategory.UNKNOWN]
            resp = await self._client.chat.completions.create(
                model=self._model,
                temperature=0.0,
                max_tokens=20,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Classify this infrastructure alert into exactly one category.\n"
                            f"Categories: {categories}\n"
                            f"Alert: {alert.alert_text[:300]}\n"
                            f"Reply with only the category value, nothing else."
                        ),
                    }
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            return IncidentCategory(raw)
        except (ValueError, Exception) as exc:  # noqa: BLE001
            logger.debug("Classifier LLM fallback failed: %s", exc)
            return IncidentCategory.UNKNOWN
