"""Query Layer — Query Fanout Generator.

Generates exactly 5 semantic rewrites of an alert, each targeting a
different retrieval angle:
  1. REMEDIATION   — step-by-step fix procedures
  2. ROOT_CAUSE    — diagnostic signals and probable causes
  3. DIAGNOSTIC    — commands and metrics to investigate
  4. DEPENDENCIES  — affected upstream/downstream services
  5. HISTORICAL    — past incidents with similar patterns
"""
from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from incidentrag.core.ai_provider import openai_client_kwargs, utility_model_name
from incidentrag.core.models import ExtractedEntities, RawAlert

logger = logging.getLogger(__name__)

FANOUT_TYPES = ["REMEDIATION", "ROOT_CAUSE", "DIAGNOSTIC", "DEPENDENCIES", "HISTORICAL"]

_FANOUT_PROMPT = """\
You are an SRE search query expert. Generate exactly 5 search queries to find relevant runbooks.

Each query must target a different angle:
1. REMEDIATION: step-by-step fix and recovery procedures
2. ROOT_CAUSE: diagnostic signals and probable causes
3. DIAGNOSTIC: kubectl/prometheus commands to investigate
4. DEPENDENCIES: affected upstream or downstream services
5. HISTORICAL: past incidents with the same pattern

Service: {service}
Environment: {environment}
Error signature: {error_signature}
Alert: {alert_text}

Return ONLY a JSON object:
{{"queries": ["query1", "query2", "query3", "query4", "query5"]}}

Each query: 5-15 words, specific, no duplicates.
"""


class QueryFanoutGenerator:
    """Generates 5 semantically distinct retrieval queries from an alert."""

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        self._client = client or AsyncOpenAI(**openai_client_kwargs())
        self._model = utility_model_name()

    async def generate(
        self,
        alert: RawAlert,
        entities: ExtractedEntities,
    ) -> list[str]:
        """Returns exactly 5 distinct search queries."""
        service = entities.service or alert.service_name or "unknown-service"
        env = entities.environment or alert.environment or "production"
        error_sig = entities.error_signature or "unknown"

        prompt = _FANOUT_PROMPT.format(
            service=service,
            environment=env,
            error_signature=error_sig,
            alert_text=alert.alert_text[:500],
        )

        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                temperature=0.3,
                max_tokens=400,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            data = json.loads(resp.choices[0].message.content or '{"queries": []}')
            queries: list[str] = data.get("queries", [])
            if isinstance(queries, dict):
                queries = list(queries.values())
        except Exception as exc:  # noqa: BLE001
            logger.warning("QueryFanout failed: %s", exc)
            queries = []

        # Ensure exactly 5 distinct queries
        seen: set[str] = set()
        unique: list[str] = []
        for q in queries:
            if q and q not in seen:
                seen.add(q)
                unique.append(q)

        fallbacks = [
            f"{error_sig} {service} remediation runbook",
            f"{error_sig} root cause analysis {service}",
            f"kubectl diagnose {service} {error_sig}",
            f"{service} dependencies blast radius {error_sig}",
            f"past incidents {service} {error_sig} resolution",
        ]
        for fb in fallbacks:
            if len(unique) >= 5:
                break
            if fb not in seen:
                unique.append(fb)

        return unique[:5]
