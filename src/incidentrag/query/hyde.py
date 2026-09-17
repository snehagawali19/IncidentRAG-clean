"""Query Layer — HyDE (Hypothetical Document Embeddings) Generator.

Generates a hypothetical runbook section of 300-500 words that *would* resolve
the described incident. Embedding this document and using it as a retrieval query
significantly improves dense search recall for incident response.

Reference: athina-ai/rag-cookbooks advanced_rag_techniques/hyde_rag.ipynb
Do NOT import from references/ — pattern reimplemented fresh.
"""
from __future__ import annotations

import logging

from openai import AsyncOpenAI

from incidentrag.core.ai_provider import openai_client_kwargs, utility_model_name
from incidentrag.core.models import ExtractedEntities, RawAlert

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a senior Site Reliability Engineer writing internal runbook documentation. "
    "Your runbooks are precise, action-oriented, and read by on-call engineers under pressure. "
    "Write in the present tense. Use numbered steps for procedures. "
    "Include specific metric thresholds, kubectl commands, and escalation criteria."
)

_PROMPT = """\
Write a hypothetical runbook section that would resolve this incident.
Target length: 300-500 words.

Incident details:
- Service: {service}
- Environment: {environment}
- Metric: {metric} = {value} (threshold: {threshold})
- Error signature: {error}
- Alert: {alert_text}

Your runbook section must include:
1. **Likely root causes** (3-5 bullet points)
2. **Immediate triage steps** (numbered, with exact commands)
3. **Mitigation procedure** (numbered steps)
4. **Verification** (how to confirm resolution)
5. **Escalation criteria** (when to page a human)

Use realistic kubectl, prometheus query, or AWS CLI commands where applicable.
"""


class HyDEGenerator:
    """Generates a hypothetical runbook section for HyDE-based retrieval."""

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        self._client = client or AsyncOpenAI(**openai_client_kwargs())
        self._model = utility_model_name()

    async def generate(self, alert: RawAlert, entities: ExtractedEntities) -> str:
        """Returns a 300-500 word hypothetical runbook section."""
        prompt = _PROMPT.format(
            service=entities.service or alert.service_name or "unknown-service",
            environment=entities.environment or alert.environment or "production",
            metric=entities.metric or "unknown",
            value=entities.metric_value or "N/A",
            threshold=entities.threshold or "N/A",
            error=entities.error_signature or "N/A",
            alert_text=alert.alert_text[:400],
        )

        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                temperature=0.4,
                max_tokens=700,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("HyDE generation failed: %s", exc)
            # Fallback: return a minimal context string so retrieval can still proceed
            svc = entities.service or alert.service_name or "service"
            err = entities.error_signature or "incident"
            return f"{err} {svc} runbook procedure triage remediation"
