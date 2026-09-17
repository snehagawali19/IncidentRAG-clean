"""Query Layer — Entity Extractor.

Parses a RawAlert into ExtractedEntities using the utility LLM (gpt-4o-mini).
Extracts canonical fields defined in ExtractedEntities from core/models.py:
  service, environment, metric, metric_value, threshold,
  host_identifier, error_signature, correlated_services.

Do NOT add fields to ExtractedEntities outside of models.py.
"""
from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from incidentrag.core.ai_provider import openai_client_kwargs, utility_model_name
from incidentrag.core.models import ExtractedEntities, RawAlert

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = """\
Extract structured entities from this infrastructure alert.
Return ONLY valid JSON with these exact keys (use null for missing fields):

{{
  "service": string | null,
  "environment": "production" | "staging" | "dev" | null,
  "metric": string | null,
  "metric_value": number | null,
  "threshold": number | null,
  "host_identifier": string | null,
  "error_signature": string | null,
  "correlated_services": [string] | []
}}

Alert: {alert_text}
"""

# Fields that map directly from LLM JSON to ExtractedEntities constructor
_ALLOWED_FIELDS = {
    "service", "environment", "metric", "metric_value",
    "threshold", "host_identifier", "error_signature", "correlated_services",
}


class EntityExtractor:
    """Extracts structured entities from raw alert text via LLM call."""

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        self._client = client or AsyncOpenAI(**openai_client_kwargs())
        self._model = utility_model_name()

    async def extract(self, alert: RawAlert) -> ExtractedEntities:
        """
        Call the utility LLM and map the JSON response to ExtractedEntities.

        On failure returns a minimal ExtractedEntities with all fields None.
        """
        prompt = _EXTRACT_PROMPT.format(alert_text=alert.alert_text[:800])
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            raw: dict = json.loads(resp.choices[0].message.content or "{}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("EntityExtractor failed: %s", exc)
            raw = {}

        # Keep only canonical ExtractedEntities fields; discard anything extra
        filtered = {
            k: v
            for k, v in raw.items()
            if k in _ALLOWED_FIELDS and v is not None
        }

        # correlated_services must be a list
        if "correlated_services" not in filtered:
            filtered["correlated_services"] = []

        return ExtractedEntities(**filtered)
