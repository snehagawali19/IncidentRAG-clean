"""Observability Layer — OpenTelemetry Tracer + Cost Tracker.

Provides:
  - get_tracer()          global tracer factory
  - trace_step()          async context manager for spans
  - CostTracker           accumulates token costs per incident
  - embed_assert          decorator to catch encoder mismatch
"""
from __future__ import annotations

import functools
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from incidentrag.core.ai_provider import embedding_model_name
from incidentrag.core.settings import settings

logger = logging.getLogger(__name__)

_provider: TracerProvider | None = None


def _init_provider() -> TracerProvider:
    global _provider
    if _provider is not None:
        return _provider
    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)
    try:
        exporter = OTLPSpanExporter(
            endpoint=settings.otel_exporter_otlp_endpoint,
            insecure=True,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
    except Exception as exc:
        logger.warning("OTEL exporter init failed (running without tracing): %s", exc)
    trace.set_tracer_provider(provider)
    _provider = provider
    return provider


def get_tracer(name: str = "incidentrag") -> trace.Tracer:
    _init_provider()
    return trace.get_tracer(name)


@asynccontextmanager
async def trace_step(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> AsyncIterator[trace.Span]:
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, str(v))
        yield span


# ── Cost Tracker ──────────────────────────────────────────────────────────────

# Rough cost per 1K tokens (USD) — update as prices change
_COST_PER_1K: dict[str, dict[str, float]] = {
    "anthropic/claude-sonnet-4-6": {"prompt": 0.003, "completion": 0.015},
    "openai/gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "openai/gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "text-embedding-3-small": {"prompt": 0.00002, "completion": 0.0},
}


class CostTracker:
    """Accumulates token usage and estimates USD cost per incident."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        step: str = "",
    ) -> None:
        rates = _COST_PER_1K.get(model, {"prompt": 0.0, "completion": 0.0})
        cost = (prompt_tokens * rates["prompt"] + completion_tokens * rates["completion"]) / 1000
        self._records.append({
            "step": step,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "estimated_usd": cost,
        })

    def total_cost(self) -> float:
        return sum(r["estimated_usd"] for r in self._records)

    def summary(self) -> dict[str, Any]:
        return {
            "records": self._records,
            "total_prompt_tokens": sum(r["prompt_tokens"] for r in self._records),
            "total_completion_tokens": sum(r["completion_tokens"] for r in self._records),
            "total_estimated_usd": self.total_cost(),
        }


# ── embed_assert decorator ────────────────────────────────────────────────────

def embed_assert(expected_model: str) -> Callable:
    """Decorator: raises AssertionError if the embedding model doesn't match."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            actual = embedding_model_name(settings)
            if actual != expected_model:
                raise AssertionError(
                    f"Embedding model mismatch: index built with '{expected_model}', "
                    f"but the configured embedding model is '{actual}'. "
                    "Rebuild the index or update settings."
                )
            return await fn(*args, **kwargs)
        return wrapper
    return decorator
