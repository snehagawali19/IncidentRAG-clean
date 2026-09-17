"""Central OpenAI-compatible client and model selection.

OpenRouter is preferred when configured and supports both chat completions and
embeddings through the OpenAI Python SDK. Direct OpenAI remains a fallback.
"""

from __future__ import annotations

from typing import Any

from incidentrag.core.settings import Settings, get_settings


def _openrouter_key(settings: Settings) -> str:
    if settings.openrouter_api_key is None:
        return ""
    key = settings.openrouter_api_key.get_secret_value().strip()
    if not key.startswith("sk-or-v1-") or len(key) <= 20:
        return ""
    return key


def using_openrouter(settings: Settings | None = None) -> bool:
    """Return whether OpenRouter is the active OpenAI-compatible provider."""
    return bool(_openrouter_key(settings or get_settings()))


def openai_client_kwargs(settings: Settings | None = None) -> dict[str, Any]:
    """Build safe SDK constructor arguments without logging credential values."""
    selected = settings or get_settings()
    key = _openrouter_key(selected)
    if key:
        return {
            "api_key": key,
            "base_url": selected.openrouter_base_url,
            "default_headers": {
                "HTTP-Referer": selected.openrouter_http_referer,
                "X-OpenRouter-Title": selected.openrouter_app_title,
            },
        }
    return {"api_key": selected.openai_api_key.strip() or "not-configured"}


def utility_model_name(settings: Settings | None = None) -> str:
    selected = settings or get_settings()
    return (
        selected.openrouter_utility_model
        if using_openrouter(selected)
        else selected.utility_model
    )


def reasoning_model_name(settings: Settings | None = None) -> str:
    selected = settings or get_settings()
    return (
        selected.openrouter_reasoning_model
        if using_openrouter(selected)
        else selected.openai_reasoning_model
    )


def embedding_model_name(settings: Settings | None = None) -> str:
    selected = settings or get_settings()
    return (
        selected.openrouter_embedding_model
        if using_openrouter(selected)
        else selected.openai_embedding_model
    )
