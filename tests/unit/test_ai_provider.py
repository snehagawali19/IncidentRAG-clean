"""Tests for automatic OpenRouter/OpenAI provider selection."""

from __future__ import annotations

from pydantic import SecretStr

from incidentrag.core.ai_provider import (
    embedding_model_name,
    openai_client_kwargs,
    reasoning_model_name,
    using_openrouter,
    utility_model_name,
)
from incidentrag.core.settings import Settings


def test_openrouter_is_preferred_for_chat_and_embeddings() -> None:
    settings = Settings(
        _env_file=None,
        openrouter_api_key=SecretStr("sk-or-v1-valid-test-key-1234567890"),
        openai_api_key="",
    )
    kwargs = openai_client_kwargs(settings)
    assert using_openrouter(settings) is True
    assert kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert kwargs["default_headers"]["X-OpenRouter-Title"] == "IncidentRAG"
    assert utility_model_name(settings) == "openai/gpt-4o-mini"
    assert reasoning_model_name(settings) == "openai/gpt-4o-mini"
    assert embedding_model_name(settings) == "openai/text-embedding-3-small"


def test_placeholder_openrouter_key_does_not_activate_provider() -> None:
    settings = Settings(
        _env_file=None,
        openrouter_api_key=SecretStr("your_openrouter_api_key"),
        openai_api_key="direct-openai-test-key",
    )
    assert using_openrouter(settings) is False
    assert openai_client_kwargs(settings) == {"api_key": "direct-openai-test-key"}
    assert utility_model_name(settings) == settings.utility_model
    assert reasoning_model_name(settings) == settings.openai_reasoning_model
    assert embedding_model_name(settings) == settings.openai_embedding_model


def test_secret_values_are_masked_in_settings_representation() -> None:
    secret = "sk-or-v1-valid-test-key-1234567890"
    settings = Settings(_env_file=None, openrouter_api_key=SecretStr(secret))
    assert secret not in repr(settings)
