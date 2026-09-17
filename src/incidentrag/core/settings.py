"""
Application settings for IncidentRAG.
All configuration is read from environment variables (or a .env file).
Uses pydantic-settings for type-safe, validated access.
"""

from __future__ import annotations

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central settings object.  Import and call `get_settings()` everywhere —
    never instantiate Settings() directly in production code so the cached
    singleton is always returned.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM API Keys ──────────────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", description="Optional Anthropic API key")
    openai_api_key: str = Field(default="", description="Optional direct OpenAI API key")

    # ── Model Names ────────────────────────────────────────────────────────
    anthropic_reasoning_model: str = Field(
        default="claude-opus-4-5",
        description="Primary Anthropic model for reasoning and generation",
    )
    openai_reasoning_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model used as fallback / secondary reasoner",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model for dense retrieval",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="sentence-transformers cross-encoder model for reranking",
    )

    # ── Neo4j ──────────────────────────────────────────────────────────────
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j Bolt URI",
    )
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(
        default="incidentrag", description="Neo4j password"
    )

    # ── Qdrant ─────────────────────────────────────────────────────────────
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant HTTP endpoint",
    )
    qdrant_api_key: str = Field(
        default="",
        description="Qdrant API key (leave empty for local unauthenticated instances)",
    )
    qdrant_collection_name: str = Field(
        default="incidentrag_chunks",
        description="Qdrant collection that stores chunk embeddings",
    )
    qdrant_embedding_dim: int = Field(
        default=1536,
        description="Embedding dimensionality — must match the index manifest",
    )

    # ── Redis ──────────────────────────────────────────────────────────────
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL (used for caching and feedback queue)",
    )

    # GitHub is a manual, read-only public issue source.  The token is optional.
    github_owner: str = Field(default="argoproj")
    github_repo: str = Field(default="argo-cd")
    github_token: SecretStr | None = Field(default=None, repr=False)
    github_max_results: int = Field(default=2, ge=1)
    github_candidate_limit: int = Field(default=10, ge=1)
    github_updated_within_days: int = Field(default=90, ge=1)
    github_api_base_url: str = Field(default="https://api.github.com")
    github_request_timeout_seconds: float = Field(default=10.0, gt=0, le=30)
    demo_mode: bool = Field(default=True)
    execution_enabled: bool = Field(default=False)
    incidentrag_api_key: SecretStr | None = Field(default=None, repr=False)
    cors_allowed_origins: str = Field(
        default="http://localhost:8501,http://localhost:8000",
        description="Comma-separated browser origins allowed to call the API",
    )

    # ── OpenTelemetry ──────────────────────────────────────────────────────
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4317",
        description="OTLP gRPC collector endpoint",
    )
    otel_service_name: str = Field(
        default="incidentrag",
        description="Service name attached to OpenTelemetry spans",
    )

    # ── Application Behaviour ──────────────────────────────────────────────
    app_environment: str = Field(
        default="development",
        description="Runtime environment tag (development | staging | production)",
    )
    max_context_tokens: int = Field(
        default=180_000,
        description="Hard token ceiling per inference call",
    )
    retrieval_top_k: int = Field(
        default=10,
        description="Number of chunks returned after reranking",
    )
    bm25_candidate_k: int = Field(
        default=50,
        description="BM25 candidate pool size before reranking",
    )
    dense_candidate_k: int = Field(
        default=50,
        description="Dense retrieval candidate pool size before reranking",
    )
    reranker_min_score: float = Field(
        default=0.0,
        description=(
            "Minimum cross-encoder relevance score. Candidates below this value "
            "are not supplied to the reasoner."
        ),
    )

    # ── Evaluation ─────────────────────────────────────────────────────────
    eval_dataset_path: str = Field(
        default="data/ground_truth/eval_set.jsonl",
        description="Path to the JSONL ground-truth evaluation dataset",
    )
    eval_min_faithfulness: float = Field(
        default=0.85,
        description="Minimum RAGAS faithfulness score required to pass the CI gate",
    )

    # ── Caching ────────────────────────────────────────────────────────────
    diskcache_dir: str = Field(
        default=".cache/embeddings",
        description="Directory for the embedding diskcache",
    )

    # ── Query Pipeline ─────────────────────────────────────────────────────
    utility_model: str = Field(
        default="gpt-4o-mini",
        description="Small LLM for classification, fanout, and entity extraction",
    )
    auto_execute_low_risk: bool = Field(
        default=False,
        description="If True, LOW-risk actions are executed without human approval",
    )

    # ── OpenRouter (optional proxy) ────────────────────────────────────────
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        repr=False,
        description="OpenRouter API key (preferred for chat and embeddings)",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter base URL",
    )
    openrouter_utility_model: str = Field(
        default="openai/gpt-4o-mini",
        description="OpenRouter model used for query understanding",
    )
    openrouter_reasoning_model: str = Field(
        default="openai/gpt-4o-mini",
        description="OpenRouter model used for assessment and grounding",
    )
    reasoning_max_output_tokens: int = Field(
        default=2_000,
        ge=256,
        le=8_192,
        description="Maximum completion tokens for a structured assessment",
    )
    openrouter_embedding_model: str = Field(
        default="openai/text-embedding-3-small",
        description="OpenRouter embedding model; dimension must match Qdrant",
    )
    openrouter_http_referer: str = Field(
        default="http://localhost:8000",
        description="Optional OpenRouter attribution URL",
    )
    openrouter_app_title: str = Field(
        default="IncidentRAG",
        description="Optional OpenRouter attribution title",
    )

    # ── Retrieval Tuning ───────────────────────────────────────────────────
    rrf_k: int = Field(
        default=60,
        description="RRF constant k in 1/(k+rank). Default 60 per the SPEC.",
    )
    recency_half_life_days: int = Field(
        default=90,
        description="Days after which a runbook's recency boost drops to 0.5",
    )

    # ── Validators ─────────────────────────────────────────────────────────
    @field_validator("app_environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"app_environment must be one of {allowed}, got '{v}'")
        return v

    @field_validator("eval_min_faithfulness")
    @classmethod
    def validate_faithfulness_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("eval_min_faithfulness must be between 0.0 and 1.0")
        return v

    @field_validator("retrieval_top_k", "bm25_candidate_k", "dense_candidate_k")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Retrieval k values must be positive integers")
        return v


_settings: Settings | None = None


def get_settings() -> Settings:
    """
    Return the cached Settings singleton.
    Reads from .env on first call; subsequent calls return the cached instance.
    Use this function everywhere instead of instantiating Settings() directly.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Module-level singleton — allows `from incidentrag.core.settings import settings`.
# Use get_settings() everywhere else to benefit from the lazy cache.
settings = get_settings()
