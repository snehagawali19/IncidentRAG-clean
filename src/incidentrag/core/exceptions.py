"""
Custom exception hierarchy for IncidentRAG.
All application exceptions inherit from IncidentRAGError for uniform catching.
"""

from __future__ import annotations


class IncidentRAGError(Exception):
    """Base class for all IncidentRAG application errors."""


class IngestionError(IncidentRAGError):
    """
    Raised when a runbook document cannot be loaded, parsed, or chunked.

    Examples:
        - Unsupported file format
        - Malformed Markdown structure
        - Chunker produces zero chunks for a non-empty document
        - SHA-256 deduplication collision with mismatched content
    """


class RetrievalError(IncidentRAGError):
    """
    Raised when the hybrid retrieval pipeline fails to return results.

    Examples:
        - Qdrant collection not found or unreachable
        - BM25 index not initialised before querying
        - RRF fusion receives empty dense *and* BM25 result sets
        - Embedding dimension mismatch detected via IndexManifest
    """


class GenerationError(IncidentRAGError):
    """
    Raised when the LLM generation step fails or returns malformed output.

    Examples:
        - LLM API call returns a non-200 status
        - Structured output parsing fails (Pydantic ValidationError wraps this)
        - Token budget exceeded before a complete response is produced
        - Model refuses the prompt (content policy)
    """


class GroundingError(IncidentRAGError):
    """
    Raised when a Claim cannot be grounded against its Evidence objects.

    This is NOT a soft verification failure (which returns verified=False on the
    Claim).  GroundingError is raised when the verifier itself encounters an
    unrecoverable error — e.g. the evidence list is empty, the grounding LLM
    call fails, or the chunk_id in Evidence does not exist in the index.
    """


class BudgetExceededError(IncidentRAGError):
    """
    Raised when the token or cost budget for a single inference call is exceeded.

    Attributes:
        budget_tokens: The configured hard ceiling in tokens.
        used_tokens:   The number of tokens that would have been consumed.
        budget_usd:    The configured hard ceiling in USD (optional).
        used_usd:      The estimated cost that would have been incurred.
    """

    def __init__(
        self,
        *,
        budget_tokens: int,
        used_tokens: int,
        budget_usd: float | None = None,
        used_usd: float | None = None,
    ) -> None:
        self.budget_tokens = budget_tokens
        self.used_tokens = used_tokens
        self.budget_usd = budget_usd
        self.used_usd = used_usd

        msg = (
            f"Token budget exceeded: {used_tokens} tokens requested, "
            f"hard ceiling is {budget_tokens}."
        )
        if budget_usd is not None and used_usd is not None:
            msg += f"  Cost ceiling: ${budget_usd:.4f}, estimated: ${used_usd:.4f}."
        super().__init__(msg)


class EncoderMismatchError(IncidentRAGError):
    """
    Raised when the query encoder does not match the index encoder.

    The IndexManifest records the embedding_model and embedding_version used to
    build the Qdrant index.  If a retrieval call is made with a different
    encoder, embeddings are incomparable and results will be silently wrong.
    This exception makes that failure loud and immediate.

    Attributes:
        index_model:   The model recorded in the IndexManifest.
        query_model:   The model currently configured for query encoding.
        index_version: The embedding_version recorded in the IndexManifest.
        query_version: The embedding_version currently configured.
    """

    def __init__(
        self,
        *,
        index_model: str,
        query_model: str,
        index_version: str,
        query_version: str,
    ) -> None:
        self.index_model = index_model
        self.query_model = query_model
        self.index_version = index_version
        self.query_version = query_version

        super().__init__(
            f"Encoder mismatch: index was built with '{index_model}' (v{index_version}) "
            f"but query uses '{query_model}' (v{query_version}).  "
            f"Re-index the collection or update the query encoder configuration."
        )
