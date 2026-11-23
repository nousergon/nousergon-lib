"""Voyage embedding wrapper for RAG document and query embeddings.

Uses Voyage voyage-3-lite (512 dimensions), optimized for retrieval on
financial text. Batch support up to 128 texts per call.

The 512d output matches the ``embedding vector(512)`` column declared
in ``rag/schema.sql`` — pgvector enforces dimension on INSERT, so any
drift between the model and the schema would be a hard failure on
ingestion.

Requires: VOYAGE_API_KEY environment variable.
"""

from __future__ import annotations

import logging
from typing import Protocol, cast


class _EmbeddingResult(Protocol):
    embeddings: list[list[float]]


class _EmbeddingClient(Protocol):
    def embed(
        self, texts: list[str], *, model: str, input_type: str
    ) -> _EmbeddingResult: ...

logger = logging.getLogger(__name__)

_client: _EmbeddingClient | None = None


def _get_client() -> _EmbeddingClient:
    global _client
    if _client is None:
        # voyageai is not declared under any pyproject.toml extra (a
        # pre-existing gap — the [rag] extra covers psycopg2/pgvector/numpy
        # only) and isn't installed in the pyright CI environment
        # (test.yml only installs
        # [dev,rag,arcticdb,quant,quant-xs,quant-stats,contracts]), so this
        # import is unresolved there regardless of local venv state.
        import voyageai  # pyright: ignore[reportMissingImports]
        from krepis.secrets import get_secret

        # VOYAGE_API_KEY is a pinned secret (alpha-engine-config-I7925) —
        # resolve through the sanctioned SSM-with-env-fallback path rather
        # than a literal os.environ.get, which consumer repos' CI forbids
        # in-repo but could not see inside this installed library.
        # The ignored optional import has no usable type information at this boundary.
        _client = cast(
            _EmbeddingClient,
            voyageai.Client(  # pyright: ignore[reportUnknownMemberType] - optional dependency is untyped
                api_key=get_secret("VOYAGE_API_KEY", required=False)
            ),
        )
    return _client


def embed_texts(
    texts: list[str],
    input_type: str = "document",
    model: str = "voyage-3-lite",
    batch_size: int = 128,
) -> list[list[float]]:
    """Embed a batch of text chunks.

    Args:
        texts: List of text strings to embed.
        input_type: 'document' for storage, 'query' for retrieval.
        model: Voyage model name.
        batch_size: Max texts per API call (Voyage limit is 128).

    Returns:
        List of embedding vectors (each 512 floats for voyage-3-lite).
    """
    client = _get_client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = client.embed(batch, model=model, input_type=input_type)
        all_embeddings.extend(result.embeddings)

    return all_embeddings


def embed_query(query: str, model: str = "voyage-3-lite") -> list[float]:
    """Embed a single query for retrieval."""
    return embed_texts([query], input_type="query", model=model)[0]
