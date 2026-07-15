"""RAG submodule — semantic retrieval over SEC filings, earnings transcripts, and theses.

Shared library code used by both alpha-engine-research (retrieval consumer:
qual analyst tools) and alpha-engine-data (ingestion producer: weekly Saturday
RAGIngestion step). Previously duplicated across both repos with drift; moved
here in nousergon-lib v0.3.0 as the single source of truth.

Top-level imports re-export the most common surface so consumers can write
``from nousergon_lib.rag import retrieve`` without reaching into submodules.

Pgvector + psycopg2 are heavy dependencies; install via the ``[rag]`` extra:

    pip install "nousergon-lib[rag] @ git+https://github.com/nousergon/nousergon-lib@v0.3.0"
"""

# Auto-load .env so RAG_DATABASE_URL and VOYAGE_API_KEY are available
# whether run from CLI, Lambda (already in env), or imported in tests.
# python-dotenv is an optional soft-dependency (not declared under any
# extra — this try/except IS the dependency contract) and isn't installed
# in the pyright CI environment (test.yml only installs
# [dev,rag,arcticdb,quant,quant-xs,quant-stats,contracts]), so the import
# is unresolved there even though it resolves fine in a dev venv that
# happens to have it.
try:
    from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed (e.g. Lambda) — env vars set externally

from .db import coerce_embedding, get_connection, is_available
from .embeddings import embed_texts
from .retrieval import (
    retrieve,
    ingest_document,
    document_exists,
)

__all__ = [
    "coerce_embedding",
    "get_connection",
    "is_available",
    "embed_texts",
    "retrieve",
    "ingest_document",
    "document_exists",
]
