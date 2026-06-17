"""Shared data-source contracts — Pydantic shapes + Protocols.

This package defines the canonical normalized shapes (``NewsArticle``,
``AnalystSnapshot``, ``FilingDocument``) and the adapter Protocols
(``NewsSource``, ``AnalystSource``, ``FilingSource``) that both
producers (alpha-engine-data) and consumers (alpha-engine-research,
alpha-engine-backtester) consume.

**Architectural pattern:** Lib defines the contract; producers
(alpha-engine-data) implement concrete adapters; consumers
(alpha-engine-research) read records produced by them via S3 / RAG
retrieval and never import adapter classes directly.

See the institutional data-revamp plan doc at
``~/Development/alpha-engine-docs/private/data-revamp-260513.md`` for
full context. PR α — lib (this); PR β — data (adapter implementations).
"""

from nousergon_lib.sources.protocols import (
    AnalystSnapshot,
    AnalystSource,
    FilingDocument,
    FilingSource,
    NewsArticle,
    NewsSource,
)

__all__ = [
    "NewsArticle",
    "AnalystSnapshot",
    "FilingDocument",
    "NewsSource",
    "AnalystSource",
    "FilingSource",
]
