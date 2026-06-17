"""Source-substrate Protocols + normalized Pydantic shapes.

Wave 1 PR α of the institutional data revamp (see
``~/Development/alpha-engine-docs/private/data-revamp-260513.md``). Each
data slot (news, filings, analyst, alt data) becomes a Protocol with
multiple adapters implementing it; concrete adapters live in
alpha-engine-data (producer-side) and consumers (alpha-engine-research)
never import them directly — they read producer outputs via S3 + the
shared RAG retrieval API.

Why Protocols over ABCs:

- Structural subtyping — third-party SDK wrappers can satisfy without
  inheriting from our base class.
- Static type-checking via ``runtime_checkable`` for explicit gating
  in the aggregator.
- No vtable overhead in hot loops.

Why Pydantic shapes (not raw dicts):

- Cross-vendor schema normalization is brittle. Pydantic gives us a
  single canonical shape with validation at the adapter boundary —
  adapter bugs surface as ``ValidationError``, not as silent
  ``KeyError`` in downstream NLP three layers deeper.
- ``extra='forbid'`` ensures vendor schema drift surfaces immediately
  instead of leaking unmapped fields through the pipeline.
- ``frozen=True`` so records are hashable + safe to share across
  threads / fan-out workers without defensive copies.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


# ── Normalized shapes ──────────────────────────────────────────────────


class NewsArticle(BaseModel):
    """One news article, normalized across all vendors.

    The canonical key for cross-vendor dedup is the composite
    (normalized title, URL host+path hash). Different vendors syndicate
    the same wire story; the aggregator's dedup catches both the URL-
    collapse case (different querystrings) and the
    title-paraphrase-on-same-URL case.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tickers: tuple[str, ...] = Field(
        description="Tickers this article concerns. Multi-ticker articles "
                    "(e.g. sector pieces) are kept as a single record with "
                    "a multi-element tuple; the aggregator's ticker-union "
                    "logic merges variants. Adapter's choice: emit once per "
                    "(article, ticker) or once with the full set."
    )
    title: str
    body_excerpt: str = Field(
        description="Lead paragraph or summary. Full-text body lives in the "
                    "RAG corpus chunk store, not in this struct — chunked at "
                    "embedding time by the ingest pipeline."
    )
    url: str
    published_at: datetime = Field(
        description="UTC publish time. Vendor wall-clock; ingest-time is "
                    "in `fetched_at`."
    )
    source: str = Field(
        description="Vendor slug: 'polygon', 'gdelt', 'yahoo_rss', "
                    "'edgar_press', 'benzinga' (paid), 'bloomberg' (paid), "
                    "'ravenpack' (paid). Joins onto the trust-weight config "
                    "downstream."
    )
    vendor_article_id: str | None = Field(
        default=None,
        description="Vendor-native unique ID for cross-reference back to "
                    "the source system (Polygon `id`, GDELT `GKGRECORDID`, "
                    "etc.). Used by ingest-side idempotency checks.",
    )
    fetched_at: datetime = Field(
        description="When this adapter pulled the article (UTC). For "
                    "freshness audit + cache-age computation."
    )
    headline_authors: tuple[str, ...] | None = Field(
        default=None,
        description="Bylines if available. None if the source doesn't "
                    "expose authors (e.g. wire feeds).",
    )
    tags: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Vendor-supplied topic / event tags. GDELT emits "
                    "structured event codes; Polygon emits keywords; "
                    "Benzinga emits Channels. Used as a soft signal for "
                    "downstream event-flag extraction.",
    )


class AnalystSnapshot(BaseModel):
    """One vendor's analyst consensus snapshot for one ticker at one
    point in time.

    Time-series of these in S3 drives self-derived revisions tracking
    (see ``alpha-engine-data/data/derived/revisions.py``, PR C).
    Adapter is responsible for vendor-string normalization at the
    boundary — downstream consumers see the canonical 5-class
    consensus_rating ladder.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str
    source: str
    fetched_at: datetime
    consensus_rating: str | None = Field(
        default=None,
        description="Categorical: 'strongBuy' | 'buy' | 'hold' | 'sell' | "
                    "'strongSell'. Vendor strings normalized at adapter "
                    "boundary.",
    )
    mean_target: float | None = Field(
        default=None, description="Mean price target (USD)."
    )
    median_target: float | None = Field(
        default=None, description="Median price target if vendor exposes it."
    )
    num_analysts: int | None = Field(
        default=None, description="Number of contributing analysts."
    )
    rating_changes_30d: tuple[dict, ...] = Field(
        default_factory=tuple,
        description="Recent upgrades/downgrades. Each entry: "
                    "{analyst, firm, action, prior_rating, new_rating, "
                    "date}. Used by downstream NLP/event-flag extraction.",
    )


class FilingDocument(BaseModel):
    """One filing document. Filings substrate (PR B). Pinned here so
    PR α can reference the shape from Protocols without forward refs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str
    form_type: str = Field(
        description="'10-K' | '10-Q' | '8-K' | '14A' | 'DEF' | 'S-1' | "
                    "'S-4' | '13D' | '13G' | '13F' | 'Form 4' | etc."
    )
    filed_date: datetime
    accession_number: str = Field(
        description="EDGAR accession (e.g. '0000320193-25-000001'). "
                    "Canonical key for filing dedup + RAG idempotency check."
    )
    title: str | None = None
    url: str
    source: str = "edgar"
    fetched_at: datetime
    body_excerpt: str = Field(
        description="Lead snippet. Full body goes to the RAG corpus, "
                    "chunked at ingest time."
    )


# ── Protocols ──────────────────────────────────────────────────────────


@runtime_checkable
class NewsSource(Protocol):
    """News adapter contract. Adapters are vendor-specific transports
    that produce normalized :class:`NewsArticle` records.

    Adapters MUST:

    - Be safely callable from concurrent contexts (own their HTTP client +
      rate-limiter; no shared mutable state).
    - Return an empty list (never raise) on transient vendor failures.
      Re-raise only on auth failures, contract-breaking schema drift, or
      configuration errors — those fail loud.
    - Normalize wall-clock timestamps to UTC.
    - Stamp ``fetched_at`` on every returned article.
    """

    name: str  # vendor slug — joins onto trust-weight config

    def fetch(
        self,
        tickers: list[str],
        *,
        hours: int = 48,
    ) -> list[NewsArticle]: ...


@runtime_checkable
class AnalystSource(Protocol):
    """Analyst data adapter contract. PR C.

    Returns ``None`` for tickers the vendor doesn't cover (e.g. micro-
    caps absent from FMP); empty fields within a returned snapshot
    indicate vendor-side coverage gaps that are typed in the shape.
    """

    name: str

    def fetch(self, ticker: str) -> AnalystSnapshot | None: ...


@runtime_checkable
class FilingSource(Protocol):
    """Filings adapter contract. PR B.

    ``form_types`` filter is advisory — adapters that don't support
    form-type pre-filtering at the vendor layer return all forms and
    let downstream filter.
    """

    name: str

    def fetch(
        self,
        tickers: list[str],
        *,
        form_types: list[str] | None = None,
        days: int = 7,
    ) -> list[FilingDocument]: ...
