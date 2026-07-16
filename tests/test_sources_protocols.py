"""Tests for ``nousergon_lib.sources`` shapes + Protocols (v0.15.0).

Wave 1 PR α of the institutional data revamp (see
``alpha-engine-docs/private/data-revamp-260513.md``). Lib defines the
contract; alpha-engine-data implements concrete adapters in PR β.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from nousergon_lib.sources import (
    AnalystSnapshot,
    AnalystSource,
    FilingDocument,
    FilingSource,
    NewsArticle,
    NewsSource,
)

# ── NewsArticle ────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TestNewsArticleShape:
    def test_canonical_construction(self):
        a = NewsArticle(
            tickers=("AAPL",),
            title="Earnings beat",
            body_excerpt="lead",
            url="https://example.com/x",
            published_at=_now(),
            source="polygon",
            fetched_at=_now(),
        )
        assert a.source == "polygon"
        assert a.tickers == ("AAPL",)
        assert a.tags == ()
        assert a.headline_authors is None
        assert a.vendor_article_id is None

    def test_frozen_blocks_assignment(self):
        a = NewsArticle(
            tickers=("AAPL",), title="t", body_excerpt="b",
            url="https://x", published_at=_now(),
            source="polygon", fetched_at=_now(),
        )
        with pytest.raises(ValidationError):
            a.title = "different"  # type: ignore[misc]

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError, match="Extra inputs are not"):
            NewsArticle(
                tickers=("AAPL",), title="t", body_excerpt="b",
                url="https://x", published_at=_now(),
                source="polygon", fetched_at=_now(),
                vendor_specific_field="oops",  # type: ignore[call-arg]
            )

    def test_multi_ticker_record(self):
        a = NewsArticle(
            tickers=("AAPL", "MSFT", "GOOGL"),
            title="Sector roundup", body_excerpt="...",
            url="https://x", published_at=_now(),
            source="polygon", fetched_at=_now(),
        )
        assert len(a.tickers) == 3

    def test_records_are_hashable(self):
        """Frozen Pydantic shapes must be hashable so they're safe to
        use in sets / dict keys across the fan-out fan-in dedup path."""
        ts = _now()
        a = NewsArticle(
            tickers=("AAPL",), title="t", body_excerpt="b",
            url="https://x", published_at=ts,
            source="polygon", fetched_at=ts,
        )
        b = NewsArticle(
            tickers=("AAPL",), title="t", body_excerpt="b",
            url="https://x", published_at=ts,
            source="polygon", fetched_at=ts,
        )
        assert {a, b} == {a}


# ── AnalystSnapshot ────────────────────────────────────────────────────


class TestAnalystSnapshotShape:
    def test_canonical_construction(self):
        s = AnalystSnapshot(
            ticker="AAPL", source="fmp", fetched_at=_now(),
            consensus_rating="buy", mean_target=250.0, num_analysts=18,
        )
        assert s.ticker == "AAPL"
        assert s.consensus_rating == "buy"
        assert s.median_target is None
        assert s.rating_changes_30d == ()

    def test_frozen(self):
        s = AnalystSnapshot(ticker="X", source="fmp", fetched_at=_now())
        with pytest.raises(ValidationError):
            s.ticker = "Y"  # type: ignore[misc]

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError, match="Extra inputs are not"):
            AnalystSnapshot(
                ticker="X", source="fmp", fetched_at=_now(),
                vendor_field="oops",  # type: ignore[call-arg]
            )


# ── FilingDocument ─────────────────────────────────────────────────────


class TestFilingDocumentShape:
    def test_canonical_construction(self):
        f = FilingDocument(
            ticker="AAPL", form_type="10-K",
            filed_date=_now(),
            accession_number="0000320193-25-000001",
            url="https://www.sec.gov/.../10-k.htm",
            fetched_at=_now(),
            body_excerpt="ITEM 1 BUSINESS...",
        )
        assert f.form_type == "10-K"
        assert f.source == "edgar"  # default

    def test_frozen(self):
        f = FilingDocument(
            ticker="X", form_type="8-K",
            filed_date=_now(), accession_number="x",
            url="https://x", fetched_at=_now(),
            body_excerpt="x",
        )
        with pytest.raises(ValidationError):
            f.ticker = "Y"  # type: ignore[misc]


# ── Protocol structural subtyping ──────────────────────────────────────


class _NewsAdapterImpl:
    """Minimal NewsSource implementation for structural-subtyping test."""
    name = "test_news"

    def fetch(self, tickers: list[str], *, hours: int = 48) -> list[NewsArticle]:
        return []


class _AnalystAdapterImpl:
    name = "test_analyst"

    def fetch(self, ticker: str) -> AnalystSnapshot | None:
        return None


class _FilingAdapterImpl:
    name = "test_filing"

    def fetch(
        self,
        tickers: list[str],
        *,
        form_types: list[str] | None = None,
        days: int = 7,
    ) -> list[FilingDocument]:
        return []


def test_news_protocol_structural_subtype():
    assert isinstance(_NewsAdapterImpl(), NewsSource)


def test_analyst_protocol_structural_subtype():
    assert isinstance(_AnalystAdapterImpl(), AnalystSource)


def test_filing_protocol_structural_subtype():
    assert isinstance(_FilingAdapterImpl(), FilingSource)


def test_class_missing_required_method_is_not_subtype():
    @dataclass
    class Incomplete:
        name: str = "x"

    assert not isinstance(Incomplete(), NewsSource)


def test_class_missing_name_attr_is_not_subtype():
    class NoName:
        def fetch(self, tickers, *, hours=48):
            return []

    assert not isinstance(NoName(), NewsSource)


# ── Lib version pin ───────────────────────────────────────────────────


def test_lib_version_bumped_to_0_15_0():
    """Pin that the sources-Protocols feature has shipped (was 0.15.0).
    Asserts >= so future version bumps don't break this gate while
    still catching accidental version regressions below 0.15.0."""
    from packaging.version import Version

    import nousergon_lib
    assert Version(nousergon_lib.__version__) >= Version("0.15.0")


# ── Re-exports from package init ──────────────────────────────────────


def test_all_shapes_and_protocols_reexported_at_package_root():
    """Consumers should import from ``nousergon_lib.sources``, not
    ``nousergon_lib.sources.protocols``."""
    from nousergon_lib.sources import (  # noqa: F401
        AnalystSnapshot,
        AnalystSource,
        FilingDocument,
        FilingSource,
        NewsArticle,
        NewsSource,
    )
