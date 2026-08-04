"""Tests for nousergon_lib.decision_set — the shared decision-set resolver
(alpha-engine-config#5809).

Mirrors the proofs that lived in ``nousergon-data/rag/pipelines/
test_rag_scope.py`` (this module's origin, config-I5700), generalized to
cover the multi-cut parameterisation this lift adds. The properties under
test are exactly the ones the issue names as non-negotiable: the O(1)
``latest.json`` pointer read, the equity-ticker regex dropping non-equity
identifiers (e.g. Metron's Treasury CUSIPs), the held-position union, and
hard-fail-no-fallback (never widens to ``signals.json::universe``).
"""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

from nousergon_lib.decision_set import (
    CUT_ATTRACTIVENESS_TOP_20,
    CUT_SCANNER_CANDIDATES,
    DecisionSetUnavailable,
    load_decision_set,
    load_decision_set_tickers,
)

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-07-30"

_MEMBERSHIP = {
    "schema_version": 1,
    "producer": "universe_membership",
    "run_date": RUN_DATE,
    "cuts": {
        "scanner_candidates": {
            "basis": "scanner_gate",
            "size": 3,
            "tickers": ["AAPL", "msft", "GOOGL"],
            "source": f"candidates/{RUN_DATE}/candidates.json::scanner_tickers",
        },
        "attractiveness_top_20": {
            "basis": "attractiveness_rank",
            "size": 2,
            "tickers": ["AAPL", "NVDA"],
            "source": f"scanner/universe/{RUN_DATE}/universe.json::attractiveness_score",
        },
        "empty_cut": {
            "basis": "scanner_gate",
            "size": 0,
            "tickers": [],
            "source": "n/a",
        },
    },
}


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _put_membership(s3, membership=None, run_date=RUN_DATE, dated=True, latest=True):
    body = json.dumps(membership if membership is not None else _MEMBERSHIP).encode("utf-8")
    if dated:
        s3.put_object(Bucket=BUCKET, Key=f"universe_membership/{run_date}/membership.json", Body=body)
    if latest:
        s3.put_object(Bucket=BUCKET, Key="universe_membership/latest.json", Body=body)


def _put_holdings(s3, tickers, as_of=RUN_DATE):
    body = json.dumps({"as_of": as_of, "tickers": tickers}).encode("utf-8")
    s3.put_object(Bucket=BUCKET, Key="metron/holdings_universe.json", Body=body)


class TestCutResolution:
    """Named-cut resolution: the caller asks for a cut by name, gets exactly
    that cut's tickers (uppercased, deduped, sorted) plus provenance."""

    def test_resolves_named_cut_scanner_candidates(self, s3):
        _put_membership(s3)
        result = load_decision_set(cut=CUT_SCANNER_CANDIDATES, bucket=BUCKET, s3_client=s3, include_held=False)
        assert result["tickers"] == ["AAPL", "GOOGL", "MSFT"]
        assert result["cut"] == "scanner_candidates"
        assert result["run_date"] == RUN_DATE
        assert result["source"].endswith("scanner_tickers")
        assert result["counts"]["scanner_candidates"] == 3

    def test_resolves_a_different_named_cut(self, s3):
        # Same artifact, different cut name — proves this is parameterised,
        # not hardcoded to one scope the way _rag_scope.py was.
        _put_membership(s3)
        result = load_decision_set(cut=CUT_ATTRACTIVENESS_TOP_20, bucket=BUCKET, s3_client=s3, include_held=False)
        assert result["tickers"] == ["AAPL", "NVDA"]
        assert result["cut"] == "attractiveness_top_20"

    def test_arbitrary_cut_name_also_resolves(self, s3):
        # Not limited to the two convenience constants — any cut the
        # producer emits is reachable by name.
        _put_membership(s3)
        result = load_decision_set(cut="scanner_candidates", bucket=BUCKET, s3_client=s3, include_held=False)
        assert result["tickers"] == ["AAPL", "GOOGL", "MSFT"]

    def test_run_date_pins_the_dated_artifact(self, s3):
        # Dated key and latest.json diverge — run_date=None uses latest,
        # run_date=RUN_DATE uses the dated key. Prevents a caller pinned to
        # a specific pipeline run from silently reading a different date.
        older = {**_MEMBERSHIP, "run_date": "2026-07-23", "cuts": {
            "scanner_candidates": {"basis": "scanner_gate", "size": 1, "tickers": ["OLD"], "source": "x"},
        }}
        _put_membership(s3, membership=older, run_date="2026-07-23", latest=False)
        _put_membership(s3, run_date=RUN_DATE, dated=True, latest=True)
        pinned = load_decision_set(cut=CUT_SCANNER_CANDIDATES, bucket=BUCKET, s3_client=s3, run_date="2026-07-23", include_held=False)
        assert pinned["tickers"] == ["OLD"]
        latest = load_decision_set(cut=CUT_SCANNER_CANDIDATES, bucket=BUCKET, s3_client=s3, include_held=False)
        assert latest["tickers"] == ["AAPL", "GOOGL", "MSFT"]

    def test_uses_o1_latest_pointer_not_a_bucket_listing(self, s3):
        # The pointer key is read directly — no list_objects_v2 call. Proven
        # by only ever writing the latest.json key (no dated partitions
        # exist at all) and still resolving successfully.
        _put_membership(s3, dated=False, latest=True)
        result = load_decision_set(cut=CUT_SCANNER_CANDIDATES, bucket=BUCKET, s3_client=s3, include_held=False)
        assert result["tickers"] == ["AAPL", "GOOGL", "MSFT"]

    def test_cut_name_is_required(self, s3):
        with pytest.raises(ValueError):
            load_decision_set(cut="", bucket=BUCKET, s3_client=s3)

    def test_tickers_convenience_wrapper_matches(self, s3):
        _put_membership(s3)
        tickers = load_decision_set_tickers(cut=CUT_SCANNER_CANDIDATES, bucket=BUCKET, s3_client=s3, include_held=False)
        assert tickers == ["AAPL", "GOOGL", "MSFT"]


class TestTickerRegex:
    """The equity-ticker regex drops non-equity identifiers — e.g. Metron's
    Treasury CUSIPs — so they never reach an equity-only ingestion source as
    a guaranteed wasted request."""

    def test_treasury_cusip_in_holdings_is_dropped(self, s3):
        _put_membership(s3)
        _put_holdings(s3, ["912828YK0", "TLT"])  # CUSIP + a real equity/ETF ticker
        result = load_decision_set(cut=CUT_SCANNER_CANDIDATES, bucket=BUCKET, s3_client=s3)
        assert "912828YK0" not in result["tickers"]
        assert "TLT" in result["tickers"]
        assert result["counts"]["rejected_non_equity"] == 1

    def test_share_class_suffix_tickers_are_kept(self, s3):
        membership = {**_MEMBERSHIP, "cuts": {
            "scanner_candidates": {
                "basis": "scanner_gate", "size": 2,
                "tickers": ["BRK.B", "BF-B"], "source": "x",
            },
        }}
        _put_membership(s3, membership=membership)
        result = load_decision_set(cut=CUT_SCANNER_CANDIDATES, bucket=BUCKET, s3_client=s3, include_held=False)
        assert result["tickers"] == ["BF-B", "BRK.B"]
        assert result["counts"]["rejected_non_equity"] == 0

    def test_lowercase_and_whitespace_are_normalized_before_matching(self, s3):
        membership = {**_MEMBERSHIP, "cuts": {
            "scanner_candidates": {
                "basis": "scanner_gate", "size": 1,
                "tickers": [" msft "], "source": "x",
            },
        }}
        _put_membership(s3, membership=membership)
        result = load_decision_set(cut=CUT_SCANNER_CANDIDATES, bucket=BUCKET, s3_client=s3, include_held=False)
        assert result["tickers"] == ["MSFT"]


class TestHeldPositionUnion:
    """A held position needs evidence whether or not it ranks this cycle."""

    def test_held_ticker_outside_the_cut_is_unioned_in(self, s3):
        _put_membership(s3)
        _put_holdings(s3, ["IBM"])  # not in scanner_candidates
        result = load_decision_set(cut=CUT_SCANNER_CANDIDATES, bucket=BUCKET, s3_client=s3)
        assert "IBM" in result["tickers"]
        assert result["counts"]["held"] == 1
        assert result["counts"]["scanner_candidates"] == 3  # cut count unaffected

    def test_include_held_false_scopes_strictly_to_the_cut(self, s3):
        _put_membership(s3)
        _put_holdings(s3, ["IBM"])
        result = load_decision_set(cut=CUT_SCANNER_CANDIDATES, bucket=BUCKET, s3_client=s3, include_held=False)
        assert "IBM" not in result["tickers"]
        assert result["counts"]["held"] == 0

    def test_missing_holdings_artifact_is_non_fatal(self, s3):
        # No holdings object written at all — resolution still succeeds,
        # scoped to the cut only. Distinguishes the holdings union (fail-soft)
        # from the membership cut (fail-loud, see TestUnavailabilityHardFails).
        _put_membership(s3)
        result = load_decision_set(cut=CUT_SCANNER_CANDIDATES, bucket=BUCKET, s3_client=s3)
        assert result["tickers"] == ["AAPL", "GOOGL", "MSFT"]
        assert result["counts"]["held"] == 0

    def test_held_and_cut_overlap_is_deduped(self, s3):
        _put_membership(s3)
        _put_holdings(s3, ["AAPL"])  # already in scanner_candidates
        result = load_decision_set(cut=CUT_SCANNER_CANDIDATES, bucket=BUCKET, s3_client=s3)
        assert result["tickers"].count("AAPL") == 1


class TestUnavailabilityHardFails:
    """Unavailability RAISES — never widens to signals.json::universe. This
    is the property the whole module exists to enforce (config#5809): a
    caller that quietly falls back to the board-wide sizing envelope is
    exactly the defect this resolver removes."""

    def test_missing_membership_artifact_raises(self, s3):
        # Nothing written at all.
        with pytest.raises(DecisionSetUnavailable, match="missing or unparseable"):
            load_decision_set(cut=CUT_SCANNER_CANDIDATES, bucket=BUCKET, s3_client=s3)

    def test_unparseable_membership_artifact_raises(self, s3):
        s3.put_object(Bucket=BUCKET, Key="universe_membership/latest.json", Body=b"{not json")
        with pytest.raises(DecisionSetUnavailable, match="missing or unparseable"):
            load_decision_set(cut=CUT_SCANNER_CANDIDATES, bucket=BUCKET, s3_client=s3)

    def test_absent_cut_name_raises(self, s3):
        _put_membership(s3)
        with pytest.raises(DecisionSetUnavailable, match="no non-empty cuts"):
            load_decision_set(cut="does_not_exist", bucket=BUCKET, s3_client=s3)

    def test_empty_cut_raises(self, s3):
        _put_membership(s3)
        with pytest.raises(DecisionSetUnavailable, match="no non-empty cuts"):
            load_decision_set(cut="empty_cut", bucket=BUCKET, s3_client=s3)

    def test_error_message_names_no_fallback_to_signals_universe(self, s3):
        # The message itself documents the refusal — a future maintainer
        # reading a stack trace must not be tempted to "fix" this by adding
        # a fallback. Assert the specific language stays in place.
        with pytest.raises(DecisionSetUnavailable, match="signals.json::universe"):
            load_decision_set(cut=CUT_SCANNER_CANDIDATES, bucket=BUCKET, s3_client=s3)

    def test_never_reads_signals_json_at_all(self, s3):
        # Plant a signals.json with a large universe in the same bucket —
        # if any fallback path existed it would resolve from this. It must
        # not: resolution still raises.
        s3.put_object(
            Bucket=BUCKET,
            Key=f"signals/{RUN_DATE}/signals.json",
            Body=json.dumps({"universe": [{"ticker": "ZZZZ"}] * 900}).encode("utf-8"),
        )
        with pytest.raises(DecisionSetUnavailable):
            load_decision_set(cut=CUT_SCANNER_CANDIDATES, bucket=BUCKET, s3_client=s3, run_date=RUN_DATE)
