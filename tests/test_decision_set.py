"""Tests for ``nousergon_lib.decision_set`` — the one definition of the
decision set and the funnel invariant that alpha-engine-config-I6630 found
violated live.
"""

from __future__ import annotations

import pytest

from nousergon_lib.decision_set import (
    ATTRACTIVENESS_FEED_TOP_N,
    FEED_CUT_NAME,
    PREDICTOR_CUT_TOP_N,
    DecisionSetContractError,
    assert_cut_nests,
    attractiveness_cut_name,
    cut_tickers,
    predictor_cut_name,
)


def _membership(**cuts) -> dict:
    """Membership artifact shaped like the real one, with only the fields
    these functions read."""
    return {
        "schema_version": 1,
        "run_date": "2026-08-07",
        "predictor_universe_cut": "attractiveness_top_20",
        "cuts": {
            name: {"basis": "attractiveness_rank", "size": len(t), "tickers": t}
            for name, t in cuts.items()
        },
    }


# ── constants ────────────────────────────────────────────────────────


def test_feed_cut_name_is_derived_from_the_width_not_a_second_literal():
    assert FEED_CUT_NAME == f"attractiveness_top_{ATTRACTIVENESS_FEED_TOP_N}"
    assert FEED_CUT_NAME == "attractiveness_top_60"


def test_predictor_cut_is_narrower_than_the_feed_set():
    # The funnel only means something if the head is strictly smaller.
    assert PREDICTOR_CUT_TOP_N < ATTRACTIVENESS_FEED_TOP_N


def test_attractiveness_cut_name_matches_the_producer_naming():
    assert attractiveness_cut_name(20) == "attractiveness_top_20"
    assert attractiveness_cut_name(60) == FEED_CUT_NAME


# ── predictor_cut_name ───────────────────────────────────────────────


def test_predictor_cut_name_reads_the_producer_published_field():
    m = _membership(attractiveness_top_20=["AAPL"])
    assert predictor_cut_name(m) == "attractiveness_top_20"


@pytest.mark.parametrize("value", [None, "", 60, []])
def test_predictor_cut_name_raises_rather_than_defaulting(value):
    m = _membership(attractiveness_top_20=["AAPL"])
    m["predictor_universe_cut"] = value
    with pytest.raises(DecisionSetContractError, match="predictor_universe_cut"):
        predictor_cut_name(m)


def test_predictor_cut_name_raises_when_the_field_is_absent():
    m = _membership(attractiveness_top_20=["AAPL"])
    del m["predictor_universe_cut"]
    with pytest.raises(DecisionSetContractError):
        predictor_cut_name(m)


# ── cut_tickers ──────────────────────────────────────────────────────


def test_cut_tickers_normalises_case_and_whitespace():
    m = _membership(attractiveness_top_20=[" aapl ", "Msft"])
    assert cut_tickers(m, "attractiveness_top_20") == ["AAPL", "MSFT"]


def test_cut_tickers_drops_empty_entries():
    m = _membership(attractiveness_top_20=["AAPL", "", "  "])
    assert cut_tickers(m, "attractiveness_top_20") == ["AAPL"]


def test_cut_tickers_raises_on_missing_cut_and_names_what_is_available():
    m = _membership(attractiveness_top_20=["AAPL"])
    with pytest.raises(DecisionSetContractError) as exc:
        cut_tickers(m, "scanner_candidates")
    assert "attractiveness_top_20" in str(exc.value)


def test_cut_tickers_raises_on_empty_cut_rather_than_returning_empty():
    # Returning [] here is how a corpus fill silently covers nothing.
    m = _membership(attractiveness_top_20=[])
    with pytest.raises(DecisionSetContractError):
        cut_tickers(m, "attractiveness_top_20")


def test_cut_tickers_raises_when_the_artifact_has_no_cuts_key():
    with pytest.raises(DecisionSetContractError):
        cut_tickers({"run_date": "2026-08-07"}, FEED_CUT_NAME)


# ── assert_cut_nests — the I6630 invariant ───────────────────────────


def test_nesting_holds_for_two_cuts_of_one_ranking():
    ranked = [f"T{i:03d}" for i in range(60)]
    m = _membership(
        attractiveness_top_20=ranked[:20],
        attractiveness_top_60=ranked,
    )
    assert_cut_nests(m, inner="attractiveness_top_20", outer=FEED_CUT_NAME)


def test_nesting_holds_when_the_two_cuts_are_identical():
    m = _membership(
        attractiveness_top_20=["AAPL", "MSFT"],
        attractiveness_top_60=["AAPL", "MSFT"],
    )
    assert_cut_nests(m, inner="attractiveness_top_20", outer=FEED_CUT_NAME)


def test_nesting_is_order_and_case_insensitive():
    m = _membership(
        attractiveness_top_20=["msft", "AAPL"],
        attractiveness_top_60=["AAPL", "MSFT", "NVDA"],
    )
    assert_cut_nests(m, inner="attractiveness_top_20", outer=FEED_CUT_NAME)


def test_nesting_fails_on_the_live_i6630_shape_and_names_the_escapees():
    # The measured 2026-08-07 state: the scored cut and the evidence cut came
    # from different rankings and overlapped on 2 of 20.
    m = _membership(
        attractiveness_top_20=["ANF", "SN", "LULU", "ROKU"],
        scanner_candidates=["ANF", "SN", "MSFT", "PLTR"],
    )
    with pytest.raises(DecisionSetContractError) as exc:
        assert_cut_nests(m, inner="attractiveness_top_20", outer="scanner_candidates")
    message = str(exc.value)
    assert "LULU" in message and "ROKU" in message
    assert "2 of 4" in message
    assert "I6630" in message


def test_nesting_fails_on_a_single_escaping_ticker():
    m = _membership(
        attractiveness_top_20=["AAPL", "MSFT"],
        attractiveness_top_60=["AAPL"],
    )
    with pytest.raises(DecisionSetContractError, match="MSFT"):
        assert_cut_nests(m, inner="attractiveness_top_20", outer=FEED_CUT_NAME)


def test_nesting_propagates_the_missing_cut_error():
    m = _membership(attractiveness_top_20=["AAPL"])
    with pytest.raises(DecisionSetContractError, match="attractiveness_top_60"):
        assert_cut_nests(m, inner="attractiveness_top_20", outer=FEED_CUT_NAME)
