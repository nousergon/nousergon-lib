"""Tests for ``nousergon_lib.universe`` — canonical universe-filter
primitive shared by alpha-engine-research and alpha-engine.
"""

from __future__ import annotations

from nousergon_lib.universe import filter_to_universe, in_universe

SP500_FIXTURE = frozenset({"AAPL", "MSFT", "GOOGL", "JPM", "XOM"})


# ── filter_to_universe: str-list inputs ───────────────────────────────


def test_filter_str_list_partitions_kept_and_dropped():
    kept, dropped = filter_to_universe(["AAPL", "TSM", "MSFT", "ASML"], SP500_FIXTURE)
    assert kept == ["AAPL", "MSFT"]
    assert dropped == ["TSM", "ASML"]


def test_filter_str_list_empty_input():
    kept, dropped = filter_to_universe([], SP500_FIXTURE)
    assert kept == []
    assert dropped == []


def test_filter_str_list_all_kept():
    kept, dropped = filter_to_universe(["AAPL", "MSFT"], SP500_FIXTURE)
    assert kept == ["AAPL", "MSFT"]
    assert dropped == []


def test_filter_str_list_all_dropped():
    kept, dropped = filter_to_universe(["TSM", "ASML"], SP500_FIXTURE)
    assert kept == []
    assert dropped == ["TSM", "ASML"]


def test_filter_str_empty_string_goes_to_dropped():
    kept, dropped = filter_to_universe(["AAPL", ""], SP500_FIXTURE)
    assert kept == ["AAPL"]
    assert dropped == [""]


# ── filter_to_universe: dict-list inputs (default key) ────────────────


def test_filter_dict_list_uses_default_ticker_key():
    items = [
        {"ticker": "AAPL", "sector": "Technology"},
        {"ticker": "TSM", "sector": "Technology"},
        {"ticker": "MSFT", "sector": "Technology"},
    ]
    kept, dropped = filter_to_universe(items, SP500_FIXTURE)
    assert kept == [items[0], items[2]]
    assert dropped == [items[1]]


def test_filter_dict_list_missing_ticker_field_goes_to_dropped():
    items = [
        {"ticker": "AAPL"},
        {"sector": "Technology"},  # no ticker key
    ]
    kept, dropped = filter_to_universe(items, SP500_FIXTURE)
    assert kept == [items[0]]
    assert dropped == [items[1]]


def test_filter_dict_list_non_string_ticker_goes_to_dropped():
    items = [
        {"ticker": "AAPL"},
        {"ticker": None},
        {"ticker": 123},
    ]
    kept, dropped = filter_to_universe(items, SP500_FIXTURE)
    assert kept == [items[0]]
    assert dropped == [items[1], items[2]]


# ── filter_to_universe: custom key ────────────────────────────────────


def test_filter_with_custom_key_callable():
    items = [
        {"sym": "AAPL"},
        {"sym": "TSM"},
        {"sym": "MSFT"},
    ]
    kept, dropped = filter_to_universe(items, SP500_FIXTURE, key=lambda x: x["sym"])
    assert kept == [items[0], items[2]]
    assert dropped == [items[1]]


def test_filter_with_custom_key_returning_none():
    items = [object(), "AAPL"]
    kept, dropped = filter_to_universe(
        items, SP500_FIXTURE,
        key=lambda x: x if isinstance(x, str) else None,
    )
    assert kept == ["AAPL"]
    assert dropped == [items[0]]


# ── filter_to_universe: universe accepts set and frozenset ────────────


def test_filter_accepts_mutable_set_universe():
    universe = {"AAPL", "MSFT"}
    kept, dropped = filter_to_universe(["AAPL", "TSM"], universe)
    assert kept == ["AAPL"]
    assert dropped == ["TSM"]


def test_filter_accepts_frozenset_universe():
    universe = frozenset({"AAPL", "MSFT"})
    kept, dropped = filter_to_universe(["AAPL", "TSM"], universe)
    assert kept == ["AAPL"]
    assert dropped == ["TSM"]


def test_filter_empty_universe_drops_everything():
    kept, dropped = filter_to_universe(["AAPL", "MSFT"], frozenset())
    assert kept == []
    assert dropped == ["AAPL", "MSFT"]


# ── filter_to_universe: order preservation ────────────────────────────


def test_filter_preserves_input_order_in_both_partitions():
    items = ["MSFT", "TSM", "AAPL", "ASML", "GOOGL"]
    kept, dropped = filter_to_universe(items, SP500_FIXTURE)
    # input order preserved within each partition
    assert kept == ["MSFT", "AAPL", "GOOGL"]
    assert dropped == ["TSM", "ASML"]


# ── in_universe: predicate ────────────────────────────────────────────


def test_in_universe_true_for_member():
    assert in_universe("AAPL", SP500_FIXTURE) is True


def test_in_universe_false_for_non_member():
    assert in_universe("TSM", SP500_FIXTURE) is False


def test_in_universe_false_for_none():
    assert in_universe(None, SP500_FIXTURE) is False


def test_in_universe_false_for_empty_string():
    assert in_universe("", SP500_FIXTURE) is False


def test_in_universe_false_for_non_string():
    # type-hint says str | None, but defend against runtime nonsense
    assert in_universe(123, SP500_FIXTURE) is False  # type: ignore[arg-type]


# ── Contract regression: divergence-impossible-by-construction ────────


def test_research_layer1_and_executor_layer2_agree_by_construction():
    """Both consumers go through ``filter_to_universe``; given the same
    universe + the same input tickers, the kept/dropped partitions are
    byte-identical. This is the contract the TSM/ASML incident demanded.
    """
    universe = frozenset({"AAPL", "MSFT", "GOOGL"})
    incumbents = [
        {"ticker": "AAPL", "sector": "Technology"},
        {"ticker": "TSM", "sector": "Technology"},
    ]
    buy_candidates = [
        {"ticker": "MSFT", "sector": "Technology", "score": 72},
        {"ticker": "ASML", "sector": "Technology", "score": 68},
    ]
    kept_r, dropped_r = filter_to_universe(incumbents, universe)
    kept_e, dropped_e = filter_to_universe(buy_candidates, universe)

    # Each layer makes the same membership decision per ticker.
    assert {item["ticker"] for item in kept_r} == {"AAPL"}
    assert {item["ticker"] for item in dropped_r} == {"TSM"}
    assert {item["ticker"] for item in kept_e} == {"MSFT"}
    assert {item["ticker"] for item in dropped_e} == {"ASML"}
