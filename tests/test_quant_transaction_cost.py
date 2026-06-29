"""Tests for nousergon_lib/quant/transaction_cost.py — √-impact cost model +
tradeability percentile."""

from __future__ import annotations

import math

import pytest

from nousergon_lib.quant.transaction_cost import (
    TransactionCostModel,
    tradeability_percentiles,
)


class TestPerSideBps:
    def test_floor_when_no_adv(self):
        m = TransactionCostModel()
        # ADV missing → impact term drops to 0 → half_spread + commission.
        assert m.per_side_bps(1_000_000, None) == pytest.approx(2.5 + 0.5)
        assert m.per_side_bps(1_000_000, 0) == pytest.approx(3.0)

    def test_sqrt_impact(self):
        m = TransactionCostModel()
        # participation = 0.01 → impact = 10 * sqrt(0.01) = 1.0 bps.
        cost = m.per_side_bps(100_000, 10_000_000)
        assert cost == pytest.approx(2.5 + 1.0 + 0.5)

    def test_impact_scales_with_sqrt_participation(self):
        m = TransactionCostModel()
        # 4x notional → 2x impact (sqrt law).
        imp1 = m.per_side_bps(100_000, 10_000_000) - 3.0
        imp4 = m.per_side_bps(400_000, 10_000_000) - 3.0
        assert imp4 == pytest.approx(2 * imp1)

    def test_min_cost_floor(self):
        m = TransactionCostModel(half_spread_bps=0.0, commission_bps=0.0, min_cost_bps=1.5)
        assert m.per_side_bps(0, None) == pytest.approx(1.5)


class TestSigmaScaling:
    def test_sigma_agnostic_when_omitted(self):
        m = TransactionCostModel()
        assert m.per_side_bps(100_000, 10_000_000) == pytest.approx(
            m.per_side_bps(100_000, 10_000_000, sigma=None, ref_sigma=None)
        )

    def test_median_sigma_reproduces_agnostic(self):
        m = TransactionCostModel()
        base = m.per_side_bps(100_000, 10_000_000)
        # sigma == ref_sigma → scale 1.0 → identical to σ-agnostic.
        assert m.per_side_bps(100_000, 10_000_000, sigma=0.3, ref_sigma=0.3) == pytest.approx(base)

    def test_higher_vol_costs_more(self):
        m = TransactionCostModel()
        imp_lo = m.per_side_bps(100_000, 10_000_000, sigma=0.15, ref_sigma=0.30) - 3.0
        imp_md = m.per_side_bps(100_000, 10_000_000, sigma=0.30, ref_sigma=0.30) - 3.0
        imp_hi = m.per_side_bps(100_000, 10_000_000, sigma=0.60, ref_sigma=0.30) - 3.0
        assert imp_lo < imp_md < imp_hi
        assert imp_lo == pytest.approx(0.5 * imp_md)  # half the vol → half the impact
        assert imp_hi == pytest.approx(2.0 * imp_md)

    def test_nonpositive_or_nan_sigma_collapses_to_agnostic(self):
        m = TransactionCostModel()
        base = m.per_side_bps(100_000, 10_000_000)
        for sigma in (0.0, -0.1, float("nan")):
            assert m.per_side_bps(100_000, 10_000_000, sigma=sigma, ref_sigma=0.3) == pytest.approx(base)


class TestRoundTrip:
    def test_round_trip_is_two_sides(self):
        m = TransactionCostModel()
        ps = m.per_side_bps(100_000, 10_000_000, sigma=0.4, ref_sigma=0.3)
        rt = m.round_trip_bps(100_000, 10_000_000, sigma=0.4, ref_sigma=0.3)
        assert rt == pytest.approx(2 * ps)


class TestFromConfig:
    def test_defaults_when_absent(self):
        m = TransactionCostModel.from_config(None)
        assert (m.half_spread_bps, m.impact_coef_bps, m.commission_bps) == (2.5, 10.0, 0.5)

    def test_overrides(self):
        m = TransactionCostModel.from_config(
            {"transaction_cost": {"half_spread_bps": 1.0, "impact_coef_bps": 20.0}}
        )
        assert m.half_spread_bps == 1.0 and m.impact_coef_bps == 20.0
        assert m.commission_bps == 0.5  # untouched key falls back


class TestCostForTurnover:
    def test_zero_turnover_zero_cost(self):
        assert TransactionCostModel().cost_for_turnover(0, 10_000_000) == 0.0

    def test_dollar_cost(self):
        m = TransactionCostModel()
        # per_side at 100k/10M = 4.0 bps → $ cost = 100000 * 4.0/1e4 = $40.
        assert m.cost_for_turnover(100_000, 10_000_000) == pytest.approx(40.0)


class TestTradeabilityPercentiles:
    def test_cheaper_ranks_higher(self):
        scores = tradeability_percentiles({"CHEAP": 5.0, "MID": 10.0, "PRICEY": 50.0})
        assert scores["CHEAP"] > scores["MID"] > scores["PRICEY"]
        assert scores["CHEAP"] == pytest.approx(100.0)  # cheapest → top percentile

    def test_none_cost_yields_none_score_and_excluded(self):
        scores = tradeability_percentiles({"A": 5.0, "B": 10.0, "GAP": None})
        assert scores["GAP"] is None
        # Ranked population is {A,B} only → B is the most expensive of the two.
        assert scores["A"] == pytest.approx(100.0)
        assert scores["B"] == pytest.approx(50.0)

    def test_ties_share_mean_rank(self):
        scores = tradeability_percentiles({"A": 10.0, "B": 10.0})
        assert scores["A"] == scores["B"] == pytest.approx(75.0)

    def test_empty_and_all_none(self):
        assert tradeability_percentiles({}) == {}
        assert tradeability_percentiles({"X": None}) == {"X": None}
