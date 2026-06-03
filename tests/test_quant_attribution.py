"""Tests for alpha_engine_lib/quant/attribution.py — Brinson-Fachler + Cariño linking."""

import math

import pytest

from alpha_engine_lib.quant.attribution import (
    BrinsonResult,
    brinson_fachler,
    link_periods,
)


class TestBrinsonFachler:
    def test_effects_sum_to_active_return(self):
        # Fundamental Brinson identity: allocation + selection + interaction =
        # R_p − R_b, when weights each sum to 1.
        wp = {"Tech": 0.6, "Energy": 0.4}
        rp = {"Tech": 0.12, "Energy": 0.03}
        wb = {"Tech": 0.5, "Energy": 0.5}
        rb = {"Tech": 0.10, "Energy": 0.05}
        res = brinson_fachler(wp, rp, wb, rb)
        assert res.total_effect == pytest.approx(res.active_return)
        assert res.portfolio_return == pytest.approx(0.6 * 0.12 + 0.4 * 0.03)
        assert res.benchmark_return == pytest.approx(0.5 * 0.10 + 0.5 * 0.05)

    def test_per_group_components(self):
        wp = {"Tech": 0.6, "Energy": 0.4}
        rp = {"Tech": 0.12, "Energy": 0.03}
        wb = {"Tech": 0.5, "Energy": 0.5}
        rb = {"Tech": 0.10, "Energy": 0.05}
        res = brinson_fachler(wp, rp, wb, rb)
        rb_total = 0.5 * 0.10 + 0.5 * 0.05
        tech = next(g for g in res.groups if g.group == "Tech")
        assert tech.allocation == pytest.approx((0.6 - 0.5) * (0.10 - rb_total))
        assert tech.selection == pytest.approx(0.5 * (0.12 - 0.10))
        assert tech.interaction == pytest.approx((0.6 - 0.5) * (0.12 - 0.10))
        assert tech.total == pytest.approx(tech.allocation + tech.selection + tech.interaction)

    def test_pure_allocation_no_selection(self):
        # Portfolio holds the benchmark's exact group returns but tilts weights →
        # selection and interaction are zero; all active return is allocation.
        rb = {"A": 0.10, "B": 0.02}
        res = brinson_fachler({"A": 0.7, "B": 0.3}, dict(rb), {"A": 0.5, "B": 0.5}, dict(rb))
        assert res.selection == pytest.approx(0.0)
        assert res.interaction == pytest.approx(0.0)
        assert res.allocation == pytest.approx(res.active_return)

    def test_out_of_benchmark_group_defaults_neutral(self):
        # A group the benchmark doesn't hold: wb=0 → selection 0; rb defaults to
        # the overall benchmark return so its allocation baseline is neutral.
        wp = {"Tech": 0.5, "Crypto": 0.5}
        rp = {"Tech": 0.10, "Crypto": 0.30}
        wb = {"Tech": 1.0}
        rb = {"Tech": 0.08}
        res = brinson_fachler(wp, rp, wb, rb)
        crypto = next(g for g in res.groups if g.group == "Crypto")
        assert crypto.selection == pytest.approx(0.0)  # wb_crypto = 0
        # Identity still holds.
        assert res.total_effect == pytest.approx(res.active_return)

    def test_no_active_bets_zero_attribution(self):
        # Identical weights and returns → zero active return, zero effects.
        w = {"A": 0.5, "B": 0.5}
        r = {"A": 0.1, "B": 0.2}
        res = brinson_fachler(dict(w), dict(r), dict(w), dict(r))
        assert res.active_return == pytest.approx(0.0)
        assert res.total_effect == pytest.approx(0.0)


class TestCarinoLinking:
    def _period(self, wp, rp, wb, rb):
        return brinson_fachler(wp, rp, wb, rb)

    def test_empty_is_zero_result(self):
        res = link_periods([])
        assert isinstance(res, BrinsonResult)
        assert res.active_return == pytest.approx(0.0)
        assert res.groups == []

    def test_single_period_unchanged(self):
        p = self._period({"A": 0.6, "B": 0.4}, {"A": 0.1, "B": 0.05}, {"A": 0.5, "B": 0.5}, {"A": 0.08, "B": 0.06})
        assert link_periods([p]) is p

    def test_linked_effects_sum_to_geometric_active_return(self):
        # The Cariño guarantee: linked total effect == cumulative geometric
        # active return, NOT the naive arithmetic sum of period active returns.
        p1 = self._period({"A": 0.6, "B": 0.4}, {"A": 0.10, "B": 0.05}, {"A": 0.5, "B": 0.5}, {"A": 0.08, "B": 0.06})
        p2 = self._period({"A": 0.6, "B": 0.4}, {"A": 0.04, "B": 0.07}, {"A": 0.5, "B": 0.5}, {"A": 0.05, "B": 0.05})
        linked = link_periods([p1, p2])
        cum_p = (1 + p1.portfolio_return) * (1 + p2.portfolio_return) - 1
        cum_b = (1 + p1.benchmark_return) * (1 + p2.benchmark_return) - 1
        assert linked.portfolio_return == pytest.approx(cum_p)
        assert linked.benchmark_return == pytest.approx(cum_b)
        assert linked.total_effect == pytest.approx(cum_p - cum_b)
        # Per-group totals still reconcile to the overall total.
        assert sum(g.total for g in linked.groups) == pytest.approx(linked.total_effect)

    def test_linking_differs_from_naive_sum(self):
        p1 = self._period({"A": 1.0}, {"A": 0.20}, {"A": 1.0}, {"A": 0.10})
        p2 = self._period({"A": 1.0}, {"A": 0.20}, {"A": 1.0}, {"A": 0.10})
        linked = link_periods([p1, p2])
        naive = p1.total_effect + p2.total_effect  # 0.10 + 0.10 = 0.20
        geometric = (1.2 * 1.2 - 1) - (1.1 * 1.1 - 1)  # 0.44 − 0.21 = 0.23
        assert linked.total_effect == pytest.approx(geometric)
        assert linked.total_effect != pytest.approx(naive)

    def test_equal_returns_period_uses_limit_no_crash(self):
        # A period where r_p == r_b exercises the L'Hôpital limit branch.
        p1 = self._period({"A": 0.5, "B": 0.5}, {"A": 0.05, "B": 0.05}, {"A": 0.5, "B": 0.5}, {"A": 0.05, "B": 0.05})
        p2 = self._period({"A": 0.6, "B": 0.4}, {"A": 0.10, "B": 0.02}, {"A": 0.5, "B": 0.5}, {"A": 0.08, "B": 0.03})
        linked = link_periods([p1, p2])
        cum_p = (1 + p1.portfolio_return) * (1 + p2.portfolio_return) - 1
        cum_b = (1 + p1.benchmark_return) * (1 + p2.benchmark_return) - 1
        assert linked.total_effect == pytest.approx(cum_p - cum_b)

    def test_total_loss_period_raises(self):
        bad = BrinsonResult(portfolio_return=-1.0, benchmark_return=0.0)
        ok = BrinsonResult(portfolio_return=0.05, benchmark_return=0.03)
        with pytest.raises(ValueError, match="100%"):
            link_periods([bad, ok])

    def test_carino_coefficient_matches_closed_form(self):
        # Spot-check the linked total against a hand-computed Cariño chain.
        p1 = self._period({"A": 1.0}, {"A": 0.10}, {"A": 1.0}, {"A": 0.05})
        p2 = self._period({"A": 1.0}, {"A": -0.05}, {"A": 1.0}, {"A": 0.00})
        linked = link_periods([p1, p2])
        cum_p = 1.10 * 0.95 - 1
        cum_b = 1.05 * 1.00 - 1
        assert linked.total_effect == pytest.approx(cum_p - cum_b)
        # Sanity: the linking coefficient identity Σ ln-ratios holds.
        expected_log = math.log(1 + cum_p) - math.log(1 + cum_b)
        actual_log = (math.log(1.10) - math.log(1.05)) + (math.log(0.95) - math.log(1.00))
        assert actual_log == pytest.approx(expected_log)
