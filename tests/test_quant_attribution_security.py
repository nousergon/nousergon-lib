"""Tests for security-level (name-level) attribution in
nousergon_lib/quant/attribution.py — active_weights, security_contributions,
reconcile, top_drivers.
"""

import pytest

from nousergon_lib.quant.attribution import (
    active_weights,
    reconcile,
    security_contributions,
    top_drivers,
)


class TestActiveWeights:
    def test_union_of_symbols(self):
        wp = {"AAPL": 0.10, "MSFT": 0.05}
        wb = {"AAPL": 0.08, "GOOG": 0.03}
        aw = active_weights(wp, wb)
        assert aw == {
            "AAPL": pytest.approx(0.02),
            "MSFT": pytest.approx(0.05),  # absent from bench -> treated as 0.0
            "GOOG": pytest.approx(-0.03),  # absent from port -> treated as 0.0
        }

    def test_empty_inputs(self):
        assert active_weights({}, {}) == {}

    def test_identical_weights_zero_active(self):
        w = {"AAPL": 0.10, "MSFT": 0.20}
        assert active_weights(dict(w), dict(w)) == {"AAPL": pytest.approx(0.0), "MSFT": pytest.approx(0.0)}


class TestSecurityContributions:
    def test_hand_computed_arithmetic(self):
        wp = {"AAPL": 0.10, "MSFT": 0.05}
        wb = {"AAPL": 0.08, "MSFT": 0.05}
        r = {"AAPL": 0.20, "MSFT": -0.10}
        res = security_contributions(wp, wb, r)
        assert res.missing == []
        aapl = next(c for c in res.contributions if c.symbol == "AAPL")
        assert aapl.port_contribution == pytest.approx(0.10 * 0.20)
        assert aapl.bench_contribution == pytest.approx(0.08 * 0.20)
        assert aapl.active_contribution == pytest.approx((0.10 - 0.08) * 0.20)
        assert aapl.active_weight == pytest.approx(0.02)
        msft = next(c for c in res.contributions if c.symbol == "MSFT")
        assert msft.active_contribution == pytest.approx(0.0)

    def test_classification_not_held(self):
        wp = {}
        wb = {"AAPL": 0.08}
        r = {"AAPL": 0.20}
        res = security_contributions(wp, wb, r)
        aapl = res.contributions[0]
        assert aapl.port_weight == pytest.approx(0.0)
        assert aapl.classification == "not_held"

    def test_classification_overweight(self):
        wp = {"AAPL": 0.10}
        wb = {"AAPL": 0.05}
        r = {"AAPL": 0.20}
        res = security_contributions(wp, wb, r)
        assert res.contributions[0].classification == "overweight"

    def test_classification_underweight(self):
        wp = {"AAPL": 0.03}
        wb = {"AAPL": 0.05}
        r = {"AAPL": 0.20}
        res = security_contributions(wp, wb, r)
        assert res.contributions[0].classification == "underweight"

    def test_classification_held(self):
        wp = {"AAPL": 0.05}
        wb = {"AAPL": 0.05}
        r = {"AAPL": 0.20}
        res = security_contributions(wp, wb, r)
        assert res.contributions[0].classification == "held"

    def test_missing_return_is_explicit_not_zero(self):
        # A symbol present in weights but absent from returns must NOT be
        # silently included with a zero contribution.
        wp = {"AAPL": 0.10, "TSLA": 0.05}
        wb = {"AAPL": 0.08}
        r = {"AAPL": 0.20}  # TSLA has no return
        res = security_contributions(wp, wb, r)
        assert res.missing == ["TSLA"]
        symbols_in_contributions = {c.symbol for c in res.contributions}
        assert "TSLA" not in symbols_in_contributions
        assert "AAPL" in symbols_in_contributions

    def test_missing_symbol_only_in_benchmark(self):
        wp = {"AAPL": 0.10}
        wb = {"AAPL": 0.08, "GOOG": 0.03}
        r = {"AAPL": 0.20}  # GOOG has no return
        res = security_contributions(wp, wb, r)
        assert res.missing == ["GOOG"]

    def test_empty_inputs(self):
        res = security_contributions({}, {}, {})
        assert res.contributions == []
        assert res.missing == []


class TestReconcile:
    def test_residual_correct_when_contributions_are_exhaustive(self):
        wp = {"AAPL": 0.6, "MSFT": 0.4}
        wb = {"AAPL": 0.5, "MSFT": 0.5}
        r = {"AAPL": 0.10, "MSFT": 0.05}
        res = security_contributions(wp, wb, r)
        portfolio_return = sum(wp[s] * r[s] for s in wp)
        benchmark_return = sum(wb[s] * r[s] for s in wb)
        rec = reconcile(res.contributions, portfolio_return, benchmark_return)
        assert rec.residual == pytest.approx(0.0)
        assert rec.within_tolerance is True

    def test_residual_nonzero_when_contributions_missing_a_symbol(self):
        # portfolio_return/benchmark_return computed over a symbol that
        # security_contributions was never given -> residual should surface
        # the gap rather than reconciling falsely.
        wp = {"AAPL": 0.5, "MSFT": 0.5}
        wb = {"AAPL": 0.5, "MSFT": 0.5}
        r = {"AAPL": 0.10, "MSFT": 0.05}
        res = security_contributions(wp, wb, r)
        # Pretend the true portfolio/benchmark return includes a third,
        # unaccounted-for security's contribution.
        portfolio_return = sum(wp[s] * r[s] for s in wp) + 0.02
        benchmark_return = sum(wb[s] * r[s] for s in wb)
        rec = reconcile(res.contributions, portfolio_return, benchmark_return)
        assert rec.residual == pytest.approx(0.02)
        assert rec.within_tolerance is False

    def test_tolerance_boundary_inclusive(self):
        wp = {"AAPL": 0.5}
        wb = {"AAPL": 0.5}
        r = {"AAPL": 0.10}
        res = security_contributions(wp, wb, r)
        portfolio_return = 0.5 * 0.10 + 1e-6
        benchmark_return = 0.5 * 0.10
        # Derive the exact residual first (float arithmetic makes a literal
        # 1e-6 tolerance an unreliable boundary), then use it as the
        # boundary itself: tolerance == |residual| must be inclusive.
        active_sum = sum(c.active_contribution for c in res.contributions)
        residual = (portfolio_return - benchmark_return) - active_sum

        at_boundary = reconcile(res.contributions, portfolio_return, benchmark_return, tolerance=abs(residual))
        assert at_boundary.residual == pytest.approx(residual)
        assert at_boundary.within_tolerance is True

        below_boundary = reconcile(
            res.contributions, portfolio_return, benchmark_return, tolerance=abs(residual) - 1e-9
        )
        assert below_boundary.within_tolerance is False

    def test_empty_contributions(self):
        rec = reconcile([], 0.05, 0.03)
        assert rec.residual == pytest.approx(0.02)
        assert rec.within_tolerance is False


class TestTopDrivers:
    def test_n_largest_by_absolute_active_contribution(self):
        wp = {"AAPL": 0.10, "MSFT": 0.05, "TSLA": 0.02}
        wb = {"AAPL": 0.05, "MSFT": 0.05, "TSLA": 0.10}
        r = {"AAPL": 0.20, "MSFT": 0.10, "TSLA": -0.30}
        res = security_contributions(wp, wb, r)
        top = top_drivers(res.contributions, 2)
        assert [c.symbol for c in top] == ["TSLA", "AAPL"]

    def test_tie_break_by_symbol_deterministic(self):
        wp = {"BBB": 0.10, "AAA": 0.05}
        wb = {"BBB": 0.05, "AAA": 0.0}
        r = {"BBB": 0.10, "AAA": 0.10}
        res = security_contributions(wp, wb, r)
        # BBB active_contribution = 0.05*0.10 = 0.005; AAA active_contribution = 0.05*0.10 = 0.005 -> tie
        top = top_drivers(res.contributions, 2)
        assert [c.symbol for c in top] == ["AAA", "BBB"]

    def test_tie_break_stable_regardless_of_input_order(self):
        wp = {"ZZZ": 0.05, "AAA": 0.05}
        wb = {"ZZZ": 0.0, "AAA": 0.0}
        r = {"ZZZ": 0.10, "AAA": 0.10}
        res = security_contributions(wp, wb, r)
        top_forward = top_drivers(res.contributions, 2)
        top_reversed = top_drivers(list(reversed(res.contributions)), 2)
        assert [c.symbol for c in top_forward] == [c.symbol for c in top_reversed] == ["AAA", "ZZZ"]

    def test_n_zero_returns_empty(self):
        wp = {"AAPL": 0.10}
        wb = {"AAPL": 0.05}
        r = {"AAPL": 0.10}
        res = security_contributions(wp, wb, r)
        assert top_drivers(res.contributions, 0) == []

    def test_n_larger_than_available_returns_all(self):
        wp = {"AAPL": 0.10}
        wb = {"AAPL": 0.05}
        r = {"AAPL": 0.10}
        res = security_contributions(wp, wb, r)
        assert len(top_drivers(res.contributions, 5)) == 1

    def test_empty_contributions(self):
        assert top_drivers([], 3) == []


class TestUnitsConsistencyWithBrinsonFachler:
    def test_same_fraction_units_as_brinson_fachler(self):
        # brinson_fachler treats weights as fractions of total and returns as
        # return-fractions (0.10 == +10%); security_contributions must use
        # the identical convention so a caller can mix group- and
        # security-level results without a unit-conversion step.
        from nousergon_lib.quant.attribution import brinson_fachler

        wp = {"Tech": 0.6, "Energy": 0.4}
        rp = {"Tech": 0.12, "Energy": 0.03}
        wb = {"Tech": 0.5, "Energy": 0.5}
        rb = {"Tech": 0.10, "Energy": 0.05}
        group_res = brinson_fachler(wp, rp, wb, rb)

        # A single-security "portfolio" mirroring one Brinson group: same
        # weight/return pair should produce the same w*r contribution math.
        sec_res = security_contributions({"Tech": 0.6}, {"Tech": 0.5}, {"Tech": 0.12})
        tech = sec_res.contributions[0]
        assert tech.port_contribution == pytest.approx(0.6 * 0.12)
        assert tech.port_contribution == pytest.approx(wp["Tech"] * rp["Tech"])
        assert group_res.portfolio_return == pytest.approx(0.6 * 0.12 + 0.4 * 0.03)
