"""Tests for nousergon_lib/quant/returns.py — XIRR, TWR, cumulative, annualize."""

from __future__ import annotations

from datetime import date

import pytest

from nousergon_lib.quant.returns import (
    CashFlow,
    ValuationPoint,
    annualize,
    cumulative_return,
    time_weighted_return,
    xirr,
)


class TestXirr:
    def test_simple_one_year_doubling(self):
        # Invest 100, get 200 exactly one year later → 100% IRR.
        flows = [CashFlow(date(2025, 1, 1), -100.0), CashFlow(date(2026, 1, 1), 200.0)]
        r = xirr(flows)
        assert r == pytest.approx(1.0, abs=1e-4)

    def test_flat_return(self):
        flows = [CashFlow(date(2025, 1, 1), -100.0), CashFlow(date(2026, 1, 1), 100.0)]
        assert xirr(flows) == pytest.approx(0.0, abs=1e-4)

    def test_known_excel_xirr_value(self):
        # Classic Excel XIRR example.
        flows = [
            CashFlow(date(2020, 1, 1), -10000.0),
            CashFlow(date(2020, 3, 1), 2000.0),
            CashFlow(date(2020, 10, 30), 4000.0),
            CashFlow(date(2021, 2, 15), 8000.0),
        ]
        r = xirr(flows)
        # Verified root (NPV≈0 at this rate): ≈ 0.4643 (46.43% annualized).
        assert r == pytest.approx(0.4643, abs=1e-3)

    def test_multiple_contributions(self):
        # Two contributions, single terminal value, mild gain.
        flows = [
            CashFlow(date(2025, 1, 1), -100.0),
            CashFlow(date(2025, 7, 1), -100.0),
            CashFlow(date(2026, 1, 1), 210.0),
        ]
        r = xirr(flows)
        assert r is not None
        assert 0.0 < r < 0.2  # a small positive money-weighted return

    def test_returns_none_without_sign_change(self):
        # All outflows → no IRR.
        flows = [CashFlow(date(2025, 1, 1), -100.0), CashFlow(date(2026, 1, 1), -50.0)]
        assert xirr(flows) is None

    def test_returns_none_too_few_flows(self):
        assert xirr([CashFlow(date(2025, 1, 1), -100.0)]) is None

    def test_loss(self):
        flows = [CashFlow(date(2025, 1, 1), -100.0), CashFlow(date(2026, 1, 1), 50.0)]
        r = xirr(flows)
        assert r == pytest.approx(-0.5, abs=1e-4)

    def test_bisection_fallback_when_newton_skipped(self):
        # max_iter=0 skips Newton entirely, forcing the bisection fallback path —
        # it must still find the same root (the robustness guarantee).
        flows = [CashFlow(date(2025, 1, 1), -100.0), CashFlow(date(2026, 1, 1), 200.0)]
        assert xirr(flows, max_iter=0) == pytest.approx(1.0, abs=1e-4)

    def test_bisection_none_when_root_outside_bracket(self):
        # A catastrophic loss whose IRR is below the -0.9999 bracket floor → None,
        # never a bogus number.
        flows = [CashFlow(date(2025, 1, 1), -100.0), CashFlow(date(2026, 1, 1), 0.0001)]
        assert xirr(flows, max_iter=0) is None


class TestTimeWeightedReturn:
    def test_no_flows_is_simple_return(self):
        pts = [
            ValuationPoint(date(2025, 1, 1), 100.0),
            ValuationPoint(date(2026, 1, 1), 130.0),
        ]
        assert time_weighted_return(pts) == pytest.approx(0.30, abs=1e-9)

    def test_neutralizes_contribution_timing(self):
        # Start 100 → grows to 110 (+10%), then 100 contributed (value 210),
        # then grows to 231 (+10%). TWR = 1.1 * 1.1 - 1 = 0.21, regardless of the
        # mid-period cash injection.
        pts = [
            ValuationPoint(date(2025, 1, 1), 100.0, flow=0.0),
            ValuationPoint(date(2025, 7, 1), 110.0, flow=100.0),
            ValuationPoint(date(2026, 1, 1), 231.0, flow=0.0),
        ]
        assert time_weighted_return(pts) == pytest.approx(0.21, abs=1e-9)

    def test_withdrawal_neutralized(self):
        # 200 → 220 (+10%), withdraw 100 (value 120), → 132 (+10%). TWR = 0.21.
        pts = [
            ValuationPoint(date(2025, 1, 1), 200.0, flow=0.0),
            ValuationPoint(date(2025, 7, 1), 220.0, flow=-100.0),
            ValuationPoint(date(2026, 1, 1), 132.0, flow=0.0),
        ]
        assert time_weighted_return(pts) == pytest.approx(0.21, abs=1e-9)

    def test_none_too_few_points(self):
        assert time_weighted_return([ValuationPoint(date(2025, 1, 1), 100.0)]) is None

    def test_none_on_nonpositive_capital(self):
        pts = [
            ValuationPoint(date(2025, 1, 1), 100.0, flow=-100.0),  # fully withdrawn
            ValuationPoint(date(2026, 1, 1), 50.0),
        ]
        assert time_weighted_return(pts) is None

    def test_twr_differs_from_mwr_on_bad_timing(self):
        # Bad timing: big contribution right before a drop. TWR (strategy) should
        # read better than MWR (investor) because TWR ignores the unlucky timing.
        pts = [
            ValuationPoint(date(2025, 1, 1), 100.0, flow=0.0),
            ValuationPoint(date(2025, 12, 1), 110.0, flow=900.0),  # +10% then add 900
            ValuationPoint(date(2026, 1, 1), 900.0, flow=0.0),  # 1010 → 900, ~-10.9%
        ]
        twr = time_weighted_return(pts)
        flows = [
            CashFlow(date(2025, 1, 1), -100.0),
            CashFlow(date(2025, 12, 1), -900.0),
            CashFlow(date(2026, 1, 1), 900.0),
        ]
        mwr = xirr(flows)
        assert twr is not None and mwr is not None
        assert twr > mwr  # strategy looks better than the investor's actual experience


class TestCumulativeAndAnnualize:
    def test_cumulative_simple(self):
        assert cumulative_return(100.0, 130.0) == pytest.approx(0.30)

    def test_cumulative_adjusts_for_contributions(self):
        # Ended at 230 but 100 of that was a contribution → real return on the
        # original 100 is (230-100)/100 - 1 = 0.30.
        assert cumulative_return(100.0, 230.0, net_contributions=100.0) == pytest.approx(0.30)

    def test_cumulative_none_on_nonpositive_begin(self):
        assert cumulative_return(0.0, 100.0) is None

    def test_annualize_one_year_is_identity(self):
        assert annualize(0.20, 365) == pytest.approx(0.20, abs=1e-9)

    def test_annualize_half_year_compounds_up(self):
        # +20% in half a year annualizes to (1.2)^2 - 1 = 0.44.
        assert annualize(0.20, 182.5) == pytest.approx(0.44, abs=1e-9)

    def test_annualize_none_on_bad_inputs(self):
        assert annualize(0.2, 0) is None
        assert annualize(-1.5, 365) is None  # worse than -100% can't annualize
