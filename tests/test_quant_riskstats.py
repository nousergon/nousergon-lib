"""Tests for nousergon_lib/quant/riskstats.py — volatility, Sharpe, Sortino, drawdown."""

from __future__ import annotations

import math

import pytest

from nousergon_lib.quant.riskstats import max_drawdown, sharpe_ratio, sortino_ratio, volatility


class TestVolatility:
    def test_constant_series_zero_vol(self):
        assert volatility([0.01, 0.01, 0.01, 0.01]) == pytest.approx(0.0)

    def test_annualization(self):
        # Daily returns with sample stdev s annualize by √252.
        returns = [0.01, -0.01, 0.01, -0.01]
        v = volatility(returns)
        # sample stdev of [.01,-.01,.01,-.01] = 0.011547..., ×√252
        assert v == pytest.approx(0.0115470 * math.sqrt(252), rel=1e-4)

    def test_none_too_few(self):
        assert volatility([0.01]) is None


class TestSharpe:
    def test_positive_excess(self):
        # Positive-mean series with real (non-zero) variance.
        returns = [0.002, 0.0005] * 126  # alternating gains, positive mean
        s = sharpe_ratio(returns)
        assert s is not None and s > 0

    def test_zero_vol_returns_none(self):
        # Constant returns → zero stdev → undefined Sharpe.
        assert sharpe_ratio([0.001, 0.001, 0.001]) is None

    def test_risk_free_reduces_sharpe(self):
        returns = [0.002, 0.0005] * 126
        s0 = sharpe_ratio(returns, risk_free_rate=0.0)
        s_hi = sharpe_ratio(returns, risk_free_rate=0.10)
        assert s0 is not None and s_hi is not None
        assert s_hi < s0

    def test_none_too_few(self):
        assert sharpe_ratio([0.01]) is None


class TestSortino:
    def test_no_downside_returns_none(self):
        # All non-negative excess returns → zero downside deviation.
        assert sortino_ratio([0.01, 0.02, 0.0, 0.03]) is None

    def test_sortino_higher_than_sharpe_with_upside_vol(self):
        # A series whose volatility is mostly upside: Sortino should exceed Sharpe
        # because it ignores the (large, benign) upside swings.
        returns = [0.05, 0.06, -0.01, 0.04, 0.05, -0.005, 0.07]
        sh = sharpe_ratio(returns)
        so = sortino_ratio(returns)
        assert sh is not None and so is not None
        assert so > sh

    def test_none_too_few(self):
        assert sortino_ratio([0.01]) is None


class TestMaxDrawdown:
    def test_monotonic_up_zero_drawdown(self):
        assert max_drawdown([100, 101, 102, 110]) == pytest.approx(0.0)

    def test_known_drawdown(self):
        # Peak 120 → trough 90 = -25%.
        assert max_drawdown([100, 120, 90, 110]) == pytest.approx(-0.25)

    def test_deepest_of_multiple(self):
        # Two drawdowns: 110→99 (-10%) and 120→84 (-30%). Worst = -30%.
        assert max_drawdown([100, 110, 99, 120, 84, 90]) == pytest.approx(-0.30)

    def test_none_too_few(self):
        assert max_drawdown([100]) is None
