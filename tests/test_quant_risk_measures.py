"""Tests for nousergon_lib/quant/risk_measures.py (VaR / CVaR estimators)."""

import pytest

from nousergon_lib.quant import risk_measures as rm


class TestNormPpf:
    def test_known_quantiles(self):
        assert rm._norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
        assert rm._norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-4)
        assert rm._norm_ppf(0.95) == pytest.approx(1.644854, abs=1e-4)
        assert rm._norm_ppf(0.99) == pytest.approx(2.326348, abs=1e-4)

    def test_symmetry(self):
        assert rm._norm_ppf(0.1) == pytest.approx(-rm._norm_ppf(0.9), abs=1e-6)

    def test_extreme_tails(self):
        # Exercises the lower- and upper-tail rational branches (p < 0.02425).
        assert rm._norm_ppf(0.005) == pytest.approx(-2.575829, abs=1e-4)
        assert rm._norm_ppf(0.995) == pytest.approx(2.575829, abs=1e-4)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            rm._norm_ppf(0.0)
        with pytest.raises(ValueError):
            rm._norm_ppf(1.0)


class TestParametric:
    def test_var_matches_closed_form(self):
        # Zero-mean series with a known stdev → VaR = z * sd.
        returns = [0.01, -0.01, 0.01, -0.01, 0.01, -0.01]  # mean 0, sd ~0.01095
        _, sd = rm._moments(returns)
        var = rm.parametric_var(returns, confidence=0.95, horizon_days=1)
        assert var == pytest.approx(1.644854 * sd, rel=1e-6)

    def test_cvar_exceeds_var(self):
        returns = [0.02, -0.015, 0.005, -0.01, 0.012, -0.008, 0.003]
        var = rm.parametric_var(returns, confidence=0.95)
        cvar = rm.parametric_cvar(returns, confidence=0.95)
        assert cvar > var > 0

    def test_horizon_scales_by_sqrt_time(self):
        returns = [0.01, -0.01, 0.01, -0.01, 0.01, -0.01]  # exactly zero mean
        v1 = rm.parametric_var(returns, confidence=0.95, horizon_days=1)
        v4 = rm.parametric_var(returns, confidence=0.95, horizon_days=4)
        # Zero mean ⇒ pure σ×√t scaling, so a 4-day VaR is exactly 2× the 1-day.
        assert v4 == pytest.approx(2 * v1, rel=1e-9)

    def test_none_when_too_short(self):
        assert rm.parametric_var([0.01]) is None
        assert rm.parametric_cvar([0.01]) is None


class TestHistorical:
    def test_var_is_loss_quantile(self):
        # 100 returns from -0.10 to +0.89: exactly 5 returns are ≤ -0.05, so the
        # 95% VaR (loss exceeded 1-in-20) lands at ~0.05.
        returns = [(-10 + i) / 100 for i in range(100)]  # -0.10 .. 0.89
        var = rm.historical_var(returns, confidence=0.95)
        assert var == pytest.approx(0.0505, abs=1e-3)

    def test_cvar_is_mean_of_tail_beyond_var(self):
        returns = [-0.10, -0.08, -0.05, 0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
        var = rm.historical_var(returns, confidence=0.90)
        cvar = rm.historical_cvar(returns, confidence=0.90)
        assert cvar >= var > 0

    def test_all_gains_gives_zero_var_and_cvar(self):
        # Every period gains → no loss in the tail → both floor at 0.
        assert rm.historical_var([0.01, 0.02, 0.03, 0.04], confidence=0.95) == 0.0
        assert rm.historical_cvar([0.01, 0.02, 0.03, 0.04], confidence=0.95) == 0.0

    def test_none_when_too_short(self):
        assert rm.historical_var([0.01]) is None
        assert rm.historical_cvar([0.01]) is None
