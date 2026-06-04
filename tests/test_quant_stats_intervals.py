"""Tests for alpha_engine_lib.quant.stats.intervals — bootstrap CI, Newey-West SE, Wilson."""

import math

import numpy as np

from alpha_engine_lib.quant.stats.intervals import (
    bootstrap_ci,
    newey_west_se,
    wilson_score_interval,
)


class TestBootstrapCI:
    def test_insufficient_data(self):
        assert bootstrap_ci([]) == {"status": "insufficient_data", "n": 0}
        assert bootstrap_ci([4.2])["status"] == "insufficient_data"

    def test_constant_sample_has_zero_width(self):
        out = bootstrap_ci([5.0, 5.0, 5.0, 5.0])
        assert out["status"] == "ok"
        assert out["estimate"] == 5.0
        assert out["ci_low"] == 5.0 and out["ci_high"] == 5.0

    def test_ci_brackets_estimate_and_is_deterministic(self):
        data = list(range(100))
        a = bootstrap_ci(data, seed=7)
        b = bootstrap_ci(data, seed=7)
        assert a == b  # seeded → reproducible (report card must be stable)
        assert a["ci_low"] <= a["estimate"] <= a["ci_high"]
        assert a["estimate"] == float(np.mean(data))
        assert a["ci_level"] == 0.95 and a["method"] == "bootstrap"

    def test_nan_dropped(self):
        out = bootstrap_ci([1.0, 2.0, float("nan"), 3.0])
        assert out["n"] == 3

    def test_custom_statistic(self):
        out = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], statistic=np.median, seed=1)
        assert out["status"] == "ok"
        assert out["estimate"] == 3.0


class TestNeweyWestSE:
    def test_insufficient_data(self):
        assert newey_west_se([2.0])["status"] == "insufficient_data"

    def test_zero_lag_matches_iid_se(self):
        # [1..5]: mean 3, gamma0 = 10/5 = 2, se = sqrt(2/5).
        out = newey_west_se([1.0, 2.0, 3.0, 4.0, 5.0], max_lags=0)
        assert out["estimate"] == 3.0
        assert out["lags"] == 0
        assert out["se"] == math.sqrt(0.4)

    def test_lags_clamped_to_n_minus_1(self):
        out = newey_west_se([1.0, 2.0, 3.0], max_lags=99)
        assert out["lags"] == 2

    def test_auto_lags_nonnegative(self):
        out = newey_west_se([float(x) for x in range(200)])
        assert out["status"] == "ok"
        assert out["lags"] >= 0
        assert out["se"] >= 0.0


class TestWilsonScoreInterval:
    def test_insufficient_data(self):
        assert wilson_score_interval(0, 0)["status"] == "insufficient_data"

    def test_known_50_of_100(self):
        # Textbook Wilson 95% interval for 50/100 is [0.4038, 0.5962].
        out = wilson_score_interval(50, 100)
        assert out["rate"] == 0.5
        assert abs(out["ci_low"] - 0.4038) < 1e-3
        assert abs(out["ci_high"] - 0.5962) < 1e-3

    def test_bounds_clamped_to_unit_interval(self):
        lo = wilson_score_interval(0, 10)
        assert lo["ci_low"] == 0.0 and 0.0 < lo["ci_high"] < 1.0
        hi = wilson_score_interval(10, 10)
        assert hi["ci_high"] == 1.0 and 0.0 < hi["ci_low"] < 1.0

    def test_successes_clamped_to_trials(self):
        out = wilson_score_interval(15, 10)
        assert out["successes"] == 10 and out["rate"] == 1.0
