"""Tests for nousergon_lib.quant.stats.calibration.

Pins:
  1. Perfectly calibrated probabilities → ECE ≈ 0.
  2. Systematically over-confident probabilities → large ECE.
  3. The scale-mismatch trap: a margin (|p_up-0.5|*2) measured against a
     direction hit-rate manufactures a structural ECE even when the underlying
     p_up is perfectly calibrated — the bug this module exists to prevent.
  4. min_bin_n drops sparse bins from the ECE sum.
  5. Empty / insufficient inputs → honest status, not a crash or a 0.0.
  6. Mismatched lengths → ValueError; custom bin_edges honored.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from nousergon_lib.quant.stats.calibration import expected_calibration_error


class TestCoreECE:
    def test_perfectly_calibrated_near_zero(self):
        # Construct outcomes whose frequency matches the predicted probability
        # exactly within each decile → ECE = 0.
        rng = np.random.default_rng(0)
        p = rng.uniform(0.0, 1.0, size=20000)
        y = (rng.uniform(0.0, 1.0, size=p.size) < p).astype(int)
        res = expected_calibration_error(p, y, n_bins=10)
        assert res["status"] == "ok"
        assert res["ece"] < 0.02

    def test_overconfident_large_ece(self):
        # Model always says 0.95 but the event happens only 50% of the time.
        p = np.full(1000, 0.95)
        y = np.tile([0, 1], 500)
        res = expected_calibration_error(p, y, n_bins=10)
        assert res["ece"] == pytest.approx(0.45, abs=1e-3)

    def test_binary_coercion(self):
        # actual_binary given as continuous alpha values → coerced via > 0.
        p = np.array([0.8, 0.8, 0.8, 0.8])
        alpha = np.array([0.03, 0.01, -0.02, 0.05])  # 3 of 4 positive
        res = expected_calibration_error(p, alpha, n_bins=10)
        # bin mean_pred 0.8 vs hit_rate 0.75 → gap 0.05
        assert res["ece"] == pytest.approx(0.05, abs=1e-9)


class TestScaleMismatchTrap:
    """The exact false-alarm mechanism: a margin is NOT a probability."""

    def test_margin_against_hitrate_manufactures_ece_on_calibrated_model(self):
        # p_up perfectly calibrated. Derive margin = |p_up-0.5|*2 and the
        # direction hit (UP correct iff y==1, DOWN correct iff y==0).
        rng = np.random.default_rng(7)
        p_up = rng.uniform(0.0, 1.0, size=20000)
        y = (rng.uniform(0.0, 1.0, size=p_up.size) < p_up).astype(int)

        margin = np.abs(p_up - 0.5) * 2.0
        direction_up = p_up >= 0.5
        hit = np.where(direction_up, y, 1 - y)  # 1 if predicted direction correct

        # WRONG (the old monitor): bin the margin, compare to hit-rate.
        wrong = expected_calibration_error(margin, hit, n_bins=10)
        # RIGHT (this module's contract): bin p_up vs the UP outcome.
        right = expected_calibration_error(p_up, y, n_bins=10)

        assert right["ece"] < 0.02          # truly calibrated
        assert wrong["ece"] > 0.15          # structural artifact from the scale mismatch


class TestGuards:
    def test_min_bin_n_drops_sparse(self):
        # 100 samples at p=0.3, plus 2 stray samples at p=0.9.
        p = np.concatenate([np.full(100, 0.3), np.full(2, 0.9)])
        y = np.concatenate([np.zeros(100), np.ones(2)])  # 0.3 bin perfectly off, 0.9 bin perfect
        res = expected_calibration_error(p, y, n_bins=10, min_bin_n=10)
        assert any(b.get("dropped_reason") for b in res["dropped_bins"])
        # Only the dense (mis-calibrated) bin counts: |0.3 - 0.0| = 0.3
        assert res["ece"] == pytest.approx(0.3, abs=1e-9)
        assert res["n"] == 100

    def test_no_data(self):
        res = expected_calibration_error([], [])
        assert res["status"] == "no_data"
        assert res["ece"] is None

    def test_insufficient_data(self):
        res = expected_calibration_error([0.7, 0.8], [1, 0], min_samples=10)
        assert res["status"] == "insufficient_data"
        assert res["ece"] is None

    def test_nan_pairs_dropped(self):
        p = np.array([0.7, np.nan, 0.7, 0.7])
        y = np.array([1, 1, 0, np.nan])
        res = expected_calibration_error(p, y)
        assert res["n_total"] == 2  # only the two finite pairs survive

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            expected_calibration_error([0.5, 0.6], [1])

    def test_custom_bin_edges(self):
        p = np.array([0.55, 0.95, 0.95])
        y = np.array([1, 1, 0])
        res = expected_calibration_error(p, y, bin_edges=[0.5, 0.9, 1.01])
        # bin [0.5,0.9): mean 0.55 hit 1.0 → gap 0.45 (n=1)
        # bin [0.9,1.01): mean 0.95 hit 0.5 → gap 0.45 (n=2)
        assert res["ece"] == pytest.approx(0.45, abs=1e-9)

    def test_bad_bin_edges_raise(self):
        with pytest.raises(ValueError):
            expected_calibration_error([0.5], [1], bin_edges=[0.5])
        with pytest.raises(ValueError):
            expected_calibration_error([0.5], [1], bin_edges=[0.5, 0.4])
