"""Tests for nousergon_lib/quant/factor_risk.py — statistical factor risk model."""

import math

import pytest

# factor_risk is the only quant module needing numpy (the [quant] extra); skip the
# whole module cleanly when a dev installs only [dev] without it.
np = pytest.importorskip("numpy")

from nousergon_lib.quant.factor_risk import (  # noqa: E402  (after importorskip guard)
    benchmark_exposure,
    estimate_factor_model,
    ledoit_wolf_cov,
    portfolio_risk,
    tracking_error,
)


def _factors(n=200, seed=0):
    rng = np.random.RandomState(seed)
    return {
        "MKT": rng.normal(0.0005, 0.01, n),
        "MOM": rng.normal(0.0, 0.006, n),
        "VAL": rng.normal(0.0, 0.005, n),
    }


class TestLedoitWolf:
    def test_sample_option_is_plain_covariance(self):
        rng = np.random.RandomState(1)
        X = rng.normal(0, 1, (500, 3))
        cov = ledoit_wolf_cov(X, shrinkage="sample")
        assert np.allclose(cov, np.cov(X, rowvar=False), atol=1e-10)

    def test_shrinkage_pulls_toward_scaled_identity(self):
        # Strongly correlated columns: LW should shrink the off-diagonals toward 0
        # (the scaled-identity target) relative to the raw sample covariance.
        rng = np.random.RandomState(2)
        base = rng.normal(0, 1, (40, 1))
        X = np.hstack([base + rng.normal(0, 0.05, (40, 1)) for _ in range(3)])
        sample = ledoit_wolf_cov(X, shrinkage="sample")
        shrunk = ledoit_wolf_cov(X, shrinkage="ledoit_wolf")
        off_sample = abs(sample[0, 1])
        off_shrunk = abs(shrunk[0, 1])
        assert off_shrunk < off_sample  # off-diagonal pulled in
        assert np.all(np.linalg.eigvalsh(shrunk) > 0)  # well-conditioned (PD)

    def test_large_clean_sample_approx_sample_cov(self):
        rng = np.random.RandomState(3)
        X = rng.normal(0, 1, (5000, 3))
        shrunk = ledoit_wolf_cov(X, shrinkage="ledoit_wolf")
        assert np.allclose(shrunk, np.cov(X, rowvar=False), atol=0.05)


class TestEstimateFactorModel:
    def test_recovers_known_betas(self):
        f = _factors()
        # Construct holdings as exact linear combos of the factors (no noise).
        y_a = 1.5 * f["MKT"] + 0.5 * f["MOM"]
        y_b = 0.8 * f["MKT"] - 0.3 * f["VAL"]
        model = estimate_factor_model({"A": y_a, "B": y_b}, f)
        assert model.factors == ["MKT", "MOM", "VAL"]
        a = model.betas[model.tickers.index("A")]
        assert a == pytest.approx([1.5, 0.5, 0.0], abs=1e-6)
        b = model.betas[model.tickers.index("B")]
        assert b == pytest.approx([0.8, 0.0, -0.3], abs=1e-6)
        # Exact fit → ~zero idiosyncratic variance.
        assert model.idio_var.max() < 1e-12

    def test_idiosyncratic_variance_captures_residual(self):
        f = _factors()
        rng = np.random.RandomState(9)
        noise = rng.normal(0, 0.02, len(f["MKT"]))
        y = 1.0 * f["MKT"] + noise
        model = estimate_factor_model({"A": y}, f)
        # Residual variance ≈ the injected noise variance.
        assert model.idio_var[0] == pytest.approx(np.var(noise, ddof=4), rel=0.25)

    def test_length_mismatch_raises(self):
        f = _factors(n=50)
        with pytest.raises(ValueError, match="expected 50"):
            estimate_factor_model({"A": [0.0] * 49}, f)

    def test_too_few_observations_raises(self):
        f = {"MKT": [0.01, 0.02, -0.01], "MOM": [0.0, 0.01, 0.0]}  # n=3, k=2 → need 4
        with pytest.raises(ValueError, match="observations"):
            estimate_factor_model({"A": [0.01, 0.0, 0.02]}, f)


class TestPortfolioRisk:
    def test_single_factor_unit_beta_zero_idio(self):
        # One factor with known variance, one holding beta=1, no idio → portfolio
        # vol == annualized factor vol.
        rng = np.random.RandomState(5)
        mkt = rng.normal(0, 0.01, 300)
        model = estimate_factor_model({"A": 1.0 * mkt}, {"MKT": mkt})
        risk = portfolio_risk(model, {"A": 1.0})
        expected = math.sqrt(np.var(mkt, ddof=1) * 252)  # ~ sample-cov vol annualized
        assert risk["total_vol"] == pytest.approx(expected, rel=0.05)
        assert risk["idio_vol"] == pytest.approx(0.0, abs=1e-6)
        assert risk["factor_exposures"]["MKT"] == pytest.approx(1.0, abs=1e-6)

    def test_total_variance_is_factor_plus_idio(self):
        f = _factors()
        rng = np.random.RandomState(7)
        ya = 1.2 * f["MKT"] + rng.normal(0, 0.02, len(f["MKT"]))
        yb = 0.7 * f["MKT"] + 0.4 * f["MOM"] + rng.normal(0, 0.015, len(f["MKT"]))
        model = estimate_factor_model({"A": ya, "B": yb}, f)
        risk = portfolio_risk(model, {"A": 0.5, "B": 0.5})
        # total_vol² ≈ factor_vol² + idio_vol²
        assert risk["total_vol"] ** 2 == pytest.approx(risk["factor_vol"] ** 2 + risk["idio_vol"] ** 2, rel=1e-6)
        # factor % contributions + idio % sum to 1.
        assert sum(risk["factor_pct_contrib"].values()) + risk["idio_pct"] == pytest.approx(1.0)

    def test_weights_renormalized_over_covered(self):
        f = _factors()
        model = estimate_factor_model({"A": 1.0 * f["MKT"], "B": 1.0 * f["MKT"]}, f)
        # Pass unnormalized weights (sum 4) + an unknown ticker → renormalized.
        r1 = portfolio_risk(model, {"A": 2.0, "B": 2.0, "Z": 5.0})
        r2 = portfolio_risk(model, {"A": 0.5, "B": 0.5})
        assert r1["total_vol"] == pytest.approx(r2["total_vol"], rel=1e-9)


class TestTrackingError:
    def test_zero_when_portfolio_matches_benchmark_exposure_no_idio(self):
        rng = np.random.RandomState(11)
        mkt = rng.normal(0, 0.01, 300)
        f = {"MKT": mkt}
        model = estimate_factor_model({"A": 1.0 * mkt}, f)  # beta 1, ~0 idio
        x_b = benchmark_exposure(mkt, f)  # benchmark IS the market → exposure ~1
        te = tracking_error(model, {"A": 1.0}, x_b)
        assert te["tracking_error"] == pytest.approx(0.0, abs=1e-5)

    def test_active_tilt_produces_tracking_error(self):
        f = _factors()
        # Portfolio loads heavily on momentum; benchmark (market) doesn't.
        ya = 1.0 * f["MKT"] + 1.5 * f["MOM"]
        model = estimate_factor_model({"A": ya}, f)
        x_b = benchmark_exposure(f["MKT"], f)  # ≈ [1, 0, 0]
        te = tracking_error(model, {"A": 1.0}, x_b)
        assert te["tracking_error"] > 0
        # The momentum tilt is the dominant active exposure.
        assert te["active_exposures"]["MOM"] == pytest.approx(1.5, abs=1e-6)
        assert abs(te["active_exposures"]["MKT"]) < 1e-6
