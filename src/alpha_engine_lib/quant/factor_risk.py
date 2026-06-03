"""Statistical factor risk model — ex-ante portfolio risk + tracking error.

Pure numpy (no sklearn/scipy), data-source-agnostic. Regresses each holding's
return series on a set of **factor return series** (e.g. market + style-ETF
spreads, or universe-wide Fama-MacBeth factor returns) to recover loadings ``B``
and idiosyncratic variance ``D``; ``F`` is the (Ledoit-Wolf-shrunk) factor-return
covariance. The structural covariance

    Σ = B · F · Bᵀ + D

then gives the portfolio's **ex-ante volatility**, its split into **factor vs
idiosyncratic** risk, **per-factor risk contributions**, and **tracking error**
vs a benchmark. Descriptive risk analytics — no advice, no trade.

The consumption layer (``portfolio_risk`` / ``tracking_error`` / decomposition)
is estimator-agnostic: it consumes any ``FactorRiskModel`` (B, F, D). Two
estimators are supported as factor *sources*: the time-series factor-ETF
regression here (``estimate_factor_model``), and — once wired — a universe-wide
cross-sectional Fama-MacBeth build (the alpha-engine predictor's ``risk_model``).
Both feed the same Σ=BFBᵀ+D core. See alpha-engine-lib OVERVIEW for the leverage
rationale.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_TRADING_DAYS = 252


def ledoit_wolf_cov(observations: np.ndarray, *, shrinkage: str = "ledoit_wolf") -> np.ndarray:
    """Covariance of an ``(n_obs, k)`` matrix, optionally Ledoit-Wolf shrunk.

    ``"ledoit_wolf"`` (default) shrinks the MLE sample covariance toward a scaled
    identity (Ledoit & Wolf 2004), which keeps a small-/noisy-sample factor
    covariance well-conditioned. ``"sample"`` returns the plain (n−1) sample
    covariance. Shrinkage intensity is estimated from the data, so for a large,
    clean sample it ≈ the sample covariance.
    """
    X = np.asarray(observations, dtype=float)
    n, p = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    if shrinkage == "sample" or n < 2 or p < 2:
        return (Xc.T @ Xc) / max(n - 1, 1)

    sample = (Xc.T @ Xc) / n  # MLE divisor (Ledoit-Wolf convention)
    mu = np.trace(sample) / p
    target = mu * np.eye(p)
    d2 = float(np.sum((sample - target) ** 2) / p)
    if d2 == 0.0:
        return sample
    # b̄² — mean squared error of the per-observation outer products vs the sample.
    b_sum = 0.0
    for t in range(n):
        xt = Xc[t][:, None]
        b_sum += float(np.sum((xt @ xt.T - sample) ** 2))
    b2 = min(b_sum / (n * n) / p, d2)
    shrink = b2 / d2  # → 0 for a clean sample, → 1 when the sample is all noise
    return shrink * target + (1.0 - shrink) * sample


def _ols(y: np.ndarray, factors: np.ndarray) -> tuple[np.ndarray, float]:
    """OLS of ``y`` on ``factors`` (+ intercept). Returns (betas, residual variance)."""
    n = len(y)
    design = np.column_stack([np.ones(n), factors])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    dof = max(n - design.shape[1], 1)
    return coef[1:], float(resid @ resid / dof)


@dataclass(frozen=True)
class FactorRiskModel:
    """Estimated factor risk model: loadings B, idio variance D, factor cov F.

    ``betas`` is ``(N, K)`` over ``tickers`` × ``factors``; ``idio_var`` is the
    ``(N,)`` daily residual variance; ``factor_cov`` is the ``(K, K)`` daily
    factor covariance. All variances are per-period (daily); annualization happens
    in the risk functions via ``periods_per_year``.
    """

    factors: list[str]
    tickers: list[str]
    betas: np.ndarray
    idio_var: np.ndarray
    factor_cov: np.ndarray
    periods_per_year: int = _TRADING_DAYS


def _factor_matrix(factor_returns: dict[str, Sequence[float]]) -> tuple[list[str], np.ndarray]:
    factors = list(factor_returns)
    mat = np.column_stack([np.asarray(factor_returns[f], dtype=float) for f in factors])
    return factors, mat


def estimate_factor_model(
    holding_returns: dict[str, Sequence[float]],
    factor_returns: dict[str, Sequence[float]],
    *,
    shrinkage: str = "ledoit_wolf",
    periods_per_year: int = _TRADING_DAYS,
) -> FactorRiskModel:
    """Fit loadings + idio variance per holding and the factor covariance.

    ``holding_returns`` and ``factor_returns`` are ``{name: return_series}`` maps,
    all aligned to the same ``n`` dates. Each holding is OLS-regressed on the
    factor matrix (with intercept). Raises ``ValueError`` if the series lengths
    disagree or there aren't enough observations to identify the factors.
    """
    factors, fmat = _factor_matrix(factor_returns)
    n, k = fmat.shape
    if n < k + 2:
        raise ValueError(f"need ≥ {k + 2} observations for {k} factors, got {n}")

    tickers: list[str] = []
    betas: list[np.ndarray] = []
    idio: list[float] = []
    for ticker, series in holding_returns.items():
        y = np.asarray(series, dtype=float)
        if len(y) != n:
            raise ValueError(f"{ticker} has {len(y)} returns, expected {n}")
        b, rv = _ols(y, fmat)
        tickers.append(ticker)
        betas.append(b)
        idio.append(rv)

    return FactorRiskModel(
        factors=factors,
        tickers=tickers,
        betas=np.array(betas).reshape(len(tickers), k),
        idio_var=np.array(idio, dtype=float),
        factor_cov=ledoit_wolf_cov(fmat, shrinkage=shrinkage),
        periods_per_year=periods_per_year,
    )


def benchmark_exposure(benchmark_returns: Sequence[float], factor_returns: dict[str, Sequence[float]]) -> np.ndarray:
    """Benchmark's factor exposure — OLS betas of its return series on the factors.

    A diversified benchmark's idiosyncratic risk is ≈ 0, so only its factor
    exposure is needed for tracking error.
    """
    _, fmat = _factor_matrix(factor_returns)
    beta, _ = _ols(np.asarray(benchmark_returns, dtype=float), fmat)
    return beta


def _weight_vector(model: FactorRiskModel, weights: dict[str, float]) -> np.ndarray:
    """Weights aligned to ``model.tickers``, renormalized to sum 1 over the covered set."""
    w = np.array([float(weights.get(t, 0.0)) for t in model.tickers])
    total = w.sum()
    return w / total if total > 0 else w


def portfolio_risk(model: FactorRiskModel, weights: dict[str, float]) -> dict:
    """Ex-ante annualized risk of the weighted portfolio, decomposed.

    Returns total / factor / idiosyncratic volatility (annualized), the
    portfolio's net factor exposures (``Bᵀw``), and each factor's share of total
    **variance** (``factor_pct_contrib`` sums to the factor share; ``idio_pct``
    is the rest).
    """
    w = _weight_vector(model, weights)
    ann = model.periods_per_year
    x = model.betas.T @ w  # (K,) portfolio factor exposure
    fx = model.factor_cov @ x
    factor_var = float(x @ fx)
    idio_var = float(np.sum(w**2 * model.idio_var))
    total_var = factor_var + idio_var

    contrib = x * fx  # per-factor variance contribution; Σ = factor_var
    # NB: zip() without strict= (added 3.10) — the lib targets 3.9; the iterables
    # are equal-length by construction (factors, contrib, x, active are all length K).
    pct = {f: (float(c) / total_var if total_var > 0 else 0.0) for f, c in zip(model.factors, contrib)}
    return {
        "total_vol": float(np.sqrt(max(total_var, 0.0) * ann)),
        "factor_vol": float(np.sqrt(max(factor_var, 0.0) * ann)),
        "idio_vol": float(np.sqrt(max(idio_var, 0.0) * ann)),
        "factor_exposures": {f: float(v) for f, v in zip(model.factors, x)},
        "factor_pct_contrib": pct,
        "idio_pct": (idio_var / total_var if total_var > 0 else 0.0),
    }


def tracking_error(model: FactorRiskModel, weights: dict[str, float], benchmark_exposures: np.ndarray) -> dict:
    """Annualized tracking error (active risk) vs a benchmark.

    Active factor exposure is ``Bᵀw − x_b``; the portfolio's idiosyncratic risk is
    treated as fully active (a diversified benchmark contributes ≈ 0 idio). Also
    returns the per-factor active exposures so you can see which tilts drive the
    deviation from the benchmark.
    """
    w = _weight_vector(model, weights)
    ann = model.periods_per_year
    active = model.betas.T @ w - np.asarray(benchmark_exposures, dtype=float)
    active_factor_var = float(active @ model.factor_cov @ active)
    active_idio_var = float(np.sum(w**2 * model.idio_var))
    te_var = active_factor_var + active_idio_var
    return {
        "tracking_error": float(np.sqrt(max(te_var, 0.0) * ann)),
        "active_factor_vol": float(np.sqrt(max(active_factor_var, 0.0) * ann)),
        "active_idio_vol": float(np.sqrt(max(active_idio_var, 0.0) * ann)),
        "active_exposures": {f: float(v) for f, v in zip(model.factors, active)},
    }
