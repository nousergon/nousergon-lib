"""Cross-sectional (Fama-MacBeth) factor risk model — the "Option A" estimator.

Complements ``quant.factor_risk`` (the "Option B" time-series factor-ETF
estimator). Both produce the inputs to the same Σ = B·F·Bᵀ + D structural
covariance consumed by ``quant.factor_risk.portfolio_risk`` / ``tracking_error``;
they differ only in how the factor returns ``f_t`` and the factor covariance
``F`` are estimated:

  - **Option B** (``factor_risk.estimate_factor_model``) — regress each holding's
    return series on a small set of *given* factor return series (market +
    style-ETF spreads). Loadings ``B`` are the regression betas. numpy-only.
  - **Option A** (here) — take *exogenous* per-ticker factor loadings ``B`` (e.g.
    fundamentals-derived style exposures) and infer the factor returns ``f_t`` by
    a cross-sectional OLS at each date (Fama-MacBeth 1973):

        r_t = B_{t-1} · f_t + ε_t

    Stacking ``f_t`` over a rolling window gives a (T × K) factor-return panel;
    ``F`` is its (Ledoit-Wolf-shrunk) covariance and ``D`` the per-ticker
    time-series variance of the residuals ε. This is the universe-wide Barra-lite
    build.

**Dependencies:** pandas (always) + scikit-learn (lazy, only for the
``ledoit_wolf``/``oas`` shrinkage estimators). Install ``nousergon-lib[quant-xs]``.
Kept in its own module so the numpy-only ``factor_risk``/``risk_measures``/etc.
consumers don't pull pandas+sklearn.

References:
  - Fama & MacBeth 1973 "Risk, Return, and Equilibrium: Empirical Tests"
    (JPE 81(3)) — cross-sectional-regression construction of factor returns
  - Grinold & Kahn 2000, _Active Portfolio Management_, Ch. 3 — canonical
    structural factor risk model
  - Menchero, Orr & Wang 2011 "The Barra US Equity Model (USE4)
    Methodology Notes" — operational reference
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


_MIN_OBS_OVER_K = 5  # require ≥ K + 5 valid observations for a stable regression


def cross_sectional_factor_returns(
    returns_t: np.ndarray,
    loadings_prev: np.ndarray,
    *,
    include_intercept: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve r_t = B_{t-1} · f_t + ε_t for one date via OLS.

    Args:
        returns_t: (N,) realized returns at time t.
        loadings_prev: (N, K) factor loadings at time t-1.
        include_intercept: if True, prepends a column of 1s to the
            loadings (the "market" factor return). f_t[0] becomes the
            cross-sectional mean return; f_t[1:] are the per-factor
            slopes. Default True.

    Returns:
        (f_t, residuals):
          • f_t: (K_eff,) factor return vector — length K+1 with
            intercept, K without.
          • residuals: (N,) per-ticker ε_t. NaN for rows where the
            inputs had NaN (preserved positionally so the caller can
            keep aligning with the universe).

    Rows with NaN in either r_t or any column of B_{t-1} are excluded
    from the regression. If fewer than K_eff + 5 valid rows remain
    (the regression is unstable), returns all-NaN for both outputs.
    """
    returns_t = np.asarray(returns_t, dtype=np.float64).ravel()
    loadings_prev = np.asarray(loadings_prev, dtype=np.float64)
    if loadings_prev.ndim != 2:
        raise ValueError(
            f"loadings_prev must be 2-D (N × K); got shape {loadings_prev.shape}"
        )
    N, K = loadings_prev.shape
    if returns_t.shape != (N,):
        raise ValueError(
            f"returns_t shape {returns_t.shape} != ({N},) matching loadings rows"
        )

    if include_intercept:
        B = np.column_stack([np.ones(N), loadings_prev])
        K_eff = K + 1
    else:
        B = loadings_prev
        K_eff = K

    valid = np.isfinite(returns_t) & np.all(np.isfinite(B), axis=1)
    n_valid = int(valid.sum())
    if n_valid < K_eff + _MIN_OBS_OVER_K:
        return np.full(K_eff, np.nan), np.full(N, np.nan)

    r_valid = returns_t[valid]
    B_valid = B[valid]

    # OLS via lstsq is rank-robust (returns minimum-norm solution if B
    # is rank-deficient). Rank-deficient B is a soft warning, not an
    # error — caller decides whether to drop low-rank dates.
    f_t, *_ = np.linalg.lstsq(B_valid, r_valid, rcond=None)

    residuals = np.full(N, np.nan)
    residuals[valid] = r_valid - B_valid @ f_t
    return f_t, residuals


def build_factor_returns_series(
    returns_panel: pd.DataFrame,
    loadings_by_date: dict[pd.Timestamp, pd.DataFrame],
    *,
    include_intercept: bool = True,
    factor_names: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loop over dates in ``returns_panel``; for each date t, run the
    cross-sectional regression r_t = B_{t-1} · f_t + ε_t.

    Args:
        returns_panel: (T × N) DataFrame indexed by date, columns are
            ticker names. r_t is the t-th row.
        loadings_by_date: mapping date_t-1 → (N × K) DataFrame of
            factor loadings for that date. Indexed by ticker, columns
            are factor names. The driver looks up loadings at the
            previous available date for each t (most recent ≤ t-1).
        include_intercept: prepends a market-factor column. See
            cross_sectional_factor_returns. Default True.
        factor_names: optional explicit order for the K factor columns.
            If provided, loadings_by_date entries are reindexed to this
            order. Default: use the order of the first loadings frame.

    Returns:
        (factor_returns_df, residuals_df):
          • factor_returns_df: (T × K_eff) — index matches returns_panel
            dates; columns are ["market", *factor_names] when intercept
            is on, [*factor_names] when off.
          • residuals_df: (T × N) — same shape as returns_panel; NaN
            where the regression was skipped or input was missing.
    """
    if returns_panel.empty:
        return pd.DataFrame(), pd.DataFrame()

    dates = list(returns_panel.index)
    tickers = list(returns_panel.columns)
    N = len(tickers)

    # Resolve canonical factor name list from the first usable loadings frame
    if factor_names is None:
        sample = next(iter(loadings_by_date.values()), None)
        if sample is None:
            raise ValueError("loadings_by_date is empty — nothing to regress against")
        factor_names = list(sample.columns)
    factor_names = list(factor_names)
    K = len(factor_names)

    col_names = (["market"] + factor_names) if include_intercept else factor_names

    f_panel = np.full((len(dates), len(col_names)), np.nan)
    eps_panel = np.full((len(dates), N), np.nan)

    sorted_loading_dates = sorted(loadings_by_date.keys())

    for i, date_t in enumerate(dates):
        prev_date = _latest_loading_date_at_or_before(sorted_loading_dates, date_t)
        if prev_date is None:
            continue
        B_df = loadings_by_date[prev_date].reindex(index=tickers, columns=factor_names)
        if B_df.empty:
            continue
        B = B_df.to_numpy(dtype=np.float64)
        r = returns_panel.iloc[i].to_numpy(dtype=np.float64)

        f_t, residuals = cross_sectional_factor_returns(
            r, B, include_intercept=include_intercept,
        )
        f_panel[i] = f_t
        eps_panel[i] = residuals

    # pd.Index(...) wraps (rather than passing the raw list) because
    # pandas' DataFrame(index=..., columns=...) stub types the parameter
    # as Axes, and a plain list[T] fails its SequenceNotStr structural
    # check (list.index()'s parameter variance isn't compatible with the
    # protocol's Any-typed index() as pyright checks it) — pandas itself
    # wraps list args in pd.Index internally, so this is a no-op at
    # runtime.
    factor_returns_df = pd.DataFrame(f_panel, index=pd.Index(dates), columns=pd.Index(col_names))
    residuals_df = pd.DataFrame(eps_panel, index=pd.Index(dates), columns=pd.Index(tickers))
    return factor_returns_df, residuals_df


def _latest_loading_date_at_or_before(
    sorted_dates: list[pd.Timestamp], cutoff: pd.Timestamp,
) -> pd.Timestamp | None:
    """Bisect for the latest loading-date strictly < cutoff (informationally
    safe: at date t we only know loadings as of date t-1)."""
    import bisect
    idx = bisect.bisect_left(sorted_dates, cutoff)
    if idx == 0:
        return None
    return sorted_dates[idx - 1]


def estimate_factor_covariance(
    factor_returns_df: pd.DataFrame,
    *,
    shrinkage: str = "ledoit_wolf",
    min_obs: int = 30,
) -> pd.DataFrame:
    """Estimate F = Cov(f_t) over the factor-return panel.

    Drops rows with any NaN (incomplete regressions). Default LW shrinkage
    mirrors the executor's portfolio_optimizer default; "sample" and "oas"
    also supported. Reuses sklearn estimators.

    Args:
        factor_returns_df: (T × K_eff) factor-return panel from
            build_factor_returns_series.
        shrinkage: estimator name. "ledoit_wolf" (default), "sample", "oas".
        min_obs: minimum clean rows required. Below floor returns an
            all-NaN F so the caller knows the build was insufficient
            (per no-silent-fails — would-be downstream consumers of F
            see NaN, not silently zero).

    Returns:
        F: (K_eff × K_eff) DataFrame, index + columns are factor names.
    """
    clean = factor_returns_df.dropna()
    K = factor_returns_df.shape[1]
    cols = list(factor_returns_df.columns)
    if len(clean) < min_obs:
        log.warning(
            "estimate_factor_covariance: only %d clean rows (need ≥%d) — "
            "returning all-NaN F", len(clean), min_obs,
        )
        return pd.DataFrame(np.full((K, K), np.nan), index=pd.Index(cols), columns=pd.Index(cols))

    if shrinkage == "ledoit_wolf":
        # Single shared Ledoit-Wolf estimator (LV1-AE.a, 2026-06-03). The numpy
        # ``quant.factor_risk.ledoit_wolf_cov`` is numerically identical to
        # sklearn's ``LedoitWolf`` (max abs diff ~1e-21, validated across
        # n∈[35,1000]) — both center the data and estimate the same shrinkage
        # intensity toward a scaled-identity target. Consolidating onto the numpy
        # impl kills the duplicate reimplementation; sklearn stays a lazy import
        # for OAS only.
        from .factor_risk import ledoit_wolf_cov
        F = ledoit_wolf_cov(clean.to_numpy(), shrinkage="ledoit_wolf")
    elif shrinkage == "oas":
        from sklearn.covariance import OAS
        F = OAS().fit(clean.to_numpy()).covariance_
    elif shrinkage == "sample":
        F = np.cov(clean.to_numpy(), rowvar=False)
    else:
        raise ValueError(f"Unknown shrinkage: {shrinkage!r}")
    return pd.DataFrame(F, index=pd.Index(cols), columns=pd.Index(cols))


def estimate_idiosyncratic_variance(
    residuals_df: pd.DataFrame,
    *,
    min_obs: int = 30,
) -> pd.Series:
    """Per-ticker D_{ii} = Var(ε_{i,t}) — diagonal of the residual cov.

    Tickers with fewer than ``min_obs`` non-NaN residual rows are
    emitted as NaN per no-silent-fails (downstream Σ = B·F·Bᵀ + D
    construction treats NaN D as "skip this name" or falls back to a
    safe default).

    Args:
        residuals_df: (T × N) residual panel from
            build_factor_returns_series.
        min_obs: minimum non-NaN observations per ticker.

    Returns:
        D: (N,) Series indexed by ticker.
    """
    out = pd.Series(np.nan, index=residuals_df.columns, dtype=np.float64)
    for ticker in residuals_df.columns:
        eps = residuals_df[ticker].dropna()
        if len(eps) < min_obs:
            continue
        # Population variance (N divisor — universe is the population for
        # cross-sectional regressions) to match the F estimator convention.
        out[ticker] = float(eps.var(ddof=0))
    return out


def build_factor_risk_model(
    returns_panel: pd.DataFrame,
    loadings_by_date: dict[pd.Timestamp, pd.DataFrame],
    *,
    include_intercept: bool = True,
    cov_shrinkage: str = "ledoit_wolf",
    min_cov_obs: int = 30,
    min_idio_obs: int = 30,
) -> dict:
    """End-to-end builder: cross-sectional regressions → F + D.

    Returns a dict with keys:
      • "factor_returns": (T × K_eff) DataFrame
      • "residuals": (T × N) DataFrame
      • "F": (K_eff × K_eff) DataFrame
      • "D": (N,) Series
      • "metadata": dict with n_dates, n_clean_dates, K_eff, n_tickers
    """
    factor_returns, residuals = build_factor_returns_series(
        returns_panel, loadings_by_date,
        include_intercept=include_intercept,
    )
    F = estimate_factor_covariance(
        factor_returns, shrinkage=cov_shrinkage, min_obs=min_cov_obs,
    )
    D = estimate_idiosyncratic_variance(residuals, min_obs=min_idio_obs)

    n_clean = int(factor_returns.dropna().shape[0])
    metadata = {
        "n_dates": int(factor_returns.shape[0]),
        "n_clean_dates": n_clean,
        "K_eff": int(factor_returns.shape[1]),
        "n_tickers": int(returns_panel.shape[1]),
        "cov_shrinkage": cov_shrinkage,
        "include_intercept": bool(include_intercept),
    }
    return {
        "factor_returns": factor_returns,
        "residuals": residuals,
        "F": F,
        "D": D,
        "metadata": metadata,
    }


__all__ = [
    "cross_sectional_factor_returns",
    "build_factor_returns_series",
    "estimate_factor_covariance",
    "estimate_idiosyncratic_variance",
    "build_factor_risk_model",
]
