"""intervals — Bootstrap confidence intervals, Newey-West SE, Wilson score intervals.

The three inference primitives the System Report Card v2 metric records require
(``MetricRecord.ci_method`` ∈ {``bootstrap``, ``newey-west``, ``wilson``}):

  - ``bootstrap_ci``          — percentile bootstrap CI for any statistic of a
                                sample (default the mean). The general-purpose
                                CI for ICs, lifts, hit-rates, Sharpe point
                                estimates where no closed form is convenient.
  - ``newey_west_se``         — heteroskedasticity-and-autocorrelation-consistent
                                (HAC) standard error of a series mean, for
                                autocorrelated daily P&L where the iid SE
                                understates uncertainty.
  - ``wilson_score_interval`` — Wilson score binomial interval for rates with
                                small N (veto-gate precision/recall, hit-rate),
                                which the normal-approximation interval handles
                                badly near 0/1.

Pure-compute; no I/O. bootstrap + Newey-West need numpy (install
``nousergon-lib[quant]``); Wilson is stdlib-only (``statistics.NormalDist``).

Reference: López de Prado, *Advances in Financial Machine Learning* (bootstrap
+ HAC); Wilson (1927) "Probable Inference, the Law of Succession, and
Statistical Inference".
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Callable, Sequence, TypedDict

import numpy as np

_DEFAULT_N_RESAMPLES = 1000


class BootstrapCIResult(TypedDict, total=False):
    status: str           # "ok" | "insufficient_data"
    n: int                # observations after NaN drop
    estimate: float       # statistic on the full sample
    ci_low: float
    ci_high: float
    ci_level: float       # e.g. 0.95
    method: str           # "bootstrap"
    n_resamples: int


class NeweyWestResult(TypedDict, total=False):
    status: str           # "ok" | "insufficient_data"
    n: int
    estimate: float       # sample mean
    se: float             # HAC standard error of the mean
    lags: int             # Bartlett-kernel lags used
    method: str           # "newey-west"


class WilsonScoreResult(TypedDict, total=False):
    status: str           # "ok" | "insufficient_data"
    n: int                # trials
    successes: int
    rate: float           # successes / trials (point estimate)
    estimate: float       # alias of rate (uniform with the other results)
    ci_low: float
    ci_high: float
    ci_level: float
    method: str           # "wilson"


def _clean(data: Sequence[float] | np.ndarray) -> np.ndarray:
    """Coerce to a 1-D float array with NaN/inf dropped."""
    arr = np.asarray(data, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def bootstrap_ci(
    data: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    ci_level: float = 0.95,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
    seed: int = 0,
) -> BootstrapCIResult:
    """Percentile bootstrap confidence interval for ``statistic`` of ``data``.

    Args:
        data: 1-D sample of observations (NaN/inf dropped).
        statistic: Callable ``(np.ndarray) -> float`` to bootstrap. Defaults to
            the mean.
        ci_level: Confidence level in (0, 1) (default 0.95).
        n_resamples: Number of bootstrap resamples (default 1000).
        seed: RNG seed for reproducibility (the report card must be stable
            across re-renders of the same cycle).

    Returns:
        A :class:`BootstrapCIResult`. ``status == "insufficient_data"`` when
        fewer than 2 finite observations remain.
    """
    arr = _clean(data)
    n = int(arr.size)
    if n < 2:
        return {"status": "insufficient_data", "n": n}

    estimate = float(statistic(arr))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot = np.fromiter(
        (statistic(arr[row]) for row in idx), dtype=float, count=n_resamples
    )
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return {"status": "insufficient_data", "n": n}

    tail = (1.0 - ci_level) / 2.0
    ci_low = float(np.percentile(boot, 100.0 * tail))
    ci_high = float(np.percentile(boot, 100.0 * (1.0 - tail)))
    return {
        "status": "ok",
        "n": n,
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": ci_level,
        "method": "bootstrap",
        "n_resamples": int(boot.size),
    }


def _auto_lags(n: int) -> int:
    """Newey-West (1994) automatic lag selection: floor(4·(n/100)^(2/9))."""
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_se(
    series: Sequence[float] | np.ndarray,
    *,
    max_lags: int | None = None,
) -> NeweyWestResult:
    """HAC (Newey-West, Bartlett kernel) standard error of the series mean.

    For autocorrelated series (daily P&L), the iid SE ``s/√n`` understates
    uncertainty. The HAC estimator inflates the long-run variance by the
    Bartlett-weighted autocovariances up to ``max_lags``.

    Args:
        series: 1-D series (NaN/inf dropped).
        max_lags: Bartlett-kernel lag truncation. ``None`` ⇒ the Newey-West
            (1994) rule ``floor(4·(n/100)^(2/9))``. Clamped to ``[0, n-1]``.

    Returns:
        A :class:`NeweyWestResult`. ``status == "insufficient_data"`` for n < 2.
    """
    x = _clean(series)
    n = int(x.size)
    if n < 2:
        return {"status": "insufficient_data", "n": n}

    lags = _auto_lags(n) if max_lags is None else int(max_lags)
    lags = max(0, min(lags, n - 1))

    e = x - x.mean()
    gamma0 = float(np.dot(e, e) / n)
    lrv = gamma0
    for j in range(1, lags + 1):
        weight = 1.0 - j / (lags + 1.0)
        gamma_j = float(np.dot(e[j:], e[:-j]) / n)
        lrv += 2.0 * weight * gamma_j
    lrv = max(lrv, 0.0)  # Bartlett kernel guarantees PSD; clamp float error.
    se = math.sqrt(lrv / n)
    return {
        "status": "ok",
        "n": n,
        "estimate": float(x.mean()),
        "se": se,
        "lags": lags,
        "method": "newey-west",
    }


def wilson_score_interval(
    successes: int,
    trials: int,
    *,
    ci_level: float = 0.95,
) -> WilsonScoreResult:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal-approximation (Wald) interval for small ``trials``
    or rates near 0/1, where Wald produces bounds outside [0, 1] and undercovers.

    Args:
        successes: Count of successes (0 ≤ successes ≤ trials).
        trials: Total trials (> 0).
        ci_level: Confidence level in (0, 1) (default 0.95).

    Returns:
        A :class:`WilsonScoreResult`. ``status == "insufficient_data"`` for
        ``trials <= 0``. Bounds are clamped to [0, 1].
    """
    if trials <= 0:
        return {"status": "insufficient_data", "n": int(max(trials, 0))}
    successes = max(0, min(int(successes), int(trials)))

    z = NormalDist().inv_cdf(1.0 - (1.0 - ci_level) / 2.0)
    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials))
    return {
        "status": "ok",
        "n": int(trials),
        "successes": int(successes),
        "rate": p,
        "estimate": p,
        "ci_low": max(0.0, center - margin),
        "ci_high": min(1.0, center + margin),
        "ci_level": ci_level,
        "method": "wilson",
    }
