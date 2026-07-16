"""Value-at-Risk and Conditional VaR (expected shortfall) of a return series.

Pure stdlib, data-source-agnostic. Two estimators, the institutional pair:

  - **parametric (Gaussian)** — assumes returns are normal, so VaR is a multiple of
    the standard deviation; CVaR follows in closed form. Smooth, but understates
    fat tails.
  - **historical** — the empirical loss quantile and the mean loss beyond it. Makes
    no distributional assumption (captures observed fat tails) but is noisier and
    bounded by the worst sample.

All outputs are **positive loss fractions** at the chosen horizon (e.g. ``0.04`` =
a 4% loss), so a 1-day 95% VaR of 0.018 reads "a 1-in-20 day loses ≥ 1.8%". Risk
*measurement* — descriptive, no advice.
"""

from __future__ import annotations

import math

_TRADING_DAYS = 252


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF via Acklam's rational approximation.

    Accurate to ~1e-9 over (0, 1) — avoids a scipy dependency for the one quantile
    the parametric estimators need. Raises ``ValueError`` outside (0, 1).
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    p_low = 0.02425
    p_high = 1 - p_low

    # Central region — a single rational approximation in (p − 0.5).
    if p_low <= p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )

    # Tails are antisymmetric: the same rational in q = √(−2 ln(tail mass)), with a
    # sign flip for the upper tail.
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        sign = 1.0
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        sign = -1.0
    return (
        sign
        * (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    )


def _norm_pdf(x: float) -> float:
    """Standard-normal probability density at ``x``."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _moments(returns: list[float]) -> tuple[float, float]:
    """(mean, sample stdev) of a return series."""
    n = len(returns)
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    return mean, math.sqrt(var)


def parametric_var(
    returns: list[float],
    *,
    confidence: float = 0.95,
    horizon_days: int = 1,
    periods_per_year: int = _TRADING_DAYS,
) -> float | None:
    """Gaussian VaR as a positive loss fraction at ``horizon_days``.

    Scales the per-period mean/σ to the horizon by √-time (drift × h, σ × √h) and
    returns ``max(0, zσ_h − μ_h)``. None if fewer than two observations. The series
    is assumed to be at the same frequency as ``periods_per_year``.
    """
    if len(returns) < 2:
        return None
    mean, sd = _moments(returns)
    z = _norm_ppf(confidence)
    mu_h = mean * horizon_days
    sd_h = sd * math.sqrt(horizon_days)
    return max(0.0, z * sd_h - mu_h)


def parametric_cvar(
    returns: list[float],
    *,
    confidence: float = 0.95,
    horizon_days: int = 1,
    periods_per_year: int = _TRADING_DAYS,
) -> float | None:
    """Gaussian CVaR (expected shortfall) as a positive loss fraction.

    Closed form for the normal tail: ``σ_h · φ(z)/(1−c) − μ_h``. The mean loss in
    the worst ``1−c`` of outcomes — always ≥ the corresponding VaR. None if < 2 obs.
    """
    if len(returns) < 2:
        return None
    mean, sd = _moments(returns)
    z = _norm_ppf(confidence)
    mu_h = mean * horizon_days
    sd_h = sd * math.sqrt(horizon_days)
    shortfall = sd_h * _norm_pdf(z) / (1 - confidence)
    return max(0.0, shortfall - mu_h)


def historical_var(returns: list[float], *, confidence: float = 0.95) -> float | None:
    """Empirical 1-period VaR — the ``1−c`` loss quantile as a positive fraction.

    Uses linear interpolation between order statistics (the standard quantile
    convention). None if fewer than two observations.
    """
    if len(returns) < 2:
        return None
    losses = sorted(-r for r in returns)  # ascending losses (gains negative)
    rank = confidence * (len(losses) - 1)
    lo = int(math.floor(rank))
    hi = min(lo + 1, len(losses) - 1)
    frac = rank - lo
    return max(0.0, losses[lo] + frac * (losses[hi] - losses[lo]))


def historical_cvar(returns: list[float], *, confidence: float = 0.95) -> float | None:
    """Empirical CVaR — the mean loss in the worst ``1−c`` tail, positive fraction.

    Averages all losses at or beyond the historical VaR threshold (falls back to
    the single worst loss when the tail is otherwise empty). None if < 2 obs.
    """
    if len(returns) < 2:
        return None
    # historical_var only returns None when len(returns) < 2 (its own
    # early-return, mirrored above), so var is never None on this path.
    var = historical_var(returns, confidence=confidence)
    assert var is not None  # noqa: S101 -- type-narrowing invariant, not input validation
    losses = [-r for r in returns]
    tail = [loss for loss in losses if loss >= var]
    if not tail:
        return max(0.0, max(losses))
    return max(0.0, sum(tail) / len(tail))
