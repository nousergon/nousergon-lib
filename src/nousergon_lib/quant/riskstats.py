"""Risk-adjusted performance statistics — descriptive measures of a return series.

Pure stdlib, data-source-agnostic. These *describe* the risk/return character of
a return stream (no advice). Conventions follow standard institutional practice:
periodic returns in, annualized risk-adjusted ratios out.
"""

from __future__ import annotations

import math

_TRADING_DAYS = 252


def volatility(returns: list[float], *, periods_per_year: int = _TRADING_DAYS) -> float | None:
    """Annualized volatility (sample stdev of periodic returns × √periods).

    None if fewer than two observations.
    """
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(var) * math.sqrt(periods_per_year)


def sharpe_ratio(
    returns: list[float],
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = _TRADING_DAYS,
) -> float | None:
    """Annualized Sharpe ratio.

    ``risk_free_rate`` is an annual rate; it's de-annualized to per-period before
    computing excess returns. None if < 2 obs or zero volatility.
    """
    if len(returns) < 2:
        return None
    rf_period = risk_free_rate / periods_per_year
    excess = [r - rf_period for r in returns]
    mean = sum(excess) / len(excess)
    var = sum((r - mean) ** 2 for r in excess) / (len(excess) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return None
    return (mean / sd) * math.sqrt(periods_per_year)


def sortino_ratio(
    returns: list[float],
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = _TRADING_DAYS,
) -> float | None:
    """Annualized Sortino ratio — Sharpe but penalizing only downside deviation.

    Downside deviation is taken against the (de-annualized) risk-free target.
    None if < 2 obs or there is no downside deviation.
    """
    if len(returns) < 2:
        return None
    rf_period = risk_free_rate / periods_per_year
    excess = [r - rf_period for r in returns]
    mean = sum(excess) / len(excess)
    downside = [min(0.0, r) for r in excess]
    dd_var = sum(d**2 for d in downside) / len(downside)
    dd = math.sqrt(dd_var)
    if dd == 0:
        return None
    return (mean / dd) * math.sqrt(periods_per_year)


def max_drawdown(values: list[float]) -> float | None:
    """Maximum peak-to-trough decline of a value/level series, as a negative fraction.

    Walks the running peak; returns the most negative ``(value/peak - 1)``
    (e.g. ``-0.25`` = a 25% drawdown). 0.0 for a monotonically non-decreasing
    series. None if fewer than two points or a non-positive peak is encountered.
    """
    if len(values) < 2:
        return None
    peak = values[0]
    worst = 0.0
    for v in values:
        if v > peak:
            peak = v
        if peak <= 0:
            return None
        dd = v / peak - 1.0
        if dd < worst:
            worst = dd
    return worst
