"""Risk-adjusted performance statistics — descriptive measures of a return series.

Pure stdlib, data-source-agnostic. These *describe* the risk/return character of
a return stream (no advice). Conventions follow standard institutional practice:
periodic returns in, annualized risk-adjusted ratios out.

This module is the fleet's ONLY implementation of these statistics (config-I7597).
A consumer that needs a variant takes it as an argument here — ``periods_per_year``
for a non-daily or non-annualized space, ``denominator`` for the downside
convention — rather than re-deriving the arithmetic locally, because independent
copies hand-aligned to agree today are copies that drift tomorrow: config-I7271
was a 1.83x Sortino error that survived in production because two functions
computed "the same" statistic differently. Consumers that genuinely cannot call
in (a vectorized 2-D kernel) carry a drift test against
``tests/test_quant_riskstats_drift_corpus.py``'s CORPUS instead.
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


_FULL = "full"
_DOWNSIDE = "downside"


def downside_deviation(
    returns: list[float],
    *,
    target: float = 0.0,
    denominator: str = _FULL,
) -> float | None:
    """Per-period downside deviation — the Sortino denominator, on its own.

    This is the single implementation of the downside-deviation maths for the
    whole fleet. It is deliberately NOT annualized: annualization belongs to the
    ratio (see :func:`sortino_ratio`), and several consumers work in a space
    where a trading-day scale does not apply at all (per-date cross-sectional
    IC, per-pick log alpha).

    ``target`` is a **per-period** target expressed in the same units as
    ``returns`` (0.0 = the standard MAR-of-zero). It is not de-annualized —
    contrast :func:`sortino_ratio`'s ``risk_free_rate``, which is annual.

    ``denominator`` selects which observations divide the sum of squared
    shortfalls:

    ``"full"`` (default, **the fleet convention**)
        RMS of ``min(0, r - target)`` over **all n** observations. Above-target
        observations contribute an explicit zero. This is Sortino (1991) /
        Satchell's definition and the convention pinned by
        alpha-engine-config-I7271.

    ``"downside"`` (**non-standard — do not use in new code**)
        RMS of the shortfalls over only the ``n_down`` below-target
        observations. Larger than ``"full"`` by ``sqrt(n / n_down)``, so it
        systematically UNDERSTATES the resulting Sortino. It exists solely so
        the call sites that ship this convention today can stop carrying their
        own copy of the arithmetic while their numbers stay put; each such call
        site names the variant explicitly at the call, which is the point —
        the divergence is now a visible argument rather than an invisible
        re-implementation. Tracked for conversion to ``"full"``.

    Returns
    -------
    float | None
        ``None`` if fewer than two observations. Under ``"full"``, ``0.0`` when
        no observation is below target (the sum of shortfalls is genuinely
        zero). Under ``"downside"``, ``None`` when no observation is below
        target (the denominator has no observations — undefined, not zero).
    """
    if denominator not in (_FULL, _DOWNSIDE):
        raise ValueError(
            f"denominator must be {_FULL!r} or {_DOWNSIDE!r}, got {denominator!r}"
        )
    if len(returns) < 2:
        return None
    shortfalls = [r - target for r in returns if r < target]
    if denominator == _DOWNSIDE:
        if not shortfalls:
            return None
        return math.sqrt(sum(d * d for d in shortfalls) / len(shortfalls))
    return math.sqrt(sum(d * d for d in shortfalls) / len(returns))


def sortino_ratio(
    returns: list[float],
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = _TRADING_DAYS,
    denominator: str = _FULL,
) -> float | None:
    """Annualized Sortino ratio — Sharpe but penalizing only downside deviation.

    Downside deviation is taken against the (de-annualized) risk-free target and
    computed by :func:`downside_deviation`; see that function for what
    ``denominator`` selects and why ``"downside"`` exists.

    None if < 2 obs or there is no downside deviation.
    """
    if len(returns) < 2:
        return None
    rf_period = risk_free_rate / periods_per_year
    excess = [r - rf_period for r in returns]
    mean = sum(excess) / len(excess)
    dd = downside_deviation(excess, target=0.0, denominator=denominator)
    if not dd:
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
