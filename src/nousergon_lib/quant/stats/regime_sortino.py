"""regime_sortino — regime-stratified, cross-sectional pick-alpha Sortino.

Stratifies per-pick risk-adjusted performance by a categorical regime label,
answering "did the regime call enable better risk-adjusted returns?" — distinct
from stratifying signal *accuracy* by regime.

Canonical-alpha conventions:
- Alpha = ``log(1 + return) − log(1 + spy_return)`` (log domain, NOT arithmetic).
- Headline metric: **Sortino** (downside-only deviation denominator), NOT raw
  Sharpe. Anchored on downside-only variance — only realizations below the
  threshold (zero alpha) enter the denominator.
- Sharpe surfaced as a SECONDARY diagnostic per stratum.

Pick-level (cross-sectional), not portfolio-level (time-series): each pick is an
independent observation, so the metric isolates regime-call quality from
position-sizing / portfolio construction. PSR + max-DD are NOT computed here —
they need time-series path-dependent data (a portfolio-level analyzer's job).

Pure-compute (numpy + pandas); no I/O. The DB loader that feeds
``stratified_sortino_by_regime`` a ``score_performance`` DataFrame stays in the
consumer (it's storage-specific); this module is the metric core.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Trading days per year — used for annualization. Mirrors quant.stats.dsr.
_TRADING_DAYS_PER_YEAR: int = 252


# Minimum picks per regime stratum before computing risk-adjusted metrics. Below
# this, the stratum reports n_picks but None metrics — too few observations to be
# statistically meaningful.
DEFAULT_MIN_PICKS_PER_STRATUM: int = 20


# Horizons reported. Picks must have NON-NULL return + spy_return on a horizon to
# count for that horizon.
SUPPORTED_HORIZONS: tuple[int, ...] = (10, 30)


# Sortino spread interpretation thresholds. Different scale than Sharpe — the
# downside-only denominator makes |spread| values typically larger for the same
# distribution.
_SORTINO_USEFUL_THRESHOLD: float = 0.3
_SORTINO_INVERTED_THRESHOLD: float = -0.3


@dataclass(frozen=True)
class StratumMetrics:
    """Per-regime risk-adjusted statistics over a horizon.

    All metrics computed over **log-domain pick alphas** (canonical framework).
    Sortino is the headline; Sharpe is a secondary diagnostic.
    """

    market_regime: str
    horizon_days: int
    n_picks: int
    # Log-alpha statistics (per-pick cross-sectional)
    mean_log_alpha: float | None
    std_log_alpha: float | None
    downside_std_log_alpha: float | None
    # Risk-adjusted metrics — annualized
    annualized_sortino: float | None       # HEADLINE
    annualized_sharpe: float | None        # secondary diagnostic
    hit_rate: float | None                 # Fraction of picks where log-alpha > 0


def _arithmetic_to_log_alpha(
    arithmetic_return: pd.Series,
    arithmetic_spy_return: pd.Series,
) -> pd.Series:
    """Convert arithmetic per-pick returns to log-domain pick alpha.

    log_alpha = log(1 + return) − log(1 + spy_return)

    Log domain is required for variance-bearing computations because log returns
    are additive in time and symmetric in sign around zero. NaN propagates; the
    caller filters those out before metric computation.
    """
    # Guard against log(0) — a return of -1.0 means the position went to zero;
    # log domain is undefined. Clip with a tiny epsilon so such picks emit a
    # large negative log return rather than crashing.
    one_plus_ret = np.maximum(1.0 + arithmetic_return, 1e-9)
    one_plus_spy = np.maximum(1.0 + arithmetic_spy_return, 1e-9)
    return np.log(one_plus_ret) - np.log(one_plus_spy)


def _annualization_factor(horizon_days: int) -> float:
    """sqrt(periods_per_year) for cross-sectional pick-alpha annualization.

    Each per-pick alpha is observed over ``horizon_days`` of forward return;
    there are _TRADING_DAYS_PER_YEAR / horizon_days such windows per year.
    Sharpe/Sortino scale by sqrt(that ratio).
    """
    return math.sqrt(_TRADING_DAYS_PER_YEAR / horizon_days)


def _annualized_sortino_from_log_alphas(
    log_alphas: np.ndarray,
    horizon_days: int,
) -> float | None:
    """Annualized Sortino on per-pick log alphas.

    Sortino = mean(log_alpha) / downside_std(log_alpha) × sqrt(periods/year)

    Downside std uses ONLY observations below zero (threshold). Returns ``None``
    on insufficient sample, near-zero downside std (IEEE-754 tolerance), or no
    downside observations at all.
    """
    if log_alphas.size < 2:
        return None
    mean = float(log_alphas.mean())
    # Downside-only deviation — RMS of the negative-side observations. Picks with
    # log_alpha > 0 are excluded from the denominator (upside, not risk).
    downside = log_alphas[log_alphas < 0.0]
    if downside.size == 0:
        # Pure upside sample — Sortino undefined but the regime is clearly
        # favorable. Caller treats None as "insufficient downside sample".
        return None
    downside_std = float(np.sqrt(np.mean(downside ** 2)))
    if not np.isfinite(downside_std) or downside_std < 1e-12:
        return None
    return mean / downside_std * _annualization_factor(horizon_days)


def _annualized_sharpe_from_log_alphas(
    log_alphas: np.ndarray,
    horizon_days: int,
) -> float | None:
    """Annualized Sharpe on per-pick log alphas — secondary diagnostic.

    Standard Sharpe (mean / sample-std × sqrt(periods/year)).
    """
    if log_alphas.size < 2:
        return None
    mean = float(log_alphas.mean())
    std = float(log_alphas.std(ddof=1))
    if not np.isfinite(std) or std < 1e-12:
        return None
    return mean / std * _annualization_factor(horizon_days)


def _downside_std(log_alphas: np.ndarray) -> float | None:
    """Downside-only RMS deviation — the Sortino denominator, surfaced
    independently of the ratio."""
    downside = log_alphas[log_alphas < 0.0]
    if downside.size == 0:
        return None
    return float(np.sqrt(np.mean(downside ** 2)))


def _stratum_metrics(
    slice_df: pd.DataFrame,
    market_regime: str,
    horizon_days: int,
    min_picks: int,
) -> StratumMetrics:
    """Compute per-stratum metrics over log-domain pick alphas.

    Returns None-padded StratumMetrics when the stratum is below ``min_picks`` —
    the caller filters those out of the headline spread metric.
    """
    return_col = f"return_{horizon_days}d"
    spy_col = f"spy_{horizon_days}d_return"
    beat_col = f"beat_spy_{horizon_days}d"

    if return_col not in slice_df.columns or spy_col not in slice_df.columns:
        return StratumMetrics(
            market_regime=market_regime,
            horizon_days=horizon_days,
            n_picks=0,
            mean_log_alpha=None,
            std_log_alpha=None,
            downside_std_log_alpha=None,
            annualized_sortino=None,
            annualized_sharpe=None,
            hit_rate=None,
        )

    populated = slice_df[slice_df[return_col].notna() & slice_df[spy_col].notna()]
    n_picks = len(populated)
    if n_picks < min_picks:
        return StratumMetrics(
            market_regime=market_regime,
            horizon_days=horizon_days,
            n_picks=n_picks,
            mean_log_alpha=None,
            std_log_alpha=None,
            downside_std_log_alpha=None,
            annualized_sortino=None,
            annualized_sharpe=None,
            hit_rate=None,
        )

    # Convert arithmetic → log domain (canonical framework)
    log_alphas = _arithmetic_to_log_alpha(
        populated[return_col], populated[spy_col],
    ).to_numpy()

    sortino = _annualized_sortino_from_log_alphas(log_alphas, horizon_days=horizon_days)
    sharpe = _annualized_sharpe_from_log_alphas(log_alphas, horizon_days=horizon_days)
    hit_rate: float | None = None
    if beat_col in populated.columns:
        beat_populated = populated[populated[beat_col].notna()]
        if len(beat_populated) > 0:
            hit_rate = float(beat_populated[beat_col].astype(bool).mean())

    return StratumMetrics(
        market_regime=market_regime,
        horizon_days=horizon_days,
        n_picks=n_picks,
        mean_log_alpha=float(log_alphas.mean()),
        std_log_alpha=float(np.std(log_alphas, ddof=1)),
        downside_std_log_alpha=_downside_std(log_alphas),
        annualized_sortino=sortino,
        annualized_sharpe=sharpe,
        hit_rate=hit_rate,
    )


def stratified_sortino_by_regime(
    df: pd.DataFrame,
    *,
    min_picks_per_stratum: int = DEFAULT_MIN_PICKS_PER_STRATUM,
    horizons: Sequence[int] = SUPPORTED_HORIZONS,
) -> list[StratumMetrics]:
    """Group ``df`` by ``market_regime``; compute Sortino + Sharpe + log-alpha +
    hit-rate per (regime, horizon) stratum.

    ``df`` is a per-pick frame carrying ``market_regime`` plus arithmetic
    ``return_{h}d`` / ``spy_{h}d_return`` (and optional ``beat_spy_{h}d``)
    columns. Returns one StratumMetrics per (regime, horizon) discovered. Strata
    below ``min_picks_per_stratum`` have None risk-adjusted metrics; n_picks
    still reflects how many were found. Rows with NaN ``market_regime`` are
    skipped.
    """
    if "market_regime" not in df.columns:
        return []

    df_with_regime = df[df["market_regime"].notna()]
    regimes = sorted(df_with_regime["market_regime"].unique())

    out: list[StratumMetrics] = []
    for regime in regimes:
        regime_slice = df_with_regime[df_with_regime["market_regime"] == regime]
        for horizon in horizons:
            out.append(
                _stratum_metrics(
                    slice_df=regime_slice,
                    market_regime=str(regime),
                    horizon_days=horizon,
                    min_picks=min_picks_per_stratum,
                )
            )
    return out


def compute_regime_spread(
    strata: Sequence[StratumMetrics],
    horizon_days: int = 10,
) -> dict[str, Any]:
    """Headline Sortino-spread metric: bull-Sortino minus bear-Sortino.

    Positive spread = the regime call enabled better downside-risk-adjusted picks
    when bull-regime was declared vs bear. Near-zero = no actionable signal;
    negative = inverted. Sharpe spread surfaced as a secondary diagnostic; the
    interpretation flag anchors on the Sortino spread.
    """
    by_regime: dict[str, StratumMetrics] = {
        s.market_regime: s for s in strata if s.horizon_days == horizon_days
    }
    bull = by_regime.get("bull")
    bear = by_regime.get("bear")
    bull_sortino = bull.annualized_sortino if bull else None
    bear_sortino = bear.annualized_sortino if bear else None
    bull_sharpe_diag = bull.annualized_sharpe if bull else None
    bear_sharpe_diag = bear.annualized_sharpe if bear else None

    spread: float | None
    sharpe_spread_diagnostic: float | None
    interpretation: str
    if bull_sortino is None or bear_sortino is None:
        spread = None
        interpretation = "insufficient_sample"
    else:
        spread = bull_sortino - bear_sortino
        if spread > _SORTINO_USEFUL_THRESHOLD:
            interpretation = "regime_signal_useful"
        elif spread > _SORTINO_INVERTED_THRESHOLD:
            interpretation = "regime_signal_neutral"
        else:
            interpretation = "regime_signal_inverted"

    if bull_sharpe_diag is None or bear_sharpe_diag is None:
        sharpe_spread_diagnostic = None
    else:
        sharpe_spread_diagnostic = bull_sharpe_diag - bear_sharpe_diag

    return {
        "horizon_days": horizon_days,
        # Headline (Sortino) — per canonical-alpha framework
        "bull_sortino": bull_sortino,
        "bear_sortino": bear_sortino,
        "neutral_sortino": (
            by_regime["neutral"].annualized_sortino
            if by_regime.get("neutral") and by_regime["neutral"].annualized_sortino is not None
            else None
        ),
        # caution_sortino preserved for grandfather attribution on rows from a
        # 4-class regime taxonomy; 3-class emissions never populate it (None).
        "caution_sortino": (
            by_regime["caution"].annualized_sortino
            if by_regime.get("caution") and by_regime["caution"].annualized_sortino is not None
            else None
        ),
        "spread_bull_minus_bear_sortino": spread,
        "interpretation": interpretation,
        "bull_n_picks": bull.n_picks if bull else 0,
        "bear_n_picks": bear.n_picks if bear else 0,
        # Sharpe — secondary diagnostic
        "diagnostic_sharpe_spread_bull_minus_bear": sharpe_spread_diagnostic,
        "diagnostic_bull_sharpe": bull_sharpe_diag,
        "diagnostic_bear_sharpe": bear_sharpe_diag,
    }


def assemble_t2_eval_payload(
    *,
    strata: Sequence[StratumMetrics],
    spread_10d: Mapping[str, Any],
    spread_30d: Mapping[str, Any],
    run_id: str,
    calendar_date: str,
    trading_day: str,
    min_picks_per_stratum: int = DEFAULT_MIN_PICKS_PER_STRATUM,
) -> dict[str, Any]:
    """Assemble the canonical eval-artifact JSON payload (pure dict build; no I/O).

    The consumer persists this via ``nousergon_lib.eval_artifacts`` writers.
    """
    strata_serialized = [
        {
            "market_regime": s.market_regime,
            "horizon_days": s.horizon_days,
            "n_picks": s.n_picks,
            "mean_log_alpha": s.mean_log_alpha,
            "std_log_alpha": s.std_log_alpha,
            "downside_std_log_alpha": s.downside_std_log_alpha,
            "annualized_sortino": s.annualized_sortino,
            "annualized_sharpe_diagnostic": s.annualized_sharpe,
            "hit_rate": s.hit_rate,
        }
        for s in strata
    ]
    return {
        "calendar_date": calendar_date,
        "trading_day": trading_day,
        "run_id": run_id,
        "schema_version": 1,
        "eval_tier": "T2_downstream_stratified_sortino",
        "min_picks_per_stratum": min_picks_per_stratum,
        "spread_10d": dict(spread_10d),
        "spread_30d": dict(spread_30d),
        "strata": strata_serialized,
        "method_metadata": {
            "annualization_basis": f"{_TRADING_DAYS_PER_YEAR}_trading_days_per_year",
            "alpha_definition": (
                "log(1+return_Nd) - log(1+spy_Nd_return) per pick cross-sectional"
            ),
            "headline_metric": "annualized_sortino (downside-only std denominator)",
            "secondary_diagnostic": "annualized_sharpe (full-sample std denominator)",
            "downside_threshold": "0.0 (log-alpha; below this is risk-bearing)",
            "interpretation_thresholds": {
                "useful_above": _SORTINO_USEFUL_THRESHOLD,
                "neutral_band": (
                    f"({_SORTINO_INVERTED_THRESHOLD}, {_SORTINO_USEFUL_THRESHOLD})"
                ),
                "inverted_below": _SORTINO_INVERTED_THRESHOLD,
            },
            "psr_max_dd_note": (
                "PSR + max DD not computed at pick-cross-sectional level "
                "(require time-series path)."
            ),
        },
    }
