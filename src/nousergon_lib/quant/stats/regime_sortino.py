"""regime_sortino — regime-stratified, cross-sectional pick-alpha Sortino.

Stratifies per-pick risk-adjusted performance by a categorical regime label,
answering "did the regime call enable better risk-adjusted returns?" — distinct
from stratifying signal *accuracy* by regime.

Canonical-alpha conventions:
- Alpha = ``log(1 + return) − log(1 + spy_return)`` (log domain, NOT arithmetic),
  where ``return`` is a DECIMAL FRACTION. The canonical ``log_alpha_{h}d`` column
  is preferred over recomputing it: a re-derivation from the wide arithmetic
  columns is both lossy (they are 2dp-rounded percent) and a standing units
  hazard. See :class:`ReturnUnits` and ``alpha-engine-config-I7661``.
- Headline metric: **Sortino** (downside-deviation denominator), NOT raw
  Sharpe. Only realizations below the threshold (zero alpha) contribute a
  shortfall; the sum of squared shortfalls is divided by the **full sample n**
  (above-target picks contribute an explicit zero), which is the Sortino (1991)
  definition and the fleet convention pinned by alpha-engine-config-I7271.
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import numpy as np
import pandas as pd

from nousergon_lib.quant import riskstats
from nousergon_lib.quant.horizons import DEFAULT_POLICY, HorizonPolicy

logger = logging.getLogger(__name__)


# Trading days per year — used for annualization. Mirrors quant.stats.dsr.
_TRADING_DAYS_PER_YEAR: int = 252


# Minimum picks per regime stratum before computing risk-adjusted metrics. Below
# this, the stratum reports n_picks but None metrics — too few observations to be
# statistically meaningful.
DEFAULT_MIN_PICKS_PER_STRATUM: int = 20


# Horizons reported — RESOLVED FROM THE FLEET HorizonPolicy, never hardcoded.
#
# This was ``(10, 30)`` until alpha-engine-config-I7661. Both were retired by
# config#1456 and orphaned by the config#1528 cutover: the long-format outcome
# store carries no horizon-30 rows at all, and the legacy wide ``return_10d`` /
# ``return_30d`` columns have had no producer since March. The metric was
# therefore computing off 34 frozen rows dated 2026-03-04..03-13 — 30 of them
# in the ``caution`` stratum, itself a regime label retired by the 3-class
# taxonomy — and republishing them weekly. Four consecutive weekly artifacts
# were identical net of run metadata, which no write-time freshness check can
# see.
#
# An incomplete migration is the root cause, and the chokepoint is the fix:
# config#1528 moved seven backtester readers onto ``HorizonPolicy`` and missed
# this one.
SUPPORTED_HORIZONS: tuple[int, ...] = DEFAULT_POLICY.all_horizons


# Sortino spread interpretation thresholds. Different scale than Sharpe — the
# downside-deviation denominator makes |spread| values typically larger for the
# same distribution.
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


class ReturnUnits(str, Enum):
    """The unit convention of an arithmetic return column.

    There is no safe default. The fleet stores the SAME quantity both ways —
    ``score_performance_outcomes`` (canonical) holds decimals, while the wide
    ``return_{h}d`` / ``spy_{h}d_return`` columns hold ``round(decimal*100, 2)``
    percent points — so a caller that does not state which it is holding is
    guessing, and a wrong guess produces plausible numbers rather than an
    error. ``alpha-engine-config-I7661``: ``log(1 + 5.55)`` is a perfectly good
    float, and that is exactly the problem.
    """

    FRACTION = "fraction"
    PERCENT = "percent"


class ReturnUnitsError(ValueError):
    """A return column's values contradict its declared units."""


# Plausibility bound for a per-pick DECIMAL return over a <= 21-trading-day
# horizon. +500% is far outside anything this metric legitimately sees, and a
# percent-point column mislabelled as a fraction blows past it immediately
# (a 5.55% move reads as +555%). Deliberately loose: this is a units tripwire,
# not an outlier filter.
_MAX_PLAUSIBLE_FRACTION: float = 5.0

# The discriminating tripwire. A percent-point column whose values happen to
# be small (a 5-day return of +2.4pp) sits comfortably inside the ±5 bound
# above while meaning something 100× different — the max alone is not enough.
# The MEDIAN separates the two conventions cleanly: per-pick decimal returns
# over a <= 21-day horizon have a median |r| of a few hundredths, while the
# same data in percent points has a median |r| of a few units. A typical pick
# moving 50% is not a return distribution this metric ever legitimately sees.
_MAX_PLAUSIBLE_MEDIAN_FRACTION: float = 0.5

# The mirror direction: a DECIMAL column read as percent divides by 100 and
# understates every alpha by two orders of magnitude — silently, since small
# numbers trip no upper bound. A per-pick return distribution whose median
# absolute move is under 5 basis points over a <= 21-day horizon is not a
# return distribution. Applied only to a column with enough rows to have a
# distribution at all.
_MIN_PLAUSIBLE_MEDIAN_FRACTION: float = 5e-4
_MEDIAN_CHECK_MIN_ROWS: int = 10


def _to_fraction(
    values: pd.Series,
    units: ReturnUnits,
    *,
    column: str,
) -> pd.Series:
    """Coerce a declared-units arithmetic return column to decimal fractions.

    Fails LOUD rather than clipping. Two conditions raise:

    * a value at or below ``-1.0`` — the position went to zero or worse, so
      ``log(1 + r)`` is genuinely undefined. The pre-I7661 code clipped
      ``1 + r`` to ``1e-9``, which turned every such row into ``log_alpha ≈
      -20.7``: an undefined quantity rendered as a finite number, and the
      direct cause of the published ``mean_log_alpha`` of -6.44 (#7237's class).
    * a value beyond ``_MAX_PLAUSIBLE_FRACTION`` after conversion — the column
      is almost certainly not in the units it was declared to be in. Raising
      here is the whole point: a future source swap must break the run, not
      quietly change what the number means.
    """
    converted = values.astype("float64")
    if units is ReturnUnits.PERCENT:
        converted = converted / 100.0
    finite = converted[np.isfinite(converted)]
    if finite.empty:
        return converted
    lo, hi = float(finite.min()), float(finite.max())
    if lo <= -1.0:
        raise ReturnUnitsError(
            f"{column}: value {lo:.6g} (declared {units.value}) implies a "
            f"return of -100% or worse, for which log(1 + r) is undefined. "
            f"Exclude or resolve these rows — do not clip them into a finite "
            f"log return (alpha-engine-config-I7661 / #7237)."
        )
    likely = (
        ReturnUnits.PERCENT if units is ReturnUnits.FRACTION else ReturnUnits.FRACTION
    )
    extreme = max(abs(lo), abs(hi))
    if extreme > _MAX_PLAUSIBLE_FRACTION:
        raise ReturnUnitsError(
            f"{column}: declared units {units.value!r} but the column spans "
            f"[{lo:.6g}, {hi:.6g}] after conversion, beyond the plausible "
            f"per-pick bound of ±{_MAX_PLAUSIBLE_FRACTION:g}. The source is "
            f"most likely {likely.value!r} (alpha-engine-config-I7661)."
        )
    median_abs = float(finite.abs().median())
    if median_abs > _MAX_PLAUSIBLE_MEDIAN_FRACTION:
        raise ReturnUnitsError(
            f"{column}: declared units {units.value!r} but the MEDIAN absolute "
            f"value is {median_abs:.6g} after conversion — a typical pick "
            f"moving {median_abs:.0%} is not a return distribution this metric "
            f"sees. The source is most likely {likely.value!r} "
            f"(alpha-engine-config-I7661)."
        )
    if (
        finite.size >= _MEDIAN_CHECK_MIN_ROWS
        and 0.0 < median_abs < _MIN_PLAUSIBLE_MEDIAN_FRACTION
    ):
        raise ReturnUnitsError(
            f"{column}: declared units {units.value!r} but the MEDIAN absolute "
            f"value is {median_abs:.6g} after conversion — under "
            f"{_MIN_PLAUSIBLE_MEDIAN_FRACTION:g} ({_MIN_PLAUSIBLE_MEDIAN_FRACTION * 1e4:.0f} bps) "
            f"over {finite.size} rows. The source is most likely "
            f"{likely.value!r} (alpha-engine-config-I7661)."
        )
    return converted


def _arithmetic_to_log_alpha(
    arithmetic_return: pd.Series,
    arithmetic_spy_return: pd.Series,
    *,
    units: ReturnUnits,
    return_column: str = "return",
    spy_column: str = "spy_return",
) -> pd.Series:
    """Convert arithmetic per-pick returns to log-domain pick alpha.

    log_alpha = log(1 + return) − log(1 + spy_return)

    Log domain is required for variance-bearing computations because log returns
    are additive in time and symmetric in sign around zero. NaN propagates; the
    caller filters those out before metric computation.

    ``units`` is REQUIRED and keyword-only — see :class:`ReturnUnits`. Prefer
    the canonical ``log_alpha_{h}d`` column where the store publishes one;
    this path exists for horizons that do not yet carry it.
    """
    ret = _to_fraction(arithmetic_return, units, column=return_column)
    spy = _to_fraction(arithmetic_spy_return, units, column=spy_column)
    return np.log1p(ret) - np.log1p(spy)


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

    Only picks below zero (the threshold) contribute a shortfall; the sum of
    squared shortfalls is divided by the **full sample** — the fleet convention
    (alpha-engine-config-I7271), identical to what every other Sortino in the
    fleet computes. Returns ``None`` on insufficient sample, and on a downside
    deviation of zero — which covers both a pure-upside sample (no pick below
    threshold) and a downside dispersion under the IEEE-754 tolerance. In both
    cases the RATIO is undefined; the denominator itself is reported separately
    by :func:`_downside_std`.

    A regime-conditional Sortino has no claim on a different denominator than a
    portfolio-level one: stratifying by regime changes WHICH observations are in
    the sample, not how a shortfall is normalised. Dividing by the below-target
    count instead (the pre-config-I7638 behaviour) makes the statistic a
    function of how many losing picks a stratum happens to contain — it inflates
    the denominator by sqrt(n / n_down) and so UNDERSTATES Sortino by that
    factor, differently in each stratum, which is precisely the comparison a
    bull-minus-bear spread makes.
    """
    if log_alphas.size < 2:
        return None
    mean = float(log_alphas.mean())
    downside_std = riskstats.downside_deviation(
        [float(x) for x in log_alphas],
        target=0.0,
        denominator="full",
    )
    if downside_std is None or not np.isfinite(downside_std) or downside_std < 1e-12:
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
    """Downside deviation — the Sortino denominator, surfaced independently of
    the ratio.

    Full-sample denominator, same convention as
    :func:`_annualized_sortino_from_log_alphas` — see the note there. A stratum
    with no pick below threshold reports ``0.0`` (the sum of shortfalls is
    genuinely zero under this convention), not ``None``; the ratio that divides
    by it still reports ``None``.
    """
    if log_alphas.size == 1:
        # A one-pick stratum (reachable with min_picks_per_stratum=1) has no
        # estimable dispersion, but this has always reported |alpha| for a
        # single negative pick. Preserved verbatim: riskstats.downside_deviation
        # declines n < 2 outright, and the two conventions coincide at n = 1
        # anyway (one shortfall over one observation).
        return float(abs(log_alphas[0])) if log_alphas[0] < 0.0 else None
    return riskstats.downside_deviation(
        [float(x) for x in log_alphas], target=0.0, denominator="full"
    )


def _empty_stratum(market_regime: str, horizon_days: int, n_picks: int = 0) -> StratumMetrics:
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


def _resolve_log_alphas(
    populated: pd.DataFrame,
    horizon_days: int,
    *,
    units: ReturnUnits,
    policy: HorizonPolicy,
    use_canonical: bool,
) -> np.ndarray:
    """Per-pick log alphas for one stratum, canonical column FIRST.

    ``score_performance_outcomes`` publishes ``log_alpha`` in decimals for the
    primary horizon, already computed by the producer that owns the definition.
    Reading it beats re-deriving it from the wide arithmetic columns, which are
    a 2dp-rounded PERCENT projection of the same numbers — lossy on top of the
    units hazard that made this metric wrong (alpha-engine-config-I7661).

    Horizons with no published log-alpha fall back to the arithmetic path,
    which now demands a stated unit convention.
    """
    cols = policy.outcome_columns(horizon_days)
    if use_canonical:
        return cast("pd.Series", populated[cols.log_alpha]).astype("float64").to_numpy()
    return _arithmetic_to_log_alpha(
        cast("pd.Series", populated[cols.stock_return]),
        cast("pd.Series", populated[cols.spy_return]),
        units=units,
        return_column=cols.stock_return,
        spy_column=cols.spy_return,
    ).to_numpy()


def _stratum_metrics(
    slice_df: pd.DataFrame,
    market_regime: str,
    horizon_days: int,
    min_picks: int,
    *,
    units: ReturnUnits,
    policy: HorizonPolicy = DEFAULT_POLICY,
) -> StratumMetrics:
    """Compute per-stratum metrics over log-domain pick alphas.

    Returns None-padded StratumMetrics when the stratum is below ``min_picks`` —
    the caller filters those out of the headline spread metric.
    """
    # Column names resolve from the HorizonPolicy chokepoint — never
    # f-string-assembled here (config#1456's root cause, EPIC config#1483).
    cols = policy.outcome_columns(horizon_days)
    return_col = cols.stock_return
    spy_col = cols.spy_return
    beat_col = cols.beat_spy

    if return_col not in slice_df.columns or spy_col not in slice_df.columns:
        return _empty_stratum(market_regime, horizon_days)

    # Boolean-mask row selection on a DataFrame always returns a DataFrame;
    # pyright's inference widens to Series|DataFrame because
    # DataFrame.__getitem__'s stub overloads also cover scalar/list-label
    # column selection.
    # Which rows can yield an alpha at all. When the canonical log-alpha
    # column is published, ITS non-nullness is the population — otherwise a
    # row could be counted as a pick and then have no alpha to contribute.
    log_col = cols.log_alpha
    use_canonical = log_col in slice_df.columns and bool(slice_df[log_col].notna().any())
    mask = (
        slice_df[log_col].notna()
        if use_canonical
        else slice_df[return_col].notna() & slice_df[spy_col].notna()
    )
    populated = cast("pd.DataFrame", slice_df[mask])
    n_picks = len(populated)
    if n_picks < min_picks:
        return _empty_stratum(market_regime, horizon_days, n_picks)

    # Canonical log-alpha where published, else an explicitly-united
    # arithmetic conversion. Both paths raise rather than clip.
    log_alphas = _resolve_log_alphas(
        populated, horizon_days, units=units, policy=policy,
        use_canonical=use_canonical,
    )

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
    units: ReturnUnits,
    min_picks_per_stratum: int = DEFAULT_MIN_PICKS_PER_STRATUM,
    horizons: Sequence[int] = SUPPORTED_HORIZONS,
    policy: HorizonPolicy = DEFAULT_POLICY,
) -> list[StratumMetrics]:
    """Group ``df`` by ``market_regime``; compute Sortino + Sharpe + log-alpha +
    hit-rate per (regime, horizon) stratum.

    ``df`` is a per-pick frame carrying ``market_regime`` plus the outcome
    columns named by ``policy.outcome_columns(h)`` — the canonical
    ``log_alpha_{h}d`` where it is published, otherwise arithmetic
    ``return_{h}d`` / ``spy_{h}d_return`` (and optional ``beat_spy_{h}d``).

    ``units`` is REQUIRED and declares the convention of those arithmetic
    columns; there is no default because the fleet stores the same quantity
    both ways and a wrong guess yields plausible numbers rather than an error
    (alpha-engine-config-I7661). It is ignored for any horizon resolved from
    the canonical log-alpha column, which is log-domain by definition.

    Returns one StratumMetrics per (regime, horizon) discovered. Strata
    below ``min_picks_per_stratum`` have None risk-adjusted metrics; n_picks
    still reflects how many were found. Rows with NaN ``market_regime`` are
    skipped.
    """
    if "market_regime" not in df.columns:
        return []

    # See the analogous cast in _stratum_metrics: boolean-mask row
    # selection on a DataFrame always returns a DataFrame.
    df_with_regime = cast("pd.DataFrame", df[df["market_regime"].notna()])
    regimes = sorted(df_with_regime["market_regime"].unique())

    out: list[StratumMetrics] = []
    for regime in regimes:
        regime_slice = cast(
            "pd.DataFrame", df_with_regime[df_with_regime["market_regime"] == regime]
        )
        for horizon in horizons:
            out.append(
                _stratum_metrics(
                    slice_df=regime_slice,
                    market_regime=str(regime),
                    horizon_days=horizon,
                    min_picks=min_picks_per_stratum,
                    units=units,
                    policy=policy,
                )
            )
    return out


def compute_regime_spread(
    strata: Sequence[StratumMetrics],
    horizon_days: int = DEFAULT_POLICY.primary_horizon,
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


@dataclass(frozen=True)
class InputWindow:
    """The date span of the rows a run actually measured.

    A stage that publishes a fresh artifact containing five-month-old inputs is
    indistinguishable, to every WRITE-TIME freshness check the fleet has, from
    one that is working — which is how four consecutive weekly
    ``regime/stratified_sortino`` artifacts came to be identical net of run
    metadata without anything noticing (alpha-engine-config-I7661). Freshness
    is a property of the INPUTS; recording them on the artifact is what makes it
    checkable.
    """

    min_score_date: str | None
    max_score_date: str | None
    n_rows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_score_date": self.min_score_date,
            "max_score_date": self.max_score_date,
            "n_rows": self.n_rows,
        }


# Slack, in calendar days, on top of the horizon itself before a run's newest
# input counts as stale. Covers the weekly cadence plus a missed cycle.
DEFAULT_INPUT_STALENESS_GRACE_DAYS: int = 14

# Artifact status vocabulary. ``unmeasurable`` exists so a run that could not
# measure anything fails LOUD instead of rendering as an empty success
# (champion-challenger-policy §7.2).
STATUS_OK: str = "ok"
STATUS_UNMEASURABLE: str = "unmeasurable"


def input_window(df: pd.DataFrame, *, date_col: str = "score_date") -> InputWindow:
    """Min/max/count of ``date_col`` over the rows fed to the metric."""
    if df is None or df.empty or date_col not in df.columns:
        return InputWindow(min_score_date=None, max_score_date=None, n_rows=0)
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dates.empty:
        return InputWindow(min_score_date=None, max_score_date=None, n_rows=len(df))
    return InputWindow(
        min_score_date=str(dates.min().date()),
        max_score_date=str(dates.max().date()),
        n_rows=len(df),
    )


def assess_input_freshness(
    window: InputWindow,
    *,
    trading_day: str,
    horizon_days: int,
    grace_days: int = DEFAULT_INPUT_STALENESS_GRACE_DAYS,
) -> tuple[str, str]:
    """Is the newest measured input recent enough for this run to mean anything?

    Returns ``(status, reason)``. The predicate is on INPUT dates, not the
    artifact's write time, and it is horizon-aware: a horizon-``h`` outcome
    cannot resolve until ``h`` trading days after the score date, so the newest
    score date a healthy run can carry is already ``h`` behind. Comparing
    against ``now`` without that allowance flags every correct run as stale —
    the mirror of the defect this guards (champion-challenger-policy §7.1).
    """
    if window.max_score_date is None or window.n_rows == 0:
        return (
            STATUS_UNMEASURABLE,
            f"no rows carried a usable {'score_date'} — nothing was measured",
        )
    try:
        newest = pd.Timestamp(window.max_score_date)
        asof = pd.Timestamp(trading_day)
    except ValueError:
        return (
            STATUS_UNMEASURABLE,
            f"unparseable dates (max_score_date={window.max_score_date!r}, "
            f"trading_day={trading_day!r})",
        )
    # Trading days → calendar days at 5 per 7.
    horizon_calendar_days = math.ceil(horizon_days * 7 / 5)
    deadline = asof - pd.Timedelta(int(horizon_calendar_days + grace_days), unit="D")
    if newest < deadline:
        age = (asof - newest).days
        return (
            STATUS_UNMEASURABLE,
            f"newest measured input is {window.max_score_date} — {age} calendar "
            f"days before trading_day {trading_day}, past the "
            f"{horizon_calendar_days}d horizon allowance + {grace_days}d grace. "
            f"The producer for this horizon has stopped (alpha-engine-config-I7661).",
        )
    return (STATUS_OK, "")


def assemble_t2_eval_payload(
    *,
    strata: Sequence[StratumMetrics],
    spread_primary: Mapping[str, Any],
    spread_diagnostic: Mapping[str, Any],
    run_id: str,
    calendar_date: str,
    trading_day: str,
    window: InputWindow,
    status: str = STATUS_OK,
    status_reason: str = "",
    units: ReturnUnits = ReturnUnits.FRACTION,
    min_picks_per_stratum: int = DEFAULT_MIN_PICKS_PER_STRATUM,
    policy: HorizonPolicy = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Assemble the canonical eval-artifact JSON payload (pure dict build; no I/O).

    The consumer persists this via ``nousergon_lib.eval_artifacts`` writers.

    Spread keys are named from the HorizonPolicy — ``spread_21d`` /
    ``spread_5d`` today. They were ``spread_10d`` / ``spread_30d`` until
    alpha-engine-config-I7661; the dashboard's Regime page had ALREADY been
    migrated to the policy-derived names (``views/15_Regime.py``, config#1456),
    so every T2 tile on that page had been rendering an em-dash against an
    artifact that never carried the keys it was reading. This rename repairs a
    broken consumer rather than breaking a working one.

    ``status`` / ``status_reason`` carry an explicit ``unmeasurable`` verdict
    rather than letting a run with no usable inputs render as an empty success,
    and ``input_window`` records the dates actually measured so freshness can be
    checked on the INPUTS.
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
        "status": status,
        "status_reason": status_reason,
        "min_picks_per_stratum": min_picks_per_stratum,
        "horizons": list(policy.all_horizons),
        f"spread_{policy.primary_horizon}d": dict(spread_primary),
        f"spread_{policy.diagnostic_horizons[0]}d": dict(spread_diagnostic),
        "input_window": window.as_dict(),
        "strata": strata_serialized,
        "method_metadata": {
            "annualization_basis": f"{_TRADING_DAYS_PER_YEAR}_trading_days_per_year",
            "return_units": units.value,
            "alpha_source": (
                "canonical log_alpha_{h}d where published, else "
                "log1p(return) - log1p(spy_return) on returns converted from "
                f"declared units {units.value!r}; a value implying <= -100% or "
                "beyond the plausible bound RAISES rather than clipping "
                "(alpha-engine-config-I7661)"
            ),
            "alpha_definition": (
                "log(1+return_Nd) - log(1+spy_Nd_return) per pick cross-sectional"
            ),
            "headline_metric": (
                "annualized_sortino (downside-deviation denominator over the "
                "full sample N)"
            ),
            "secondary_diagnostic": "annualized_sharpe (full-sample std denominator)",
            "downside_threshold": "0.0 (log-alpha; below this is risk-bearing)",
            "downside_denominator": (
                "full sample N (alpha-engine-config-I7271; before "
                "alpha-engine-config-I7638 this was the below-target count, "
                "which understated every Sortino by sqrt(n / n_down))"
            ),
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
