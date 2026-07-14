"""
series_contract.py — L2 per-series data-contract validation gates.

**Why this exists.** alpha-engine-config#2456 (L2 of the market-value-
integrity framework, alpha-engine-config#1277). L1 (``artifact_freshness``,
this package) answers "did the artifact land at all, on time" — an
ARTIFACT-shaped question. L2 answers a SERIES-shaped question that L1
structurally cannot: "is the price series itself internally sound" —
schema-correct, sane, not stale per-symbol, gap-free against the real NYSE
calendar, free of unexplained outsized moves, and date-ordered. Same
governing principle as L1: no number a human trades on should be trusted
unless it has passed a gate.

**Proximate trigger.** alpha-engine-config#1276 — a bad SPY close silently
entered the settled store because nothing checked it against the prior
history at write time in a calendar-aware way. ``nousergon-data``'s
``validators/price_validator.py`` already covers schema-adjacent OHLC/
sanity/volume checks at write time (``validate_today_row``) and a NAIVE
calendar-DAY gap check on full history (``validate_parquet``'s
``MAX_GAP_TRADING_DAYS`` logic) — the gap check doesn't consult a real
trading calendar, so a gap that happens to straddle a holiday can dodge it,
it has no explicit duplicate/out-of-order date check, and its outlier
threshold is a fixed 50% rather than scaled to the series' own realized
volatility. This module supplies exactly those three gaps as reusable,
calendar-aware, vol-scaled primitives — plus schema/sanity/staleness
checks with the same structured-result contract — so every repo that reads
price series (data, predictor, backtester) can share one validation
substrate instead of re-deriving calendar/vol logic per repo.

**Public surface — the six L2 gates, each returning a
:class:`GateResult`:**

- :func:`check_schema` — required fields present, correct (numeric-
  coercible) types.
- :func:`check_sanity` — price > 0, no negative/zero closes.
- :func:`check_staleness` — per-SERIES (not per-artifact) recency: the
  series' newest observation must be within ``max_age_trading_days`` of
  the reference "as-of" trading day. Distinct from
  :mod:`nousergon_lib.artifact_freshness`, which judges whether an S3
  OBJECT was written on schedule — a stale series can hide inside a
  fresh artifact (the artifact landed on time, but its last row is
  stale because the producer silently stopped updating one symbol).
- :func:`check_continuity` — calendar-aware: no missing NYSE trading
  days between the series' first and last observation, cross-referenced
  against :mod:`krepis.trading_calendar`. This is the check that would
  have caught the 2026-06-24 gap referenced in the parent epic.
- :func:`check_outlier` — dynamic: flags ``|move| > n_sigma × trailing
  realized volatility`` rather than a fixed percentage. See
  :data:`DEFAULT_OUTLIER_N_SIGMA` for the sizing rationale.
- :func:`check_calendar_monotonic` — dates strictly increasing, no
  duplicate or out-of-order entries.

:func:`validate_series` runs all six and returns a
:class:`SeriesContractReport`; :func:`quarantine_decision` reduces a
report to the caller's action (promote vs. quarantine) plus the alarm
text to route through the existing flow-doctor convention (see below).

**Alarming.** This module is pure — no logging, no network calls, no
flow-doctor coupling — mirroring :mod:`nousergon_lib.artifact_freshness`'s
"pure probe, caller alerts" split. Consumers already running inside a
flow-doctor-initialized entrypoint (``setup_logging(...,
flow_doctor_yaml=...)`` at module import time — see
``nousergon-data/weekly_collector.py``) alarm by calling
``logger.error(...)`` with :attr:`SeriesContractReport.summary`; flow-
doctor's ERROR-level log handler captures it automatically and fans out
per the entrypoint's ``flow-doctor.yaml`` (email / GitHub issue / S3
changelog / Telegram via :mod:`nousergon_lib.flow_doctor_fleet`). Contexts
without an initialized singleton (a Lambda) should call
``notify_via_flow_doctor`` / the ``flow_doctor_fleet`` helpers directly.
This module does not duplicate either path.

**Quarantine.** A gate failure means: do not promote the datum into the
settled store. :func:`quarantine_decision` is the pure reducer callers use
to decide whether to skip the write; the actual skip/continue wiring is
caller-side (see ``nousergon-data/builders/daily_append.py``'s
``validate_today_row`` call site, which this module supplements rather
than replaces for the checks ``price_validator`` already owns).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Literal, Mapping, Sequence

import pandas as pd

from krepis.trading_calendar import is_trading_day, previous_trading_day

GateName = Literal[
    "schema",
    "sanity",
    "staleness",
    "continuity",
    "outlier",
    "calendar_monotonic",
]

GATE_NAMES: tuple[GateName, ...] = (
    "schema",
    "sanity",
    "staleness",
    "continuity",
    "outlier",
    "calendar_monotonic",
)

Severity = Literal["block", "warn"]

# Gates that are unambiguous data corruption (schema violations, non-
# positive prices, out-of-order/duplicate dates) default to "block" — the
# datum must not be promoted. Staleness / continuity / outlier can each
# legitimately arise from an operational gap or a real market event rather
# than corruption, so they default to "warn" (quarantine-and-alarm, not a
# hard promotion refusal) UNLESS the caller's use case wants a harder gate
# (e.g. a backtester ingestion path may want continuity to block). Callers
# needing stricter behavior pass their own ``block_gates`` override to
# :func:`quarantine_decision`.
DEFAULT_BLOCK_GATES: frozenset[GateName] = frozenset({
    "schema",
    "sanity",
    "calendar_monotonic",
})

# ── Outlier sizing (issue's gotcha) ──────────────────────────────────────────
# The threshold must be sized against the series' OWN historical realized
# volatility distribution, not a fixed percentage — a fixed % (the
# ``price_validator.MAX_DAILY_RETURN=0.50`` this module supplements) is
# either too loose for a low-vol name (SPY rarely moves 50% intraday-close-
# to-close, so a real 15% gap-down on a shock day sails under a 50%
# threshold undetected) or too tight for a high-vol name (a leveraged/small-
# cap name can legitimately move >50% on a real catalyst). Scaling by
# trailing realized vol makes the threshold self-calibrating per series.
#
# N=6 is the conservative default the issue's gotcha asks for: sized against
# a normal-ish daily-return distribution, a 6-sigma move has a vanishingly
# small false-positive rate under normal trading, so the gate only fires on
# moves at the extreme tail — not every elevated-vol day. Genuine large
# moves (earnings surprises, FOMC-day repricings, guidance cuts) DO
# sometimes clear 6 sigma — that's a real quarantine-and-alarm case, not a
# false positive — but a 6-sigma bar is deliberately looser than a naive
# "3-sigma" statistical-outlier convention, which would false-positive on a
# meaningful fraction of ordinary earnings reactions and train operators to
# ignore the alarm (the exact operational nuisance the issue's gotcha warns
# against). Tune per-series via ``n_sigma`` if a name's realized behavior
# warrants it; this default is a starting point, not a universal constant.
DEFAULT_OUTLIER_N_SIGMA: float = 6.0

# Trailing window (trading days) for the realized-volatility estimate the
# outlier gate scales against. 20 trading days (~1 calendar month) is long
# enough to smooth single-day noise out of the vol estimate but short
# enough to adapt to a genuine regime shift (post-split, post-earnings
# elevated-vol regime) within about a month rather than baking in a stale
# pre-shift vol estimate for a full year.
DEFAULT_VOL_WINDOW: int = 20

# Minimum trailing observations required before the outlier gate will fire
# at all — below this, the trailing-vol estimate is too noisy to trust
# (the sample stdev of a 3-point window swings wildly), so the gate
# abstains (returns ok=True with a reason) rather than false-positiving off
# a garbage estimate.
MIN_VOL_OBSERVATIONS: int = 5

# Default per-series staleness floor, expressed in TRADING days (not
# calendar days) — distinguishes "the series' own newest row is old" from
# L1's per-ARTIFACT freshness floor (which is judged in wall-clock / cycle
# terms against when the S3 OBJECT was last written). 1 trading day: the
# series' newest observation should be no older than the last closed
# trading day as of the reference "as-of" date — a series that stopped
# advancing while its sibling artifact keeps getting touched (e.g. a
# manifest rewrite that re-touches every file's LastModified without
# actually refreshing every symbol) is exactly the failure L1 cannot see.
DEFAULT_STALENESS_MAX_TRADING_DAYS: int = 1

REQUIRED_OHLCV_FIELDS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")
NUMERIC_OHLCV_FIELDS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")


# ── Result types ──────────────────────────────────────────────────────────


@dataclass
class GateResult:
    """One gate's outcome.

    Attributes:
        gate: Which of :data:`GATE_NAMES` produced this result.
        ok: ``True`` iff the gate found no violation.
        severity: Caller-facing default severity for a failure
            (``"block"`` / ``"warn"``) — see :data:`DEFAULT_BLOCK_GATES`.
            Meaningless when ``ok`` is ``True``.
        reason: Human-readable diagnostic — routed into the alarm body.
        detail: Structured extra context (violating dates, computed
            thresholds, etc.) for programmatic consumers / dashboards.
    """

    gate: GateName
    ok: bool
    severity: Severity = "warn"
    reason: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class SeriesContractReport:
    """Aggregate of all six gate results for one series/symbol.

    Attributes:
        series_id: Caller-supplied identifier (ticker, e.g. ``"SPY"``).
        results: One :class:`GateResult` per gate that ran (a gate may
            be skipped — e.g. outlier on an all-NaN series — in which
            case it is omitted rather than reported ``ok=False``).
        passed: ``True`` iff every gate in :attr:`results` is ``ok``.
        failing: The subset of :attr:`results` with ``ok=False``.
        summary: One-line human summary for the alarm body.
    """

    series_id: str
    results: tuple[GateResult, ...]

    @property
    def passed(self) -> bool:
        return all(r.ok for r in self.results)

    @property
    def failing(self) -> tuple[GateResult, ...]:
        return tuple(r for r in self.results if not r.ok)

    @property
    def summary(self) -> str:
        if self.passed:
            return f"series_contract[{self.series_id}]: all {len(self.results)} gates OK"
        parts = [f"{r.gate}={r.reason}" for r in self.failing]
        return f"series_contract[{self.series_id}]: FAILED " + "; ".join(parts)


@dataclass
class QuarantineDecision:
    """Reduction of a :class:`SeriesContractReport` to a caller action.

    Attributes:
        quarantine: ``True`` iff at least one failing gate is in the
            effective block set — the caller MUST NOT promote this
            datum into the settled store.
        alarm: ``True`` iff at least one gate failed at all (block or
            warn) — the caller SHOULD alarm even when not quarantining,
            per the issue's "quarantine + alarm, do not sit silent"
            requirement for every gate failure.
        blocking_gates / warning_gates: Gate names partitioned by the
            effective severity used for this decision.
        message: Alarm text ready to pass to ``logger.error(...)`` (or
            ``notify_via_flow_doctor``) in a flow-doctor-initialized
            context.
    """

    quarantine: bool
    alarm: bool
    blocking_gates: tuple[GateName, ...]
    warning_gates: tuple[GateName, ...]
    message: str


def quarantine_decision(
    report: SeriesContractReport,
    *,
    block_gates: Iterable[GateName] = DEFAULT_BLOCK_GATES,
) -> QuarantineDecision:
    """Reduce a report to the promote/quarantine + alarm decision.

    ``block_gates`` overrides which failing gates trigger quarantine
    (default :data:`DEFAULT_BLOCK_GATES`); every OTHER failing gate
    still alarms, it just doesn't refuse the write — mirrors
    ``validators/price_validator.py``'s block/warn severity split so a
    caller can upgrade e.g. ``outlier`` to block during a known-quiet
    observation window, matching that module's
    ``DAILY_APPEND_BLOCK_ANOMALY_TYPES`` escape hatch.
    """
    block_set = frozenset(block_gates)
    blocking = tuple(r.gate for r in report.failing if r.gate in block_set)
    warning = tuple(r.gate for r in report.failing if r.gate not in block_set)
    return QuarantineDecision(
        quarantine=bool(blocking),
        alarm=not report.passed,
        blocking_gates=blocking,
        warning_gates=warning,
        message=report.summary,
    )


# ── Gate 1: schema ────────────────────────────────────────────────────────


def check_schema(
    df: pd.DataFrame,
    series_id: str,
    *,
    required_fields: Sequence[str] = REQUIRED_OHLCV_FIELDS,
    numeric_fields: Sequence[str] = NUMERIC_OHLCV_FIELDS,
) -> GateResult:
    """Required fields present + numeric-coercible.

    ``df`` is a date-indexed OHLCV frame (one or more rows). Missing
    columns or a column that cannot be interpreted as numeric (object
    dtype containing non-numeric values) both fail. An all-NaN numeric
    column passes THIS gate (NaN-ness is a sanity/coverage question,
    not a schema question) — :func:`check_sanity` catches non-positive
    values, and NaN already fails ``price_validator``'s
    ``nan_or_inf``-class checks for the feature surface.
    """
    missing = [c for c in required_fields if c not in df.columns]
    if missing:
        return GateResult(
            gate="schema",
            ok=False,
            severity="block",
            reason=f"missing required field(s): {missing}",
            detail={"missing_fields": missing},
        )

    non_numeric: list[str] = []
    for col in numeric_fields:
        if col not in df.columns:
            continue
        try:
            pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError):
            non_numeric.append(col)

    if non_numeric:
        return GateResult(
            gate="schema",
            ok=False,
            severity="block",
            reason=f"non-numeric value(s) in field(s): {non_numeric}",
            detail={"non_numeric_fields": non_numeric},
        )

    return GateResult(gate="schema", ok=True)


# ── Gate 2: sanity ────────────────────────────────────────────────────────


def check_sanity(
    df: pd.DataFrame,
    series_id: str,
    *,
    price_field: str = "Close",
) -> GateResult:
    """Price > 0 — no negative/zero closes.

    Only inspects non-null values in ``price_field``; NaN rows are out
    of scope for this gate (coverage, not sanity). Reports up to the
    first 5 offending dates in ``detail`` for a compact alarm body.
    """
    if price_field not in df.columns:
        return GateResult(
            gate="sanity",
            ok=False,
            severity="block",
            reason=f"{price_field!r} column absent — cannot evaluate sanity",
        )

    prices = pd.to_numeric(df[price_field], errors="coerce")
    bad_mask = prices.notna() & (prices <= 0)
    if not bad_mask.any():
        return GateResult(gate="sanity", ok=True)

    bad_idx = df.index[bad_mask]
    bad_dates = [_fmt_date(d) for d in bad_idx[:5]]
    return GateResult(
        gate="sanity",
        ok=False,
        severity="block",
        reason=(
            f"{int(bad_mask.sum())} non-positive {price_field} value(s), "
            f"e.g. {bad_dates}"
        ),
        detail={"n_bad": int(bad_mask.sum()), "sample_dates": bad_dates},
    )


# ── Gate 3: staleness (per-series) ───────────────────────────────────────


def check_staleness(
    df: pd.DataFrame,
    series_id: str,
    *,
    as_of: date,
    max_age_trading_days: int = DEFAULT_STALENESS_MAX_TRADING_DAYS,
) -> GateResult:
    """Per-SERIES staleness: the newest observation must be recent.

    Distinct from :mod:`nousergon_lib.artifact_freshness`, which judges
    an S3 OBJECT's ``LastModified`` — a manifest can be freshly
    rewritten (fresh per L1) while carrying a stale row for one symbol
    that silently stopped updating. This gate looks at the SERIES'
    own last-observed date against ``as_of`` (the reference trading
    day — typically ``krepis.trading_calendar.last_closed_trading_day()``
    at call time; passed explicitly here to keep this function pure/
    testable).

    ``max_age_trading_days=1`` (default) means the newest row must be
    dated on or after the trading day immediately before ``as_of`` —
    i.e. the series must have advanced through (at latest) the prior
    close. Widen for series with a known reporting lag.
    """
    if df.empty:
        return GateResult(
            gate="staleness",
            ok=False,
            severity="warn",
            reason="series has zero observations",
        )

    newest = _to_date(df.index.max())
    floor = as_of
    for _ in range(max_age_trading_days):
        floor = previous_trading_day(floor)

    if newest >= floor:
        return GateResult(gate="staleness", ok=True)

    age_trading_days = _trading_days_between(newest, as_of)
    return GateResult(
        gate="staleness",
        ok=False,
        severity="warn",
        reason=(
            f"newest observation {newest.isoformat()} is "
            f"{age_trading_days} trading day(s) behind as_of "
            f"{as_of.isoformat()} (max allowed {max_age_trading_days})"
        ),
        detail={
            "newest_date": newest.isoformat(),
            "as_of": as_of.isoformat(),
            "age_trading_days": age_trading_days,
        },
    )


def _trading_days_between(start: date, end: date) -> int:
    """Count NYSE trading days strictly between ``start`` and ``end``."""
    from krepis.trading_calendar import count_trading_days

    return count_trading_days(start, end)


# ── Gate 4: continuity (calendar-aware) ──────────────────────────────────


def check_continuity(
    df: pd.DataFrame,
    series_id: str,
    *,
    max_missing_days_report: int = 10,
) -> GateResult:
    """No missing NYSE trading days between the series' first and last date.

    Cross-references the expected trading-day set — every date ``d``
    with ``krepis.trading_calendar.is_trading_day(d)`` between the
    series' min and max index date, inclusive — against the dates
    actually present. This is calendar-aware in a way
    ``price_validator.validate_parquet``'s ``MAX_GAP_TRADING_DAYS``
    logic is not: that check flags a run of >5 CALENDAR days between
    consecutive rows, which both under- and over-fires relative to the
    real NYSE calendar (a holiday-adjacent single missing trading day
    can produce a calendar-day gap short enough to dodge the flag; a
    multi-holiday stretch can trip it even though every real trading
    day is present). This gate instead builds the exact expected set
    and reports the exact missing dates — the check that would have
    caught the 2026-06-24 gap referenced in the parent epic.

    Only trading days are expected — weekends and NYSE holidays are
    correctly absent and never reported as gaps.
    """
    if len(df) < 2:
        return GateResult(gate="continuity", ok=True, reason="fewer than 2 rows; nothing to span")

    dates = sorted({_to_date(d) for d in df.index})
    start, end = dates[0], dates[-1]
    present = set(dates)

    expected = _expected_trading_days(start, end)
    missing = sorted(expected - present)

    if not missing:
        return GateResult(gate="continuity", ok=True)

    sample = [d.isoformat() for d in missing[:max_missing_days_report]]
    return GateResult(
        gate="continuity",
        ok=False,
        severity="warn",
        reason=(
            f"{len(missing)} missing NYSE trading day(s) between "
            f"{start.isoformat()} and {end.isoformat()}, e.g. {sample}"
        ),
        detail={"n_missing": len(missing), "missing_dates": sample},
    )


def _expected_trading_days(start: date, end: date) -> set[date]:
    """Every NYSE trading day in ``[start, end]`` inclusive."""
    from datetime import timedelta

    out: set[date] = set()
    d = start
    while d <= end:
        if is_trading_day(d):
            out.add(d)
        d += timedelta(days=1)
    return out


# ── Gate 5: outlier (vol-scaled) ─────────────────────────────────────────


def check_outlier(
    df: pd.DataFrame,
    series_id: str,
    *,
    price_field: str = "Close",
    n_sigma: float = DEFAULT_OUTLIER_N_SIGMA,
    vol_window: int = DEFAULT_VOL_WINDOW,
    min_observations: int = MIN_VOL_OBSERVATIONS,
) -> GateResult:
    """Flag ``|move| > n_sigma × trailing realized volatility``.

    Deliberately NOT a fixed-percentage threshold (see
    :data:`DEFAULT_OUTLIER_N_SIGMA` for the sizing rationale the
    issue's gotcha asks for). Trailing realized vol is the rolling
    stdev of daily simple returns over ``vol_window`` trading days,
    computed causally (each day's threshold uses only PRIOR days' vol,
    never including the day itself) so a huge move cannot inflate its
    own detection threshold — the classic self-defeating-outlier-
    detector bug.

    Abstains (``ok=True`` with an explanatory reason, no violation
    reported) when there are fewer than ``min_observations`` trailing
    returns available, or when the trailing vol estimate is exactly
    zero (a flat-lined/halted series — a real anomaly, but not this
    gate's anomaly; ``check_sanity``/``check_staleness`` are better
    suited to a frozen series).
    """
    if price_field not in df.columns or len(df) < min_observations + 2:
        return GateResult(
            gate="outlier", ok=True,
            reason="insufficient history for a trailing-vol estimate",
        )

    prices = pd.to_numeric(df[price_field], errors="coerce")
    prices = prices[prices.notna() & (prices > 0)]
    if len(prices) < min_observations + 2:
        return GateResult(
            gate="outlier", ok=True,
            reason="insufficient positive-price history for a trailing-vol estimate",
        )

    prices = prices.sort_index()
    returns = prices.pct_change()

    # Causal trailing vol: shift(1) so day t's threshold uses only
    # returns strictly before t.
    trailing_vol = returns.rolling(window=vol_window, min_periods=min_observations).std()
    trailing_vol = trailing_vol.shift(1)

    threshold = n_sigma * trailing_vol
    violation_mask = (
        returns.abs() > threshold
    ) & threshold.notna() & (threshold > 0)

    if not violation_mask.any():
        return GateResult(gate="outlier", ok=True)

    bad_idx = returns.index[violation_mask]
    samples = []
    for d in bad_idx[:5]:
        samples.append({
            "date": _fmt_date(d),
            "move": round(float(returns.loc[d]), 6),
            "threshold": round(float(threshold.loc[d]), 6),
        })
    return GateResult(
        gate="outlier",
        ok=False,
        severity="warn",
        reason=(
            f"{int(violation_mask.sum())} move(s) exceeded "
            f"{n_sigma}x trailing {vol_window}d realized vol, "
            f"e.g. {samples}"
        ),
        detail={"n_violations": int(violation_mask.sum()), "samples": samples},
    )


# ── Gate 6: calendar-monotonic ────────────────────────────────────────────


def check_calendar_monotonic(df: pd.DataFrame, series_id: str) -> GateResult:
    """Dates strictly increasing — no duplicate or out-of-order entries.

    Compares the raw index ORDER (not a sorted copy) so an out-of-order
    entry is distinguished from a duplicate: a duplicate date anywhere
    in the index fails regardless of position; a date that is present
    but appears after a LATER date in index order fails even if no
    value repeats.
    """
    idx = list(df.index)
    dates = [_to_date(d) for d in idx]

    seen: dict[date, int] = {}
    duplicates: list[str] = []
    for pos, d in enumerate(dates):
        if d in seen:
            duplicates.append(d.isoformat())
        else:
            seen[d] = pos

    out_of_order: list[str] = []
    for i in range(1, len(dates)):
        if dates[i] < dates[i - 1]:
            out_of_order.append(f"{dates[i-1].isoformat()}->{dates[i].isoformat()}")

    if not duplicates and not out_of_order:
        return GateResult(gate="calendar_monotonic", ok=True)

    reasons = []
    if duplicates:
        reasons.append(f"{len(duplicates)} duplicate date(s): {duplicates[:5]}")
    if out_of_order:
        reasons.append(f"{len(out_of_order)} out-of-order transition(s): {out_of_order[:5]}")

    return GateResult(
        gate="calendar_monotonic",
        ok=False,
        severity="block",
        reason="; ".join(reasons),
        detail={"duplicates": duplicates[:20], "out_of_order": out_of_order[:20]},
    )


# ── Orchestration ─────────────────────────────────────────────────────────


def validate_series(
    df: pd.DataFrame,
    series_id: str,
    *,
    as_of: date | None = None,
    price_field: str = "Close",
    required_fields: Sequence[str] = REQUIRED_OHLCV_FIELDS,
    numeric_fields: Sequence[str] = NUMERIC_OHLCV_FIELDS,
    max_age_trading_days: int = DEFAULT_STALENESS_MAX_TRADING_DAYS,
    n_sigma: float = DEFAULT_OUTLIER_N_SIGMA,
    vol_window: int = DEFAULT_VOL_WINDOW,
    run_gates: Sequence[GateName] = GATE_NAMES,
) -> SeriesContractReport:
    """Run the requested L2 gates and return the aggregate report.

    ``df`` is a date-indexed OHLCV frame for one series/symbol
    (typically the full history read for a write-time check, or a
    freshly-refreshed slice for a batch check). ``as_of`` is required
    only when ``"staleness"`` is in ``run_gates`` (typically the
    caller's ``krepis.trading_calendar.last_closed_trading_day()`` at
    call time — passed explicitly rather than defaulted internally to
    keep every gate pure/deterministic for tests).

    Gates that structurally cannot run against a given frame (e.g.
    ``calendar_monotonic``/``continuity`` on an empty frame) still
    produce a ``GateResult`` — ``ok=True`` with an explanatory
    ``reason`` — rather than being silently omitted, so
    :attr:`SeriesContractReport.results` always has one entry per
    requested gate.
    """
    results: list[GateResult] = []

    if "schema" in run_gates:
        results.append(
            check_schema(
                df, series_id,
                required_fields=required_fields,
                numeric_fields=numeric_fields,
            )
        )

    if "sanity" in run_gates:
        results.append(check_sanity(df, series_id, price_field=price_field))

    if "staleness" in run_gates:
        if as_of is None:
            results.append(GateResult(
                gate="staleness", ok=True,
                reason="as_of not supplied; staleness gate skipped",
            ))
        else:
            results.append(
                check_staleness(
                    df, series_id,
                    as_of=as_of, max_age_trading_days=max_age_trading_days,
                )
            )

    if "continuity" in run_gates:
        if df.empty:
            results.append(GateResult(gate="continuity", ok=True, reason="empty series"))
        else:
            results.append(check_continuity(df, series_id))

    if "outlier" in run_gates:
        results.append(
            check_outlier(
                df, series_id,
                price_field=price_field, n_sigma=n_sigma, vol_window=vol_window,
            )
        )

    if "calendar_monotonic" in run_gates:
        if df.empty:
            results.append(GateResult(gate="calendar_monotonic", ok=True, reason="empty series"))
        else:
            results.append(check_calendar_monotonic(df, series_id))

    return SeriesContractReport(series_id=series_id, results=tuple(results))


# ── Small helpers ──────────────────────────────────────────────────────────


def _to_date(value: Any) -> date:
    """Coerce a pandas Timestamp / datetime / date / str index value to date."""
    if isinstance(value, date) and not hasattr(value, "hour"):
        return value
    ts = pd.Timestamp(value)
    return ts.date()


def _fmt_date(value: Any) -> str:
    return _to_date(value).isoformat()


__all__ = [
    "GateName",
    "GATE_NAMES",
    "Severity",
    "DEFAULT_BLOCK_GATES",
    "DEFAULT_OUTLIER_N_SIGMA",
    "DEFAULT_VOL_WINDOW",
    "MIN_VOL_OBSERVATIONS",
    "DEFAULT_STALENESS_MAX_TRADING_DAYS",
    "REQUIRED_OHLCV_FIELDS",
    "NUMERIC_OHLCV_FIELDS",
    "GateResult",
    "SeriesContractReport",
    "QuarantineDecision",
    "quarantine_decision",
    "check_schema",
    "check_sanity",
    "check_staleness",
    "check_continuity",
    "check_outlier",
    "check_calendar_monotonic",
    "validate_series",
]
