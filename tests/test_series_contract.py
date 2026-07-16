"""
Unit tests for ``nousergon_lib.series_contract`` (alpha-engine-config#2456,
the L2 per-series data-contract validation gates).

Covers the six gates individually plus the ``validate_series`` /
``quarantine_decision`` orchestration:

  1. schema — happy path + missing field + non-numeric field.
  2. sanity — happy path + zero/negative close.
  3. staleness — happy path + stale series (per-series, distinct from
     artifact_freshness's per-artifact axis).
  4. continuity — calendar-aware: a gap that straddles a synthetic NYSE
     holiday must NOT false-positive; a real missing trading day must.
  5. outlier — vol-scaled: a big move in a HIGH-vol series must not fire
     while the same absolute move in a LOW-vol series does.
  6. calendar_monotonic — duplicate dates + out-of-order dates.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from nousergon_lib.series_contract import (
    DEFAULT_BLOCK_GATES,
    GATE_NAMES,
    check_calendar_monotonic,
    check_continuity,
    check_outlier,
    check_sanity,
    check_schema,
    check_staleness,
    quarantine_decision,
    validate_series,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _ohlcv(dates, closes, *, volume=1_000_000):
    """Build a minimal OHLCV frame indexed by the given dates."""
    n = len(dates)
    closes = list(closes)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [volume] * n,
        },
        index=pd.to_datetime(dates),
    )


def _bdate_series(start="2026-06-01", periods=30, base=100.0, daily_vol=0.01, seed=7):
    """Seeded pseudo-random-walk OHLCV series — NOT a perfectly smooth
    arithmetic ramp. A deterministic linear ramp has near-zero return
    variance (each day's % move shrinks as the price rises), which makes
    the vol-scaled outlier gate's ``n_sigma x trailing_std`` threshold
    degenerate (tiny std -> tiny threshold -> everything "fails"). A
    seeded random walk with a realistic per-day vol gives the outlier
    gate a non-degenerate trailing-vol estimate to scale against, same as
    a real price series would.
    """
    import random

    rng = random.Random(seed)
    dates = pd.bdate_range(start, periods=periods)
    closes = [base]
    for _ in range(periods - 1):
        closes.append(closes[-1] * (1.0 + rng.gauss(0.0, daily_vol)))
    return _ohlcv(dates, closes)


# ── Gate 1: schema ────────────────────────────────────────────────────────


class TestCheckSchema:
    def test_happy_path(self):
        df = _bdate_series()
        result = check_schema(df, "SPY")
        assert result.ok
        assert result.gate == "schema"

    def test_missing_field(self):
        df = _bdate_series().drop(columns=["Volume"])
        result = check_schema(df, "SPY")
        assert not result.ok
        assert result.severity == "block"
        assert "Volume" in result.detail["missing_fields"]

    def test_non_numeric_field(self):
        df = _bdate_series()
        df["Close"] = df["Close"].astype(object)
        df.iloc[3, df.columns.get_loc("Close")] = "not-a-number"
        result = check_schema(df, "SPY")
        assert not result.ok
        assert "Close" in result.detail["non_numeric_fields"]


# ── Gate 2: sanity ────────────────────────────────────────────────────────


class TestCheckSanity:
    def test_happy_path(self):
        df = _bdate_series()
        result = check_sanity(df, "SPY")
        assert result.ok

    def test_zero_close(self):
        df = _bdate_series()
        df.iloc[5, df.columns.get_loc("Close")] = 0.0
        result = check_sanity(df, "SPY")
        assert not result.ok
        assert result.severity == "block"
        assert result.detail["n_bad"] == 1

    def test_negative_close(self):
        df = _bdate_series()
        df.iloc[5, df.columns.get_loc("Close")] = -12.5
        result = check_sanity(df, "SPY")
        assert not result.ok
        assert result.detail["n_bad"] == 1

    def test_missing_price_field(self):
        df = _bdate_series().drop(columns=["Close"])
        result = check_sanity(df, "SPY")
        assert not result.ok
        assert result.severity == "block"


# ── Gate 3: staleness (per-series, distinct from artifact freshness) ───────


class TestCheckStaleness:
    def test_fresh_series(self):
        # Series' last row is a Friday; as_of is the following Monday
        # (last_closed_trading_day would resolve to that Friday) — within
        # the default max_age_trading_days=1 floor.
        dates = pd.bdate_range("2026-06-01", periods=10)  # ends Mon 2026-06-15... adjust
        df = _ohlcv(dates, [100 + i for i in range(10)])
        last_date = dates[-1].date()
        result = check_staleness(df, "SPY", as_of=last_date)
        assert result.ok

    def test_stale_series(self):
        dates = pd.bdate_range("2026-06-01", periods=10)
        df = _ohlcv(dates, [100 + i for i in range(10)])
        # as_of is 10 business days after the series' last row — well past
        # the 1-trading-day default floor.
        as_of = (dates[-1] + pd.tseries.offsets.BDay(10)).date()
        result = check_staleness(df, "SPY", as_of=as_of)
        assert not result.ok
        assert result.severity == "warn"
        assert result.detail["age_trading_days"] > 1

    def test_empty_series(self):
        df = _ohlcv([], [])
        result = check_staleness(df, "SPY", as_of=date(2026, 6, 15))
        assert not result.ok


# ── Gate 4: continuity (calendar-aware, the config#1276 gap check) ─────────


class TestCheckContinuity:
    def test_clean_series_no_gaps(self):
        df = _bdate_series(start="2026-06-01", periods=15)
        result = check_continuity(df, "SPY")
        assert result.ok

    def test_holiday_adjacent_gap_is_not_flagged(self):
        # 2026-07-03 (Fri, July 4th observed) is an NYSE holiday per
        # krepis.trading_calendar.NYSE_HOLIDAYS. A series with rows on the
        # Thursday before and the Monday after (skipping Fri 7/3 + the
        # weekend) has NO missing trading day and must read clean.
        dates = [
            "2026-06-30", "2026-07-01", "2026-07-02",  # Tue/Wed/Thu
            # Fri 2026-07-03 = holiday (skipped, correct)
            # Sat/Sun weekend (skipped, correct)
            "2026-07-06",  # Mon
            "2026-07-07",  # Tue
        ]
        df = _ohlcv(dates, [100, 101, 102, 103, 104])
        result = check_continuity(df, "SPY")
        assert result.ok, result.reason

    def test_genuine_missing_trading_day_is_flagged(self):
        # Same window as above, but ALSO drops Wed 2026-07-01 — a real
        # trading day with no holiday excuse. This is the shape of the
        # 2026-06-24 gap referenced in the parent epic.
        dates = [
            "2026-06-30",  # Tue
            # Wed 2026-07-01 MISSING — genuine gap, no holiday excuse
            "2026-07-02",  # Thu
            "2026-07-06",  # Mon (Fri 7/3 is a holiday, correctly absent)
        ]
        df = _ohlcv(dates, [100, 101, 102])
        result = check_continuity(df, "SPY")
        assert not result.ok
        assert result.severity == "warn"
        assert "2026-07-01" in result.detail["missing_dates"]

    def test_single_row_series_ok(self):
        df = _ohlcv(["2026-06-01"], [100])
        result = check_continuity(df, "SPY")
        assert result.ok


# ── Gate 5: outlier (vol-scaled, not a fixed percentage) ───────────────────


class TestCheckOutlier:
    def test_no_outlier_in_stable_series(self):
        df = _bdate_series(daily_vol=0.008)
        result = check_outlier(df, "SPY")
        assert result.ok, result.reason

    def test_large_move_flagged_in_low_vol_series(self):
        # Low daily vol (~0.1% moves, seeded random walk so the trailing
        # std is small-but-non-degenerate) then one huge 20% jump — should
        # dwarf a vol-scaled threshold built from the quiet trailing window.
        import random

        rng = random.Random(11)
        dates = pd.bdate_range("2026-06-01", periods=25)
        closes = [100.0]
        for _ in range(23):
            closes.append(closes[-1] * (1.0 + rng.gauss(0.0, 0.001)))
        closes.append(closes[-1] * 1.20)  # 20% jump on the last day
        df = _ohlcv(dates, closes)
        result = check_outlier(df, "SPY", vol_window=20, min_observations=5)
        assert not result.ok
        assert result.severity == "warn"
        assert result.detail["n_violations"] >= 1

    def test_same_absolute_move_not_flagged_in_high_vol_series(self):
        # Same 20% final-day jump, but the trailing window is itself
        # high-vol (large daily swings) — a vol-scaled gate must NOT fire
        # here, proving the threshold is dynamic rather than a fixed %.
        import itertools

        dates = pd.bdate_range("2026-06-01", periods=25)
        closes = [100.0]
        swings = itertools.cycle([1.15, 0.87, 1.12, 0.90, 1.18, 0.85])
        for s in itertools.islice(swings, 23):
            closes.append(closes[-1] * s)
        closes.append(closes[-1] * 1.20)  # same 20% jump as the low-vol case
        df = _ohlcv(dates, closes)
        result = check_outlier(df, "SPY", vol_window=20, min_observations=5)
        assert result.ok, result.reason

    def test_insufficient_history_abstains(self):
        df = _bdate_series(periods=3)
        result = check_outlier(df, "SPY", min_observations=5)
        assert result.ok
        assert "insufficient" in result.reason


# ── Gate 6: calendar-monotonic ──────────────────────────────────────────────


class TestCheckCalendarMonotonic:
    def test_happy_path(self):
        df = _bdate_series()
        result = check_calendar_monotonic(df, "SPY")
        assert result.ok

    def test_duplicate_date(self):
        dates = ["2026-06-01", "2026-06-02", "2026-06-02", "2026-06-03"]
        df = _ohlcv(dates, [100, 101, 101.5, 102])
        result = check_calendar_monotonic(df, "SPY")
        assert not result.ok
        assert result.severity == "block"
        assert "2026-06-02" in result.detail["duplicates"]

    def test_out_of_order_dates(self):
        dates = ["2026-06-01", "2026-06-05", "2026-06-03"]
        df = _ohlcv(dates, [100, 101, 102])
        result = check_calendar_monotonic(df, "SPY")
        assert not result.ok
        assert result.severity == "block"
        assert result.detail["out_of_order"]

    def test_empty_series_ok(self):
        df = _ohlcv([], [])
        result = check_calendar_monotonic(df, "SPY")
        assert result.ok


# ── Orchestration: validate_series + quarantine_decision ───────────────────


class TestValidateSeries:
    def test_all_gates_pass_on_clean_series(self):
        df = _bdate_series(start="2026-06-01", periods=30)
        as_of = df.index[-1].date()
        report = validate_series(df, "SPY", as_of=as_of)
        assert report.passed
        assert {r.gate for r in report.results} == set(GATE_NAMES)

    def test_report_captures_multiple_failures(self):
        dates = ["2026-06-01", "2026-06-02", "2026-06-02"]  # duplicate
        df = _ohlcv(dates, [100, -5, 101])  # + negative close
        report = validate_series(df, "SPY", run_gates=("sanity", "calendar_monotonic"))
        assert not report.passed
        failing_gates = {r.gate for r in report.failing}
        assert failing_gates == {"sanity", "calendar_monotonic"}
        assert "FAILED" in report.summary

    def test_staleness_skipped_without_as_of(self):
        df = _bdate_series()
        report = validate_series(df, "SPY", run_gates=("staleness",))
        result = report.results[0]
        assert result.ok
        assert "as_of not supplied" in result.reason


class TestQuarantineDecision:
    def test_block_gate_failure_quarantines(self):
        df = _bdate_series()
        df.iloc[0, df.columns.get_loc("Close")] = 0.0  # sanity block-default
        report = validate_series(df, "SPY", run_gates=("sanity",))
        decision = quarantine_decision(report)
        assert decision.quarantine
        assert decision.alarm
        assert "sanity" in decision.blocking_gates

    def test_warn_gate_failure_alarms_but_does_not_quarantine(self):
        # continuity is warn-by-default in DEFAULT_BLOCK_GATES.
        dates = ["2026-06-30", "2026-07-02", "2026-07-06"]  # missing 07-01
        df = _ohlcv(dates, [100, 101, 102])
        report = validate_series(df, "SPY", run_gates=("continuity",))
        decision = quarantine_decision(report)
        assert not decision.quarantine
        assert decision.alarm
        assert "continuity" in decision.warning_gates

    def test_all_pass_no_alarm(self):
        df = _bdate_series(start="2026-06-01", periods=30)
        as_of = df.index[-1].date()
        report = validate_series(df, "SPY", as_of=as_of)
        decision = quarantine_decision(report)
        assert not decision.quarantine
        assert not decision.alarm

    def test_caller_can_override_block_set(self):
        dates = ["2026-06-30", "2026-07-02", "2026-07-06"]
        df = _ohlcv(dates, [100, 101, 102])
        report = validate_series(df, "SPY", run_gates=("continuity",))
        decision = quarantine_decision(
            report, block_gates=DEFAULT_BLOCK_GATES | {"continuity"}
        )
        assert decision.quarantine
        assert "continuity" in decision.blocking_gates
