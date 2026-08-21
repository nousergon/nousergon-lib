"""alpha-engine-config-I7661 — units, source, and input-freshness for regime_sortino.

Measured 2026-08-18/21 against live state:

* ``s3://alpha-engine-research/regime/stratified_sortino/latest.json``
  (run_id 2608151626, trading_day 2026-08-14) publishes
  ``caution``/10d ``mean_log_alpha = -6.436916136981`` and
  ``annualized_sortino = -1.9051784434380492`` — computed by feeding 2dp
  PERCENT points into ``log(1 + r)``, which expects a decimal fraction, with a
  clip of ``1 + r`` to ``1e-9`` turning every pick worse than -1 percent point
  into ``log_alpha ≈ -20.7``.
* ``research.db``: only 34 rows carry a paired ``return_10d`` +
  ``spy_10d_return``, all dated 2026-03-04..03-13, 30 of them in the
  ``caution`` stratum — itself a regime label retired by the 3-class taxonomy.
  The long-format ``score_performance_outcomes`` store carries horizons
  {1,3,5,10,15,21} through 2026-07-10 in DECIMALS, with ``log_alpha``
  populated on 534/534 rows at the primary horizon 21 and none at all at 30.
* The four weekly artifacts named in the issue are NOT byte-identical (their
  ETags differ) but are identical net of ``run_id``/date fields — every metric
  is the same number four weeks running.
"""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from nousergon_lib.quant.horizons import DEFAULT_POLICY
from nousergon_lib.quant.stats.regime_sortino import (
    STATUS_OK,
    STATUS_UNMEASURABLE,
    SUPPORTED_HORIZONS,
    InputWindow,
    ReturnUnits,
    ReturnUnitsError,
    _arithmetic_to_log_alpha,
    assess_input_freshness,
    input_window,
    stratified_sortino_by_regime,
)

_FRACTION = ReturnUnits.FRACTION
_PERCENT = ReturnUnits.PERCENT


# ── 1. The horizons are policy-derived, not the retired pair ────────────────


def test_supported_horizons_track_the_fleet_policy():
    assert SUPPORTED_HORIZONS == DEFAULT_POLICY.all_horizons == (21, 5)
    assert 10 not in SUPPORTED_HORIZONS
    assert 30 not in SUPPORTED_HORIZONS


# ── 2. The units bug ────────────────────────────────────────────────────────


class TestUnits:
    def test_percent_points_declared_as_fractions_raise(self):
        """The live defect, exactly: 5.55 means 5.55%, not +555%."""
        with pytest.raises(ReturnUnitsError, match="most likely 'percent'"):
            _arithmetic_to_log_alpha(
                pd.Series([5.55, 3.20]), pd.Series([1.10, 1.10]),
                units=_FRACTION,
            )

    def test_a_small_percent_column_is_caught_by_the_median_not_the_max(self):
        """The max-only bound misses this: a 5-day return of +2.4pp reads as a
        merely-enormous fraction (+240%), inside ±5. The MEDIAN says percent."""
        with pytest.raises(ReturnUnitsError, match="MEDIAN absolute"):
            _arithmetic_to_log_alpha(
                pd.Series([2.4, 3.1, 1.9, 2.8]), pd.Series([1.1] * 4),
                units=_FRACTION,
            )

    def test_percent_declared_correctly_converts(self):
        out = _arithmetic_to_log_alpha(
            pd.Series([5.55]), pd.Series([1.10]), units=_PERCENT,
        )
        expected = math.log1p(0.0555) - math.log1p(0.0110)
        assert out.iloc[0] == pytest.approx(expected, abs=1e-12)

    def test_a_pick_at_or_below_minus_one_raises_instead_of_clipping(self):
        """The 1e-9 clip turned an UNDEFINED log return into a finite -20.7 and
        is the direct cause of the published mean_log_alpha of -6.44."""
        with pytest.raises(ReturnUnitsError, match="-100% or worse"):
            _arithmetic_to_log_alpha(
                pd.Series([-1.0]), pd.Series([0.01]), units=_FRACTION,
            )

    def test_the_exact_published_number_is_no_longer_reachable(self):
        """Guard-fails-without-the-fix: the pre-fix code produced roughly
        -20.7 per row here (clip to 1e-9); the fixed code refuses the input."""
        percent_points = pd.Series([-3.2, -5.5, -2.0])
        spy = pd.Series([1.1, 1.1, 1.1])
        with pytest.raises(ReturnUnitsError):
            _arithmetic_to_log_alpha(percent_points, spy, units=_FRACTION)
        # Declared honestly, the same rows are ordinary small negative alphas.
        honest = _arithmetic_to_log_alpha(percent_points, spy, units=_PERCENT)
        assert honest.min() > -0.20
        assert honest.max() < 0.0

    def test_units_is_required_on_the_public_entry(self):
        df = pd.DataFrame([{"market_regime": "bull", "return_21d": 0.05,
                            "spy_21d_return": 0.02}])
        with pytest.raises(TypeError):
            stratified_sortino_by_regime(df)  # type: ignore[call-arg]

    def test_nan_propagates_rather_than_raising(self):
        out = _arithmetic_to_log_alpha(
            pd.Series([float("nan"), 0.05]), pd.Series([0.02, 0.02]),
            units=_FRACTION,
        )
        assert math.isnan(out.iloc[0])
        assert out.iloc[1] == pytest.approx(math.log1p(0.05) - math.log1p(0.02))


# ── 3. The canonical log-alpha source ───────────────────────────────────────


class TestCanonicalLogAlphaSource:
    def _df(self, *, with_log_alpha: bool) -> pd.DataFrame:
        rows = []
        for i in range(25):
            r = 0.02 + 0.001 * i
            spy = 0.01
            row = {
                "market_regime": "bull",
                "score_date": "2026-07-10",
                # Wide columns in the PERCENT convention attach_outcomes emits.
                "return_21d": round(r * 100, 2),
                "spy_21d_return": round(spy * 100, 2),
                "return_5d": round(r * 100, 2),
                "spy_5d_return": round(spy * 100, 2),
            }
            if with_log_alpha:
                # Canonical store value: full-precision decimal log-alpha.
                row["log_alpha_21d"] = math.log1p(r) - math.log1p(spy)
            rows.append(row)
        return pd.DataFrame(rows)

    def test_primary_horizon_reads_the_canonical_column_not_the_wide_one(self):
        """Declared PERCENT would still be lossy (2dp rounding). The canonical
        column wins, and the result is exact."""
        df = self._df(with_log_alpha=True)
        strata = stratified_sortino_by_regime(
            df, units=_PERCENT, min_picks_per_stratum=5,
        )
        primary = next(s for s in strata if s.horizon_days == 21)
        expected = float(
            (df["log_alpha_21d"]).mean()
        )
        assert primary.n_picks == 25
        assert primary.mean_log_alpha == pytest.approx(expected, abs=1e-15)

    def test_diagnostic_horizon_falls_back_to_the_declared_units(self):
        """The long store publishes no log_alpha at the diagnostic horizon, so
        that horizon must still convert — under a stated convention."""
        df = self._df(with_log_alpha=True)
        strata = stratified_sortino_by_regime(
            df, units=_PERCENT, min_picks_per_stratum=5,
        )
        diag = next(s for s in strata if s.horizon_days == 5)
        assert diag.n_picks == 25
        assert diag.mean_log_alpha is not None
        assert -0.5 < diag.mean_log_alpha < 0.5

    def test_wrong_units_on_the_fallback_horizon_raise(self):
        df = self._df(with_log_alpha=True)
        with pytest.raises(ReturnUnitsError):
            stratified_sortino_by_regime(
                df, units=_FRACTION, min_picks_per_stratum=5,
            )

    def test_no_canonical_column_uses_the_arithmetic_path(self):
        df = self._df(with_log_alpha=False)
        strata = stratified_sortino_by_regime(
            df, units=_PERCENT, min_picks_per_stratum=5,
        )
        primary = next(s for s in strata if s.horizon_days == 21)
        assert primary.n_picks == 25
        assert primary.mean_log_alpha is not None


# ── 4. Freshness on the INPUTS, not the write time ──────────────────────────


class TestInputFreshness:
    def test_input_window_reports_the_measured_span(self):
        df = pd.DataFrame({"score_date": ["2026-03-04", "2026-03-13", None]})
        w = input_window(df)
        assert w.min_score_date == "2026-03-04"
        assert w.max_score_date == "2026-03-13"
        assert w.n_rows == 3

    def test_the_frozen_march_window_is_unmeasurable_against_august(self):
        """The live artifact's actual inputs: 2026-03-04..03-13, published
        2026-08-14. Four weekly runs in a row and nothing said a word."""
        w = InputWindow("2026-03-04", "2026-03-13", 34)
        status, reason = assess_input_freshness(
            w, trading_day="2026-08-14", horizon_days=21,
        )
        assert status == STATUS_UNMEASURABLE
        assert "2026-03-13" in reason
        assert "154" in reason  # calendar days between the two dates

    def test_a_healthy_lagged_window_is_ok(self):
        """A horizon-21 outcome cannot resolve for 21 trading days, so a
        correct run's newest input is ALREADY ~a month old. Flagging that
        would be the mirror defect."""
        w = InputWindow("2026-06-01", "2026-07-10", 534)
        status, reason = assess_input_freshness(
            w, trading_day="2026-08-14", horizon_days=21,
        )
        assert status == STATUS_OK
        assert reason == ""

    def test_empty_input_is_unmeasurable_not_an_empty_success(self):
        status, reason = assess_input_freshness(
            InputWindow(None, None, 0), trading_day="2026-08-14", horizon_days=21,
        )
        assert status == STATUS_UNMEASURABLE
        assert "nothing was measured" in reason

    def test_shorter_horizon_gets_a_shorter_allowance(self):
        w = InputWindow("2026-06-01", "2026-07-10", 100)
        assert assess_input_freshness(
            w, trading_day="2026-08-14", horizon_days=5,
        )[0] == STATUS_UNMEASURABLE


def test_decimals_declared_as_percent_are_caught_by_the_lower_median_bound():
    """The mirror direction: a decimal column read as percent divides by 100
    and understates every alpha silently — no upper bound can see it."""
    decimals = pd.Series([0.03, -0.02, 0.045, -0.031, 0.028] * 4)
    with pytest.raises(ReturnUnitsError, match="bps"):
        _arithmetic_to_log_alpha(
            decimals, pd.Series([0.01] * 20), units=_PERCENT,
        )


def test_the_lower_bound_needs_a_distribution_to_judge():
    """Fewer rows than a distribution: no verdict, rather than a false one."""
    out = _arithmetic_to_log_alpha(
        pd.Series([0.0003, -0.0002]), pd.Series([0.0001, 0.0001]),
        units=_FRACTION,
    )
    assert len(out) == 2
