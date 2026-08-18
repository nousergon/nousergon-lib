"""Golden corpus for the fleet-wide risk-ratio drift tests (config-I7597).

`riskstats.sharpe_ratio` / `sortino_ratio` / `downside_deviation` are the fleet's
only implementation of these statistics. Consumers that cannot call them
directly (a vectorized 2-D sweep kernel, for one) keep a drift test that pins
their own output against the same series listed here.

This module is the LIBRARY end of that contract: it pins the library's answer on
every series, including the degenerate ones (zero volatility, no downside days,
n < 2), from values written out by hand from the definition rather than from a
lib call. A library change that moves any of these numbers fails here first, and
the consumer drift tests then say who else it moved.

Keep `CORPUS` byte-identical to the copies in:
  crucible-backtester/tests/test_riskstats_drift.py
  crucible-predictor/tests/test_riskstats_drift.py
  crucible-dashboard/tests/test_riskstats_drift.py
  crucible-evaluator/tests/test_riskstats_drift.py
"""

from __future__ import annotations

import math

import pytest

from nousergon_lib.quant import riskstats

# name -> series. Fixed for all time; append, never edit.
CORPUS: dict[str, list[float]] = {
    "mixed": [0.01, -0.02, 0.015, -0.005, 0.03, -0.01, 0.0, 0.02, -0.03, 0.005],
    "all_positive": [0.01, 0.02, 0.005, 0.03, 0.015],
    "all_negative": [-0.01, -0.02, -0.005, -0.04],
    "all_zero": [0.0, 0.0, 0.0, 0.0, 0.0],
    "zero_vol_positive": [0.01] * 8,
    "zero_vol_negative": [-0.01] * 8,
    "two_obs": [0.01, -0.01],
    "single_obs": [0.02],
    "empty": [],
    "tiny_downside": [0.01, 0.02, 0.03, -1e-9],
}


def _ref_sharpe(r: list[float], ppy: int = 252) -> float | None:
    """Sharpe written out from the definition — no lib call."""
    if len(r) < 2:
        return None
    mean = sum(r) / len(r)
    var = sum((x - mean) ** 2 for x in r) / (len(r) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return None
    return (mean / sd) * math.sqrt(ppy)


def _ref_dd_full(r: list[float], target: float = 0.0) -> float | None:
    """Downside deviation, n-denominator, from the definition — no lib call."""
    if len(r) < 2:
        return None
    return math.sqrt(sum(min(0.0, x - target) ** 2 for x in r) / len(r))


def _ref_dd_downside(r: list[float], target: float = 0.0) -> float | None:
    """Downside deviation, n_down-denominator, from the definition."""
    if len(r) < 2:
        return None
    short = [x - target for x in r if x < target]
    if not short:
        return None
    return math.sqrt(sum(d * d for d in short) / len(short))


def _ref_sortino(r: list[float], ppy: int = 252, denominator: str = "full") -> float | None:
    if len(r) < 2:
        return None
    mean = sum(r) / len(r)
    dd = _ref_dd_full(r) if denominator == "full" else _ref_dd_downside(r)
    if not dd:
        return None
    return (mean / dd) * math.sqrt(ppy)


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_sharpe_matches_definition(name: str) -> None:
    r = CORPUS[name]
    got, want = riskstats.sharpe_ratio(r), _ref_sharpe(r)
    if want is None:
        assert got is None, f"{name}: expected undefined, got {got}"
    else:
        assert got == pytest.approx(want, rel=1e-12, abs=1e-12), name


@pytest.mark.parametrize("name", sorted(CORPUS))
@pytest.mark.parametrize("denominator", ["full", "downside"])
def test_downside_deviation_matches_definition(name: str, denominator: str) -> None:
    r = CORPUS[name]
    got = riskstats.downside_deviation(r, denominator=denominator)
    want = _ref_dd_full(r) if denominator == "full" else _ref_dd_downside(r)
    if want is None:
        assert got is None, f"{name}/{denominator}: expected None, got {got}"
    else:
        assert got == pytest.approx(want, rel=1e-12, abs=1e-15), f"{name}/{denominator}"


@pytest.mark.parametrize("name", sorted(CORPUS))
@pytest.mark.parametrize("denominator", ["full", "downside"])
def test_sortino_matches_definition(name: str, denominator: str) -> None:
    r = CORPUS[name]
    got = riskstats.sortino_ratio(r, denominator=denominator)
    want = _ref_sortino(r, denominator=denominator)
    if want is None:
        assert got is None, f"{name}/{denominator}: expected undefined, got {got}"
    else:
        assert got == pytest.approx(want, rel=1e-12, abs=1e-12), f"{name}/{denominator}"


def test_degenerate_cases_are_undefined_not_zero() -> None:
    """The load-bearing sentinels: undefined is None, never a measured 0.0."""
    assert riskstats.sharpe_ratio(CORPUS["zero_vol_positive"]) is None
    assert riskstats.sortino_ratio(CORPUS["zero_vol_positive"]) is None
    assert riskstats.sortino_ratio(CORPUS["all_positive"]) is None
    assert riskstats.sharpe_ratio(CORPUS["single_obs"]) is None
    assert riskstats.sharpe_ratio(CORPUS["empty"]) is None
    assert riskstats.downside_deviation(CORPUS["single_obs"]) is None
    # n-denominator: no downside days is a genuine zero deviation...
    assert riskstats.downside_deviation(CORPUS["all_positive"]) == 0.0
    # ...but the n_down variant has no observations at all — undefined.
    assert riskstats.downside_deviation(CORPUS["all_positive"], denominator="downside") is None


def test_downside_variant_understates_sortino_by_sqrt_n_over_ndown() -> None:
    """The exact size of the divergence the "downside" variant carries."""
    r = CORPUS["mixed"]
    n = len(r)
    n_down = sum(1 for x in r if x < 0)
    full = riskstats.sortino_ratio(r, denominator="full")
    down = riskstats.sortino_ratio(r, denominator="downside")
    assert full is not None and down is not None
    assert full / down == pytest.approx(math.sqrt(n / n_down), rel=1e-12)


def test_unknown_denominator_raises() -> None:
    with pytest.raises(ValueError, match="denominator must be"):
        riskstats.downside_deviation(CORPUS["mixed"], denominator="n_down")
