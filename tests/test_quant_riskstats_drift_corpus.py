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
import pathlib

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


def _ref_sortino(r: list[float], ppy: int = 252) -> float | None:
    if len(r) < 2:
        return None
    mean = sum(r) / len(r)
    dd = _ref_dd_full(r)
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
def test_downside_deviation_matches_definition(name: str) -> None:
    r = CORPUS[name]
    got = riskstats.downside_deviation(r)
    want = _ref_dd_full(r)
    if want is None:
        assert got is None, f"{name}: expected None, got {got}"
    else:
        assert got == pytest.approx(want, rel=1e-12, abs=1e-15), name


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_sortino_matches_definition(name: str) -> None:
    r = CORPUS[name]
    got = riskstats.sortino_ratio(r)
    want = _ref_sortino(r)
    if want is None:
        assert got is None, f"{name}: expected undefined, got {got}"
    else:
        assert got == pytest.approx(want, rel=1e-12, abs=1e-12), name


def test_degenerate_cases_are_undefined_not_zero() -> None:
    """The load-bearing sentinels: undefined is None, never a measured 0.0."""
    assert riskstats.sharpe_ratio(CORPUS["zero_vol_positive"]) is None
    assert riskstats.sortino_ratio(CORPUS["zero_vol_positive"]) is None
    assert riskstats.sortino_ratio(CORPUS["all_positive"]) is None
    assert riskstats.sharpe_ratio(CORPUS["single_obs"]) is None
    assert riskstats.sharpe_ratio(CORPUS["empty"]) is None
    assert riskstats.downside_deviation(CORPUS["single_obs"]) is None
    # n-denominator: no downside days is a genuine zero deviation, measured...
    assert riskstats.downside_deviation(CORPUS["all_positive"]) == 0.0
    # ...and a ratio over a zero denominator is still undefined, never infinite.
    assert riskstats.sortino_ratio(CORPUS["all_positive"]) is None


def test_retired_downside_denominator_is_rejected() -> None:
    """The n_down convention cannot return by argument (config-I7618 #5).

    A source grep only guards the repo it runs in; this guards every caller in
    every repo, because the arithmetic is gone and the name raises.
    """
    retired = "down" + "side"  # spelled in pieces: the source guard below
    for fn in (riskstats.downside_deviation, riskstats.sortino_ratio):
        with pytest.raises(ValueError, match="retired"):
            fn(CORPUS["mixed"], denominator=retired)


def test_unknown_denominator_raises() -> None:
    with pytest.raises(ValueError, match="denominator must be"):
        riskstats.downside_deviation(CORPUS["mixed"], denominator="n_down")


def test_no_file_in_this_repo_asks_for_the_downside_denominator() -> None:
    """config-I7618 deliverable 5, this repo's half.

    Mirrors ``crucible-backtester``'s guard of the same name (PR #699), with one
    difference: here NOTHING is exempt, tests included — which is why
    ``test_retired_downside_denominator_is_rejected`` above spells the retired
    name in pieces. There is no longer any legitimate use of the literal
    anywhere in this repo, so the guard needs no carve-out, and the last place
    the n_down arithmetic could have hidden is the library that used to own it.

    A per-call-site numeric test only catches the sites it already knows about;
    this catches the NEXT one. The fleet-wide half of the guard is the
    ``ValueError`` itself: the arithmetic is gone, so a call site in ANY repo
    fails loudly rather than silently computing the retired convention.
    """
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    skip_dirs = {".git", ".venv", "venv", "build", "dist", "__pycache__",
                 "node_modules", ".worktrees"}
    # Spelled in pieces so this detector is not its own first offender.
    retired = "down" + "side"
    needles = (f'denominator="{retired}"', f"denominator='{retired}'")
    offenders = []
    for path in repo_root.rglob("*.py"):
        if any(part in skip_dirs for part in path.relative_to(repo_root).parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue  # a comment may explain the variant it does not use
            if any(needle in line for needle in needles):
                offenders.append(f"{path.relative_to(repo_root)}:{i}")
    assert not offenders, (
        f'denominator="{retired}" is the retired n_down convention '
        "config-I7271 ruled against and config-I7618/-I7638 removed. Use "
        'denominator="full". Offenders: ' + ", ".join(offenders)
    )
