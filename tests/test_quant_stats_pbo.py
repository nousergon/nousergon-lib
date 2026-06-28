"""Tests for nousergon_lib.quant.stats.pbo — CSCV Probability of Backtest
Overfitting (Bailey, Borwein, López de Prado & Zhu 2014).

Pins (mirrors the predictor's L4582 cscv_pbo unit tests, the source impl):
  1. A spec that dominates at every split → PBO 0 (winner always generalizes).
  2. A spec that alternates around a constant rival → PBO 0.5.
  3. <2 specs or <min_splits clean rows → status="insufficient" (honest N/A).
  4. Non-finite rows are dropped, not fabricated.
  5. spec_ids label the selected_counts; default ids are positional.
"""
from __future__ import annotations

import pytest

# quant.stats is the [quant-stats] extra. Skip cleanly when deps are absent.
np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from nousergon_lib.quant.stats.pbo import cscv_pbo


def test_dominant_spec_is_zero():
    # Spec 1 beats spec 0 at EVERY split → the IS pick is always spec 1 and it
    # always ranks top OOS → PBO 0.
    matrix = [[0.01, 0.05]] * 8
    out = cscv_pbo(matrix, spec_ids=["a", "b"])
    assert out["status"] == "ok"
    assert out["pbo"] == 0.0
    assert out["selected_counts"] == {"b": 8}
    assert out["n_splits"] == 8
    assert out["n_specs"] == 2
    assert out["spec_ids"] == ["a", "b"]


def test_alternating_spec_is_half():
    # Spec A alternates 0/1 around spec B's constant 0.4: whenever the held-out
    # split is one of A's bad rows, the IS mean (over the good rows) still picks
    # A and it underperforms OOS → half the splits are mistakes.
    a = [0.0, 1.0] * 4
    b = [0.4] * 8
    out = cscv_pbo(list(zip(a, b)), spec_ids=["a", "b"])
    assert out["status"] == "ok"
    assert out["pbo"] == pytest.approx(0.5)


def test_insufficient_specs_and_splits():
    one_spec = cscv_pbo([[0.1]] * 8)
    assert one_spec["status"] == "insufficient"
    assert "needs >=2" in one_spec["reason"]
    assert np.isnan(one_spec["pbo"])

    few_rows = cscv_pbo([[0.1, 0.2]] * 3)
    assert few_rows["status"] == "insufficient"
    assert "min_splits" in few_rows["reason"]
    assert np.isnan(few_rows["pbo"])


def test_drops_nonfinite_rows():
    rows = [[0.01, 0.05]] * 6 + [[float("nan"), 0.05], [0.01, float("inf")]]
    out = cscv_pbo(rows)
    assert out["status"] == "ok"
    assert out["n_splits"] == 6  # the 2 dirty rows dropped, not fabricated


def test_default_spec_ids_are_positional():
    out = cscv_pbo([[0.01, 0.05, 0.02]] * 5)
    assert out["status"] == "ok"
    assert out["spec_ids"] == [0, 1, 2]
    # spec at column index 1 dominates → all selections land on it.
    assert out["selected_counts"] == {1: 5}


def test_min_splits_override_allows_fewer_rows():
    out = cscv_pbo([[0.01, 0.05]] * 2, min_splits=2)
    assert out["status"] == "ok"
    assert out["n_splits"] == 2


def test_mean_logit_present_on_ok():
    out = cscv_pbo([[0.01, 0.05]] * 8, spec_ids=["a", "b"])
    assert "mean_logit" in out
    assert np.isfinite(out["mean_logit"])
