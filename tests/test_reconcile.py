"""Tests for nousergon_lib.reconcile — the data-migration parity gate."""

from __future__ import annotations

import pytest

from nousergon_lib.reconcile import reconcile_frame_dicts


def _frames(pd):
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    a = {
        "AAA": pd.DataFrame({"Close": [1.0, 2, 3, 4, 5]}, index=idx),
        "BBB": pd.DataFrame({"Close": [10.0, 11, 12, 13, 14]}, index=idx),
    }
    return idx, a


def test_identical_stores_pass():
    pd = pytest.importorskip("pandas")
    _, a = _frames(pd)
    b = {k: v.copy() for k, v in a.items()}
    rep = reconcile_frame_dicts(a, b)
    assert rep.passed
    assert rep.ticker_sets_match
    assert rep.rowcounts_match
    assert rep.max_abs_value_delta == 0.0
    assert rep.n_cells_over_epsilon == 0
    assert "PASS" in rep.summary()
    assert rep.as_metrics()["passed"] is True


def test_value_delta_over_epsilon_fails_and_locates_worst_cell():
    pd = pytest.importorskip("pandas")
    idx, a = _frames(pd)
    b = {k: v.copy() for k, v in a.items()}
    b["BBB"].loc[idx[2], "Close"] = 12.5  # +0.5 vs 12.0
    rep = reconcile_frame_dicts(a, b, epsilon=1e-6)
    assert not rep.passed
    assert rep.n_cells_over_epsilon == 1
    assert rep.max_abs_value_delta == pytest.approx(0.5)
    assert rep.worst_cell[0] == "BBB"
    assert rep.worst_cell[1] == "Close"
    assert rep.worst_cell[2].startswith("2025-01-03")


def test_within_epsilon_passes():
    pd = pytest.importorskip("pandas")
    idx, a = _frames(pd)
    b = {k: v.copy() for k, v in a.items()}
    b["AAA"].loc[idx[0], "Close"] += 1e-9  # below default epsilon
    rep = reconcile_frame_dicts(a, b, epsilon=1e-6)
    assert rep.passed


def test_ticker_set_asymmetry_fails_when_required_else_reported():
    pd = pytest.importorskip("pandas")
    _, a = _frames(pd)
    b = {"AAA": a["AAA"].copy()}  # missing BBB
    rep = reconcile_frame_dicts(a, b)
    assert rep.only_in_a == frozenset({"BBB"})
    assert not rep.passed  # require_ticker_match defaults True
    rep2 = reconcile_frame_dicts(a, b, require_ticker_match=False)
    assert rep2.passed  # values agree on the common ticker


def test_rowcount_delta_reported_but_not_fatal_by_default():
    pd = pytest.importorskip("pandas")
    idx, a = _frames(pd)
    b = {k: v.copy() for k, v in a.items()}
    b["AAA"] = b["AAA"].iloc[1:]  # one fewer boundary row, overlap identical
    rep = reconcile_frame_dicts(a, b)
    assert rep.rowcount_deltas == {"AAA": 1}
    assert not rep.rowcounts_match
    assert rep.passed  # overlap agrees; boundary slice is expected
    strict = reconcile_frame_dicts(a, b, require_rowcount_match=True)
    assert not strict.passed


def test_asymmetric_nan_is_a_mismatch():
    pd = pytest.importorskip("pandas")
    idx, a = _frames(pd)
    b = {k: v.copy() for k, v in a.items()}
    b["AAA"].loc[idx[0], "Close"] = float("nan")
    rep = reconcile_frame_dicts(a, b)
    assert not rep.passed
    assert rep.n_cells_over_epsilon == 1


def test_both_nan_is_not_a_mismatch():
    pd = pytest.importorskip("pandas")
    idx, a = _frames(pd)
    a["AAA"].loc[idx[0], "Close"] = float("nan")
    b = {k: v.copy() for k, v in a.items()}
    rep = reconcile_frame_dicts(a, b)
    assert rep.passed
    assert rep.n_cells_over_epsilon == 0


def test_compares_only_on_date_intersection():
    pd = pytest.importorskip("pandas")
    a = {"AAA": pd.DataFrame(
        {"Close": [1.0, 2, 3]}, index=pd.date_range("2025-01-01", periods=3))}
    # b shifted: overlaps on 2025-01-02..03 with identical values
    b = {"AAA": pd.DataFrame(
        {"Close": [2.0, 3, 4]}, index=pd.date_range("2025-01-02", periods=3))}
    rep = reconcile_frame_dicts(a, b)
    assert rep.passed  # only the 2 overlapping dates compared, and they agree
    assert rep.n_cells_compared == 2


def test_as_metrics_is_json_able():
    pd = pytest.importorskip("pandas")
    import json

    _, a = _frames(pd)
    b = {k: v.copy() for k, v in a.items()}
    rep = reconcile_frame_dicts(a, b)
    json.dumps(rep.as_metrics())  # must not raise
