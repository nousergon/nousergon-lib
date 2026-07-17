"""Local-lmdb integration tests for the bitemporal schema helpers
(config#2459): ``open_preliminary_lib``, ``read_settled_only``,
``write_correction``, and the bitemporal column constants.

Unlike ``tests/test_arcticdb.py`` (stub-based, no real arcticdb needed),
these exercise a REAL ArcticDB instance backed by a local ``lmdb://`` file
URI (no S3/AWS credentials, no network) — this is the closest thing to a
production round-trip we can validate from a sandbox with no path to the
live ``alpha-engine-research`` bucket. ``pytest.importorskip("arcticdb")``
keeps this file a no-op skip (not a failure) on any environment that
doesn't have the ``[arcticdb]`` extra installed, matching the rest of the
suite's precedent for arcticdb-optional tests (see
``nousergon-data/tests/test_prune_delisted_tickers.py``'s
``_real_arctic`` helper, the direct precedent this file follows).
"""

from __future__ import annotations

import pytest

adb = pytest.importorskip("arcticdb")
pd = pytest.importorskip("pandas")

from nousergon_lib import arcticdb as ae_arctic  # noqa: E402


def _local_arctic(tmp_path):
    return adb.Arctic(f"lmdb://{tmp_path}")


def _ohlcv(n=3, start="2026-07-01"):
    idx = pd.date_range(start, periods=n, freq="B", name="date")
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [101.0 + i for i in range(n)],
            "Low": [99.0 + i for i in range(n)],
            "Close": [100.5 + i for i in range(n)],
            "Volume": [1_000_000] * n,
        },
        index=idx,
    )


# ── open_preliminary_lib: physical separation ───────────────────────────────


def test_open_preliminary_lib_is_a_distinct_library_from_universe(tmp_path, monkeypatch):
    """The preliminary library must be a structurally separate ArcticDB
    library from ``universe`` — writing a symbol into one must not make it
    visible from the other. This is the physical-separation guarantee the
    issue asks for ('not just convention')."""
    arctic = _local_arctic(tmp_path)
    monkeypatch.setattr(ae_arctic, "open_arctic", lambda bucket, region=None: arctic)

    prelim_lib = ae_arctic.open_preliminary_lib("any-bucket", create_if_missing=True)
    universe_lib = ae_arctic.open_universe_lib("any-bucket", create_if_missing=True)

    assert prelim_lib.name != universe_lib.name
    assert ae_arctic.PRELIMINARY_LIB != ae_arctic.UNIVERSE_LIB

    prelim_lib.write("AAPL", _ohlcv())
    assert prelim_lib.has_symbol("AAPL")
    assert not universe_lib.has_symbol("AAPL"), (
        "a symbol written to the preliminary library must not leak into "
        "the settled universe library — physical separation, not a flag"
    )


# ── read_settled_only: read-path chokepoint ─────────────────────────────────


def test_read_settled_only_never_reads_preliminary_library(tmp_path, monkeypatch):
    """Even if a caller writes a symbol ONLY to the preliminary library
    (never to universe), read_settled_only must not find it — it hard-codes
    the universe library and offers no way to point it at preliminary."""
    arctic = _local_arctic(tmp_path)
    monkeypatch.setattr(ae_arctic, "open_arctic", lambda bucket, region=None: arctic)

    prelim_lib = ae_arctic.open_preliminary_lib("any-bucket", create_if_missing=True)
    prelim_lib.write("AAPL", _ohlcv())

    # Blind Exception is intentional: the raised type/message legitimately
    # varies by environment — a real arcticdb raises NoSuchSymbolException,
    # while the CI-mocked path fails at library-open — so neither an exception
    # class nor a message `match=` is stable here. The contract under test is
    # simply "read_settled_only refuses to fall through to preliminary data".
    with pytest.raises(Exception):  # noqa: B017 -- see comment above
        ae_arctic.read_settled_only("any-bucket", "AAPL")


def test_read_settled_only_returns_universe_data_when_present(tmp_path, monkeypatch):
    arctic = _local_arctic(tmp_path)
    monkeypatch.setattr(ae_arctic, "open_arctic", lambda bucket, region=None: arctic)

    universe_lib = ae_arctic.open_universe_lib("any-bucket", create_if_missing=True)
    df = _ohlcv()
    universe_lib.write("SPY", df)

    out = ae_arctic.read_settled_only("any-bucket", "SPY")
    pd.testing.assert_frame_equal(out, df, check_freq=False)


def test_read_settled_only_drops_unsettled_rows_when_settled_col_present(
    tmp_path, monkeypatch
):
    """Defense-in-depth: even within the universe library, if a frame
    carries a ``settled`` column, rows marked not-settled must never be
    returned by the chokepoint."""
    arctic = _local_arctic(tmp_path)
    monkeypatch.setattr(ae_arctic, "open_arctic", lambda bucket, region=None: arctic)

    universe_lib = ae_arctic.open_universe_lib("any-bucket", create_if_missing=True)
    df = _ohlcv(n=4)
    df[ae_arctic.SETTLED_COL] = [True, True, False, False]
    universe_lib.write("MSFT", df)

    out = ae_arctic.read_settled_only("any-bucket", "MSFT")
    assert len(out) == 2
    assert out[ae_arctic.SETTLED_COL].all()


def test_read_settled_only_alerts_loudly_when_dropping_unsettled_rows(
    tmp_path, monkeypatch
):
    """An unsettled row inside the SETTLED library is itself a
    data-integrity violation (the physical split is supposed to make
    this impossible) — read_settled_only must not just silently filter
    it out, it must alert via alert_gate_failure so the violation
    surfaces instead of being swallowed."""
    from unittest.mock import MagicMock

    import nousergon_lib.gate_alerts as gate_alerts

    arctic = _local_arctic(tmp_path)
    monkeypatch.setattr(ae_arctic, "open_arctic", lambda bucket, region=None: arctic)

    universe_lib = ae_arctic.open_universe_lib("any-bucket", create_if_missing=True)
    df = _ohlcv(n=3)
    df[ae_arctic.SETTLED_COL] = [True, False, True]
    universe_lib.write("BADROW", df)

    mock_alert = MagicMock()
    monkeypatch.setattr(gate_alerts, "alert_gate_failure", mock_alert)

    out = ae_arctic.read_settled_only("any-bucket", "BADROW")

    assert len(out) == 2
    mock_alert.assert_called_once()
    _, kwargs = mock_alert.call_args
    assert kwargs["layer"] == "L0"
    assert kwargs["series"] == "BADROW"
    assert kwargs["severity"] == "error"
    assert "1 unsettled" in kwargs["detail"]


def test_read_settled_only_no_alert_when_all_rows_settled(tmp_path, monkeypatch):
    """The common case (every row settled, or no settled column at all)
    must NOT fire an alert — only an actual violation should."""
    from unittest.mock import MagicMock

    import nousergon_lib.gate_alerts as gate_alerts

    arctic = _local_arctic(tmp_path)
    monkeypatch.setattr(ae_arctic, "open_arctic", lambda bucket, region=None: arctic)

    universe_lib = ae_arctic.open_universe_lib("any-bucket", create_if_missing=True)
    df = _ohlcv(n=2)
    df[ae_arctic.SETTLED_COL] = [True, True]
    universe_lib.write("GOODROW", df)

    mock_alert = MagicMock()
    monkeypatch.setattr(gate_alerts, "alert_gate_failure", mock_alert)

    ae_arctic.read_settled_only("any-bucket", "GOODROW")

    mock_alert.assert_not_called()


def test_read_settled_only_treats_missing_settled_col_as_all_settled(
    tmp_path, monkeypatch
):
    """Backward-compat: symbols written before this PR (no ``settled``
    column at all) must round-trip through read_settled_only unchanged —
    matches the additive-migration contract (like total_return_close)."""
    arctic = _local_arctic(tmp_path)
    monkeypatch.setattr(ae_arctic, "open_arctic", lambda bucket, region=None: arctic)

    universe_lib = ae_arctic.open_universe_lib("any-bucket", create_if_missing=True)
    df = _ohlcv()
    universe_lib.write("OLD_SYMBOL_NO_BITEMPORAL_COLS", df)

    out = ae_arctic.read_settled_only("any-bucket", "OLD_SYMBOL_NO_BITEMPORAL_COLS")
    pd.testing.assert_frame_equal(out, df, check_freq=False)


def test_read_settled_only_as_of_forwards_to_lib_read(tmp_path, monkeypatch):
    """as_of must reach a bitemporal 'what did we believe as of version N'
    query — pins the parameter forwarding into lib.read."""
    arctic = _local_arctic(tmp_path)
    monkeypatch.setattr(ae_arctic, "open_arctic", lambda bucket, region=None: arctic)

    universe_lib = ae_arctic.open_universe_lib("any-bucket", create_if_missing=True)
    df_v0 = _ohlcv()
    universe_lib.write("SPY", df_v0, prune_previous_versions=False)
    df_v1 = df_v0.copy()
    df_v1.iloc[0, df_v1.columns.get_loc("Close")] = -1.0
    universe_lib.write("SPY", df_v1, prune_previous_versions=False)

    out_v0 = ae_arctic.read_settled_only("any-bucket", "SPY", as_of=0)
    pd.testing.assert_frame_equal(out_v0, df_v0, check_freq=False)

    out_head = ae_arctic.read_settled_only("any-bucket", "SPY")
    pd.testing.assert_frame_equal(out_head, df_v1, check_freq=False)


# ── write_correction: versioned correction records ──────────────────────────


def test_write_correction_does_not_overwrite_prior_version(tmp_path, monkeypatch):
    """The core bitemporal-correction contract: writing a correction must
    NOT destroy the original value — it must still be readable as-of its
    own version."""
    arctic = _local_arctic(tmp_path)
    lib = arctic.get_library("universe", create_if_missing=True)

    original = _ohlcv()
    v0 = lib.write("SPY", original, prune_previous_versions=False)

    corrected = original.copy()
    corrected.iloc[1, corrected.columns.get_loc("Close")] = 12345.0
    v1 = ae_arctic.write_correction(
        lib, "SPY", corrected, reason="bad print corrected by exchange",
        source="polygon",
    )

    assert v1.version == v0.version + 1

    still_original = lib.read("SPY", as_of=v0.version).data
    pd.testing.assert_frame_equal(still_original, original, check_freq=False)

    head = lib.read("SPY").data
    pd.testing.assert_frame_equal(head, corrected, check_freq=False)


def test_write_correction_queryable_as_of_original_timestamp(tmp_path):
    """Bitemporal query: as-of a timestamp taken right after the original
    write (before the correction), the ORIGINAL value must still be what
    comes back — 'what did we believe on date X'."""
    import time

    arctic = _local_arctic(tmp_path)
    lib = arctic.get_library("universe", create_if_missing=True)

    original = _ohlcv()
    v0 = lib.write("SPY", original, prune_previous_versions=False)
    t_after_v0 = pd.Timestamp(v0.timestamp) + pd.Timedelta(milliseconds=1)

    time.sleep(0.05)
    corrected = original.copy()
    corrected.iloc[0, corrected.columns.get_loc("Close")] = -999.0
    ae_arctic.write_correction(
        lib, "SPY", corrected, reason="typo fix", source="ops",
    )

    as_of_before_correction = lib.read("SPY", as_of=t_after_v0).data
    pd.testing.assert_frame_equal(as_of_before_correction, original, check_freq=False)


def test_write_correction_stamps_reason_source_timestamp_metadata(tmp_path):
    """The correction's reason/source/timestamp must be queryable —
    the audit-trail contract (scope item 3: 'reason/source')."""
    arctic = _local_arctic(tmp_path)
    lib = arctic.get_library("universe", create_if_missing=True)

    lib.write("SPY", _ohlcv(), prune_previous_versions=False)
    corrected = _ohlcv()
    corrected.iloc[0, corrected.columns.get_loc("Close")] = 1.0

    vi = ae_arctic.write_correction(
        lib, "SPY", corrected,
        reason="vendor restated close", source="polygon",
    )

    meta = lib.read_metadata("SPY", as_of=vi.version).metadata
    assert meta["correction"] is True
    assert meta["reason"] == "vendor restated close"
    assert meta["source"] == "polygon"
    assert "correction_time" in meta
    # Must parse as a real ISO-8601 timestamp.
    pd.Timestamp(meta["correction_time"])


def test_write_correction_explicit_correction_time_is_used(tmp_path):
    arctic = _local_arctic(tmp_path)
    lib = arctic.get_library("universe", create_if_missing=True)
    lib.write("SPY", _ohlcv(), prune_previous_versions=False)

    fixed_ts = pd.Timestamp("2026-07-10T12:00:00Z")
    vi = ae_arctic.write_correction(
        lib, "SPY", _ohlcv(), reason="r", source="s", correction_time=fixed_ts,
    )
    meta = lib.read_metadata("SPY", as_of=vi.version).metadata
    assert meta["correction_time"] == fixed_ts.isoformat()


def test_write_correction_uses_is_none_not_truthiness_for_correction_time(tmp_path):
    """correction_time must be checked with `is None`, not truthiness —
    a falsy-but-explicit timestamp (e.g. the Unix epoch, which some
    pandas/numpy paths render as falsy in boolean context) must be
    honored as given, not silently replaced with 'now'."""
    arctic = _local_arctic(tmp_path)
    lib = arctic.get_library("universe", create_if_missing=True)
    lib.write("SPY", _ohlcv(), prune_previous_versions=False)

    epoch_ts = pd.Timestamp(0, tz="UTC")
    vi = ae_arctic.write_correction(
        lib, "SPY", _ohlcv(), reason="r", source="s", correction_time=epoch_ts,
    )
    meta = lib.read_metadata("SPY", as_of=vi.version).metadata
    assert meta["correction_time"] == epoch_ts.isoformat()


# ── bitemporal column constants ──────────────────────────────────────────────


def test_bitemporal_cols_are_six_fields_matching_issue_scope():
    """Pins the exact 6-field bitemporal contract from config#2459's scope
    item 1: settled, as_of, source (reused PROVENANCE_COL — not
    re-declared here), source_tier, valid_date, knowledge_time."""
    assert ae_arctic.SETTLED_COL == "settled"
    assert ae_arctic.AS_OF_COL == "as_of"
    assert ae_arctic.SOURCE_TIER_COL == "source_tier"
    assert ae_arctic.VALID_DATE_COL == "valid_date"
    assert ae_arctic.KNOWLEDGE_TIME_COL == "knowledge_time"
    assert ae_arctic.BITEMPORAL_COLS == (
        "settled", "as_of", "source_tier", "valid_date", "knowledge_time",
    )


def test_preliminary_lib_name_is_distinct_and_not_in_universe_or_macro():
    assert ae_arctic.PRELIMINARY_LIB not in (ae_arctic.UNIVERSE_LIB, ae_arctic.MACRO_LIB)
