"""Tests for nousergon_lib.arcticdb helpers.

These tests cover URI construction + error-wrapping shapes using
stubs — they don't require arcticdb to be installed.
"""

from __future__ import annotations

import sys
import types

import pytest

from nousergon_lib import arcticdb as ae_arctic

# ── URI construction ────────────────────────────────────────────────────────


def test_arctic_uri_default_region(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    uri = ae_arctic.arctic_uri("test-bucket")
    assert uri == (
        "s3s://s3.us-east-1.amazonaws.com:test-bucket"
        "?path_prefix=arcticdb&aws_auth=true"
    )


def test_arctic_uri_region_from_env(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    uri = ae_arctic.arctic_uri("test-bucket")
    assert "s3.eu-west-1.amazonaws.com" in uri


def test_arctic_uri_explicit_region_overrides_env(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    uri = ae_arctic.arctic_uri("test-bucket", region="us-west-2")
    assert "s3.us-west-2.amazonaws.com" in uri


def test_arctic_uri_matches_existing_preflight_format(monkeypatch):
    """Guard against drift — the uri string must be byte-identical to what
    preflight.check_arcticdb_fresh currently builds. Any consumer that
    interacts with existing ArcticDB libraries depends on this exact URI
    (different URIs point at different library indexes, even if the bucket
    is the same)."""
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    expected = (
        "s3s://s3.us-east-1.amazonaws.com:alpha-engine-research"
        "?path_prefix=arcticdb&aws_auth=true"
    )
    assert ae_arctic.arctic_uri("alpha-engine-research") == expected


# ── ImportError handling ─────────────────────────────────────────────────────


def test_import_helper_raises_runtimeerror_when_arcticdb_missing(monkeypatch):
    """When arcticdb is not installed, _import_arcticdb must wrap the
    ImportError in a RuntimeError with install guidance."""
    # Simulate arcticdb being absent by blocking the import.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "arcticdb":
            raise ImportError("No module named 'arcticdb'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(RuntimeError, match="nousergon-lib\\[arcticdb\\]"):
        ae_arctic._import_arcticdb()


# ── open_universe_lib / open_macro_lib error wrapping ────────────────────────


class _StubArctic:
    """Stub that raises on get_library to exercise the error wrapper.

    Records the ``create_if_missing`` value it was called with so tests can
    assert the flag is threaded through unchanged.
    """

    def __init__(self, raise_on_get: bool = False):
        self._raise = raise_on_get
        self.last_create_if_missing = None

    def get_library(self, name, create_if_missing=False):
        self.last_create_if_missing = create_if_missing
        if self._raise:
            raise RuntimeError(f"fake get_library failure for {name}")
        return _StubLibrary(name, create_if_missing=create_if_missing)


class _StubLibrary:
    def __init__(self, name, symbols=None, create_if_missing=False):
        self._name = name
        self._symbols = symbols if symbols is not None else ["A", "B", "C"]
        self.create_if_missing = create_if_missing

    def list_symbols(self):
        return list(self._symbols)


def _stub_arcticdb_module():
    mod = types.ModuleType("arcticdb")
    mod.Arctic = lambda uri: _StubArctic(raise_on_get=False)
    return mod


def test_open_universe_lib_wraps_errors_with_bucket_context(monkeypatch):
    """When get_library raises, open_universe_lib must raise a RuntimeError
    whose message names the bucket, URI, and library — so an operator can
    see *which* endpoint is failing."""
    mod = types.ModuleType("arcticdb")
    mod.Arctic = lambda uri: _StubArctic(raise_on_get=True)
    monkeypatch.setitem(sys.modules, "arcticdb", mod)

    with pytest.raises(RuntimeError) as exc_info:
        ae_arctic.open_universe_lib("broken-bucket")
    msg = str(exc_info.value)
    assert "universe" in msg
    assert "broken-bucket" in msg
    assert "uri=" in msg


def test_open_macro_lib_wraps_errors(monkeypatch):
    mod = types.ModuleType("arcticdb")
    mod.Arctic = lambda uri: _StubArctic(raise_on_get=True)
    monkeypatch.setitem(sys.modules, "arcticdb", mod)

    with pytest.raises(RuntimeError) as exc_info:
        ae_arctic.open_macro_lib("broken-bucket")
    assert "macro" in str(exc_info.value)


def test_open_universe_lib_returns_library_on_success(monkeypatch):
    monkeypatch.setitem(sys.modules, "arcticdb", _stub_arcticdb_module())
    lib = ae_arctic.open_universe_lib("test-bucket")
    assert isinstance(lib, _StubLibrary)
    assert lib._name == "universe"


# ── create_if_missing pass-through ───────────────────────────────────────────


def test_open_universe_lib_defaults_create_if_missing_false(monkeypatch):
    """Read-only consumers (the default) must NOT create the library —
    ``create_if_missing`` defaults to ``False`` and is threaded to
    ``get_library`` unchanged."""
    stub = _StubArctic(raise_on_get=False)
    mod = types.ModuleType("arcticdb")
    mod.Arctic = lambda uri: stub
    monkeypatch.setitem(sys.modules, "arcticdb", mod)

    lib = ae_arctic.open_universe_lib("test-bucket")
    assert stub.last_create_if_missing is False
    assert lib.create_if_missing is False


def test_open_universe_lib_passes_create_if_missing_true(monkeypatch):
    """Producer cold-start path: ``create_if_missing=True`` must reach
    ``get_library`` so a fresh bucket bootstraps the library."""
    stub = _StubArctic(raise_on_get=False)
    mod = types.ModuleType("arcticdb")
    mod.Arctic = lambda uri: stub
    monkeypatch.setitem(sys.modules, "arcticdb", mod)

    lib = ae_arctic.open_universe_lib("test-bucket", create_if_missing=True)
    assert stub.last_create_if_missing is True
    assert lib.create_if_missing is True


def test_open_macro_lib_defaults_create_if_missing_false(monkeypatch):
    stub = _StubArctic(raise_on_get=False)
    mod = types.ModuleType("arcticdb")
    mod.Arctic = lambda uri: stub
    monkeypatch.setitem(sys.modules, "arcticdb", mod)

    lib = ae_arctic.open_macro_lib("test-bucket")
    assert stub.last_create_if_missing is False
    assert lib.create_if_missing is False


def test_open_macro_lib_passes_create_if_missing_true(monkeypatch):
    stub = _StubArctic(raise_on_get=False)
    mod = types.ModuleType("arcticdb")
    mod.Arctic = lambda uri: stub
    monkeypatch.setitem(sys.modules, "arcticdb", mod)

    lib = ae_arctic.open_macro_lib("test-bucket", create_if_missing=True)
    assert stub.last_create_if_missing is True
    assert lib.create_if_missing is True


# ── get_universe_symbols ─────────────────────────────────────────────────────


def test_get_universe_symbols_returns_set(monkeypatch):
    monkeypatch.setitem(sys.modules, "arcticdb", _stub_arcticdb_module())
    symbols = ae_arctic.get_universe_symbols("test-bucket")
    assert symbols == {"A", "B", "C"}
    assert isinstance(symbols, set)


def test_get_universe_symbols_raises_if_list_fails(monkeypatch):
    class _BrokenLibrary(_StubLibrary):
        def list_symbols(self):
            raise RuntimeError("list failure")

    mod = types.ModuleType("arcticdb")
    class _ArcticWithBrokenLib:
        def get_library(self, name, create_if_missing=False):
            return _BrokenLibrary(name)
    mod.Arctic = lambda uri: _ArcticWithBrokenLib()
    monkeypatch.setitem(sys.modules, "arcticdb", mod)

    with pytest.raises(RuntimeError, match="list_symbols"):
        ae_arctic.get_universe_symbols("test-bucket")


# ── PIT membership: pit_membership_as_of (pure) ──────────────────────────────

# Producer keying (build_pit_membership): each key is an index *change date*
# and its value is the roster held *immediately before* that change. So for
# changes on 2020-06-01 (X added) and 2022-03-01 (Y removed):
#   - membership[2020-06-01] = roster before 2020-06-01 (no X yet)
#   - membership[2022-03-01] = roster before 2022-03-01 (has X, still has Y)
_PIT_MEMBERSHIP = {
    "2020-06-01": ["AAA", "BBB"],           # before X was added
    "2022-03-01": ["AAA", "BBB", "XXX"],    # after X added, before Y removed
}


def test_pit_membership_as_of_between_changes():
    """A date in [2020-06-01, 2022-03-01) gets the snapshot keyed at the
    NEXT change date (membership 'before 2022-03-01')."""
    import datetime

    got = ae_arctic.pit_membership_as_of(_PIT_MEMBERSHIP, datetime.date(2021, 1, 1))
    assert got == {"AAA", "BBB", "XXX"}


def test_pit_membership_as_of_before_first_change():
    """A date before the earliest change date gets the earliest snapshot."""
    import datetime

    got = ae_arctic.pit_membership_as_of(_PIT_MEMBERSHIP, datetime.date(2019, 1, 1))
    assert got == {"AAA", "BBB"}


def test_pit_membership_as_of_on_change_date_uses_next_snapshot():
    """On a change date D itself, membership is 'after D' — so the snapshot
    keyed at D (which is 'before D') must NOT be returned; the next snapshot
    is (if any). On 2020-06-01 the roster is 'after 2020-06-01' == the
    2022-03-01 snapshot (before the next change)."""
    import datetime

    got = ae_arctic.pit_membership_as_of(_PIT_MEMBERSHIP, datetime.date(2020, 6, 1))
    assert got == {"AAA", "BBB", "XXX"}


def test_pit_membership_as_of_after_last_change_returns_none():
    """On/after the latest change date there is no pre-change snapshot — the
    roster is the *current* one (lives in ArcticDB), so return None so the
    caller falls back to list_symbols()."""
    import datetime

    assert ae_arctic.pit_membership_as_of(_PIT_MEMBERSHIP, datetime.date(2022, 3, 1)) is None
    assert ae_arctic.pit_membership_as_of(_PIT_MEMBERSHIP, datetime.date(2025, 1, 1)) is None


def test_pit_membership_as_of_uppercases_and_dedupes():
    import datetime

    membership = {"2021-01-01": ["aaa", "Bbb"], "2023-01-01": ["aaa"]}
    got = ae_arctic.pit_membership_as_of(membership, datetime.date(2022, 1, 1))
    assert got == {"AAA"}


def test_pit_membership_as_of_accepts_datetime():
    import datetime

    got = ae_arctic.pit_membership_as_of(
        _PIT_MEMBERSHIP, datetime.datetime(2021, 1, 1, 15, 30)
    )
    assert got == {"AAA", "BBB", "XXX"}


# ── PIT membership: get_universe_symbols(as_of=...) end-to-end ───────────────


class _FakeS3:
    """Minimal boto3-like S3 client serving one JSON object for get_object."""

    def __init__(self, body: bytes, *, raise_missing: bool = False):
        self._body = body
        self._raise = raise_missing

    def get_object(self, Bucket, Key):  # noqa: N803 - boto3 kwarg names
        if self._raise:
            raise RuntimeError(f"NoSuchKey: {Key}")

        class _Body:
            def __init__(self, b):
                self._b = b

            def read(self):
                return self._b

        return {"Body": _Body(self._body)}


def _constituents_doc(membership: dict) -> bytes:
    import json

    return json.dumps(
        {"schema_version": 1, "membership": membership, "n_snapshots": len(membership)}
    ).encode()


def test_get_universe_symbols_as_of_returns_pit_set(monkeypatch):
    """as_of resolves the PIT snapshot from S3 — ArcticDB list_symbols is
    NOT consulted when the map covers the date (no arcticdb import needed)."""
    import datetime

    # Ensure any accidental ArcticDB use would blow up loudly.
    monkeypatch.delitem(sys.modules, "arcticdb", raising=False)
    s3 = _FakeS3(_constituents_doc(_PIT_MEMBERSHIP))

    got = ae_arctic.get_universe_symbols(
        "b", as_of=datetime.date(2021, 1, 1), s3_client=s3
    )
    assert got == {"AAA", "BBB", "XXX"}


def test_get_universe_symbols_as_of_after_last_change_falls_back_to_current(monkeypatch):
    """When as_of is past the last recorded change, the roster IS the current
    one — get_universe_symbols must fall back to ArcticDB list_symbols()."""
    import datetime

    monkeypatch.setitem(sys.modules, "arcticdb", _stub_arcticdb_module())  # A,B,C
    s3 = _FakeS3(_constituents_doc(_PIT_MEMBERSHIP))

    got = ae_arctic.get_universe_symbols(
        "b", as_of=datetime.date(2025, 1, 1), s3_client=s3
    )
    assert got == {"A", "B", "C"}  # current roster from list_symbols()


def test_get_universe_symbols_omitted_as_of_is_current_behavior(monkeypatch):
    """Non-breaking default: omitting as_of must be byte-identical to before —
    the current ArcticDB roster, no S3 PIT read at all."""
    monkeypatch.setitem(sys.modules, "arcticdb", _stub_arcticdb_module())

    def _boom(*a, **k):  # any PIT read would be a regression
        raise AssertionError("PIT map must not be read when as_of is omitted")

    monkeypatch.setattr(ae_arctic, "_load_constituent_membership", _boom)

    assert ae_arctic.get_universe_symbols("b") == {"A", "B", "C"}


def test_get_universe_symbols_as_of_missing_map_raises(monkeypatch):
    """A missing PIT map under as_of is a hard precondition failure (silently
    falling back to the current roster would reintroduce survivorship bias)."""
    import datetime

    s3 = _FakeS3(b"", raise_missing=True)
    with pytest.raises(RuntimeError, match="PIT constituent map"):
        ae_arctic.get_universe_symbols("b", as_of=datetime.date(2021, 1, 1), s3_client=s3)


def test_get_universe_symbols_as_of_malformed_map_raises(monkeypatch):
    import datetime

    s3 = _FakeS3(b'{"schema_version": 1}')  # no membership key
    with pytest.raises(RuntimeError, match="no usable 'membership'"):
        ae_arctic.get_universe_symbols("b", as_of=datetime.date(2021, 1, 1), s3_client=s3)


# ── load_universe_ohlcv ──────────────────────────────────────────────────────


def _stub_arctic_with_ohlcv(monkeypatch, frames, symbols=None):
    """Install an arcticdb stub whose universe lib serves ``frames``.

    ``frames`` is ``{ticker: DataFrame}``; ``.read(sym, date_range, columns)``
    returns an object with ``.data`` sliced to the date_range (and columns if
    given), mirroring the real arcticdb read contract.
    """
    syms = list(symbols) if symbols is not None else list(frames)

    class _ReadResult:
        def __init__(self, data):
            self.data = data

    class _Lib:
        def list_symbols(self):
            return syms

        def read(self, sym, date_range=None, columns=None):
            df = frames[sym]
            if date_range is not None:
                lo, hi = date_range
                # Real ArcticDB stores tz-naive indexes; compare on a
                # tz-naive view so a tz-aware test fixture still slices
                # (production strips tz after .read, not before).
                ix = df.index
                if getattr(ix, "tz", None) is not None:
                    ix = ix.tz_localize(None)
                df = df[(ix >= lo) & (ix <= hi)]
            if columns is not None:
                df = df[list(columns)]
            return _ReadResult(df.copy())

    class _Arctic:
        def get_library(self, name, create_if_missing=False):
            return _Lib()

    mod = types.ModuleType("arcticdb")
    mod.Arctic = lambda uri: _Arctic()
    monkeypatch.setitem(sys.modules, "arcticdb", mod)


def test_load_universe_ohlcv_returns_slim_cache_shape(monkeypatch):
    pd = pytest.importorskip("pandas")
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    frames = {
        "AAA": pd.DataFrame({"Close": [1.0, 2, 3, 4, 5], "Volume": [10] * 5}, index=idx),
        "BBB": pd.DataFrame({"Close": [9.0, 8, 7, 6, 5], "Volume": [20] * 5}, index=idx),
    }
    _stub_arctic_with_ohlcv(monkeypatch, frames)

    out = ae_arctic.load_universe_ohlcv("b", lookback_days=3650, end="2025-12-31")

    assert set(out) == {"AAA", "BBB"}
    for df in out.values():
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is None
        assert df.index.is_monotonic_increasing
    assert out["AAA"]["Close"].tolist() == [1, 2, 3, 4, 5]


def test_load_universe_ohlcv_default_symbols_from_library(monkeypatch):
    pd = pytest.importorskip("pandas")
    idx = pd.date_range("2025-01-01", periods=2, freq="D")
    frames = {"XYZ": pd.DataFrame({"Close": [1.0, 2]}, index=idx)}
    _stub_arctic_with_ohlcv(monkeypatch, frames, symbols=["XYZ"])

    out = ae_arctic.load_universe_ohlcv("b", lookback_days=3650, end="2025-12-31")
    assert list(out) == ["XYZ"]


def test_load_universe_ohlcv_drops_tz_and_dupes(monkeypatch):
    pd = pytest.importorskip("pandas")
    idx = pd.DatetimeIndex(
        ["2025-01-02", "2025-01-01", "2025-01-02"], tz="US/Eastern"
    )
    frames = {"DUP": pd.DataFrame({"Close": [2.0, 1.0, 99.0]}, index=idx)}
    _stub_arctic_with_ohlcv(monkeypatch, frames)

    out = ae_arctic.load_universe_ohlcv("b", lookback_days=3650, end="2025-12-31")
    df = out["DUP"]
    assert df.index.tz is None
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates
    assert len(df) == 2  # 3 rows, one duplicate date collapsed
    # keep="last" then sort -> 2025-01-01 row=1.0, deduped 2025-01-02 row=99.0
    assert df["Close"].tolist() == [1.0, 99.0]


def test_load_universe_ohlcv_skips_failures_partial_load(monkeypatch):
    pd = pytest.importorskip("pandas")
    idx = pd.date_range("2025-01-01", periods=2, freq="D")
    good = pd.DataFrame({"Close": [1.0, 2]}, index=idx)

    class _ReadResult:
        def __init__(self, data):
            self.data = data

    class _Lib:
        def list_symbols(self):
            return ["GOOD", "BAD", "EMPTY"]

        def read(self, sym, date_range=None, columns=None):
            if sym == "BAD":
                raise RuntimeError("boom")
            if sym == "EMPTY":
                return _ReadResult(good.iloc[0:0])
            return _ReadResult(good.copy())

    class _Arctic:
        def get_library(self, name, create_if_missing=False):
            return _Lib()

    mod = types.ModuleType("arcticdb")
    mod.Arctic = lambda uri: _Arctic()
    monkeypatch.setitem(sys.modules, "arcticdb", mod)

    out = ae_arctic.load_universe_ohlcv("b", lookback_days=3650, end="2025-12-31")
    assert list(out) == ["GOOD"]  # BAD raised, EMPTY empty -> both dropped


def test_load_universe_ohlcv_empty_universe_returns_empty(monkeypatch):
    pytest.importorskip("pandas")
    _stub_arctic_with_ohlcv(monkeypatch, {}, symbols=[])
    assert ae_arctic.load_universe_ohlcv("b") == {}


# ── load_macro_series ────────────────────────────────────────────────────────


def test_load_macro_series_reads_requested_symbols(monkeypatch):
    pd = pytest.importorskip("pandas")
    idx = pd.date_range("2025-01-01", periods=4, freq="D")
    frames = {
        "VIX": pd.DataFrame({"Close": [18.0, 19, 20, 21]}, index=idx),
        "TNX": pd.DataFrame({"Close": [4.0, 4.1, 4.2, 4.3]}, index=idx),
        "XLK": pd.DataFrame({"Close": [200.0, 201, 202, 203]}, index=idx),
        "features": pd.DataFrame({"Close": [0.0]}, index=idx[:1]),  # non-price
    }
    _stub_arctic_with_ohlcv(monkeypatch, frames)

    out = ae_arctic.load_macro_series(
        "b", ["VIX", "TNX", "XLK"], lookback_days=3650, end="2025-12-31"
    )
    # Only the requested symbols — the heterogeneous 'features' key is NOT
    # read (required-symbols contract protects against it).
    assert set(out) == {"VIX", "TNX", "XLK"}
    assert out["VIX"]["Close"].tolist() == [18, 19, 20, 21]
    for df in out.values():
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is None


def test_load_macro_series_requires_symbols_returns_empty(monkeypatch):
    pytest.importorskip("pandas")
    _stub_arctic_with_ohlcv(monkeypatch, {"VIX": None}, symbols=["VIX"])
    # No symbols requested -> empty (no read-all default for the macro lib).
    assert ae_arctic.load_macro_series("b", []) == {}


def test_load_macro_series_partial_load_skips_failures(monkeypatch):
    pd = pytest.importorskip("pandas")
    idx = pd.date_range("2025-01-01", periods=2, freq="D")
    good = pd.DataFrame({"Close": [1.0, 2]}, index=idx)

    class _Res:
        def __init__(self, data):
            self.data = data

    class _Lib:
        def list_symbols(self):
            return ["GLD", "MISSING"]

        def read(self, sym, date_range=None, columns=None):
            if sym == "MISSING":
                raise RuntimeError("no such symbol")
            return _Res(good.copy())

    class _Arctic:
        def get_library(self, name, create_if_missing=False):
            return _Lib()

    mod = types.ModuleType("arcticdb")
    mod.Arctic = lambda uri: _Arctic()
    monkeypatch.setitem(sys.modules, "arcticdb", mod)

    out = ae_arctic.load_macro_series(
        "b", ["GLD", "MISSING"], lookback_days=3650, end="2025-12-31"
    )
    assert list(out) == ["GLD"]


def test_load_macro_series_shares_normalization_with_universe(monkeypatch):
    """The shared core gives macro reads identical tz/dup normalization."""
    pd = pytest.importorskip("pandas")
    idx = pd.DatetimeIndex(
        ["2025-01-02", "2025-01-01", "2025-01-02"], tz="US/Eastern"
    )
    frames = {"USO": pd.DataFrame({"Close": [2.0, 1.0, 99.0]}, index=idx)}
    _stub_arctic_with_ohlcv(monkeypatch, frames)

    out = ae_arctic.load_macro_series(
        "b", ["USO"], lookback_days=3650, end="2025-12-31"
    )
    df = out["USO"]
    assert df.index.tz is None
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates
    assert df["Close"].tolist() == [1.0, 99.0]  # keep="last" then sort
