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
