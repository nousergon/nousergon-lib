"""I/O pool sizing and chunked ``read_batch`` keep the same frames.

The lmdb cases use a local file URI. They do not open S3 or AWS. The
subprocess cases start a fresh interpreter so an explicit
``ARCTICDB_VersionStore_NumIOThreads_int`` is what arcticdb records at
import, and so the helper's default does not depend on this process
already having imported arcticdb.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from nousergon_lib import arcticdb as ae_arctic

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
_IO_ENV = ae_arctic.IO_THREADS_ENV


def test_explicit_env_wins_over_the_argument(monkeypatch):
    monkeypatch.setenv(_IO_ENV, "7")
    assert ae_arctic.configure_arcticdb_io_threads(32) == "7"
    assert os.environ[_IO_ENV] == "7"


def test_missing_env_uses_the_default(monkeypatch):
    monkeypatch.delenv(_IO_ENV, raising=False)
    assert ae_arctic.configure_arcticdb_io_threads() == "32"
    assert os.environ[_IO_ENV] == "32"


def test_loaded_package_receives_the_explicit_value_not_the_default(monkeypatch):
    """``set_config_int`` outranks the environment, so it must be fed the env."""
    monkeypatch.setenv(_IO_ENV, "7")
    calls: list[tuple[str, int]] = []
    pkg = types.ModuleType("arcticdb")
    pkg.__path__ = []  # type: ignore[attr-defined]
    config = types.ModuleType("arcticdb.config")

    def _set_config_int(label: str, value: int) -> None:
        calls.append((label, value))

    config.set_config_int = _set_config_int
    monkeypatch.setitem(sys.modules, "arcticdb", pkg)
    monkeypatch.setitem(sys.modules, "arcticdb.config", config)

    assert ae_arctic.configure_arcticdb_io_threads(32) == "7"
    assert calls == [(ae_arctic.IO_THREADS_CONFIG_KEY, 7)]


def test_loaded_package_receives_the_default_when_env_is_unset(monkeypatch):
    monkeypatch.delenv(_IO_ENV, raising=False)
    calls: list[tuple[str, int]] = []
    pkg = types.ModuleType("arcticdb")
    pkg.__path__ = []  # type: ignore[attr-defined]
    config = types.ModuleType("arcticdb.config")
    config.set_config_int = lambda label, value: calls.append((label, value))
    monkeypatch.setitem(sys.modules, "arcticdb", pkg)
    monkeypatch.setitem(sys.modules, "arcticdb.config", config)

    assert ae_arctic.configure_arcticdb_io_threads() == "32"
    assert calls == [(ae_arctic.IO_THREADS_CONFIG_KEY, 32)]


def test_non_package_stub_does_not_call_into_a_cached_config(monkeypatch):
    """A ``ModuleType`` stub must not reach a previously imported real setter."""
    monkeypatch.delenv(_IO_ENV, raising=False)
    calls: list[tuple[object, ...]] = []
    stub = types.ModuleType("arcticdb")
    config = types.ModuleType("arcticdb.config")
    config.set_config_int = lambda *args: calls.append(args)
    monkeypatch.setitem(sys.modules, "arcticdb", stub)
    monkeypatch.setitem(sys.modules, "arcticdb.config", config)

    assert ae_arctic.configure_arcticdb_io_threads(32) == "32"
    assert calls == []


def test_non_int_explicit_env_is_kept_and_not_pushed(monkeypatch, caplog):
    monkeypatch.setenv(_IO_ENV, "nope")
    calls: list[tuple[str, int]] = []
    pkg = types.ModuleType("arcticdb")
    pkg.__path__ = []  # type: ignore[attr-defined]
    config = types.ModuleType("arcticdb.config")
    config.set_config_int = lambda label, value: calls.append((label, value))
    monkeypatch.setitem(sys.modules, "arcticdb", pkg)
    monkeypatch.setitem(sys.modules, "arcticdb.config", config)

    assert ae_arctic.configure_arcticdb_io_threads(32) == "nope"
    assert calls == []
    assert "not an int" in caplog.text


def _fresh(code: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _child_env(**extra: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop(_IO_ENV, None)
    env["PYTHONPATH"] = str(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env.update(extra)
    return env


def test_fresh_interpreter_default_is_visible_to_arcticdb():
    pytest.importorskip("arcticdb")
    code = (
        "import os, sys\n"
        f"os.environ.pop({_IO_ENV!r}, None)\n"
        "from nousergon_lib.arcticdb import configure_arcticdb_io_threads\n"
        "assert 'arcticdb' not in sys.modules\n"
        "print(configure_arcticdb_io_threads())\n"
        "import arcticdb\n"
        "from arcticdb.config import get_config_int\n"
        "print(get_config_int('VersionStore.NumIOThreads'))\n"
    )
    proc = _fresh(code, _child_env())
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-2:] == ["32", "32"]


def test_fresh_interpreter_explicit_env_beats_the_default():
    pytest.importorskip("arcticdb")
    code = (
        "import sys\n"
        "from nousergon_lib.arcticdb import configure_arcticdb_io_threads\n"
        "assert 'arcticdb' not in sys.modules\n"
        "print(configure_arcticdb_io_threads(32))\n"
        "import arcticdb\n"
        "from arcticdb.config import get_config_int\n"
        "print(get_config_int('VersionStore.NumIOThreads'))\n"
    )
    proc = _fresh(code, _child_env(**{_IO_ENV: "7"}))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-2:] == ["7", "7"]


def _imports_arcticdb(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[0] == "arcticdb" for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").split(".")[0] == "arcticdb"
    return False


def _under_type_checking(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    current: ast.AST | None = node
    while current is not None:
        current = parents.get(current)
        if isinstance(current, ast.If) and isinstance(current.test, ast.Name):
            if current.test.id == "TYPE_CHECKING":
                return True
    return False


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.AST | None:
    current: ast.AST | None = node
    while current is not None:
        current = parents.get(current)
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
    return None


def _configured_before(func: ast.AST, import_node: ast.AST) -> bool:
    import_line = getattr(import_node, "lineno", 0)
    for sub in ast.walk(func):
        lineno = getattr(sub, "lineno", None)
        if lineno is None or lineno >= import_line:
            continue
        if not isinstance(sub, ast.Call):
            continue
        called = sub.func
        name = called.attr if isinstance(called, ast.Attribute) else getattr(called, "id", "")
        if name == "configure_arcticdb_io_threads":
            return True
    return False


def test_every_runtime_arcticdb_import_configures_the_pool_first():
    offenders: list[str] = []
    for path in sorted((_SRC / "nousergon_lib").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        for node in ast.walk(tree):
            if not _imports_arcticdb(node) or _under_type_checking(node, parents):
                continue
            func = _enclosing_function(node, parents)
            rel = path.relative_to(_REPO)
            # This helper is how configure pushes the winning value through
            # set_config_int. It imports arcticdb.config only after
            # configure_arcticdb_io_threads has chosen the env value, and it
            # cannot call configure itself (that would recurse).
            if isinstance(func, ast.FunctionDef) and func.name == "_apply_loaded_io_threads":
                continue
            if func is None or not _configured_before(func, node):
                offenders.append(f"{rel}:{getattr(node, 'lineno', '?')}")
    apply_src = (_SRC / "nousergon_lib" / "arcticdb.py").read_text(encoding="utf-8")
    assert apply_src.count("_apply_loaded_io_threads(") == 2
    assert offenders == []


class _BatchReads:
    def read_batch(self, requests):
        out = []
        for req in requests:
            try:
                out.append(self.read(req.symbol, date_range=req.date_range, columns=req.columns))
            except Exception as exc:  # noqa: BLE001 - stub mirrors DataError
                out.append(types.SimpleNamespace(symbol=req.symbol, exception_string=str(exc)))
        return out


def test_read_batch_is_chunked_at_150_and_frames_stay_with_their_symbols(monkeypatch):
    pd = pytest.importorskip("pandas")
    idx = pd.date_range("2025-01-01", periods=2, freq="D")
    n = ae_arctic.READ_BATCH_CHUNK_SIZE + 1
    symbols = [f"S{i:04d}" for i in range(n)]
    frames = {
        sym: pd.DataFrame({"Close": [float(i), float(i) + 0.5]}, index=idx)
        for i, sym in enumerate(symbols)
    }
    sizes: list[int] = []

    class _Lib(_BatchReads):
        def read(self, sym, date_range=None, columns=None):
            class _R:
                data = frames[sym].copy()

            return _R()

        def read_batch(self, requests):
            sizes.append(len(requests))
            return _BatchReads.read_batch(self, requests)

    class _Arctic:
        def get_library(self, name, create_if_missing=False):
            return _Lib()

    mod = types.ModuleType("arcticdb")
    mod.Arctic = lambda uri: _Arctic()
    mod.ReadRequest = lambda **kw: types.SimpleNamespace(**kw)
    mod.DataError = type("DataError", (), {})
    monkeypatch.setitem(sys.modules, "arcticdb", mod)

    out = ae_arctic.load_universe_ohlcv(
        "b", symbols=symbols, lookback_days=3650, end="2025-12-31",
    )
    assert sizes == [ae_arctic.READ_BATCH_CHUNK_SIZE, 1]
    assert set(out) == set(symbols)
    for sym, frame in frames.items():
        assert out[sym]["Close"].tolist() == frame["Close"].tolist()


def test_macro_read_batch_uses_the_same_chunks(monkeypatch):
    pd = pytest.importorskip("pandas")
    idx = pd.date_range("2025-01-01", periods=2, freq="D")
    symbols = ["GLD", "TNX", "USO", "VIX", "VIX3M"]
    frames = {
        sym: pd.DataFrame({"Close": [float(i), float(i) + 1]}, index=idx)
        for i, sym in enumerate(symbols)
    }
    sizes: list[int] = []

    class _Lib(_BatchReads):
        def read(self, sym, date_range=None, columns=None):
            class _R:
                data = frames[sym].copy()

            return _R()

        def read_batch(self, requests):
            sizes.append(len(requests))
            return _BatchReads.read_batch(self, requests)

    class _Arctic:
        def get_library(self, name, create_if_missing=False):
            return _Lib()

    mod = types.ModuleType("arcticdb")
    mod.Arctic = lambda uri: _Arctic()
    mod.ReadRequest = lambda **kw: types.SimpleNamespace(**kw)
    mod.DataError = type("DataError", (), {})
    monkeypatch.setitem(sys.modules, "arcticdb", mod)
    monkeypatch.setattr(ae_arctic, "READ_BATCH_CHUNK_SIZE", 2)

    out = ae_arctic.load_macro_series(
        "b", symbols, lookback_days=3650, end="2025-12-31",
    )
    assert sizes == [2, 2, 1]
    assert set(out) == set(symbols)
    for sym, frame in frames.items():
        assert out[sym]["Close"].tolist() == frame["Close"].tolist()


def _normalize(pd, df):
    if df is None or df.empty:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df[~df.index.duplicated(keep="last")].sort_index()


def _legacy_per_symbol(pd, lib, symbols, *, lookback_days, end, columns):
    """The pre-chunk contract: one ``lib.read`` per symbol, then normalize."""
    end_ts = pd.Timestamp(end).normalize()
    if end_ts.tz is not None:
        end_ts = end_ts.tz_localize(None)
    start_ts = end_ts - pd.Timedelta(days=lookback_days)
    out = {}
    for sym in sorted(set(symbols)):
        try:
            raw = lib.read(
                sym,
                date_range=(start_ts, end_ts),
                columns=list(columns) if columns is not None else None,
            ).data
        except Exception:
            raw = None
        if raw is None:
            continue
        df = _normalize(pd, raw)
        if df is None:
            continue
        out[sym] = df
    return out


class _CountingLib:
    def __init__(self, inner):
        self._inner = inner
        self.sizes: list[int] = []

    def read_batch(self, requests):
        self.sizes.append(len(list(requests)))
        return self._inner.read_batch(requests)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _install_local_arctic(monkeypatch, arctic):
    wrappers: dict[str, _CountingLib] = {}

    class _Arctic:
        def get_library(self, name, create_if_missing=False):
            if name not in wrappers:
                wrappers[name] = _CountingLib(
                    arctic.get_library(name, create_if_missing=create_if_missing)
                )
            return wrappers[name]

    monkeypatch.setattr(ae_arctic, "open_arctic", lambda bucket, region=None: _Arctic())
    return wrappers


def _assert_same_frames(pd, left, right):
    assert set(left) == set(right)
    for sym in left:
        pd.testing.assert_frame_equal(left[sym], right[sym], check_freq=False)


def test_lmdb_chunked_universe_matches_the_old_read(tmp_path, monkeypatch):
    """Chunked ``load_universe_ohlcv`` matches one batch and per-symbol reads."""
    adb = pytest.importorskip("arcticdb")
    pd = pytest.importorskip("pandas")
    arctic = adb.Arctic(f"lmdb://{tmp_path / 'store'}")
    lib = arctic.get_library("universe", create_if_missing=True)

    in_window = pd.bdate_range("2024-06-03", periods=4)
    dup_index = pd.DatetimeIndex(["2024-06-04", "2024-06-04", "2024-06-05"])
    stale = pd.bdate_range("2020-01-02", periods=3)
    n = ae_arctic.READ_BATCH_CHUNK_SIZE
    symbols = [f"S{i:04d}" for i in range(n)]
    for i, sym in enumerate(symbols):
        close = float(i + 1)
        lib.write(
            sym,
            pd.DataFrame({"Open": close, "Close": close + 0.25, "Volume": 10.0}, index=in_window),
        )
    lib.write(
        "DUP",
        pd.DataFrame({"Open": [1.0, 2.0, 3.0], "Close": [1.5, 9.0, 3.5]}, index=dup_index),
    )
    lib.write("STALE", pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=stale))

    requested = symbols + ["DUP", "STALE", "MISSING"]
    end = "2024-06-15"
    lookback = 40
    wrappers = _install_local_arctic(monkeypatch, arctic)

    monkeypatch.setattr(ae_arctic, "READ_BATCH_CHUNK_SIZE", 10**9)
    single = ae_arctic.load_universe_ohlcv(
        "local", symbols=requested, lookback_days=lookback, end=end,
    )
    assert wrappers["universe"].sizes == [len(requested)]

    wrappers["universe"].sizes.clear()
    monkeypatch.setattr(ae_arctic, "READ_BATCH_CHUNK_SIZE", n)
    chunked = ae_arctic.load_universe_ohlcv(
        "local", symbols=requested, lookback_days=lookback, end=end,
    )
    # sorted: DUP, MISSING, S0000.., STALE. One full chunk, then the tail.
    assert wrappers["universe"].sizes == [n, len(requested) - n]
    _assert_same_frames(pd, chunked, single)

    legacy = _legacy_per_symbol(
        pd, lib, requested, lookback_days=lookback, end=end, columns=None,
    )
    _assert_same_frames(pd, chunked, legacy)
    assert "MISSING" not in chunked
    assert "STALE" not in chunked
    assert "DUP" in chunked
    # keep="last" on the duplicated 2024-06-04 row, then sort.
    assert chunked["DUP"]["Close"].tolist() == [9.0, 3.5]

    narrowed = ae_arctic.load_universe_ohlcv(
        "local",
        symbols=["S0000", "DUP"],
        lookback_days=lookback,
        end=end,
        columns=["Close"],
    )
    legacy_narrow = _legacy_per_symbol(
        pd, lib, ["S0000", "DUP"], lookback_days=lookback, end=end, columns=["Close"],
    )
    _assert_same_frames(pd, narrowed, legacy_narrow)
    assert list(narrowed["S0000"].columns) == ["Close"]


def test_lmdb_chunked_macro_matches_the_old_read(tmp_path, monkeypatch):
    adb = pytest.importorskip("arcticdb")
    pd = pytest.importorskip("pandas")
    arctic = adb.Arctic(f"lmdb://{tmp_path / 'store'}")
    lib = arctic.get_library("macro", create_if_missing=True)
    idx = pd.bdate_range("2024-05-01", periods=6)
    symbols = ["GLD", "TNX", "USO", "VIX", "VIX3M"]
    for i, sym in enumerate(symbols):
        lib.write(sym, pd.DataFrame({"Close": [float(i + 1)] * len(idx)}, index=idx))

    wrappers = _install_local_arctic(monkeypatch, arctic)
    monkeypatch.setattr(ae_arctic, "READ_BATCH_CHUNK_SIZE", 2)
    out = ae_arctic.load_macro_series(
        "local", symbols, lookback_days=3650, end="2024-06-15", columns=["Close"],
    )
    assert wrappers["macro"].sizes == [2, 2, 1]
    legacy = _legacy_per_symbol(
        pd, lib, symbols, lookback_days=3650, end="2024-06-15", columns=["Close"],
    )
    _assert_same_frames(pd, out, legacy)


def test_local_bench_frames_match():
    """The timing script's identity check holds on a tiny local store."""
    pytest.importorskip("arcticdb")
    pytest.importorskip("pandas")
    import importlib.util

    path = _REPO / "scripts" / "bench_arcticdb_local_read.py"
    spec = importlib.util.spec_from_file_location("bench_arcticdb_local_read", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.run(n_symbols=4, n_rows=6, repeats=1)
    assert result["label"] == "local"
    assert result["backend"] == "lmdb"
    assert result["identical"] is True
