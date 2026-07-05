"""
ArcticDB helpers: uniform library-open path + common read patterns.

Centralizes the ``adb.Arctic(uri).get_library("...")`` boilerplate that was
duplicated across predictor, research, backtester, data, and executor. Every
site was constructing the same S3 URI string by hand — one escape bug in
that string (path_prefix= query param collapsing under shell double-quote
interpolation) surfaced 2026-04-21 during the SNDK incident.

Using this module guarantees that:

- The S3 URI format stays consistent everywhere (single source of truth).
- Library-open failures raise a uniform ``RuntimeError`` with bucket
  context, so downstream errors have a consistent shape.
- ``arcticdb`` is imported lazily inside each function, so this module
  stays importable on consumers that don't install the ``[arcticdb]``
  optional extra (e.g. lightweight CLI tools that only use the logging
  submodule).

Requires the ``arcticdb`` optional extra
(``nousergon-lib[arcticdb]``) to actually call any function here.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from arcticdb.version_store.library import Library

log = logging.getLogger(__name__)

# Library name constants — these match what every alpha-engine module uses.
# Centralized so a rename propagates from one place.
UNIVERSE_LIB = "universe"
MACRO_LIB = "macro"


def arctic_uri(bucket: str, *, region: str | None = None) -> str:
    """Return the canonical ArcticDB S3 URI for ``bucket``.

    Format: ``s3s://s3.{region}.amazonaws.com:{bucket}?path_prefix=arcticdb&aws_auth=true``

    ``region`` defaults to ``AWS_REGION`` env var, then ``us-east-1``.
    """
    region = region or os.environ.get("AWS_REGION", "us-east-1")
    return (
        f"s3s://s3.{region}.amazonaws.com:{bucket}"
        "?path_prefix=arcticdb&aws_auth=true"
    )


def _import_arcticdb():
    """Lazy import helper with a uniform error message."""
    try:
        import arcticdb as adb
    except ImportError as exc:
        raise RuntimeError(
            "arcticdb is not importable — install "
            "nousergon-lib[arcticdb] or add arcticdb to the deploy "
            f"image: {exc}"
        ) from exc
    return adb


def open_arctic(bucket: str, *, region: str | None = None):
    """Return an ``arcticdb.Arctic`` instance pointed at ``bucket``.

    Raises ``RuntimeError`` if ``arcticdb`` is not installed.
    """
    adb = _import_arcticdb()
    return adb.Arctic(arctic_uri(bucket, region=region))


def open_universe_lib(
    bucket: str, *, region: str | None = None, create_if_missing: bool = False
) -> "Library":
    """Open the ``universe`` library on ``bucket``.

    Raises ``RuntimeError`` on any library-open failure, with bucket and
    URI in the message so the operator can see which endpoint is failing.

    ``create_if_missing`` defaults to ``False`` so read-only consumers keep
    strict "must already exist" semantics; producer modules that bootstrap
    the library on a fresh bucket (cold start) pass ``True`` to preserve
    create-on-missing behavior.
    """
    arctic = open_arctic(bucket, region=region)
    try:
        return arctic.get_library(UNIVERSE_LIB, create_if_missing=create_if_missing)
    except Exception as exc:
        raise RuntimeError(
            f"ArcticDB {UNIVERSE_LIB!r} library open failed on bucket "
            f"{bucket!r} (uri={arctic_uri(bucket, region=region)}): {exc}"
        ) from exc


def open_macro_lib(
    bucket: str, *, region: str | None = None, create_if_missing: bool = False
) -> "Library":
    """Open the ``macro`` library on ``bucket``.

    Raises ``RuntimeError`` on any library-open failure.

    ``create_if_missing`` defaults to ``False`` so read-only consumers keep
    strict "must already exist" semantics; producer modules that bootstrap
    the library on a fresh bucket (cold start) pass ``True`` to preserve
    create-on-missing behavior.
    """
    arctic = open_arctic(bucket, region=region)
    try:
        return arctic.get_library(MACRO_LIB, create_if_missing=create_if_missing)
    except Exception as exc:
        raise RuntimeError(
            f"ArcticDB {MACRO_LIB!r} library open failed on bucket "
            f"{bucket!r} (uri={arctic_uri(bucket, region=region)}): {exc}"
        ) from exc


def get_universe_symbols(bucket: str, *, region: str | None = None) -> set[str]:
    """Return the set of symbols currently present in the universe library.

    Common use case: filtering tickers against "what's actually in
    ArcticDB right now" before passing to downstream code that hard-fails
    on missing symbols (e.g. the executor's load_daily_vwap / load_atr_14_pct
    guards, or the backtester's simulate replay of historical signals).

    Raises ``RuntimeError`` on library-open or list failure — an
    ArcticDB health problem is a pipeline-level precondition, not
    something to silently paper over with an empty set.
    """
    lib = open_universe_lib(bucket, region=region)
    try:
        symbols = set(lib.list_symbols())
    except Exception as exc:
        raise RuntimeError(
            f"ArcticDB {UNIVERSE_LIB}.list_symbols() failed on bucket "
            f"{bucket!r}: {exc}"
        ) from exc
    log.info("ArcticDB %s symbols available: %d", UNIVERSE_LIB, len(symbols))
    return symbols


# Default OHLCV columns. ``None`` (the load_universe_ohlcv default) reads the
# full stored frame so the result is a faithful slim-cache equivalent; pass
# this explicitly to narrow the read for perf when only prices are needed.
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# The slim cache (alpha-engine-data collectors/slim_cache.py) writes a ~2-year
# tail slice of the full price_cache parquets. 730d is the parity-equivalent
# default so a load_universe_ohlcv() result lines up with load_slim_cache()
# over the same window; widen via lookback_days for backtester-style reads.
_SLIM_EQUIVALENT_LOOKBACK_DAYS = 730


def _load_arctic_frames(
    lib,
    symbols,
    *,
    lookback_days: int,
    end,
    columns,
    max_workers: int,
    label: str,
) -> "dict[str, 'pd.DataFrame']":
    """Shared read core for the universe + macro ArcticDB readers.

    Reads a date-windowed slice of each ``symbols`` entry out of an
    already-opened ArcticDB ``lib``, normalizing exactly like
    ``load_slim_cache`` (tz-naive monotonic ``DatetimeIndex``, dup dates
    collapsed keep=last) so a ``reconcile`` against slim compares
    like-for-like. ``label`` only tunes log messages.

    Contract (mirrors ``load_slim_cache``): individual symbol read failures
    / empty frames are logged at WARNING and dropped — the caller decides
    how to handle a partial load.
    """
    import pandas as pd  # lazy: only needed with the [arcticdb] extra
    from concurrent.futures import ThreadPoolExecutor, as_completed

    symbols = sorted(set(symbols))
    if not symbols:
        return {}

    end_ts = (
        pd.Timestamp(end) if end is not None else pd.Timestamp.now(tz="UTC")
    ).normalize()
    if end_ts.tz is not None:
        end_ts = end_ts.tz_localize(None)
    start_ts = end_ts - pd.Timedelta(days=lookback_days)

    def _read(sym: str):
        read_kwargs = {"date_range": (start_ts, end_ts)}
        if columns is not None:
            read_kwargs["columns"] = list(columns)
        df = lib.read(sym, **read_kwargs).data
        if df is None or df.empty:
            return sym, None
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return sym, df

    out: "dict[str, pd.DataFrame]" = {}
    errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_read, s): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                ticker, df = fut.result()
            except Exception as exc:  # noqa: BLE001 - partial-load contract
                log.warning("ArcticDB %s read failed for %s: %s", label, sym, exc)
                errors += 1
                continue
            if df is None:
                log.warning(
                    "ArcticDB %s returned empty frame for %s", label, sym
                )
                errors += 1
                continue
            out[ticker] = df

    log.info(
        "%s: %d symbols OK, %d errors (window %s..%s)",
        label,
        len(out),
        errors,
        start_ts.date(),
        end_ts.date(),
    )
    return out


def load_universe_ohlcv(
    bucket: str,
    *,
    symbols=None,
    lookback_days: int = _SLIM_EQUIVALENT_LOOKBACK_DAYS,
    end=None,
    columns=None,
    max_workers: int = 20,
    region: str | None = None,
) -> "dict[str, 'pd.DataFrame']":
    """Load a ticker -> OHLCV DataFrame dict from the ArcticDB **universe** lib.

    This is the **single source of truth** for "read a 2y-ish OHLCV slice per
    ticker out of ArcticDB" — the read+dedup+normalize idiom that was
    copy-pasted into predictor ``inference/stages/load_prices.py`` and is
    needed again by the data macro-breadth / feature-compute and backtester
    exit-timing consumers as they migrate off the
    ``predictor/price_cache_slim/`` parquet tier. Returns the **same shape**
    as ``alpha-engine-data store.parquet_loader.load_slim_cache`` (ticker ->
    DataFrame with a tz-naive monotonic ``DatetimeIndex``) so it is a drop-in
    substitute for slim-cache reads.

    Contract (mirrors ``load_slim_cache``): individual ticker read failures /
    empty frames are logged at WARNING and dropped from the result — the
    caller decides how to handle a partial load. Returns ``{}`` if the
    universe library has no symbols.

    Note: the universe lib holds equities + SPY only. Macro/index series
    (VIX, TNX, IRX, GLD, USO, VIX3M) and sector ETFs live in the **macro**
    lib — use :func:`load_macro_series` for those.

    Args:
        bucket: S3 bucket backing ArcticDB.
        symbols: iterable of tickers to read; ``None`` reads every symbol
            currently in the universe library (``get_universe_symbols``).
        lookback_days: window size; default matches the slim cache's ~2y tail.
        end: window end (``pd.Timestamp``/str); ``None`` -> today (normalized).
        columns: columns to read; ``None`` reads the full stored frame (true
            slim-cache equivalent). Pass ``OHLCV_COLUMNS`` to narrow for perf.
        max_workers: ThreadPool width for the per-ticker reads.
        region: AWS region override (defaults via ``arctic_uri``).
    """
    if symbols is None:
        symbols = get_universe_symbols(bucket, region=region)
    symbols = sorted(set(symbols))
    if not symbols:
        log.warning(
            "load_universe_ohlcv: universe library %r is empty on %r",
            UNIVERSE_LIB,
            bucket,
        )
        return {}
    lib = open_universe_lib(bucket, region=region)
    return _load_arctic_frames(
        lib,
        symbols,
        lookback_days=lookback_days,
        end=end,
        columns=columns,
        max_workers=max_workers,
        label="load_universe_ohlcv",
    )


def load_macro_series(
    bucket: str,
    symbols,
    *,
    lookback_days: int = _SLIM_EQUIVALENT_LOOKBACK_DAYS,
    end=None,
    columns=None,
    max_workers: int = 20,
    region: str | None = None,
) -> "dict[str, 'pd.DataFrame']":
    """Load a symbol -> OHLCV DataFrame dict from the ArcticDB **macro** lib.

    The macro-lib analog of :func:`load_universe_ohlcv`, sharing the exact
    same read+dedup+tz-normalize core so a ``reconcile`` against the slim
    cache compares like-for-like. The macro lib holds the index/macro series
    (SPY, VIX, VIX3M, TNX, IRX, GLD, USO) and the sector ETFs (XL*) that the
    universe lib does **not** carry — i.e. exactly what
    ``alpha-engine-data features/compute.py::_extract_macro`` needs as it
    migrates off ``predictor/price_cache_slim/``.

    ``symbols`` is **required** (no read-all default): the macro lib is
    heterogeneous and contains non-price composite keys (e.g. a ``features``
    symbol written by the data backfill) that must not be read as OHLCV.
    Pass the explicit set the caller needs.

    Same partial-load contract as :func:`load_universe_ohlcv`: per-symbol
    read failures / empty frames are logged at WARNING and dropped. Returns
    ``{}`` if ``symbols`` is empty.

    Args:
        bucket: S3 bucket backing ArcticDB.
        symbols: explicit iterable of macro/ETF symbols to read (required).
        lookback_days: window size; default matches the slim cache's ~2y tail.
        end: window end (``pd.Timestamp``/str); ``None`` -> today (normalized).
        columns: columns to read; ``None`` reads the full stored frame.
        max_workers: ThreadPool width for the per-symbol reads.
        region: AWS region override (defaults via ``arctic_uri``).
    """
    symbols = sorted(set(symbols))
    if not symbols:
        log.warning("load_macro_series: no symbols requested on %r", bucket)
        return {}
    lib = open_macro_lib(bucket, region=region)
    return _load_arctic_frames(
        lib,
        symbols,
        lookback_days=lookback_days,
        end=end,
        columns=columns,
        max_workers=max_workers,
        label="load_macro_series",
    )
