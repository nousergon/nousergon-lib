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
(``alpha-engine-lib[arcticdb]``) to actually call any function here.
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
            "alpha-engine-lib[arcticdb] or add arcticdb to the deploy "
            f"image: {exc}"
        ) from exc
    return adb


def open_arctic(bucket: str, *, region: str | None = None):
    """Return an ``arcticdb.Arctic`` instance pointed at ``bucket``.

    Raises ``RuntimeError`` if ``arcticdb`` is not installed.
    """
    adb = _import_arcticdb()
    return adb.Arctic(arctic_uri(bucket, region=region))


def open_universe_lib(bucket: str, *, region: str | None = None) -> "Library":
    """Open the ``universe`` library on ``bucket``.

    Raises ``RuntimeError`` on any library-open failure, with bucket and
    URI in the message so the operator can see which endpoint is failing.
    """
    arctic = open_arctic(bucket, region=region)
    try:
        return arctic.get_library(UNIVERSE_LIB)
    except Exception as exc:
        raise RuntimeError(
            f"ArcticDB {UNIVERSE_LIB!r} library open failed on bucket "
            f"{bucket!r} (uri={arctic_uri(bucket, region=region)}): {exc}"
        ) from exc


def open_macro_lib(bucket: str, *, region: str | None = None) -> "Library":
    """Open the ``macro`` library on ``bucket``.

    Raises ``RuntimeError`` on any library-open failure.
    """
    arctic = open_arctic(bucket, region=region)
    try:
        return arctic.get_library(MACRO_LIB)
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
    """Load a ticker -> OHLCV DataFrame dict from the ArcticDB universe lib.

    This is the **single source of truth** for "read a 2y-ish OHLCV slice per
    ticker out of ArcticDB" — the read+dedup+normalize idiom that was
    copy-pasted into predictor ``inference/stages/load_prices.py`` and is
    needed again by the data macro-breadth and backtester exit-timing
    consumers as they migrate off the ``predictor/price_cache_slim/`` parquet
    tier. Returns the **same shape** as
    ``alpha-engine-data store.parquet_loader.load_slim_cache`` (ticker ->
    DataFrame with a tz-naive monotonic ``DatetimeIndex``) so it is a drop-in
    substitute for slim-cache reads.

    Contract (mirrors ``load_slim_cache``): individual ticker read failures /
    empty frames are logged at WARNING and dropped from the result — the
    caller decides how to handle a partial load. Returns ``{}`` if the
    universe library has no symbols.

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
    import pandas as pd  # lazy: only needed with the [arcticdb] extra
    from concurrent.futures import ThreadPoolExecutor, as_completed

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

    end_ts = (
        pd.Timestamp(end) if end is not None else pd.Timestamp.now(tz="UTC")
    ).normalize()
    if end_ts.tz is not None:
        end_ts = end_ts.tz_localize(None)
    start_ts = end_ts - pd.Timedelta(days=lookback_days)

    lib = open_universe_lib(bucket, region=region)

    def _read(sym: str):
        read_kwargs = {"date_range": (start_ts, end_ts)}
        if columns is not None:
            read_kwargs["columns"] = list(columns)
        df = lib.read(sym, **read_kwargs).data
        if df is None or df.empty:
            return sym, None
        # Mirror load_slim_cache normalization exactly so a reconcile against
        # it compares like-for-like: tz-naive monotonic DatetimeIndex, no
        # duplicate dates (historical dup-row residue, see load_prices.py).
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return sym, df

    price_data: "dict[str, pd.DataFrame]" = {}
    errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_read, s): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                ticker, df = fut.result()
            except Exception as exc:  # noqa: BLE001 - partial-load contract
                log.warning("ArcticDB universe read failed for %s: %s", sym, exc)
                errors += 1
                continue
            if df is None:
                log.warning("ArcticDB universe returned empty frame for %s", sym)
                errors += 1
                continue
            price_data[ticker] = df

    log.info(
        "load_universe_ohlcv: %d tickers OK, %d errors (window %s..%s)",
        len(price_data),
        errors,
        start_ts.date(),
        end_ts.date(),
    )
    return price_data
