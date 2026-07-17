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
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:  # pragma: no cover
    from datetime import date

    import pandas as pd
    from arcticdb.version_store.library import Library, VersionedItem

log = logging.getLogger(__name__)

# Library name constants — these match what every alpha-engine module uses.
# Centralized so a rename propagates from one place.
UNIVERSE_LIB = "universe"
MACRO_LIB = "macro"

# Preliminary/intraday values library (market-value-integrity L0/L5,
# config#2459). Physically SEPARATE from ``universe`` — mirrors the
# ``delisted_history`` precedent (config#1943 Leg 3, see
# ``nousergon-data store.arctic_store.get_delisted_history_lib``): a new
# consumer that reads prices for a "published"/"final" artifact must go
# through :func:`read_settled_only`, which structurally can never touch this
# library, rather than relying on per-caller discipline to filter out
# preliminary rows. Producers (intraday/preliminary collectors) write here
# directly via ``open_preliminary_lib``; nothing in the settled read path
# ever opens this library.
PRELIMINARY_LIB = "preliminary"

# Bitemporal schema fields (config#2459 scope item 1). Additive to the
# existing canonical universe/macro layout — same discipline as
# ``TOTAL_RETURN_COL`` (nousergon-data store.arctic_store): absent on
# every symbol written before this PR lands, and a no-op there. New
# writers add these six columns; nothing in the existing OHLCV+source+
# FEATURES contract changes shape or order because of them.
#
#   settled         bool      True once a value is treated as final/
#                              published-report-eligible; False for
#                              preliminary/intraday rows (also enforced
#                              structurally by the preliminary/universe
#                              library split — this column is a same-
#                              library-would-be-redundant belt for anyone
#                              inspecting a raw frame, not the sole guard).
#   as_of           Timestamp UTC capture time — when this datum was
#                              observed/ingested, independent of the
#                              trading day it describes ("knowledge time"
#                              axis half A).
#   source_tier     str       Coarse vendor-quality tier (e.g. "primary",
#                              "secondary", "derived") — distinct from the
#                              existing ``PROVENANCE_COL`` ("source"),
#                              which names the specific vendor
#                              (polygon/yfinance/fred). ``source`` already
#                              satisfies the bitemporal "source" field
#                              (config#2459 asks for a source field among
#                              the six; it does not ask for a second,
#                              redundant vendor-name column) — see
#                              ``to_arctic_canonical`` in nousergon-data
#                              for how the existing column is reused rather
#                              than duplicated.
#   valid_date      date      The trading/business date this datum
#                              describes (the pre-existing DataFrame index
#                              already carries this for OHLCV rows; this
#                              column exists for symbols/records where the
#                              valid date and the physical index key can
#                              diverge, e.g. a correction record whose
#                              index is the correction's own timestamp).
#   knowledge_time  Timestamp When the system's BELIEF about this datum
#                              was last updated — equal to ``as_of`` on
#                              first write, advanced on every correction
#                              (see ``write_correction`` below). Together
#                              with ``valid_date`` this is the bitemporal
#                              pair that answers "what did we believe on
#                              date X" queries.
SETTLED_COL: str = "settled"
AS_OF_COL: str = "as_of"
SOURCE_TIER_COL: str = "source_tier"
VALID_DATE_COL: str = "valid_date"
KNOWLEDGE_TIME_COL: str = "knowledge_time"

# Ordered list of the new bitemporal columns, appended (in this order)
# after PROVENANCE_COL/FEATURES in the canonical layout — see
# ``to_arctic_canonical``'s ``BITEMPORAL_COLS`` splice.
BITEMPORAL_COLS: tuple[str, ...] = (
    SETTLED_COL,
    AS_OF_COL,
    SOURCE_TIER_COL,
    VALID_DATE_COL,
    KNOWLEDGE_TIME_COL,
)

# Point-in-time constituent membership map, written weekly by
# alpha-engine-data ``collectors/historical_constituents.py`` (#490/#645) to
# the SAME bucket that backs ArcticDB, as a plain S3 JSON object (NOT an
# ArcticDB symbol). The document shape is::
#
#     {"schema_version": 1, "membership": {"YYYY-MM-DD": [tickers], ...}, ...}
#
# where each ``membership`` key is an index *change date* and its value is the
# roster that held *immediately before* that change took effect (see the
# producer's ``build_pit_membership`` docstring). The as-of lookup semantics
# in :func:`pit_membership_as_of` follow directly from that "before-the-change"
# keying.
HISTORICAL_CONSTITUENTS_KEY = "market_data/historical_constituents.json"


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
) -> Library:
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
) -> Library:
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


def open_preliminary_lib(
    bucket: str, *, region: str | None = None, create_if_missing: bool = False
) -> "Library":
    """Open the ``preliminary`` library on ``bucket`` (config#2459).

    Structurally separate from :data:`UNIVERSE_LIB` — the physical-library
    split that lets intraday/preliminary values exist without any risk of
    a "final report" consumer accidentally reading them (mirrors the
    ``delisted_history`` precedent). Producers writing intraday/preliminary
    ticks call this directly; nothing that reads for a published artifact
    should — use :func:`read_settled_only` instead, which never opens this
    library.

    Raises ``RuntimeError`` on any library-open failure, matching
    :func:`open_universe_lib` / :func:`open_macro_lib`.

    ``create_if_missing`` defaults to ``False`` (read-only-by-default
    convention); the preliminary-data producer passes ``True`` to bootstrap
    on a fresh bucket.
    """
    arctic = open_arctic(bucket, region=region)
    try:
        return arctic.get_library(PRELIMINARY_LIB, create_if_missing=create_if_missing)
    except Exception as exc:
        raise RuntimeError(
            f"ArcticDB {PRELIMINARY_LIB!r} library open failed on bucket "
            f"{bucket!r} (uri={arctic_uri(bucket, region=region)}): {exc}"
        ) from exc


def read_settled_only(
    bucket: str,
    symbol: str,
    *,
    region: str | None = None,
    as_of=None,
    **read_kwargs,
) -> "pd.DataFrame | None":
    """THE read-path chokepoint for a published/final-report price read
    (config#2459 scope item 2).

    Opens :data:`UNIVERSE_LIB` — never :data:`PRELIMINARY_LIB` — so a
    caller cannot accidentally read preliminary/intraday data into a
    "final" artifact just by getting the library name wrong; the function
    itself hard-codes the settled library and offers no parameter that
    could point it at ``preliminary``. Returns ``None`` iff ArcticDB's own
    read returns ``None`` data (rare — e.g. some empty-symbol read modes);
    callers that always expect a frame should check for this the same way
    they would guard any other ArcticDB read.

    If the bitemporal ``settled`` column (:data:`SETTLED_COL`) is present
    in the returned frame, rows where it is falsy are dropped — belt on
    top of the physical library split, not a substitute for it (most
    existing symbols predate the column and have none, in which case
    every row is treated as settled, matching current behavior
    byte-for-byte). Because the whole point of the physical split is that
    an unsettled row should NEVER be able to land in :data:`UNIVERSE_LIB`
    in the first place, finding one here is itself a data-integrity
    violation, not a routine occurrence — so unlike a normal filter this
    logs at ERROR (best-effort import of :func:`gate_alerts
    .alert_gate_failure`, so a monitored caller's flow-doctor singleton
    picks it up the same way any other L0-layer gate failure would)
    before silently continuing to drop the rows. Never raises for the
    alerting side-effect itself — a broken alert path must not turn a
    read into a hard failure.

    ``as_of`` forwards to ``lib.read`` for a bitemporal "what did we
    believe as of version/timestamp X" query (correction-record
    versioning, config#2459 scope item 3) — an int version, a
    ``pd.Timestamp``, or ``None`` for the current head version. A plain
    ``str`` is ArcticDB *snapshot-name* lookup, not a point-in-time query
    — pass a ``pd.Timestamp``/``int`` for the bitemporal use case.

    Any other ``read_kwargs`` (e.g. ``columns``, ``date_range``) forward
    to ``lib.read`` unchanged. Note: passing ``columns=[...]`` without
    ``SETTLED_COL`` in the list makes the settled-row filter a no-op (the
    column isn't in the returned frame to filter on) — the physical
    library split is still the load-bearing guarantee in that case.

    Raises ``RuntimeError`` on library-open failure (via
    :func:`open_universe_lib`); propagates ArcticDB's own exception types
    (e.g. ``NoSuchVersionException``) for read failures — callers that want
    a soft-fail should catch at the call site, matching the rest of this
    module's read helpers.
    """
    lib = open_universe_lib(bucket, region=region)
    result = lib.read(symbol, as_of=as_of, **read_kwargs)
    # arcticdb's Library.read() return type is a union that includes
    # LazyDataFrame/ExpressionNode (only reachable when the caller passes
    # lazy=True, which read_kwargs never does here) — cast matches the
    # existing precedent in preflight.py (BasePreflight.check_arcticdb_*)
    # rather than inventing a second pattern for the same stub gap.
    df = cast("pd.DataFrame | None", result.data)
    if df is not None and not df.empty and SETTLED_COL in df.columns:
        mask = df[SETTLED_COL].fillna(False).astype(bool)
        if not bool(mask.all()):
            _n_dropped = int((~mask).sum())
            try:
                from nousergon_lib.gate_alerts import alert_gate_failure

                alert_gate_failure(
                    layer="L0",
                    series=symbol,
                    detail=(
                        f"read_settled_only dropped {_n_dropped} unsettled "
                        f"row(s) found INSIDE the settled library "
                        f"({UNIVERSE_LIB!r}) — the physical preliminary/"
                        f"settled split should make this impossible; a "
                        f"producer likely wrote an unsettled row to the "
                        f"wrong library"
                    ),
                    severity="error",
                )
            except Exception:  # noqa: BLE001 — alerting must never break a read.
                log.error(
                    "read_settled_only: dropped %d unsettled row(s) for "
                    "%r found inside %r (alert_gate_failure unavailable)",
                    _n_dropped, symbol, UNIVERSE_LIB,
                )
            df = cast("pd.DataFrame", df[mask])
    return df


def write_correction(
    lib,
    symbol: str,
    df: "pd.DataFrame",
    *,
    reason: str,
    source: str,
    correction_time=None,
) -> "VersionedItem":
    """Write a post-hoc correction to a previously-published settled
    value as a NEW ArcticDB version — never an in-place overwrite
    (config#2459 scope item 3).

    This is deliberately a thin wrapper, not a new storage mechanism:
    ArcticDB already versions every ``lib.write``, and THIS call always
    passes ``prune_previous_versions=False`` (hard-coded below, not a
    caller option) so the version this call creates never prunes an
    earlier one out from under itself. Investigated and confirmed against
    a local ``lmdb://`` instance (see ``tests/test_arcticdb_bitemporal.py``
    ``test_write_correction_*``): a symbol written, then "corrected",
    keeps its original version fully readable by version index AND by a
    timestamp taken at/after the original write and before the
    correction. This is the SOTA approach the issue asks for — no side
    table needed; the version history IS the bitemporal correction log.

    ``df`` is the FULL corrected frame (same identity-preserving
    discipline as every other write site in this codebase — see
    ``nousergon-data corporate_actions.migrate_symbol`` / this module's
    ``delisted_history`` precedent) — pass the complete series with the
    correction applied, not a delta.

    ``reason`` and ``source`` are stamped into the write's metadata dict
    (queryable via ``lib.read(symbol, as_of=...).metadata`` or
    ``lib.read_metadata(symbol)``) along with ``correction_time`` (UTC,
    defaults to ``pd.Timestamp.now(tz="UTC")``) — this is the audit trail
    scope item 3 asks for (reason/source/timestamp per correction),
    riding on ArcticDB's native per-write metadata rather than a bespoke
    side-table.

    Returns the ``VersionedItem`` ArcticDB gives back from the write
    (carries the new version number + write timestamp) so the caller can
    log/alert on it.

    Caveat this function CANNOT enforce on its own: the "prior value
    stays queryable forever" guarantee depends on every OTHER writer that
    ever touches this symbol also avoiding version pruning. A later,
    ordinary write to the same symbol with ``prune_previous_versions=True``
    (the convention several producers in this fleet already use — see
    ``nousergon-data builders/daily_append.py``) can still prune away a
    correction's predecessor version. This helper only controls the one
    write it makes; it is not a library-wide pruning policy and does not
    (cannot, from a single call site) prevent a differently-configured
    writer from pruning history later. Treat "never prune the ``universe``
    library" as an operational invariant the correction-audit-trail
    feature depends on, not something this function alone guarantees.
    """
    import pandas as pd

    ts = correction_time if correction_time is not None else pd.Timestamp.now(tz="UTC")
    metadata = {
        "correction": True,
        "reason": reason,
        "source": source,
        "correction_time": ts.isoformat(),
    }
    return lib.write(symbol, df, metadata=metadata, prune_previous_versions=False)


def _load_constituent_membership(
    bucket: str, *, s3_client=None
) -> dict[str, list[str]]:
    """Read the PIT constituent membership map from S3.

    Returns the ``{change_date: [tickers]}`` ``membership`` sub-dict of
    ``market_data/historical_constituents.json`` (see
    :data:`HISTORICAL_CONSTITUENTS_KEY`). ``s3_client`` is an optional
    boto3-like client injection point for tests; when ``None`` a
    ``boto3.client("s3")`` is constructed (region resolves from the standard
    AWS env, matching the producer collector).

    Raises ``RuntimeError`` if the object is missing or malformed — an
    ``as_of`` backtest that silently fell back to the current roster would
    reintroduce exactly the survivorship bias this map exists to remove, so
    a missing PIT map is a hard precondition failure, not a soft fallback.
    """
    if s3_client is None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - boto3 is a core dep
            raise RuntimeError(
                f"boto3 is required to read {HISTORICAL_CONSTITUENTS_KEY}: {exc}"
            ) from exc
        s3_client = boto3.client("s3")

    import json

    try:
        resp = s3_client.get_object(
            Bucket=bucket, Key=HISTORICAL_CONSTITUENTS_KEY
        )
        doc = json.loads(resp["Body"].read())
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read PIT constituent map "
            f"s3://{bucket}/{HISTORICAL_CONSTITUENTS_KEY}: {exc}. The weekly "
            f"producer (alpha-engine-data historical_constituents) must have "
            f"written it before an as_of backtest can run."
        ) from exc

    membership = doc.get("membership")
    if not isinstance(membership, dict) or not membership:
        raise RuntimeError(
            f"PIT constituent map s3://{bucket}/{HISTORICAL_CONSTITUENTS_KEY} "
            f"has no usable 'membership' object (got {type(membership).__name__})."
        )
    return membership


def pit_membership_as_of(
    membership: Mapping[str, list[str] | set[str]], as_of: date
) -> set[str] | None:
    """Resolve as-of-date index membership from a PIT ``membership`` map.

    ``membership`` is ``{change_date: tickers}`` where each key is an index
    change date and its value is the roster that held *immediately before*
    that change (the producer's ``build_pit_membership`` keying). So the
    roster in effect *on* ``as_of`` is the snapshot for the **earliest change
    date strictly greater than** ``as_of``:

      * for change dates ``D1 < D2 < D3``, ``membership[D2]`` is "after D1,
        before D2", i.e. the roster for any day in ``[D1, D2)``;
      * an ``as_of`` before the first change date -> ``membership[D1]``;
      * an ``as_of`` on/after the latest change date -> ``None`` (the map
        only stores pre-change snapshots; the roster after the last change is
        the *current* roster, which lives in ArcticDB, not this map). Callers
        treat ``None`` as "use the current universe".

    Pure (no IO) so it is cheaply unit-testable and can be called per-date in
    a tight backtest loop.
    """
    from datetime import date as _date
    from datetime import datetime as _datetime

    if isinstance(as_of, _datetime):
        as_of = as_of.date()

    def _parse(key: str) -> _date:
        return _datetime.strptime(key, "%Y-%m-%d").date()

    # Smallest change date strictly greater than as_of -> that snapshot is
    # "membership before this change" == membership as of as_of.
    best_key: str | None = None
    best_dt: _date | None = None
    for key in membership:
        try:
            dt = _parse(key)
        except (ValueError, TypeError):
            continue
        if dt > as_of and (best_dt is None or dt < best_dt):
            best_dt = dt
            best_key = key

    if best_key is None:
        return None
    return {str(t).upper() for t in membership[best_key]}


def get_universe_symbols(
    bucket: str,
    *,
    as_of: date | None = None,
    region: str | None = None,
    s3_client=None,
) -> set[str]:
    """Return the set of universe symbols, optionally as-of a past date.

    Default (``as_of=None``) — behavior is **identical** to before: return
    the symbols currently present in the ArcticDB universe library
    (``list_symbols()``). Common use case: filtering tickers against "what's
    actually in ArcticDB right now" before passing to downstream code that
    hard-fails on missing symbols (e.g. the executor's load_daily_vwap /
    load_atr_14_pct guards, or the backtester's simulate replay of historical
    signals).

    ``as_of`` (a ``datetime.date``) — return the **point-in-time** index
    membership that held on that date, read from the weekly PIT constituent
    map (:data:`HISTORICAL_CONSTITUENTS_KEY`). This removes look-ahead
    *inclusion* survivorship bias: a backtest date no longer sees today's
    constituent snapshot applied to all history. When ``as_of`` falls
    on/after the most recent recorded index change (i.e. the PIT map has no
    pre-change snapshot past it), the membership *is* the current roster, so
    this falls back to the live ``list_symbols()`` set — identical to the
    default. ``s3_client`` is an optional boto3-like injection point for the
    PIT-map read (tests).

    Raises ``RuntimeError`` on library-open or list failure, or (with
    ``as_of``) on a missing/malformed PIT map — an ArcticDB health problem or
    an absent survivorship map is a pipeline-level precondition, not
    something to silently paper over with an empty or biased set.
    """
    if as_of is not None:
        membership = _load_constituent_membership(bucket, s3_client=s3_client)
        pit = pit_membership_as_of(membership, as_of)
        if pit is not None:
            log.info(
                "ArcticDB %s PIT membership as_of %s: %d symbols",
                UNIVERSE_LIB,
                as_of,
                len(pit),
            )
            return pit
        # as_of is on/after the latest recorded change -> current roster.
        log.info(
            "PIT membership as_of %s is on/after the latest index change — "
            "using current ArcticDB %s roster",
            as_of,
            UNIVERSE_LIB,
        )

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
    end: pd.Timestamp | str | None,
    columns,
    max_workers: int,
    label: str,
) -> dict[str, pd.DataFrame]:
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
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import pandas as pd  # lazy: only needed with the [arcticdb] extra

    symbols = sorted(set(symbols))
    if not symbols:
        return {}

    # pd.Timestamp(...)'s constructor stub includes a NaT-producing branch
    # (its generic fallback for unparseable input); `end` is documented as
    # pd.Timestamp/str/None and this is a controlled internal call, so NaT
    # can't actually occur here.
    end_ts = cast(
        "pd.Timestamp",
        pd.Timestamp(end) if end is not None else pd.Timestamp.now(tz="UTC"),
    ).normalize()
    if end_ts.tz is not None:
        end_ts = end_ts.tz_localize(None)
    start_ts = end_ts - pd.Timedelta(days=lookback_days)

    def _read(sym: str):
        # dict[str, Any]: this dict is heterogeneous kwargs (a tuple value
        # today, optionally a list value below) — the plain-literal
        # inference would otherwise pin the value type to the first entry.
        read_kwargs: dict[str, Any] = {"date_range": (start_ts, end_ts)}
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

    out: dict[str, pd.DataFrame] = {}
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
    end: pd.Timestamp | str | None = None,
    columns=None,
    max_workers: int = 20,
    region: str | None = None,
) -> dict[str, pd.DataFrame]:
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
    end: pd.Timestamp | str | None = None,
    columns=None,
    max_workers: int = 20,
    region: str | None = None,
) -> dict[str, pd.DataFrame]:
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
