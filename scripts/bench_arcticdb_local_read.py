#!/usr/bin/env python3
"""Time chunked vs single-batch ArcticDB reads on a local ``lmdb://`` store.

This process never opens S3, AWS, or a production bucket. The numbers it
prints are local-disk timings. They are not the scanner's 45-47s to 5-6s
measurement, which was 150 symbols against the live library.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from nousergon_lib import arcticdb as ae_arctic


def _frame(n_rows: int, close_0: float) -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=n_rows, freq="B")
    close = close_0 + pd.Series(range(n_rows), index=index, dtype="float64")
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1_000_000.0,
        },
        index=index,
    )


def _identical(left: dict[str, pd.DataFrame], right: dict[str, pd.DataFrame]) -> bool:
    if set(left) != set(right):
        return False
    for sym in left:
        try:
            pd.testing.assert_frame_equal(left[sym], right[sym], check_freq=False)
        except AssertionError:
            return False
    return True


# LMDB allows one Arctic instance per path in a process. A second open logs
# a warning and is not a fair thread-pool comparison, so 3 vs 32 each get a
# fresh interpreter and their own store. The env is set before arcticdb imports.
_THREAD_PROBE = """
import json, os, sys, tempfile, time
from pathlib import Path

import pandas as pd

n_threads, n_symbols, n_rows, repeats = (int(arg) for arg in sys.argv[1:])
os.environ["ARCTICDB_VersionStore_NumIOThreads_int"] = str(n_threads)
import arcticdb as adb
from arcticdb.config import get_config_int

seen = get_config_int("VersionStore.NumIOThreads")
if seen != n_threads:
    raise SystemExit(f"pool {seen} != {n_threads}")
symbols = [f"S{i:04d}" for i in range(n_symbols)]
index = pd.date_range("2024-01-02", periods=n_rows, freq="B")
with tempfile.TemporaryDirectory(prefix="arcticdb-bench-threads-") as tmp:
    uri = f"lmdb://{Path(tmp) / 'store'}"
    lib = adb.Arctic(uri).get_library("universe", create_if_missing=True)
    for i, sym in enumerate(symbols):
        close = float(i + 1) + pd.Series(range(n_rows), index=index, dtype="float64")
        lib.write(sym, pd.DataFrame({"Close": close}, index=index))
    requests = [adb.ReadRequest(symbol=sym, columns=["Close"]) for sym in symbols]
    lib.read_batch(requests)
    started = time.perf_counter()
    for _ in range(repeats):
        lib.read_batch(requests)
    elapsed = (time.perf_counter() - started) / repeats
print(json.dumps({"seconds": elapsed, "threads": seen}))
"""


def _time_threads(n_threads: int, n_symbols: int, n_rows: int, repeats: int) -> float:
    proc = subprocess.run(
        [sys.executable, "-c", _THREAD_PROBE, str(n_threads), str(n_symbols), str(n_rows), str(repeats)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    return float(payload["seconds"])


def run(
    *,
    n_symbols: int = 40,
    n_rows: int = 80,
    repeats: int = 3,
) -> dict[str, object]:
    """Build a throwaway lmdb library and time the old and new read shapes.

    ``single_batch_s`` is one ``read_batch`` of every symbol (the previous
    shape). ``chunked_s`` is :func:`_load_arctic_frames` at
    ``READ_BATCH_CHUNK_SIZE``. ``threads_3_s`` and ``threads_32_s`` are the
    same full-library ``read_batch`` in fresh interpreters whose pool was set
    to 3 and 32 before arcticdb imported. All four are local lmdb, not S3.
    """
    import arcticdb as adb

    if n_symbols < 1 or n_rows < 1 or repeats < 1:
        raise ValueError("n_symbols, n_rows, and repeats must be >= 1")

    ae_arctic.configure_arcticdb_io_threads()
    symbols = [f"S{i:04d}" for i in range(n_symbols)]
    with tempfile.TemporaryDirectory(prefix="arcticdb-bench-") as tmp:
        uri = f"lmdb://{Path(tmp) / 'store'}"
        arctic = adb.Arctic(uri)
        lib = arctic.get_library("universe", create_if_missing=True)
        for i, sym in enumerate(symbols):
            lib.write(sym, _frame(n_rows, float(i + 1)))

        saved_chunk = ae_arctic.READ_BATCH_CHUNK_SIZE
        try:
            ae_arctic.READ_BATCH_CHUNK_SIZE = max(n_symbols, 1)
            started = time.perf_counter()
            single = ae_arctic._load_arctic_frames(
                lib,
                symbols,
                lookback_days=10_000,
                end="2025-12-31",
                columns=["Close"],
                label="bench-single",
            )
            single_s = time.perf_counter() - started

            ae_arctic.READ_BATCH_CHUNK_SIZE = saved_chunk
            started = time.perf_counter()
            chunked = ae_arctic._load_arctic_frames(
                lib,
                symbols,
                lookback_days=10_000,
                end="2025-12-31",
                columns=["Close"],
                label="bench-chunked",
            )
            chunked_s = time.perf_counter() - started
        finally:
            ae_arctic.READ_BATCH_CHUNK_SIZE = saved_chunk

    threads_3_s = _time_threads(3, n_symbols, n_rows, repeats)
    threads_32_s = _time_threads(32, n_symbols, n_rows, repeats)

    return {
        "label": "local",
        "backend": "lmdb",
        "n_symbols": n_symbols,
        "n_rows": n_rows,
        "repeats": repeats,
        "chunk_size": saved_chunk,
        "single_batch_s": round(single_s, 6),
        "chunked_s": round(chunked_s, 6),
        "threads_3_s": round(threads_3_s, 6),
        "threads_32_s": round(threads_32_s, 6),
        "identical": _identical(single, chunked),
        "io_threads_env": ae_arctic.configure_arcticdb_io_threads(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=int, default=40)
    parser.add_argument("--rows", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    result = run(n_symbols=args.symbols, n_rows=args.rows, repeats=args.repeats)
    print(
        "local lmdb: "
        f"single_batch {result['single_batch_s']}s, "
        f"chunked {result['chunked_s']}s, "
        f"3 threads {result['threads_3_s']}s, "
        f"32 threads {result['threads_32_s']}s, "
        f"identical={result['identical']} "
        f"({result['n_symbols']} symbols x {result['n_rows']} rows)"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
