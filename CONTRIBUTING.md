# Contributing

Thank you for your interest. Before any contribution can be accepted, please
read the two policies below — they are required and exist to keep the
project's licensing options intact.

## 1. Developer Certificate of Origin (DCO)

All commits must be signed off (`git commit -s`), certifying the
[Developer Certificate of Origin 1.1](https://developercertificate.org/).
Pull requests containing commits without a `Signed-off-by:` line will not be
merged.

## 2. Inbound license

By submitting a contribution, you agree that your contribution is licensed to
the project under the **MIT License**, regardless of the project's outbound
license. This permits the project to distribute your contribution under its
current license (see LICENSE) and under commercial licenses. If you cannot
contribute under these terms, please open an issue instead of a pull request.

## Scope

Issues and discussions are welcome. Substantial changes should start as an
issue before any code is written.

## Running the test suite

`pip install -e ".[dev]"` alone is not enough to run every test green — several
test modules import an optional extra's dependency at module load time (or, for
the RAG parquet/ANN tests, inside a fixture) and will fail rather than skip if
it's absent without the extra installed too:

```bash
pip install -e ".[dev,rag,rag-parquet,rag-local-ann,arcticdb,quant,quant-xs,quant-stats,contracts,github_app]"
PYTHONPATH=$PWD/src python3 -m pytest tests/ -q
```

That's the same extras set `.github/workflows/test.yml` installs. In
particular, `tests/test_rag_parquet_mirror.py` and `tests/test_rag_local_ann.py`
need `[rag-parquet]` (pandas + pyarrow) and, for the HNSW-index tests,
`[rag-local-ann]` (adds hnswlib) — without them those tests skip explicitly
rather than fail (nousergon-lib-I265).


## Date-axis review chokepoint (nousergon/alpha-engine-config#1613)

This repo consumes `krepis.dates` (`now_dual()` / `.trading_day` /
`last_closed_trading_day` / `session_for_timestamp`) and/or
`krepis.trading_calendar` (`session_date()`). Two axes exist and must not be
conflated:

- **knowledge axis** (`trading_day`, via `now_dual()`) — the last CLOSED
  NYSE session; correct for anything computed FROM data (predictions,
  signals, features, eval joins, freshness checks).
- **event axis** (`session_date()`) — the session a physical event belongs
  to (fills, NAV marks, account snapshots).

See the `krepis/dates.py` module docstring for the full doctrine and
nousergon/alpha-engine-config#1610 / #1613 for the ratifying fleet audit.
When reviewing a PR that adds a new date-axis callsite, classify it against
this doctrine before approving.
