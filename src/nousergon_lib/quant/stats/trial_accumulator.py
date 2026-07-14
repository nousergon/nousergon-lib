"""trial_accumulator — cumulative multiple-testing trial count (config#2454).

DSR (Deflated Sharpe Ratio, Bailey & Lopez de Prado 2014) needs ``n_trials``:
the total number of strategy configurations trialed, since inception, before
the deployed configuration was selected. That count is NOT scoped to one
sweep producer — the backtester runs (at least) 4 independent producers that
each generate candidate configurations:

  1. ``run_optimizer_param_sweep_stage``   (risk_aversion x tcost_bps cells)
  2. ``run_gamma_sweep_stage``             (alpha_uncertainty_penalty cells)
  3. ``run_cov_estimator_sweep_stage``     (covariance-estimator cells)
  4. ``run_predictor_param_sweep``         (vectorized/random predictor combos)

Operator-confirmed design (config#1153 audit, resolved 2026-07-13): the
multiple-testing count is the CUMULATIVE SUM, since inception, across ALL
4 producers — not any single producer's per-cycle count. This module is the
shared chokepoint (lift-to-chokepoint rule — >=2 producers, in this case 4,
already need it) both the writer (crucible-backtester, all 4 call sites) and
the reader (crucible-evaluator's DSR metric) depend on.

Persisted artifact (single shared JSON, NOT per-date):

    s3://{bucket}/backtest/cumulative_trial_count.json
    {
      "total": 1234,
      "last_updated": "2026-07-14",
      "by_producer": {
        "optimizer_param_sweep": 456,
        "gamma_sweep": 210,
        "cov_estimator_sweep": 320,
        "predictor_param_sweep": 248
      }
    }

Concurrency: multiple producers can finish in the same cycle (or even the
same second, if run_date's phases are parallelized later), so increments use
an S3-ETag conditional PUT (read-modify-write, ``IfMatch``) with bounded
retry rather than a blind read-then-put — the same "single accurate counter
under concurrent writers" problem the krepis conditional-PUT primitives
(``krepis.locks``) exist for, but this is a monotonic-counter merge rather
than a mutual-exclusion lock, so it gets its own tiny helper instead of
reusing the writer-lock context manager.

Skip-detection: a producer cycle that reused a prior marker's result (e.g.
``predictor_param_sweep`` via ``PhaseRegistry(..., supports_auto_skip=True)``
when ``ctx.skipped`` is True) generated ZERO new trials that cycle. Callers
MUST NOT call :func:`increment_trial_count` for a skipped/reused cycle — this
module does not attempt to infer skip state itself (it has no visibility into
the caller's phase registry), it only guards against the trivial ``n_trials
<= 0`` no-op case.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TypedDict

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "alpha-engine-research"
DEFAULT_KEY = "backtest/cumulative_trial_count.json"
_MAX_RETRIES = 5
_RETRY_BASE_SLEEP_SECONDS = 0.1


class CumulativeTrialCount(TypedDict):
    total: int
    last_updated: str
    by_producer: dict[str, int]


def _client(s3_client=None):
    if s3_client is not None:
        return s3_client
    import boto3

    return boto3.client("s3")


def _empty_state() -> CumulativeTrialCount:
    return {"total": 0, "last_updated": "", "by_producer": {}}


def _get_with_etag(
    s3_client, bucket: str, key: str,
) -> tuple[CumulativeTrialCount, str | None]:
    """Read the current counter + its ETag. Returns (state, etag).

    ``etag=None`` means the object does not exist yet — the caller does an
    unconditional create (``IfNoneMatch="*"``) instead of a conditional
    update.
    """
    from botocore.exceptions import ClientError

    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404"):
            return _empty_state(), None
        raise
    body = obj["Body"].read()
    etag = obj.get("ETag")
    try:
        state = json.loads(body)
    except json.JSONDecodeError:
        logger.warning(
            "trial_accumulator: malformed JSON at s3://%s/%s — treating as empty",
            bucket, key,
        )
        return _empty_state(), etag
    # Defensive defaults so a hand-edited or partially-written artifact
    # doesn't KeyError the merge below.
    state.setdefault("total", 0)
    state.setdefault("last_updated", "")
    state.setdefault("by_producer", {})
    return state, etag  # type: ignore[return-value]


def read_cumulative_trial_count(
    bucket: str = DEFAULT_BUCKET, *, key: str = DEFAULT_KEY, s3_client=None,
) -> CumulativeTrialCount:
    """Read the current cumulative trial count. Returns the empty state
    (``total=0``) if the artifact does not exist yet — callers (e.g. the DSR
    metric) should treat that as "not enough history for a trial count yet"
    rather than an error; the caller decides whether ``total=0`` should
    render as N/A (compute_dsr requires ``n_trials >= 1``).
    """
    state, _etag = _get_with_etag(_client(s3_client), bucket, key)
    return state


def increment_trial_count(
    producer: str,
    n_trials: int,
    run_date: str,
    *,
    bucket: str = DEFAULT_BUCKET,
    key: str = DEFAULT_KEY,
    s3_client=None,
) -> CumulativeTrialCount:
    """Add ``n_trials`` to the running total under ``by_producer[producer]``.

    Callers MUST only call this for a cycle that actually ran new trials —
    see module docstring's Skip-detection section. This function itself only
    guards the trivial case: ``n_trials <= 0`` is a no-op (logged, not
    raised, since a producer legitimately reporting 0 cells — e.g. an
    empty/disabled sweep grid — shouldn't crash the pipeline).

    Uses conditional PUT (``IfMatch`` the ETag just read, or ``IfNoneMatch:
    "*"`` if the artifact doesn't exist yet) with bounded retry so concurrent
    incrementers (two producers landing in the same run) merge correctly
    instead of a last-write-wins clobber.
    """
    if n_trials <= 0:
        logger.info(
            "trial_accumulator: skipping increment for producer=%s "
            "n_trials=%d (non-positive, no-op)",
            producer, n_trials,
        )
        return read_cumulative_trial_count(bucket, key=key, s3_client=s3_client)

    s3 = _client(s3_client)
    from botocore.exceptions import ClientError

    last_state: CumulativeTrialCount = _empty_state()
    for attempt in range(_MAX_RETRIES):
        state, etag = _get_with_etag(s3, bucket, key)
        new_total = int(state["total"]) + int(n_trials)
        new_by_producer = dict(state["by_producer"])
        new_by_producer[producer] = int(new_by_producer.get(producer, 0)) + int(n_trials)
        new_state: CumulativeTrialCount = {
            "total": new_total,
            "last_updated": run_date,
            "by_producer": new_by_producer,
        }
        last_state = new_state
        body = json.dumps(new_state, indent=2, sort_keys=True).encode("utf-8")
        put_kwargs = dict(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
        if etag is None:
            put_kwargs["IfNoneMatch"] = "*"
        else:
            put_kwargs["IfMatch"] = etag
        try:
            s3.put_object(**put_kwargs)
            logger.info(
                "trial_accumulator: producer=%s +%d trials → total=%d "
                "(s3://%s/%s, attempt=%d)",
                producer, n_trials, new_total, bucket, key, attempt + 1,
            )
            return new_state
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in ("PreconditionFailed", "412"):
                # Someone else wrote between our GET and PUT. Backoff + retry
                # with a fresh read — the merge is commutative (sum), so a
                # retry against the new state is always correct.
                sleep_s = _RETRY_BASE_SLEEP_SECONDS * (2 ** attempt)
                logger.info(
                    "trial_accumulator: conditional PUT race on attempt %d "
                    "(producer=%s) — retrying in %.2fs",
                    attempt + 1, producer, sleep_s,
                )
                time.sleep(sleep_s)
                continue
            raise

    raise RuntimeError(
        f"trial_accumulator: failed to increment producer={producer!r} "
        f"n_trials={n_trials} after {_MAX_RETRIES} conditional-PUT retries "
        f"(persistent contention at s3://{bucket}/{key})"
    )


def backfill_cumulative_trial_count(
    per_producer_totals: dict[str, int],
    run_date: str,
    *,
    bucket: str = DEFAULT_BUCKET,
    key: str = DEFAULT_KEY,
    s3_client=None,
    overwrite: bool = False,
) -> CumulativeTrialCount:
    """One-time seed of the counter from historical per-cycle sums.

    ``per_producer_totals`` is the ALREADY-SUMMED historical trial count per
    producer (e.g. ``{"optimizer_param_sweep": 4230, ...}``), typically
    computed by scanning every dated ``backtest/{run_date}/{producer}.json``
    archive in S3 and summing each cycle's ``n_trials`` field (see
    ``crucible-backtester``'s ``scripts/backfill_trial_count.py``).

    Refuses to clobber an existing non-empty artifact unless
    ``overwrite=True`` — the backfill is meant to run exactly once, before
    any producer has incremented the counter live; running it twice (or
    after live increments have already started) would double-count history.
    """
    s3 = _client(s3_client)
    existing, etag = _get_with_etag(s3, bucket, key)
    if existing["total"] != 0 and not overwrite:
        raise RuntimeError(
            f"trial_accumulator: refusing backfill — s3://{bucket}/{key} "
            f"already has total={existing['total']} (pass overwrite=True to "
            f"force). The backfill is a one-time seed and should not run "
            f"against a counter that has already accrued live increments."
        )

    new_state: CumulativeTrialCount = {
        "total": sum(int(v) for v in per_producer_totals.values()),
        "last_updated": run_date,
        "by_producer": {k: int(v) for k, v in per_producer_totals.items()},
    }
    body = json.dumps(new_state, indent=2, sort_keys=True).encode("utf-8")
    put_kwargs = dict(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    if etag is not None:
        put_kwargs["IfMatch"] = etag
    else:
        put_kwargs["IfNoneMatch"] = "*"
    s3.put_object(**put_kwargs)
    logger.info(
        "trial_accumulator: backfilled s3://%s/%s total=%d by_producer=%s",
        bucket, key, new_state["total"], new_state["by_producer"],
    )
    return new_state
