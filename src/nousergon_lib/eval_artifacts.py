"""
eval_artifacts.py — Canonical S3 layout for eval-style judgment artifacts.

Codifies the institutional partition template that emerged from two
parallel implementations:

- alpha-engine-research ``evals/orchestrator.py`` — LLM-as-judge rubric
  evaluations of decision-capture artifacts (shipped 2026-05-08;
  alpha-engine-research #143/#144/#145).
- alpha-engine-predictor ``analysis/triple_barrier_cutover_runner.py`` —
  Stage 3 cutover-gate evaluations of parallel triple-barrier predictions
  (shipped 2026-05-10; alpha-engine-predictor #129+).

Both pipelines share the same shape: a JUDGMENT artifact produced by a
batch invocation against captured source data, with a need to preserve
forensic capture across same-day re-runs. This module is the single
source of truth for the partition / run-identifier / sidecar conventions
those pipelines share, so future eval-style pipelines (Stage 4
continuous regime feature gate, Stage 5 meta-labeling, etc.) ship
LdP-correct dating + run-identifier discipline by default.

Canonical S3 layout::

    {prefix}/
      {run_id}.json    ← per-invocation artifact (YYMMDDHHMM encodes date)
      latest.json      ← single-fetch operator UX (mirror)

Where:

- ``prefix`` is the eval pipeline's S3 prefix (e.g.,
  ``predictor/variant_gates/triple_barrier``).
- ``run_id`` is a structured timestamp produced by
  :func:`new_eval_run_id` — ``YYMMDDHHMM`` (year, month, day, hour,
  minute, UTC). Sortable lexicographically across the entire prefix —
  no date partition needed because the timestamp itself encodes the
  date. Listings yield chronological order automatically.

The flat layout (no ``{calendar_date}/`` sub-partition) is deliberate:
once the run_id is timestamp-encoded, a date prefix is pure redundancy.
A weekly-cadence eval pipeline accumulates ~52 entries/year — trivial
for S3 list operations even over multi-year history. Date scoping is
still possible by listing with ``StartAfter="{prefix}/2605"`` (everything
in May 2026) etc.

The artifact payload still carries both ``calendar_date`` (UTC wall-clock)
and ``trading_day`` (last closed NYSE session) per
``DATE_CONVENTIONS.md`` dual-tracking — those are FACTS about the
artifact for downstream join queries, distinct from the path's role as
addressing.

Same-minute collisions are by design (see ``new_eval_run_id`` docstring):
production cron cadence makes them effectively impossible. Sub-minute
re-runs would overwrite — for tests and deterministic-id needs, callers
inject explicit ``run_id`` strings to whatever helper consumes them.

The ``latest.json`` sidecar provides a stable single-fetch endpoint for
dashboards / evaluator email rendering / operator scripts.

Use this only for **eval-style judgment artifacts** — pipelines that
produce a verdict against captured source data and may run multiple
times per day. Config-recommendation artifacts (assembler, regression
monitor, cost-anomaly) use simple ``{date}.json`` partitioning since
they're single-writer-per-day and overwrite-on-rerun is the desired
semantic (the latest verdict IS the canonical config).

Example::

    from nousergon_lib.dates import now_dual
    from nousergon_lib.eval_artifacts import (
        new_eval_run_id, eval_artifact_key, eval_latest_key,
    )

    dual = now_dual()
    run_id = new_eval_run_id()
    payload = {
        "calendar_date": dual.calendar_date,
        "trading_day": dual.trading_day,
        "run_id": run_id,
        "verdict": ...,
    }
    dated_key = eval_artifact_key(
        prefix="predictor/variant_gates/triple_barrier",
        run_id=run_id,
    )
    latest_key = eval_latest_key(
        prefix="predictor/variant_gates/triple_barrier",
    )
    s3.put_object(Bucket=bucket, Key=dated_key, Body=...)
    s3.put_object(Bucket=bucket, Key=latest_key, Body=...)  # mirror
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# Stable filename for the operator-UX single-fetch sidecar. Constant
# rather than configurable so dashboards / scripts can hard-code it.
EVAL_LATEST_FILENAME: str = "latest.json"


def new_eval_run_id(*, now: datetime | None = None) -> str:
    """Mint a structured-timestamp run identifier in ``YYMMDDHHMM`` form.

    Returns the UTC wall-clock moment formatted as a 10-character
    ``YYMMDDHHMM`` string. Sortable lexicographically — the partition
    listing automatically yields chronological order. Human-readable
    in path listings, S3 console UI, and operator dashboards.

    Replaces the prior UUIDv4 convention (used in the early eval-judge
    +  triple-barrier-gate implementations) — UUIDs are globally unique
    but provide no temporal information at the path level. Operators
    routinely needed to open each JSON to see when it ran; the
    structured timestamp encodes that in the filename itself.

    Collision profile: minute granularity. Two runs firing in the same
    UTC minute would collide and overwrite. In production this is
    essentially impossible (Sat SF cron fires once weekly; ad-hoc
    operator runs are sparse). For tests and deterministic-id needs,
    callers can construct any ``YYMMDDHHMM``-shaped string and pass it
    explicitly to whatever helper consumes a ``run_id``.

    Args:
        now: optional UTC datetime override (testing / deterministic
            replay). When None, uses ``datetime.now(timezone.utc)``.

    Returns:
        10-character string ``YYMMDDHHMM``. Example: a run at 2026-05-10
        14:37 UTC returns ``"2605101437"``.
    """
    moment = now if now is not None else datetime.now(timezone.utc)
    if moment.tzinfo is None:
        # Treat naive datetimes as UTC (consistent with dates.now_dual).
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.strftime("%y%m%d%H%M")


def eval_artifact_key(
    prefix: str,
    run_id: str,
    *,
    basename: str = "result.json",
) -> str:
    """Format the canonical S3 key for an eval-style artifact.

    Returns ``{prefix}/{run_id}.json`` when ``basename`` is the default;
    otherwise ``{prefix}/{run_id}_{basename}`` (multi-file-per-run
    pipelines). No date sub-partition — the YYMMDDHHMM run_id encodes
    the date itself, so listings yield chronological order across the
    full prefix without a ``{calendar_date}/`` partition.

    Two forms supported:

    - **Single-file-per-run pipelines** (e.g., cutover gate): one JSON
      per invocation, default basename.
    - **Multi-file-per-run pipelines** (e.g., eval-judge with per-stage
      outputs): caller supplies per-file basename like
      ``"haiku_eval.json"``, ``"sonnet_escalation.json"``, etc. The
      run_id prefix keeps files for the same run grouped in path
      listings.

    Trailing/leading slashes on ``prefix`` are normalized away. ``run_id``
    is NOT validated here — callers should derive it via
    :func:`new_eval_run_id`.

    Args:
        prefix: S3 prefix root for the eval pipeline. Example:
            ``"predictor/variant_gates/triple_barrier"``.
        run_id: ``YYMMDDHHMM`` string from :func:`new_eval_run_id` (or
            any caller-supplied identifier — the function does not
            constrain shape, only formats the key).
        basename: per-file name. Defaults to ``"result.json"`` →
            simplified to ``{run_id}.json``. Any other basename →
            ``{run_id}_{basename}`` to preserve run-id grouping in
            sub-file listings.

    Returns:
        Fully-formatted S3 key string.
    """
    prefix_clean = prefix.strip("/")
    if basename == "result.json":
        return f"{prefix_clean}/{run_id}.json"
    return f"{prefix_clean}/{run_id}_{basename}"


def eval_latest_key(prefix: str) -> str:
    """Format the canonical S3 key for the operator-UX latest sidecar.

    Returns ``{prefix}/latest.json``. Pure mirror of the most-recently-
    written dated artifact for the pipeline; the dated key remains the
    forensic source of truth so re-runs are preserved.

    Trailing/leading slashes on ``prefix`` are normalized away.

    Args:
        prefix: S3 prefix root for the eval pipeline.

    Returns:
        S3 key string for the latest sidecar.
    """
    return f"{prefix.strip('/')}/{EVAL_LATEST_FILENAME}"


# ─ Canonical readers ──────────────────────────────────────────────────
# Symmetric counterpart to the write helpers above. Multiple repos
# need to read eval-artifact-shaped artifacts (regime substrate from
# research + predictor + executor + dashboard; future eval pipelines
# similarly) — these helpers are the single source of truth for the
# sidecar-resolution + history-listing patterns so consumers don't
# each reimplement them with slightly different fail-graceful semantics.


def load_latest_eval_artifact(
    s3_client: Any,
    *,
    bucket: str,
    prefix: str,
) -> dict | None:
    """Load the most recent eval-style artifact via canonical sidecar pointer.

    Resolution sequence (all-or-nothing, returns None on any failure):

    1. ``s3_client.get_object`` on ``{prefix}/latest.json``
    2. Parse sidecar JSON, extract ``artifact_key``
    3. ``s3_client.get_object`` on the artifact key
    4. Parse + return the artifact payload

    Returns ``None`` and logs at INFO level on any failure mode:
    missing sidecar, malformed sidecar, missing artifact_key in
    sidecar, missing artifact body, parse errors, transient S3 hiccups.
    Callers handle the None case to fall back to whatever default
    behavior makes sense for their domain (regime substrate: macro
    agent falls back to LLM + post-LLM-guardrail; etc.).

    The ``s3_client`` parameter is a boto3-like S3 client — any object
    that responds to ``get_object(Bucket=, Key=)`` with a dict whose
    ``Body`` field is readable. Pass a real ``boto3.client("s3")`` in
    production, or an in-memory stub in tests.

    Args:
        s3_client: boto3-like S3 client.
        bucket: S3 bucket name (e.g. ``"alpha-engine-research"``).
        prefix: S3 prefix root for the pipeline (e.g. ``"regime"``,
            ``"predictor/variant_gates/triple_barrier"``).

    Returns:
        Parsed artifact payload dict, or ``None`` if unavailable.
    """
    sidecar_key = eval_latest_key(prefix)
    try:
        sidecar_obj = s3_client.get_object(Bucket=bucket, Key=sidecar_key)
        sidecar = json.loads(sidecar_obj["Body"].read())
    except Exception as e:
        logger.info(
            "[load_latest_eval_artifact] sidecar read failed at s3://%s/%s (%s) — "
            "no artifact available yet",
            bucket, sidecar_key, type(e).__name__,
        )
        return None

    artifact_key = sidecar.get("artifact_key") if isinstance(sidecar, dict) else None
    if not artifact_key:
        logger.warning(
            "[load_latest_eval_artifact] sidecar at s3://%s/%s lacks artifact_key",
            bucket, sidecar_key,
        )
        return None

    try:
        body_obj = s3_client.get_object(Bucket=bucket, Key=artifact_key)
        return json.loads(body_obj["Body"].read())
    except Exception as e:
        logger.warning(
            "[load_latest_eval_artifact] artifact body read failed at s3://%s/%s (%s)",
            bucket, artifact_key, type(e).__name__,
        )
        return None


def list_eval_artifacts(
    s3_client: Any,
    *,
    bucket: str,
    prefix: str,
    n_recent: int | None = None,
) -> list[dict]:
    """List eval-style artifacts under ``prefix``, oldest → newest by run_id.

    Lists ``{prefix}/{YYMMDDHHMM}.json`` keys (canonical eval_artifacts
    shape — flat layout, no calendar_date sub-partition), filters out:

    - the ``latest.json`` sidecar (pure pointer, not an artifact)
    - any nested keys (none expected under the canonical shape)
    - filenames that aren't 10-digit run_ids (defensive — catches
      accidentally-written files)
    - non-``.json`` files

    Sorts lexicographically by run_id (= chronological since the
    timestamp encoding is left-padded), then loads up to ``n_recent``
    most-recent payloads. ``n_recent=None`` returns all.

    Partial progress on fetch failures — one S3 hiccup on a single
    artifact doesn't drop the rest of the window. Failed reads are
    logged at WARNING and silently skipped.

    Args:
        s3_client: boto3-like S3 client.
        bucket: S3 bucket name.
        prefix: S3 prefix root for the pipeline.
        n_recent: Cap on number of most-recent artifacts to load.
            ``None`` = all. Default ``None``.

    Returns:
        List of parsed artifact payload dicts, oldest → newest.
        Empty list when no artifacts exist (pre-deploy state).
    """
    prefix_clean = prefix.strip("/")
    list_prefix = f"{prefix_clean}/"

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        run_ids: list[tuple[str, str]] = []  # (run_id, artifact_key)
        for page in paginator.paginate(Bucket=bucket, Prefix=list_prefix):
            for obj in page.get("Contents", []):
                key = obj.get("Key", "")
                rel = key[len(list_prefix):]
                if "/" in rel or not rel.endswith(".json"):
                    continue
                run_id = rel[:-len(".json")]
                if run_id == "latest" or not run_id.isdigit() or len(run_id) != 10:
                    continue
                run_ids.append((run_id, key))
    except Exception as e:
        logger.warning(
            "[list_eval_artifacts] listing failed at s3://%s/%s (%s)",
            bucket, list_prefix, type(e).__name__,
        )
        return []

    run_ids.sort()  # lexicographic = chronological with YYMMDDHHMM
    if n_recent is not None and len(run_ids) > n_recent:
        run_ids = run_ids[-n_recent:]

    out: list[dict] = []
    for _run_id, key in run_ids:
        try:
            body_obj = s3_client.get_object(Bucket=bucket, Key=key)
            out.append(json.loads(body_obj["Body"].read()))
        except Exception as e:
            logger.warning(
                "[list_eval_artifacts] body read failed at s3://%s/%s (%s) — "
                "skipping this artifact",
                bucket, key, type(e).__name__,
            )
            continue
    return out
