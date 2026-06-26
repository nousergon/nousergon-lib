"""
artifact_resolution.py — the CONSUMER half of the artifact-resilience principle
(alpha-engine-config#1190): resolve a dated S3 artifact to the FRESHEST instance
within a trailing window, never to one exact run-date key.

**Why this exists (the "clean Saturday run" redefinition).** The Saturday /
weekday / EOD Step Functions are allowed to FAIL and be retried / redriven
multiple times across days; a single continuous green execution is NOT a
precondition any consumer may assume. "Clean Saturday run" is therefore
redefined: not "one uninterrupted green execution," but "the freshest required
artifacts each exist within their freshness window," however many partial /
retried / off-cycle runs produced them (Brian, 2026-06-23: "there should be NO
PART OF THIS SYSTEM that gets crippled simply because the saturday sf didn't run
continuously in a single push"). A consumer that reads ``backtest/{run_date}/x``
at the EXACT run_date reads N/A the moment the producer ran a day off-cycle —
even though the artifact it needs HAS been produced.

**Single source of truth.** Before this module, the windowed-resolution pattern
had been re-implemented independently at least four times:

  - ``crucible-evaluator`` ``grading.artifacts.get_json_windowed`` — the keystone
    (freshest JSON within 10d, skips corrupt mid-writes).
  - ``crucible-executor`` ``executor.signal_reader.read_signals_with_fallback``
    — latest-pointer-then-date-scan (14d) for ``signals/{date}/signals.json``.
  - ``crucible-executor`` ``executor.eod_reconcile._load_signals_from_s3`` — an
    ad-hoc copy of the same backward scan.
  - ``crucible-predictor`` / ``crucible-backtester`` ``load_universe`` /
    ``backtest`` — further mirrors of "most recent signals within N days".

Each copy drifts (different windows, some skip corrupt mid-writes and some
don't, some try a ``latest.json`` pointer and some don't). This module is the
ONE place the rule lives so every consumer resolves artifacts identically; the
sibling freshness MONITOR (``nousergon_lib.artifact_freshness``) already obeys
the same "freshest within a window, never the exact key" rule on the alerting
side.

**Public surface:**

- :func:`resolve_windowed_artifact` — generic ``(s3, bucket, key_template,
  run_date) -> ResolvedArtifact``. Walks back one calendar day at a time from
  ``run_date`` and returns the freshest existing object key within the window.
  Optionally tries a fixed ``latest`` pointer first.
- :func:`get_json_windowed` — JSON convenience wrapper returning
  ``(doc, src_date, age_days, key)`` — the exact tuple shape the evaluator
  keystone exposes, so it is a drop-in for ``grading.artifacts.get_json_windowed``.
- :data:`DEFAULT_ARTIFACT_MAX_AGE_DAYS` — 10 (a weekly artifact + slack for
  multi-day recovery).

**Fail-loud posture** (``[[feedback_no_silent_fails]]``): a *missing* key
(``NoSuchKey`` / 404) is a legitimate "keep walking back" signal; a *corrupt /
half-written* JSON candidate (a crashed mid-write from a failed pipeline attempt)
is SKIPPED and the scan continues to the last good instance; any *other* S3 error
(auth / throttle / wrong-bucket / network) is an upstream contract violation and
is RAISED — never graded on a partial read we can't explain.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import dataclass
from typing import Any

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


# Resilience keystone window (config#1190 — the "clean Saturday run"
# redefinition): the consumer must reflect whatever artifacts EXIST, accumulated
# across PARTIAL / RETRIED / OFF-CYCLE pipeline runs — never require every
# artifact at a single continuous run_date. A weekly artifact + slack for
# multi-day recovery → 10 days. (The executor signals reader uses 14 to also
# cover a fully-missed weekly cycle + buffer; callers override as needed.)
DEFAULT_ARTIFACT_MAX_AGE_DAYS = 10

# S3 error codes that mean "this key does not exist" (so: keep walking back),
# as opposed to a real upstream failure we must surface.
_NOT_FOUND_CODES = frozenset({"NoSuchKey", "404", "NotFound"})


@dataclass
class ResolvedArtifact:
    """The outcome of one windowed resolution.

    Attributes:
        key: The resolved S3 key of the freshest instance, or ``None`` if no
            instance exists in the window.
        src_date: ISO date the resolved instance is keyed under (``None`` if
            none found, or the literal ``run_date`` for a non-ISO run_date that
            fell through to an exact read). This is the real provenance — surface
            it as the consumer's ``source_path`` so staleness stays visible.
        age_days: Calendar days between ``run_date`` and ``src_date`` (0 for an
            exact-date hit; ``None`` if none found).
        used_pointer: ``True`` when the fixed ``latest`` pointer satisfied the
            read before any date scan (so ``src_date`` came from the pointer's
            payload, if the caller resolved it).
    """

    key: str | None = None
    src_date: str | None = None
    age_days: int | None = None
    used_pointer: bool = False

    @property
    def found(self) -> bool:
        return self.key is not None


def _is_not_found(err: ClientError) -> bool:
    code = err.response.get("Error", {}).get("Code")
    return code in _NOT_FOUND_CODES


def _head_exists(s3: Any, bucket: str, key: str) -> bool:
    """True iff ``key`` exists. Raises on any non-404 ClientError."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if _is_not_found(e):
            return False
        logger.error("S3 HEAD failed for s3://%s/%s: %s", bucket, key, e)
        raise
    return True


def get_json(s3: Any, bucket: str, key: str) -> dict | None:
    """Read one JSON object from S3.

    Returns the parsed dict, or ``None`` if the key does not exist
    (``NoSuchKey`` / 404). Raises on any other ``ClientError`` (a real S3
    problem — auth / throttle / wrong-bucket / network) and on malformed JSON
    (the caller decides whether a corrupt candidate is skippable — see
    :func:`get_json_windowed`).
    """
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if _is_not_found(e):
            return None
        logger.error("S3 read failed for s3://%s/%s: %s", bucket, key, e)
        raise
    return json.loads(resp["Body"].read())


def _walk_back_dates(run_date: str, max_age_days: int):
    """Yield ``(iso_date, age_days)`` from ``run_date`` backward, inclusive.

    Raises ``ValueError`` if ``run_date`` is not an ISO date — callers handle
    the non-ISO fall-through (an exact read at the literal value).
    """
    day = _dt.date.fromisoformat(run_date)
    for delta in range(max_age_days + 1):
        yield (day - _dt.timedelta(days=delta)).isoformat(), delta


def resolve_windowed_artifact(
    s3: Any,
    bucket: str,
    key_template: str,
    run_date: str,
    *,
    max_age_days: int = DEFAULT_ARTIFACT_MAX_AGE_DAYS,
    latest_pointer_key: str | None = None,
) -> ResolvedArtifact:
    """Resolve ``key_template`` to the FRESHEST existing instance at/before
    ``run_date`` within ``max_age_days``, walking back one calendar day at a time.

    ``key_template`` carries a ``{date}`` placeholder, e.g.
    ``"backtest/{date}/e2e_lift.json"``. Existence is checked with HEAD (cheap,
    no body download) — use :func:`get_json_windowed` when you also want the
    parsed JSON and corrupt-candidate skipping.

    If ``latest_pointer_key`` is given (a fixed, non-dated key such as
    ``"signals/latest.json"`` that the producer writes alongside the dated file),
    it is HEADed FIRST; a hit short-circuits the date scan and returns it with
    ``used_pointer=True`` (the caller resolves the real ``src_date`` from the
    pointer payload if it needs provenance).

    Returns a :class:`ResolvedArtifact`; ``.found is False`` when no instance
    exists in the window. Raises on any non-404 S3 error (an upstream contract
    violation we must not silently grade through).
    """
    if latest_pointer_key is not None and _head_exists(s3, bucket, latest_pointer_key):
        return ResolvedArtifact(key=latest_pointer_key, used_pointer=True)

    try:
        candidates = list(_walk_back_dates(run_date, max_age_days))
    except (ValueError, TypeError):
        # Non-ISO run_date — fall back to a single exact read at the literal value.
        key = key_template.format(date=run_date)
        if _head_exists(s3, bucket, key):
            return ResolvedArtifact(key=key, src_date=run_date, age_days=0)
        return ResolvedArtifact()

    for iso, age in candidates:
        key = key_template.format(date=iso)
        if _head_exists(s3, bucket, key):
            return ResolvedArtifact(key=key, src_date=iso, age_days=age)
    return ResolvedArtifact()


def get_json_windowed(
    s3: Any,
    bucket: str,
    key_template: str,
    run_date: str,
    *,
    max_age_days: int = DEFAULT_ARTIFACT_MAX_AGE_DAYS,
) -> tuple[dict | None, str | None, int | None, str | None]:
    """Resolve a dated JSON artifact to the FRESHEST instance at/before
    ``run_date`` within ``max_age_days``, walking back one calendar day at a time.

    Returns ``(doc, src_date, age_days, key)`` — or ``(None, None, None, None)``
    if no instance exists in the window. ``key_template`` carries a ``{date}``
    placeholder, e.g. ``"backtest/{date}/e2e_lift.json"``.

    This is the consumer half of the artifact-resilience principle: a producer
    that ran on a partial / earlier attempt this week still grades, instead of
    the metric reading N/A because THIS run_date's pipeline didn't reach that
    stage. The returned ``key`` (carrying the real artifact date) is meant to be
    used as the consumer's ``source_path`` so provenance — and any staleness —
    stays visible, never silently graded as "today".

    A corrupt / empty / partially-written candidate (a crashed mid-write from a
    failed pipeline attempt) is NOT a usable instance — it is SKIPPED and the
    scan keeps walking back to the last GOOD one. A real S3 ``ClientError`` (auth
    / throttle / wrong-bucket / network) still propagates.

    This is the single source of truth lifted from
    ``crucible-evaluator.grading.artifacts.get_json_windowed`` (config#1190).
    """
    try:
        candidates = list(_walk_back_dates(run_date, max_age_days))
    except (ValueError, TypeError):
        # Non-ISO run_date — fall back to a single exact read at the literal value.
        key = key_template.format(date=run_date)
        doc = get_json(s3, bucket, key)
        return (doc, run_date, 0, key) if doc is not None else (None, None, None, None)

    for iso, age in candidates:
        key = key_template.format(date=iso)
        try:
            doc = get_json(s3, bucket, key)
        except (json.JSONDecodeError, ValueError) as e:
            # A corrupt / empty / partially-written artifact (a crashed mid-write
            # from a failed pipeline attempt) is NOT a usable instance — skip it
            # and keep walking back to the last GOOD one. This is the resilience
            # point: a half-written file from a non-continuous run must not crash
            # the consumer. A real S3 ClientError still propagates (get_json raises).
            logger.warning("Skipping corrupt artifact s3://%s/%s: %s", bucket, key, e)
            continue
        if doc is not None:
            return doc, iso, age, key
    return None, None, None, None
