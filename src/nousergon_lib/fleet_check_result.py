#!/usr/bin/env python3
"""The fleet check-result envelope: one shape every scheduled check publishes,
and one console surface renders (config-I5548 / I5507).

WHY A SHARED EMITTER RATHER THAN A COPY PER CHECK
-------------------------------------------------
`check_iam_grant_usage.py` built the envelope inline as the first producer.
This is adoption two through four (`check_scheduled_workflow_health`,
`deploy_release_standard_sweep`, `lib_pin_drift_sweep`) — the second adoption is
the fleet's consolidation trigger, and four hand-rolled copies of a schema the
console parses is exactly how a contract drifts into four dialects.

WHY IT LIVES IN THE LIB (2026-07-31, alpha-engine-config-I5863)
---------------------------------------------------------------
It was a module inside `alpha-engine-config/scripts/`, which was correct while
every producer lived in that one repo. `crucible-dashboard` is now a producer
too — the dashboard box publishes its own memory-headroom check, and cgroup
facts exist only on the box, so that producer cannot be re-homed into a GHA
sweep beside its siblings. A second REPO is `shared-code-policy.md` §2's lift
trigger: "copy once; on the second consumer, lift it."

`alpha-engine-config` still carries its own copy for now. Its five production
checks run on bare runners with no lib install, so making them import this
would mean adding a network install to five scheduled workflows in the same
change that ships a new check — staged deliberately rather than silently, per
`principles.md` §2.4, and the interim duplication is held legitimate by a
contract test in that repo asserting the two modules still agree on the
envelope they produce (`shared-code-policy.md` §5.1's fork-detection backstop).
Migration is tracked in alpha-engine-config-I5865.

THIS MODULE IS THE PRODUCER HALF OF A CROSS-REPO CONTRACT. The consumer is
`crucible-dashboard/loaders/fleet_checks_loader.py`. Changing a field name here
without a paired dashboard change silently blanks a console row.

THE CONTRACT
------------
    s3://alpha-engine-research/ops/checks/{check_id}/latest.json

    { "schema_version": 1,
      "check_id":  "lib_pin_drift",              # == the S3 path segment
      "label":     "Lib-pin drift (co-install pair)",
      "ran_at":    "2026-07-29T15:00:00+00:00",  # when the check ACTUALLY ran
      "status":    "ok" | "attention" | "error",
      "summary":   "one line an operator can act on",
      "cadence_minutes": 1440,                   # what it CLAIMS its cadence is
      "deep_link": "https://…",                  # optional, per-check evidence
      "findings":  [{"key": "...", "detail": "..."}] }

The console (`crucible-dashboard` `loaders/fleet_checks_loader.py`) discovers
producers **by S3 prefix**, so a check appears on its first successful publish
with no console deploy. That is the whole point: checks reported to Telegram
only because surfacing one used to cost a dashboard PR.

`ran_at` + `cadence_minutes` are what let the console mark a check STALE when
it stops publishing, whatever status it last wrote — the last thing a dying
check writes is almost always "ok". Emitting an honest `cadence_minutes` is
therefore not decoration; understating it makes a check page early, overstating
it lets a dead check read healthy for longer.

WHY EMISSION NEVER RAISES
-------------------------
A check's job is its check. If S3 is unavailable, the right outcome is a logged
warning and the check's own verdict still reaching its exit code and logs — not
a green check turned red by its telemetry. The console renders a missing
artifact as `unreadable`, never `ok`, so a silent emit failure degrades to a
visible gap rather than a false all-clear.
"""

from __future__ import annotations

import json
import logging
import os

# `datetime.UTC` is 3.11+, and this package declares requires-python >=3.9.
# The alpha-engine-config original could use it because that repo runs 3.12
# only; the lib's CI matrix starts at 3.9, which is where it broke on the lift.
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

RESEARCH_BUCKET = "alpha-engine-research"
CHECKS_PREFIX = "ops/checks/"
SCHEMA_VERSION = 1

STATUS_OK = "ok"
STATUS_ATTENTION = "attention"
STATUS_ERROR = "error"
_VALID = (STATUS_OK, STATUS_ATTENTION, STATUS_ERROR)


def build(
    *,
    check_id: str,
    label: str,
    status: str,
    summary: str,
    cadence_minutes: int | None,
    findings: list[dict] | None = None,
    deep_link: str | None = None,
    now: datetime | None = None,
) -> dict:
    """The envelope. Pure — every producer's tests assert against this.

    `cadence_minutes=None` is the honest declaration for an EVENT-DRIVEN
    producer — one with no clock, only a trigger (a merge, a queue message,
    an EventBridge event pattern) — and is not a default anyone falls into by
    omission: every existing call site names a number
    (`alpha-engine-config-I9033`). The console's `checks-envelope` adapter
    already treats a missing `cadence_minutes` as "no freshness input" and
    skips staleness entirely rather than computing one against a manufactured
    number — the alternative, flooring an unscheduled workflow's cadence to
    this publisher's own 4-hour observation interval, downgrades it to
    `MISSED` a few hours after every legitimate merge-triggered run, which is
    the false positive this parameter exists to prevent. `None` still writes
    the key (`"cadence_minutes": null`) rather than omitting it — the
    consumer contract enumerates the field as always-present.
    """
    if status not in _VALID:
        raise ValueError(f"status must be one of {_VALID}, got {status!r}")
    if not check_id or "/" in check_id:
        raise ValueError(f"check_id must be a single path segment, got {check_id!r}")
    if cadence_minutes is not None and cadence_minutes <= 0:
        raise ValueError(
            f"cadence_minutes must be positive or None, got {cadence_minutes!r} "
            f"— the console derives its staleness threshold from it, and a "
            f"zero or negative cadence makes every publish instantly stale"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "check_id": check_id,
        "label": label,
        "ran_at": (now or datetime.now(timezone.utc)).isoformat(),
        "status": status,
        "summary": summary,
        "cadence_minutes": cadence_minutes,
        "deep_link": deep_link,
        "findings": list(findings or []),
    }


def key_for(check_id: str) -> str:
    return f"{CHECKS_PREFIX}{check_id}/latest.json"


#: Set to "1" to let a test process actually publish. The ONLY legitimate user
#: is a test that is deliberately exercising this function's S3 write against a
#: throwaway bucket; nothing in a normal suite should need it.
ALLOW_TEST_WRITES_ENV = "FLEET_CHECK_RESULT_ALLOW_TEST_WRITES"


def _running_under_test() -> bool:
    """Is this process a test runner?

    `PYTEST_CURRENT_TEST` is set by pytest for the duration of every test and
    unset outside one, so it answers "am I inside a test right now", not the
    weaker "was this interpreter started by pytest" that `sys.modules` would
    answer. `UNITTEST_CURRENT_TEST` covers the stdlib runner some box scripts
    still use.
    """
    return bool(os.environ.get("PYTEST_CURRENT_TEST")
                or os.environ.get("UNITTEST_CURRENT_TEST"))


def emit(envelope: dict, *, dry_run: bool = False) -> str | None:
    """Publish an envelope. Returns the s3:// URI, or None if nothing was
    written.

    Never raises — see the module docstring. A check must not go red because
    its telemetry did.

    A TEST PROCESS NEVER WRITES (alpha-engine-config-I9052). `ops/checks/<id>/
    latest.json` is a single mutable production object that the console reads
    as a component's live state, so a producer test that reaches this function
    unstubbed does not fail — it silently republishes its own fixture as the
    fleet's truth. That is what happened on 2026-08-31: nous-ergon-ops'
    `test_a_failed_restore_keeps_the_deadman_armed` drove
    `router_degraded_mode_drill.main([])` with `restore_router` raising, and
    the real `emit_check_envelope` published `status: error`, `summary:
    "CRITICAL: restore failed: did not come back"`, `deep_link: "s3://test"`
    over the genuine 2026-08-25 drill run. The console rendered the drill
    FAILED, which withheld `ARMED` from the Lambda probe row anchored to it and
    held the fleet transparency gap at 1.

    The guard lives HERE rather than in each producer's tests because the
    exposure is the whole class: every `emit`/`emit_result` call site in every
    repo, present and future, is one unstubbed test away from the same write,
    and the `except Exception` below means it fails silently and invisibly on
    any machine WITHOUT credentials — so the defect is undetectable exactly
    where it is harmless and lands in production exactly where it is not.
    Stubbing the two known tests fixes two tests; this fixes the class.
    """
    key = key_for(envelope["check_id"])
    uri = f"s3://{RESEARCH_BUCKET}/{key}"
    if dry_run:
        logger.info("[dry-run] would publish %s (%s)", uri, envelope["status"])
        return None
    if _running_under_test() and os.environ.get(ALLOW_TEST_WRITES_ENV) != "1":
        logger.warning(
            "REFUSING to publish %s from a test process — %s is a live console "
            "surface. Stub this call in the test, or set %s=1 if the write is "
            "the thing under test.", uri, key, ALLOW_TEST_WRITES_ENV)
        return None
    try:
        import boto3
        boto3.client("s3").put_object(
            Bucket=RESEARCH_BUCKET, Key=key,
            Body=json.dumps(envelope, indent=2).encode(),
            ContentType="application/json",
        )
    except Exception:  # noqa: BLE001 — a failed publish must not fail the check
        logger.warning(
            "could not publish check result to %s — the console will render this "
            "check as `unreadable` (never `ok`), so the gap is visible", uri,
            exc_info=True,
        )
        return None
    return uri


def emit_result(
    *,
    check_id: str,
    label: str,
    status: str,
    summary: str,
    cadence_minutes: int | None,
    findings: list[dict] | None = None,
    deep_link: str | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
) -> str | None:
    """build + emit, for the common one-line call site."""
    return emit(
        build(check_id=check_id, label=label, status=status, summary=summary,
              cadence_minutes=cadence_minutes, findings=findings,
              deep_link=deep_link, now=now),
        dry_run=dry_run,
    )
