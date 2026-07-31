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
from datetime import UTC, datetime

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
    cadence_minutes: int,
    findings: list[dict] | None = None,
    deep_link: str | None = None,
    now: datetime | None = None,
) -> dict:
    """The envelope. Pure — every producer's tests assert against this."""
    if status not in _VALID:
        raise ValueError(f"status must be one of {_VALID}, got {status!r}")
    if not check_id or "/" in check_id:
        raise ValueError(f"check_id must be a single path segment, got {check_id!r}")
    if cadence_minutes <= 0:
        raise ValueError(
            f"cadence_minutes must be positive, got {cadence_minutes!r} — the "
            f"console derives its staleness threshold from it, and a zero or "
            f"negative cadence makes every publish instantly stale"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "check_id": check_id,
        "label": label,
        "ran_at": (now or datetime.now(UTC)).isoformat(),
        "status": status,
        "summary": summary,
        "cadence_minutes": cadence_minutes,
        "deep_link": deep_link,
        "findings": list(findings or []),
    }


def key_for(check_id: str) -> str:
    return f"{CHECKS_PREFIX}{check_id}/latest.json"


def emit(envelope: dict, *, dry_run: bool = False) -> str | None:
    """Publish an envelope. Returns the s3:// URI, or None if nothing was
    written.

    Never raises — see the module docstring. A check must not go red because
    its telemetry did."""
    key = key_for(envelope["check_id"])
    uri = f"s3://{RESEARCH_BUCKET}/{key}"
    if dry_run:
        logger.info("[dry-run] would publish %s (%s)", uri, envelope["status"])
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
    cadence_minutes: int,
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
