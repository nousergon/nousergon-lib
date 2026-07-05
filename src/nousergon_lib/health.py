"""
Consolidated module-health *enrichment* writer/reader for the Alpha Engine
fleet.

**Why this exists.** Five copy-pasted health-writer implementations drifted
across the fleet — ``crucible-executor`` (``executor/health_status.py``),
``crucible-backtester`` (``health_status.py``), ``crucible-predictor``
(``health_status.py``), ``crucible-research`` (``health_status.py``), and
``nousergon-data`` (an inline ``_write_module_health`` in
``weekly_collector.py``). All five PUT the *same* JSON schema to
``s3://alpha-engine-research/health/{module}.json`` after every run, and
three of them additionally re-implement identical ``read_health`` /
``check_upstream_health`` readers. This module is the single consolidation
target (config#1727, Phase C of epic config#1724) the five call sites will
later import from.

**Framing: enrichment, not gating.** These stamps answer the honest *"why"*
on a freshness alert and back the dashboard's System Health page — they are
NOT a gating authority. The executor's safety gate migrates to *independent*
freshness in Phase A (config#1725); until that lands, the self-reported
stamp is still safety-critical, which is exactly why the five call-site
migrations are deferred to a follow-up sequenced *after* Phase A. This
module is therefore purely additive: it creates the target, it does not
move any caller.

**Public surface:**

- :class:`Deliverable` — one declared per-run output artifact
  (``name`` / ``required`` / ``produced`` / ``detail``).
- :func:`derive_status` — pure ``(deliverables, error, warnings) → status``
  computing ``"ok"`` / ``"degraded"`` / ``"failed"``. The
  required-deliverable-missing → not-``"ok"`` invariant is *structural*
  (see the function docstring): the "all good" branch is only reachable
  when no required deliverable is absent, so a caller cannot pass a status
  string that bypasses it.
- :func:`write_health` — build the canonical payload (the eight legacy keys
  + the new ``deliverables`` field) and PUT it to ``health/{module}.json``.
  The S3 client is injectable for unit-testing without live S3.
- :func:`read_health` / :func:`check_upstream_health` — the readers ported
  verbatim-in-behaviour from the existing copies.
- :data:`HEALTH_KEYS` — the canonical module → S3-key map derived from the
  five modules' names.
- :data:`STATUSES` — the closed set of status strings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final, List, Literal, Optional

logger = logging.getLogger(__name__)

# ── Canonical constants ──────────────────────────────────────────────────────

#: The default research bucket every fleet module writes its health stamp to.
#: Matches the live ``s3://alpha-engine-research/health/{module}.json`` path.
DEFAULT_HEALTH_BUCKET: Final[str] = "alpha-engine-research"

Status = Literal["ok", "degraded", "failed"]

#: Closed set of status strings, worst → best. Ordering is load-bearing for
#: any consumer that wants to reduce a fleet of stamps to a single worst-case.
STATUSES: Final[tuple[str, ...]] = ("failed", "degraded", "ok")

#: Canonical module → ``health/`` S3 key map, derived from the five copied
#: implementations' module names. Keys are the logical module names callers
#: pass as ``module_name``; values are the S3 keys they resolve to. Kept as
#: an explicit registry (rather than an f-string at every call site) so the
#: dashboard / upstream-health readers share one source of truth for *which*
#: stamps exist.
#:
#: ``data`` is the ``nousergon-data`` weekly collector's module name (its
#: inline ``_write_module_health`` is called with ``module_name="daily_data"``
#: on the daily path and per-phase names elsewhere; ``data`` is the umbrella
#: entry the dashboard groups under). The four ``crucible-*`` modules write
#: under their short names.
HEALTH_KEYS: Final[dict[str, str]] = {
    "data": "health/data.json",
    "research": "health/research.json",
    "predictor": "health/predictor.json",
    "backtester": "health/backtester.json",
    "executor": "health/executor.json",
}


def health_key(module_name: str) -> str:
    """Resolve the ``health/`` S3 key for ``module_name``.

    Uses :data:`HEALTH_KEYS` for the five canonical modules and falls back to
    the ``health/{module_name}.json`` convention for any other name, so the
    writer stays a drop-in for the legacy per-module f-string.
    """
    return HEALTH_KEYS.get(module_name, f"health/{module_name}.json")


# ── Deliverables ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Deliverable:
    """One declared output artifact of a module run.

    ``required`` deliverables gate the ``"ok"`` status: if any required
    deliverable did not materialise (``produced=False``), the run cannot be
    ``"ok"`` (see :func:`derive_status`). ``detail`` is a free-form human note
    surfaced on the dashboard's System Health page (e.g. row counts, the S3
    key written, or *why* it is absent).
    """

    name: str
    required: bool = True
    produced: bool = False
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the ``deliverables`` array in the health payload."""
        return {
            "name": self.name,
            "required": self.required,
            "produced": self.produced,
            "detail": self.detail,
        }


# ── Status derivation ────────────────────────────────────────────────────────


def missing_required(deliverables: List[Deliverable]) -> List[Deliverable]:
    """Return the required deliverables that were not produced.

    Factored out of :func:`derive_status` so the "is this run structurally
    allowed to be ok?" question has one name, and so callers/tests can assert
    on *which* deliverable is the blocker.
    """
    return [d for d in deliverables if d.required and not d.produced]


def derive_status(
    deliverables: List[Deliverable],
    error: Optional[str] = None,
    warnings: Optional[List[str]] = None,
) -> Status:
    """Derive the run status from its outcome — pure, no side effects.

    Precedence (worst wins):

    1. ``error`` present                       → ``"failed"``
    2. a *required* deliverable is absent       → ``"failed"``
    3. warnings present, or a *non-required*
       deliverable is absent                    → ``"degraded"``
    4. otherwise                                → ``"ok"``

    **Structural invariant.** It is impossible for this function to return
    ``"ok"`` while a required deliverable is absent. The ``"ok"`` value is
    *only* returned from the final branch, and that branch is guarded by
    ``not missing_required(...)``. Status is *derived* from the deliverables
    here rather than accepted as a caller-supplied string, so there is no
    parameter a caller can set to force ``"ok"`` past a missing required
    deliverable — the check cannot be bypassed. :func:`write_health` calls
    this internally and does not accept a ``status`` override, so the
    invariant holds end-to-end at the write boundary too.
    """
    warnings = warnings or []
    blockers = missing_required(deliverables)

    if error:
        return "failed"
    if blockers:
        return "failed"
    # Reaching here guarantees: no error AND no missing required deliverable.
    if warnings or any(not d.produced for d in deliverables):
        return "degraded"
    return "ok"


# ── Writer ───────────────────────────────────────────────────────────────────


def build_health_payload(
    module_name: str,
    deliverables: List[Deliverable],
    run_date: str,
    duration_seconds: float,
    summary: Optional[dict] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    """Build the canonical health payload without touching S3.

    Returns the eight legacy keys the five copies write —
    ``module`` / ``status`` / ``last_success`` / ``run_date`` /
    ``duration_seconds`` / ``summary`` / ``warnings`` / ``error`` — plus the
    new ``deliverables`` array. ``status`` is *derived* (never accepted) via
    :func:`derive_status`, and ``last_success`` is nulled on ``"failed"`` so
    downstream staleness checks can distinguish "ran and failed today" from
    "hasn't run in N hours" (behaviour preserved from the legacy writers).
    """
    warnings = warnings or []
    status = derive_status(deliverables, error=error, warnings=warnings)
    return {
        "module": module_name,
        "status": status,  # "ok" | "degraded" | "failed"
        "last_success": (
            datetime.now(timezone.utc).isoformat() if status != "failed" else None
        ),
        "run_date": run_date,
        "duration_seconds": round(duration_seconds, 1),
        "summary": summary or {},
        "warnings": warnings,
        "error": error,
        "deliverables": [d.to_dict() for d in deliverables],
    }


def write_health(
    module_name: str,
    deliverables: List[Deliverable],
    run_date: str,
    duration_seconds: float,
    *,
    summary: Optional[dict] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[str] = None,
    bucket: str = DEFAULT_HEALTH_BUCKET,
    s3_client: Any = None,
) -> dict[str, Any]:
    """Derive status, build the payload, and PUT it to ``health/{module}.json``.

    The status is *derived* from ``deliverables`` / ``error`` / ``warnings``
    (see :func:`derive_status`) — there is deliberately no ``status``
    parameter, so a caller cannot stamp ``"ok"`` over a missing required
    deliverable. Returns the payload that was written (so callers can log /
    assert on it).

    ``s3_client`` is injectable: pass a client (real or fake) to make the
    write unit-testable without live S3. When omitted a real
    ``boto3.client("s3")`` is constructed lazily — importing this module does
    not require boto3 credentials.

    Write failures are logged and swallowed (never raised): health is
    enrichment, so a health-write outage must never take down the run it is
    describing (behaviour preserved from the legacy writers).
    """
    payload = build_health_payload(
        module_name=module_name,
        deliverables=deliverables,
        run_date=run_date,
        duration_seconds=duration_seconds,
        summary=summary,
        warnings=warnings,
        error=error,
    )
    key = health_key(module_name)
    try:
        if s3_client is None:
            import boto3

            s3_client = boto3.client("s3")
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(payload, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info("Health status written: %s → %s", module_name, payload["status"])
    except Exception as exc:  # pragma: no cover - defensive; enrichment only
        logger.warning("Failed to write health status for %s: %s", module_name, exc)
    return payload


# ── Readers (ported from the existing copies) ────────────────────────────────


def read_health(
    module_name: str,
    *,
    bucket: str = DEFAULT_HEALTH_BUCKET,
    s3_client: Any = None,
) -> Optional[dict]:
    """Read the health JSON for ``module_name``. Returns ``None`` if absent.

    Ported from the copies' ``read_health``: any read failure (missing key,
    permission, malformed JSON) collapses to ``None`` so a downstream
    upstream-health check treats "no stamp" and "unreadable stamp"
    identically. ``s3_client`` is injectable as in :func:`write_health`.
    """
    try:
        if s3_client is None:
            import boto3

            s3_client = boto3.client("s3")
        obj = s3_client.get_object(Bucket=bucket, Key=health_key(module_name))
        return json.loads(obj["Body"].read())
    except Exception:
        return None


def check_upstream_health(
    modules: List[str],
    *,
    bucket: str = DEFAULT_HEALTH_BUCKET,
    max_age_hours: float = 48,
    s3_client: Any = None,
) -> dict[str, dict]:
    """Check the health of multiple upstream modules.

    Returns ``{module: {"status": str, "age_hours": float, "stale": bool}}``.
    A module with no stamp is ``{"status": "unknown", "age_hours": -1,
    "stale": True}``. ``age_hours`` is derived from ``last_success`` (nulled
    on failed runs → treated as ``-1`` / stale). Ported from the copies'
    ``check_upstream_health`` with the S3 client made injectable.
    """
    results: dict[str, dict] = {}
    now = datetime.now(timezone.utc)
    for mod in modules:
        health = read_health(mod, bucket=bucket, s3_client=s3_client)
        if health is None:
            results[mod] = {"status": "unknown", "age_hours": -1, "stale": True}
            continue
        age_hours = -1.0
        if health.get("last_success"):
            try:
                last = datetime.fromisoformat(health["last_success"])
                age_hours = (now - last).total_seconds() / 3600
            except (ValueError, TypeError):
                pass
        results[mod] = {
            "status": health.get("status", "unknown"),
            "age_hours": round(age_hours, 1),
            "stale": age_hours < 0 or age_hours > max_age_hours,
        }
    return results
