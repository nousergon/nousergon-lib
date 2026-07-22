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
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Final, Literal

logger = logging.getLogger(__name__)

#: Grace window subtracted from the verification cutoff (``now -
#: duration_seconds``) to absorb clock skew between the writer and S3, and
#: the gap between an artifact's own write and this module's write_health()
#: call at the end of the same run. See :func:`verify_fresh_s3`.
_ARTIFACT_FRESHNESS_BUFFER_SECONDS: Final[int] = 60

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
#: entry the dashboard groups under). Values match the live
#: ``ARTIFACT_REGISTRY.yaml`` ``health_*`` rows (config#1728) — NOT the stale
#: ``health/{module}.json`` shorthand that was never written for data/predictor.
HEALTH_KEYS: Final[dict[str, str]] = {
    "data": "health/daily_data.json",
    "daily_data": "health/daily_data.json",
    "research": "health/research.json",
    "predictor": "health/predictor_inference.json",
    "predictor_inference": "health/predictor_inference.json",
    "predictor_training": "health/predictor_training.json",
    "predictor_health_check": "health/predictor_health_check.json",
    "backtester": "health/backtester.json",
    "executor": "health/executor.json",
    "eod_reconcile": "health/eod_reconcile.json",
}

#: ``health_*`` rows in ``ARTIFACT_REGISTRY.yaml`` — artifact_id → S3 key.
#: Single source for registry validator + dashboard alignment tests (config#1728).
REGISTRY_HEALTH_ARTIFACTS: Final[dict[str, str]] = {
    "health_alpha_engine_data": "health/daily_data.json",
    "health_alpha_engine_research": "health/research.json",
    "health_alpha_engine_predictor": "health/predictor_inference.json",
    "health_alpha_engine_backtester": "health/backtester.json",
}

#: Candidate ``health/`` filenames per logical module for staleness checks.
#: Used by ``alpha-engine-dashboard/health_checker.py`` (config#1728).
HEALTH_CHECK_CANDIDATES: Final[dict[str, tuple[str, ...]]] = {
    "data": ("daily_data.json", "data_phase1.json", "data_phase2.json"),
    "executor": ("executor.json",),
    "predictor": (
        "predictor_inference.json",
        "predictor_training.json",
        "predictor_health_check.json",
    ),
    "research": ("research.json",),
    "backtester": ("backtester.json",),
}

#: Modules shown on the dashboard System Health panel — (module_name, bucket, stale_hrs).
DASHBOARD_HEALTH_MODULES: Final[tuple[tuple[str, str, int], ...]] = (
    ("research", "research", 8 * 24),
    ("predictor_training", "research", 8 * 24),
    ("predictor_inference", "research", 4 * 24),
    ("executor", "research", 4 * 24),
    ("eod_reconcile", "trades", 4 * 24),
)


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

    ``artifact_key`` is an OPTIONAL S3 key backing a ``produced=True`` claim.
    When set, :func:`write_health` re-derives ``produced`` via
    :func:`verify_fresh_s3` before deriving status — turning "trust the
    boolean" into "verify the S3 object" (config#3153: every instance of a
    stage self-reporting ``produced=True`` while its artifact silently went
    stale was caught by a human, never by CI). Leave unset (the default) to
    keep the pre-existing self-reported behavior — verification is opt-in
    per deliverable, since not every deliverable has a single natural S3
    artifact to check (e.g. "sent notification email").
    """

    name: str
    required: bool = True
    produced: bool = False
    detail: str = ""
    artifact_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the ``deliverables`` array in the health payload."""
        return {
            "name": self.name,
            "required": self.required,
            "produced": self.produced,
            "detail": self.detail,
            "artifact_key": self.artifact_key,
        }


def verify_fresh_s3(
    deliverable: Deliverable,
    *,
    bucket: str = DEFAULT_HEALTH_BUCKET,
    s3_client: Any = None,
    since: datetime | None = None,
) -> Deliverable:
    """Re-derive ``produced`` by checking ``artifact_key`` against live S3.

    No-op (returns ``deliverable`` unchanged) when ``artifact_key`` is
    ``None`` or ``produced`` is already ``False`` — there is nothing to
    verify or disprove.

    Otherwise ``head_object``\\s the key (metadata only — never downloads the
    artifact just to confirm it exists):

    - Confirmed absent (404 / NoSuchKey / NotFound) → returns a copy with
      ``produced=False`` and a ``detail`` note. This is a real, definitive
      signal: the run claimed to have produced the artifact and it isn't
      there.
    - Present but ``LastModified`` predates ``since`` (minus a small clock-
      skew/timing buffer) → same downgrade, with a stale-artifact note. This
      is what catches the config-I3053 failure mode: an artifact that
      exists but wasn't actually touched by *this* run, so ``produced=True``
      would otherwise be believed on a stale file.
    - Present and fresh (or ``since`` not given) → returns a copy with
      ``produced=True`` confirmed.
    - Any other S3 error (permission, network, throttling) → the claim is
      left AS DECLARED and a warning is logged. Mirrors
      ``crucible-backtester``'s ``PhaseRegistry._first_missing_artifact``
      (L4524): don't silently flip a trust decision on incomplete
      information — only a *confirmed* absence or staleness downgrades the
      claim.
    """
    if deliverable.artifact_key is None or not deliverable.produced:
        return deliverable

    if s3_client is None:
        import boto3

        s3_client = boto3.client("s3")

    try:
        head = s3_client.head_object(Bucket=bucket, Key=deliverable.artifact_key)
    except Exception as exc:
        code = ""
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            code = response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NoSuchBucket", "NotFound"):
            logger.warning(
                "verify_fresh_s3: deliverable %r claims produced=True but "
                "s3://%s/%s is absent — downgrading to produced=False "
                "(config#3153 artifact-verified marker).",
                deliverable.name, bucket, deliverable.artifact_key,
            )
            return replace(
                deliverable,
                produced=False,
                detail=(
                    f"{deliverable.detail} "
                    f"[S3-verified: s3://{bucket}/{deliverable.artifact_key} absent]"
                ).strip(),
            )
        # Transient / permission error: fail open on the claim, don't guess.
        logger.warning(
            "verify_fresh_s3: could not verify s3://%s/%s for deliverable %r "
            "(%s) — leaving produced=%s as declared.",
            bucket, deliverable.artifact_key, deliverable.name, exc, deliverable.produced,
        )
        return deliverable

    if since is not None:
        last_modified = head.get("LastModified")
        cutoff = since - timedelta(seconds=_ARTIFACT_FRESHNESS_BUFFER_SECONDS)
        if last_modified is not None and last_modified < cutoff:
            logger.warning(
                "verify_fresh_s3: deliverable %r claims produced=True but "
                "s3://%s/%s was last modified %s (before this run's %s cutoff) "
                "— downgrading to produced=False (config#3153).",
                deliverable.name, bucket, deliverable.artifact_key, last_modified, cutoff,
            )
            return replace(
                deliverable,
                produced=False,
                detail=(
                    f"{deliverable.detail} "
                    f"[S3-verified: s3://{bucket}/{deliverable.artifact_key} "
                    f"stale — last modified {last_modified}, expected >= {cutoff}]"
                ).strip(),
            )

    return replace(
        deliverable,
        detail=(
            f"{deliverable.detail} "
            f"[S3-verified: s3://{bucket}/{deliverable.artifact_key}]"
        ).strip(),
    )


# ── Status derivation ────────────────────────────────────────────────────────


def missing_required(deliverables: list[Deliverable]) -> list[Deliverable]:
    """Return the required deliverables that were not produced.

    Factored out of :func:`derive_status` so the "is this run structurally
    allowed to be ok?" question has one name, and so callers/tests can assert
    on *which* deliverable is the blocker.
    """
    return [d for d in deliverables if d.required and not d.produced]


def derive_status(
    deliverables: list[Deliverable],
    error: str | None = None,
    warnings: list[str] | None = None,
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
    deliverables: list[Deliverable],
    run_date: str,
    duration_seconds: float,
    summary: dict | None = None,
    warnings: list[str] | None = None,
    error: str | None = None,
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
    deliverables: list[Deliverable],
    run_date: str,
    duration_seconds: float,
    *,
    summary: dict | None = None,
    warnings: list[str] | None = None,
    error: str | None = None,
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

    Any deliverable declaring ``artifact_key`` is re-verified against S3 via
    :func:`verify_fresh_s3` before status is derived — a ``produced=True``
    claim only survives if the artifact actually exists and was touched
    within this run's window (``now - duration_seconds``, config#3153).
    Deliverables without ``artifact_key`` are untouched, so callers that
    don't opt in see no behavior change and pay no extra S3 round-trip.
    """
    run_started_at = datetime.now(timezone.utc) - timedelta(seconds=duration_seconds)
    verified_deliverables = deliverables
    if any(d.artifact_key is not None and d.produced for d in deliverables):
        try:
            verify_client = s3_client
            if verify_client is None:
                import boto3

                verify_client = boto3.client("s3")
            verified_deliverables = [
                verify_fresh_s3(
                    d, bucket=bucket, s3_client=verify_client, since=run_started_at,
                )
                for d in deliverables
            ]
            if s3_client is None:
                s3_client = verify_client  # reuse for the put_object below
        except Exception as exc:  # pragma: no cover - defensive; enrichment only
            logger.warning(
                "Failed to verify deliverable artifacts for %s: %s", module_name, exc,
            )

    payload = build_health_payload(
        module_name=module_name,
        deliverables=verified_deliverables,
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
) -> dict | None:
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
    modules: list[str],
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
