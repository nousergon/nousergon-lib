"""
Artifact-freshness substrate for the load-bearing-S3-artifact absence
bug class.

**Why this exists.** 2026-05-17 → 2026-05-27 `pit_parity.json` incident:
a load-bearing artifact (open ROADMAP item L3293's manual-flip gate
preconditioning input) was silently absent for 11 days across 4
consecutive Saturday SF firings. Brian's manual audit caught it only
because he asked "what are priority actionable items" and we walked
the gate's preconditions. The narrow PR (#255) fixed the proximate
``copy.deepcopy(s3_resource)`` bug but not the SHAPE of the failure:
neither flow-doctor, SF Catch, nor substrate-health-check noticed
absence-of-artifact because all three are *event-driven* — failure →
alert. None watches for silence.

Same class as:

* 2026-05-18 factor-profiles orphan (``s3://alpha-engine-research/factors/``
  empty for ~2 weeks before audit caught it).
* 2026-05-23 missing ``signals.json`` (visible only after SF Catch fired
  on the downstream Predictor inference Lambda's hard-fail).
* 2026-05-15 Wave-1 ``latest_weekly.json`` pointer staleness.

The architectural fix is *absence-driven* monitoring: a declarative
registry of load-bearing artifacts + their SLAs, walked by a probe
that fires :func:`alpha_engine_lib.alerts.publish` on misses past SLA.
This module is the substrate; PR 2 ships the registry SoT
(``alpha-engine-config/private-docs/ARTIFACT_REGISTRY.yaml``); PR 3
ships the freshness-monitor Lambda that wires the two together.

**Public surface:**

- :data:`CADENCE_SYMBOLS` — supported cron-cadence symbols
  (``saturday_sf`` / ``weekday_sf`` / ``eod_sf`` / ``continuous``).
- :class:`ArtifactSpec` — registry-row dataclass.
- :class:`CheckResult` — single-probe outcome dataclass.
- :func:`check_freshness` — pure ``(s3_client, spec, now) → CheckResult``.
  No side effects, no alerting. The Lambda is responsible for
  consuming the result and routing to :func:`alpha_engine_lib.alerts.publish`.
- :func:`resolve_dedup_key` — pure ``(spec, now) → str`` producing the
  stable per-cycle dedup key used by ``alerts.publish``.
- :func:`resolve_current_cycle` — pure helper exposing the
  ``(cycle_start_utc, cycle_window_label)`` for testability.

**Design invariants** (mirror the plan doc at
``~/Development/alpha-engine-docs/private/artifact-freshness-monitor-260527.md``):

1. **Pure.** No S3 calls except via the injected ``s3_client``. No
   datetime.now() — caller passes ``now`` for testability.
2. **Calendar-aware.** ``weekday_sf`` / ``eod_sf`` cycles resolve via
   :mod:`alpha_engine_lib.trading_calendar` — NYSE holidays = no
   check (returns ``state='fresh'`` with a holiday reason so the
   Lambda short-circuits the alert path).
3. **Recovery-aware.** When ``spec.recovery_key_template`` is set,
   a 404 / stale on the canonical key falls through to a HEAD on the
   recovery key. Either fresh ⇒ overall fresh.
4. **Probe-failure separated.** ``403`` / ``InvalidBucketName`` /
   ``EndpointConnectionError`` ⇒ ``probe_failed`` (critical, routes
   to operator). ``404`` ⇒ ``missing`` (the canonical missing-artifact
   path, severity per spec).
5. **Grace-period.** Specs younger than ``grace_period_cycles`` cycles
   suppress to ``grace_period`` state — newly-onboarded producers
   don't false-alarm on their first emissions.

**Composes with:**

- :func:`alpha_engine_lib.alerts.publish` — the alert chokepoint the
  Lambda calls with ``dedup_key=resolve_dedup_key(spec, now)``.
- :mod:`alpha_engine_lib.trading_calendar` — NYSE-holiday substrate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Final, Literal

from alpha_engine_lib.trading_calendar import is_trading_day


# ── Cadence symbols ──────────────────────────────────────────────────────────
# Plan §4 PR 2 commits the registry to a closed set of cadence symbols
# rather than free-form cron strings. The symbols encode the cron
# expression AND the calendar-awareness rule. New symbols added here
# require both a cycle-resolution rule in ``resolve_current_cycle`` and
# a window-label rule in ``resolve_dedup_key``.
CadenceSymbol = Literal["saturday_sf", "weekday_sf", "eod_sf", "continuous"]

CADENCE_SYMBOLS: Final[frozenset[str]] = frozenset(
    {"saturday_sf", "weekday_sf", "eod_sf", "continuous"}
)

# Cron-tick hours (UTC) per the live EventBridge rules. Source:
# ``~/Development/CLAUDE.md`` § Architecture diagrams (Saturday SF at
# 09:00 UTC, Weekday SF at 13:00 UTC). EOD SF is daemon-triggered
# post-close so its "expected cron" anchors to a conservative 21:00 UTC
# (after the daemon shutdown + EOD reconcile typically completes).
_SATURDAY_SF_CRON_UTC: Final[int] = 9
_WEEKDAY_SF_CRON_UTC: Final[int] = 13
_EOD_SF_ANCHOR_UTC: Final[int] = 21

CheckState = Literal["fresh", "stale", "missing", "probe_failed", "grace_period"]


# ── Spec ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArtifactSpec:
    """One row of the ``ARTIFACT_REGISTRY.yaml`` registry.

    Attributes:
        artifact_id: Stable string identifier, unique across the
            registry. Used as the dedup-key prefix.
        s3_bucket: Bucket holding the artifact (typically
            ``alpha-engine-research``).
        s3_key_template: Key with optional ``{date}`` / ``{trading_day}``
            placeholders. Resolved per-cycle via :func:`_format_key`.
        cadence: Cron-cadence symbol — one of :data:`CADENCE_SYMBOLS`.
        sla_minutes_after_cron: Grace window in minutes after the
            cycle's cron tick during which the artifact's absence is
            still acceptable. Past this window, ``check_freshness``
            classifies the artifact ``missing``.
        severity: Routed to :func:`alpha_engine_lib.alerts.publish` on
            a miss. One of ``"warning"`` / ``"critical"``.
        owner_repo: Producing repo identifier
            (``alpha-engine-research`` / etc.) — used by the dashboard
            surface for grouping and by the CI-guard cascade
            (Phase 4 PRs).
        created_at: Date the spec was added to the registry. Drives
            the ``grace_period_cycles`` short-circuit.
        grace_period_cycles: Number of cadence cycles after
            ``created_at`` during which the check returns
            ``state="grace_period"`` regardless of artifact presence.
            Default ``2`` per plan §3 invariant 7.
        recovery_key_template: Optional alternate key checked when the
            canonical key is missing / stale. Enables the
            recovery-SF-substitution semantic from plan §3 invariant 3.
            ``None`` disables substitution.
        calendar_aware: When ``True`` (the institutional default for
            weekday-keyed cadences), NYSE-holiday cycles short-circuit
            to ``state="fresh"`` regardless of artifact presence —
            the cron is expected to not have fired. Default ``True``.
        interval_minutes: Required only for ``cadence="continuous"``;
            defines the cycle window length. Ignored otherwise.
    """

    artifact_id: str
    s3_bucket: str
    s3_key_template: str
    cadence: CadenceSymbol
    sla_minutes_after_cron: int
    severity: Literal["warning", "critical"]
    owner_repo: str
    created_at: date
    grace_period_cycles: int = 2
    recovery_key_template: str | None = None
    calendar_aware: bool = True
    interval_minutes: int | None = None

    def __post_init__(self) -> None:
        if self.cadence not in CADENCE_SYMBOLS:
            raise ValueError(
                f"ArtifactSpec.cadence={self.cadence!r} not in "
                f"{sorted(CADENCE_SYMBOLS)}"
            )
        if self.severity not in ("warning", "critical"):
            raise ValueError(
                f"ArtifactSpec.severity={self.severity!r} must be "
                "'warning' or 'critical'"
            )
        if self.sla_minutes_after_cron < 0:
            raise ValueError(
                f"ArtifactSpec.sla_minutes_after_cron must be >= 0, "
                f"got {self.sla_minutes_after_cron}"
            )
        if self.grace_period_cycles < 0:
            raise ValueError(
                f"ArtifactSpec.grace_period_cycles must be >= 0, "
                f"got {self.grace_period_cycles}"
            )
        if self.cadence == "continuous":
            if self.interval_minutes is None or self.interval_minutes <= 0:
                raise ValueError(
                    "ArtifactSpec.cadence='continuous' requires "
                    "interval_minutes > 0"
                )


# ── Result ──────────────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    """One ``check_freshness`` outcome.

    Attributes:
        state: Outcome class. ``fresh`` ⇒ artifact present + within
            the current cycle's last_modified window (or substituted
            by recovery). ``stale`` ⇒ object exists but last_modified
            predates the current cycle's start (the pointer-pattern
            case). ``missing`` ⇒ 404. ``probe_failed`` ⇒ S3 client
            error other than 404 (the monitor itself is broken).
            ``grace_period`` ⇒ spec younger than
            ``grace_period_cycles`` cycles; alert suppressed by design.
        last_modified: Object's ``LastModified`` from the canonical
            (or recovery) HEAD. ``None`` for ``missing`` / ``probe_failed``.
        sla_violated_by_minutes: For ``missing`` / ``stale``, how far
            past the SLA grace the breach is in minutes. ``0`` for
            ``fresh`` / ``grace_period`` / ``probe_failed``.
        reason: Human-readable diagnostic; routed into the alert body
            and the dashboard surface.
        canonical_key: The resolved canonical key the probe HEADed.
        recovery_substituted: ``True`` when the canonical key was
            missing / stale but the recovery key was fresh (i.e. the
            recovery-SF-substitution semantic kicked in).
    """

    state: CheckState
    last_modified: datetime | None = None
    sla_violated_by_minutes: int = 0
    reason: str = ""
    canonical_key: str = ""
    recovery_substituted: bool = False


# ── Cycle resolution ────────────────────────────────────────────────────────


def _utc(now: datetime) -> datetime:
    """Coerce ``now`` to UTC-aware. Naive ⇒ assumed UTC."""
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _most_recent_weekday(now_utc: datetime) -> date:
    """Most recent calendar weekday (Mon-Fri) at or before ``now_utc``.

    Does NOT consult the NYSE calendar — callers needing trading-day
    semantics use :func:`alpha_engine_lib.trading_calendar.last_closed_trading_day`
    directly. This helper is the fallback for non-calendar-aware specs.
    """
    d = now_utc.date()
    while d.weekday() > 4:
        d -= timedelta(days=1)
    return d


def _most_recent_saturday(now_utc: datetime) -> date:
    """Most recent calendar Saturday at or before ``now_utc.date()``."""
    d = now_utc.date()
    # weekday(): Mon=0, Sat=5
    days_back = (d.weekday() - 5) % 7
    return d - timedelta(days=days_back)


def resolve_current_cycle(
    spec: ArtifactSpec, now: datetime
) -> tuple[datetime, str]:
    """Return ``(cycle_cron_tick_utc, cycle_window_label)`` for the
    most recent cycle whose cron tick is at or before ``now``.

    The cycle label is the stable string used in the dedup key — same
    label across all probes within the cycle so :func:`alerts.publish`
    dedup collapses retries to one alert per cycle.

    For ``saturday_sf``: label is ISO year-week
    (``"2026-W22"``); cron tick is Saturday at
    :data:`_SATURDAY_SF_CRON_UTC` UTC.

    For ``weekday_sf`` / ``eod_sf``: label is the cycle's
    trading-day ISO date (``"2026-05-27"``); cron tick is that day
    at the cadence-specific UTC hour. If ``spec.calendar_aware``,
    holidays are skipped via
    :func:`alpha_engine_lib.trading_calendar.last_closed_trading_day`;
    otherwise a plain weekday fallback is used.

    For ``continuous``: label is
    ``f"continuous_{interval_minutes}m_{bucket_index}"`` where the
    bucket is ``floor((now_epoch - cron_epoch_zero) / interval)``;
    cron tick is the bucket's start.
    """
    now_utc = _utc(now)

    if spec.cadence == "saturday_sf":
        # Find the most recent Saturday whose 09:00 UTC tick has passed.
        sat = _most_recent_saturday(now_utc)
        tick = datetime(
            sat.year, sat.month, sat.day,
            _SATURDAY_SF_CRON_UTC, 0, tzinfo=timezone.utc,
        )
        if tick > now_utc:
            sat = sat - timedelta(days=7)
            tick = datetime(
                sat.year, sat.month, sat.day,
                _SATURDAY_SF_CRON_UTC, 0, tzinfo=timezone.utc,
            )
        iso_year, iso_week, _ = sat.isocalendar()
        return tick, f"{iso_year}-W{iso_week:02d}"

    if spec.cadence in ("weekday_sf", "eod_sf"):
        cron_hour = (
            _WEEKDAY_SF_CRON_UTC
            if spec.cadence == "weekday_sf"
            else _EOD_SF_ANCHOR_UTC
        )
        # Walk back to the most recent calendar weekday whose cron hour
        # has passed in UTC. The cycle is the calendar weekday — NYSE
        # holidays are NOT snapped away here. The holiday gate in
        # :func:`check_freshness` is what suppresses the alert on
        # holidays (state="fresh" with a holiday reason), preserving
        # one distinct cycle per calendar day so dedup keys don't
        # collide with the prior trading day's actual probe.
        d = _most_recent_weekday(now_utc)
        tick = datetime(
            d.year, d.month, d.day, cron_hour, 0, tzinfo=timezone.utc,
        )
        if tick > now_utc:
            # Today's tick hasn't fired yet — step back one weekday.
            d -= timedelta(days=1)
            while d.weekday() > 4:
                d -= timedelta(days=1)
            tick = datetime(
                d.year, d.month, d.day, cron_hour, 0, tzinfo=timezone.utc,
            )
        return tick, d.isoformat()

    if spec.cadence == "continuous":
        # Bucket the wall-clock UTC into interval_minutes-wide windows.
        # Bucket index = floor(epoch_min / interval_min).
        assert spec.interval_minutes is not None  # validated in __post_init__
        epoch_min = int(now_utc.timestamp() // 60)
        bucket = epoch_min // spec.interval_minutes
        tick = datetime.fromtimestamp(
            bucket * spec.interval_minutes * 60, tz=timezone.utc,
        )
        return tick, f"continuous_{spec.interval_minutes}m_{bucket}"

    # Unreachable — __post_init__ validates the symbol set.
    raise ValueError(f"unknown cadence {spec.cadence!r}")


# ── Key formatting ──────────────────────────────────────────────────────────


def _format_key(template: str, cycle_label: str, cycle_tick: datetime) -> str:
    """Substitute cycle placeholders into the key template.

    Supported placeholders:

    - ``{date}``: ISO date of the cycle (Saturday for saturday_sf;
      trading day for weekday_sf / eod_sf; bucket-start date for
      continuous).
    - ``{trading_day}``: alias for ``{date}`` (registry convention —
      semantic name for date-keyed trading-day artifacts).
    - ``{cycle_label}``: the raw cycle label (e.g. ``"2026-W22"``).

    Templates without placeholders pass through unchanged (the
    pointer-pattern case — same key across cycles, freshness inferred
    from ``last_modified``).
    """
    iso = cycle_tick.date().isoformat()
    return template.format(
        date=iso,
        trading_day=iso,
        cycle_label=cycle_label,
    )


# ── Dedup-key resolution ────────────────────────────────────────────────────


def resolve_dedup_key(spec: ArtifactSpec, now: datetime) -> str:
    """Return the stable per-cycle dedup key for
    :func:`alpha_engine_lib.alerts.publish`.

    Shape: ``freshness_{artifact_id}_{cycle_window_label}``. Same
    label across all probes within the cycle collapses 4× / hour
    retries (sub-15min cron granularity) to at most one alert per
    cycle per artifact.
    """
    _, label = resolve_current_cycle(spec, now)
    return f"freshness_{spec.artifact_id}_{label}"


# ── Probe helpers ───────────────────────────────────────────────────────────


def _classify_client_error(err: Any) -> tuple[str, str]:
    """Classify a boto3 ClientError into ``(state, reason)``.

    Returns either ``("missing", ...)`` for the 404 / NoSuchKey path
    or ``("probe_failed", ...)`` for any other client error
    (403 / InvalidBucketName / EndpointConnectionError / etc.).
    """
    # boto3 stores the HTTP status in ``err.response["ResponseMetadata"]["HTTPStatusCode"]``
    # and the canonical error code under ``err.response["Error"]["Code"]``.
    # We avoid importing botocore here to keep the module dep-light;
    # duck-type the shape instead.
    code = ""
    status = 0
    try:
        code = err.response.get("Error", {}).get("Code", "") or ""
        status = int(err.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
    except (AttributeError, TypeError, ValueError):
        # Non-ClientError exception (network, malformed response).
        return ("probe_failed", f"probe error: {err!r}")

    # S3 HEAD returns 404 with code "404" for missing keys; the
    # higher-level resource API uses "NoSuchKey". Treat both as missing.
    if status == 404 or code in ("404", "NoSuchKey", "NotFound"):
        return ("missing", "S3 HEAD returned 404 (key not present)")

    return (
        "probe_failed",
        f"S3 HEAD failed with status={status} code={code!r}",
    )


def _head_object(
    s3_client: Any, bucket: str, key: str
) -> tuple[str, datetime | None, str]:
    """HEAD an S3 object. Return ``(state, last_modified, reason)``
    where state is one of ``"present"`` / ``"missing"`` / ``"probe_failed"``.
    """
    try:
        resp = s3_client.head_object(Bucket=bucket, Key=key)
    except Exception as err:  # noqa: BLE001 — duck-typed boto error classification
        state, reason = _classify_client_error(err)
        return state, None, reason

    last_modified = resp.get("LastModified")
    if last_modified is not None and last_modified.tzinfo is None:
        last_modified = last_modified.replace(tzinfo=timezone.utc)
    return "present", last_modified, "HEAD ok"


# ── check_freshness — the core public function ──────────────────────────────


def check_freshness(
    s3_client: Any, spec: ArtifactSpec, now: datetime
) -> CheckResult:
    """Probe ``spec`` and return the classified outcome.

    Pure with respect to side effects beyond the ``s3_client.head_object``
    call (no logging, no alerting, no DDB / S3 marker writes). The
    Lambda is responsible for routing the result to
    :func:`alpha_engine_lib.alerts.publish` with
    ``dedup_key=resolve_dedup_key(spec, now)``.

    The probe walks five steps:

    1. **Grace-period gate.** If ``(now - spec.created_at)`` is shorter
       than ``spec.grace_period_cycles`` cycles, return
       ``state="grace_period"`` immediately — newly-onboarded
       producers don't false-alarm on their first emissions.
    2. **Calendar-holiday gate.** When ``spec.calendar_aware`` and
       the resolved cycle's date is NOT a trading day, return
       ``state="fresh"`` with a holiday reason — the cron didn't
       fire, so the artifact's absence is correct.
    3. **HEAD canonical.** Resolve the template into the canonical
       key. HEAD. Classify the response.
    4. **Stale check.** When the canonical key is present, compare
       ``LastModified`` against ``cycle_cron_tick``. Older ⇒ ``stale``
       (the pointer-pattern case); same-or-newer ⇒ ``fresh``.
    5. **Recovery substitution.** When the canonical key is
       missing / stale AND ``spec.recovery_key_template`` is set,
       HEAD the recovery key. If the recovery key is fresh, override
       to ``state="fresh"`` with ``recovery_substituted=True``.

    SLA-violated-by-minutes is computed as
    ``(now - cycle_cron_tick - sla_minutes_after_cron)`` for the
    ``missing`` / ``stale`` paths; clipped at zero so the field
    is always non-negative.
    """
    now_utc = _utc(now)
    cycle_tick, cycle_label = resolve_current_cycle(spec, now_utc)

    # ── 1. Grace period ─────────────────────────────────────────────────
    cycle_seconds = _cycle_length_seconds(spec)
    age_seconds = (now_utc - datetime.combine(
        spec.created_at, datetime.min.time(), tzinfo=timezone.utc,
    )).total_seconds()
    if age_seconds < spec.grace_period_cycles * cycle_seconds:
        return CheckResult(
            state="grace_period",
            reason=(
                f"spec age {age_seconds / 3600:.1f}h < "
                f"{spec.grace_period_cycles} cycles × "
                f"{cycle_seconds / 3600:.1f}h = grace period"
            ),
            canonical_key=_format_key(
                spec.s3_key_template, cycle_label, cycle_tick,
            ),
        )

    # ── 2. Calendar-holiday short-circuit ───────────────────────────────
    if spec.calendar_aware and spec.cadence in ("weekday_sf", "eod_sf"):
        if not is_trading_day(cycle_tick.date()):
            return CheckResult(
                state="fresh",
                reason=(
                    f"NYSE holiday {cycle_tick.date().isoformat()} — "
                    "cron did not fire, absence is correct"
                ),
                canonical_key=_format_key(
                    spec.s3_key_template, cycle_label, cycle_tick,
                ),
            )

    # ── 3. HEAD canonical ───────────────────────────────────────────────
    canonical_key = _format_key(spec.s3_key_template, cycle_label, cycle_tick)
    head_state, last_modified, reason = _head_object(
        s3_client, spec.s3_bucket, canonical_key,
    )

    # ── 4. Stale check (only when canonical present) ────────────────────
    canonical_state: CheckState
    if head_state == "present":
        assert last_modified is not None
        if last_modified < cycle_tick:
            canonical_state = "stale"
            reason = (
                f"object present but last_modified={last_modified.isoformat()} "
                f"< cycle_tick={cycle_tick.isoformat()}"
            )
        else:
            return CheckResult(
                state="fresh",
                last_modified=last_modified,
                reason="canonical HEAD fresh",
                canonical_key=canonical_key,
            )
    elif head_state == "missing":
        canonical_state = "missing"
    else:
        # probe_failed bypasses the recovery substitution — the
        # monitor itself is broken; the operator needs to know.
        return CheckResult(
            state="probe_failed",
            reason=reason,
            canonical_key=canonical_key,
        )

    # ── 5. Recovery substitution ────────────────────────────────────────
    if spec.recovery_key_template is not None:
        recovery_key = _format_key(
            spec.recovery_key_template, cycle_label, cycle_tick,
        )
        rec_state, rec_last_modified, _rec_reason = _head_object(
            s3_client, spec.s3_bucket, recovery_key,
        )
        if rec_state == "present" and rec_last_modified is not None:
            if rec_last_modified >= cycle_tick:
                return CheckResult(
                    state="fresh",
                    last_modified=rec_last_modified,
                    reason=(
                        "canonical missing/stale; recovery key "
                        f"{recovery_key} satisfies cycle"
                    ),
                    canonical_key=canonical_key,
                    recovery_substituted=True,
                )

    # ── SLA-violation arithmetic ─────────────────────────────────────────
    sla_deadline = cycle_tick + timedelta(minutes=spec.sla_minutes_after_cron)
    violated_minutes = int(
        max(0, (now_utc - sla_deadline).total_seconds() // 60)
    )

    return CheckResult(
        state=canonical_state,
        last_modified=last_modified,
        sla_violated_by_minutes=violated_minutes,
        reason=reason,
        canonical_key=canonical_key,
    )


# ── Internals ───────────────────────────────────────────────────────────────


def _cycle_length_seconds(spec: ArtifactSpec) -> float:
    """Approximate cycle length in seconds — used for the grace-period
    arithmetic. The exact length doesn't need to track NYSE holidays
    (the grace period is a coarse cold-start gate, not an SLA).
    """
    if spec.cadence == "saturday_sf":
        return 7 * 24 * 3600
    if spec.cadence in ("weekday_sf", "eod_sf"):
        return 24 * 3600
    if spec.cadence == "continuous":
        assert spec.interval_minutes is not None
        return spec.interval_minutes * 60
    raise ValueError(f"unknown cadence {spec.cadence!r}")
