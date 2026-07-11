"""
Artifact-freshness substrate for the load-bearing-S3-artifact absence
bug class.

**Why this exists.** 2026-05-17 → 2026-05-27 `pit_parity.json` incident:
a load-bearing artifact (open ROADMAP item L3293's manual-flip gate
preconditioning input) was silently absent for 11 days across 4
consecutive Weekly Freshness SF firings. Brian's manual audit caught it only
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
that fires :func:`nousergon_lib.alerts.publish` on misses past SLA.
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
  consuming the result and routing to :func:`nousergon_lib.alerts.publish`.
- :func:`resolve_dedup_key` — pure ``(spec, now) → str`` producing the
  stable per-cycle dedup key used by ``alerts.publish``.
- :func:`resolve_current_cycle` — pure helper exposing the
  ``(cycle_start_utc, cycle_window_label)`` for testability.

**Design invariants** (mirror the plan doc at
``~/Development/alpha-engine-docs/private/artifact-freshness-monitor-260527.md``):

1. **Pure.** No S3 calls except via the injected ``s3_client``. No
   datetime.now() — caller passes ``now`` for testability.
2. **Calendar-aware.** ``weekday_sf`` / ``eod_sf`` cycles resolve via
   :mod:`nousergon_lib.trading_calendar` — NYSE holidays = no
   check (returns ``state='fresh'`` with a holiday reason so the
   Lambda short-circuits the alert path). ``continuous`` artifacts
   declare their producer calendar via ``run_calendar`` (``trading_days``
   / ``all_days`` / ``market_hours``), which drives BOTH the idle
   short-circuit and a trading-day-aware freshness floor — so a daily
   trading-day producer is not flagged stale across the weekend gap.
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

- :func:`nousergon_lib.alerts.publish` — the alert chokepoint the
  Lambda calls with ``dedup_key=resolve_dedup_key(spec, now)``.
- :mod:`nousergon_lib.trading_calendar` — NYSE-holiday substrate.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Final, Literal

from nousergon_lib.trading_calendar import (
    is_trading_day,
    last_closed_trading_day,
    previous_trading_day,
)


# ── Cadence symbols ──────────────────────────────────────────────────────────
# Plan §4 PR 2 commits the registry to a closed set of cadence symbols
# rather than free-form cron strings. The symbols encode the cron
# expression AND the calendar-awareness rule. New symbols added here
# require both a cycle-resolution rule in ``resolve_current_cycle`` and
# a window-label rule in ``resolve_dedup_key``.
CadenceSymbol = Literal[
    "saturday_sf", "weekday_sf", "eod_sf", "continuous", "event_driven"
]

CADENCE_SYMBOLS: Final[frozenset[str]] = frozenset(
    {"saturday_sf", "weekday_sf", "eod_sf", "continuous", "event_driven"}
)

# ``event_driven`` is the cadence for artifacts written ONLY when a gated
# producer decides to write (an optimizer that clears its ``_MIN_SAMPLES``
# floor, a one-way latch, a challenger that beats its champion). Their
# ABSENCE is correct, so they must NEVER self-page on age — the historical
# workaround was ``grace_period_cycles: 999`` (a magic number that
# permanently BLINDS the row: a row that never alerts on age also loses all
# liveness signal, so a genuinely-dead producer is indistinguishable from
# one that correctly declined to write). ``event_driven`` replaces that
# whack-a-mole with an explicit, validator-enforced contract: the row's
# own freshness short-circuits to ``fresh`` (never stale/missing on age),
# and its producer-liveness is DELEGATED to an independent, separately
# freshness-monitored proxy named by ``ArtifactSpec.liveness_via`` (e.g. an
# unconditional per-run report the producer stage always writes). The
# proxy going stale is what pages — so liveness is never silently lost, it
# is relocated to a signal that CAN go stale. ``build_dependency_graph``
# enforces that every ``event_driven`` row names a real, non-``event_driven``
# anchor (so liveness can never chain into another never-paging row).
# See config#1718 (this primitive) + config#1726 (registry application).

# Cron-tick hours (UTC) per the live EventBridge rules. Source:
# ``~/Development/CLAUDE.md`` § Architecture diagrams (Weekly Freshness SF at
# 09:00 UTC, Pre-open Trading SF at 13:00 UTC). Post-close Trading SF is daemon-triggered
# post-close so its "expected cron" anchors to a conservative 21:00 UTC
# (after the daemon shutdown + EOD reconcile typically completes).
_SATURDAY_SF_CRON_UTC: Final[int] = 9
_WEEKDAY_SF_CRON_UTC: Final[int] = 13
_EOD_SF_ANCHOR_UTC: Final[int] = 21

CheckState = Literal["fresh", "stale", "missing", "probe_failed", "grace_period"]


# ── Run-calendar symbols ─────────────────────────────────────────────────────
# The producer's actual run calendar — the single source of truth for a
# ``continuous`` artifact's calendar-awareness (Brian's directive 2026-06-28:
# "all freshness checks tied to trading day unless there is a clear reason
# not to"). Subsumed the prior ad-hoc ``active_trading_days_only`` boolean
# (removed in v0.102.0, config#1334) and parameterizes the idle
# short-circuit AND the freshness floor:
#
# - ``trading_days`` — producer runs only on NYSE session days (the
#   DEFAULT-by-principle for new continuous specs). Non-trading days
#   short-circuit to ``fresh``; on trading days the floor is computed in
#   TRADING-DAY terms so the weekend/holiday gap before a session is not
#   counted against the producer (kills the Monday-morning false positive
#   that ``active_trading_days_only`` alone left unfixed).
# - ``all_days`` — producer genuinely runs every calendar day (the
#   DOCUMENTED exception, e.g. a 7-day GHA cron or the 24/7 self-monitor
#   heartbeat). Wall-clock floor ``now - (interval + sla)``. Each use must
#   be justified — it is wrong for any producer that skips weekends.
# - ``market_hours`` — producer runs only within the NYSE session window on
#   session days; requires ``active_hours_utc``. Idle outside the window
#   (overnight / weekends / holidays); inside, the rolling floor applies so
#   a mid-session death is still caught.
RunCalendarSymbol = Literal["trading_days", "all_days", "market_hours"]

RUN_CALENDAR_SYMBOLS: Final[frozenset[str]] = frozenset(
    {"trading_days", "all_days", "market_hours"}
)


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
        severity: Routed to :func:`nousergon_lib.alerts.publish` on
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
        run_calendar: ``continuous``-only. The producer's actual run
            calendar — one of :data:`RUN_CALENDAR_SYMBOLS`
            (``trading_days`` / ``all_days`` / ``market_hours``). The
            single source of truth for a continuous artifact's
            calendar-awareness: it drives BOTH the idle short-circuit and
            the freshness floor (see :func:`_resolve_run_calendar` for the
            resolution precedence and :func:`_freshness_floor` for the
            floor it selects). ``None`` ⇒ resolved from ``active_hours_utc``
            (set ⇒ ``market_hours``) for backward-compatibility, else
            ``all_days``; new continuous specs should set this explicitly
            and default to ``trading_days``.
        active_hours_utc: ``continuous``-only ``(start, end)`` UTC-hour
            bounds — REQUIRED when ``run_calendar="market_hours"`` (the
            session window). When set, the check short-circuits to
            ``state="fresh"`` outside ``[start, end)`` UTC — the producer
            is idle by design (e.g. the executor daemon writing
            ``open_orders`` only during the NYSE session). Inside the
            window the normal recency floor still applies, so a producer
            that dies mid-window is still caught.
        produces: Optional lineage edge — the ``artifact_id``s this
            artifact's producer *stage* also emits. Advisory metadata for
            the dashboard/stage grouping; the DAG walk keys off
            ``depends_on``. Default ``()`` (declared none).
        depends_on: Optional lineage edges — the ``artifact_id``s this
            artifact's producer READS to produce it (artifact → upstream
            artifact). Turns the flat registry into a DAG so
            :func:`localize_root_causes` can walk a downstream miss back to
            the first missing upstream (the actual failed stage) and
            :func:`leaf_alert_decisions` can suppress the blocked-downstream
            cascade. An empty ``depends_on`` marks a DAG *root* — a miss
            there is its own root cause and pages as it does today, so
            unedged specs are behaviour-preserving. Default ``()``.
        liveness_via: REQUIRED for ``cadence="event_driven"``, forbidden
            otherwise. The ``artifact_id`` of the independent, separately
            freshness-monitored proxy that carries this row's
            producer-liveness (an unconditional per-run report the producer
            stage always writes, regardless of whether it applied a change).
            The event-driven row itself short-circuits to ``fresh`` (its
            absence is correct), so its liveness rides this anchor: the
            anchor going stale is what pages. :func:`build_dependency_graph`
            enforces that the target exists and is NOT itself
            ``event_driven`` (liveness cannot chain into another
            never-paging row). ``None`` for every non-event-driven cadence.
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
    run_calendar: RunCalendarSymbol | None = None
    active_hours_utc: tuple[int, int] | None = None
    produces: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    liveness_via: str | None = None

    def __post_init__(self) -> None:
        if self.cadence not in CADENCE_SYMBOLS:
            raise ValueError(
                f"ArtifactSpec.cadence={self.cadence!r} not in "
                f"{sorted(CADENCE_SYMBOLS)}"
            )
        self._validate_liveness_via()
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
        self._validate_active_window()
        self._validate_lineage()

    def _validate_liveness_via(self) -> None:
        """Validate the ``event_driven`` ↔ ``liveness_via`` coupling.

        Per-spec half of the invariant: an ``event_driven`` row MUST name a
        liveness anchor (else it is silently blind — the exact
        ``grace_period_cycles: 999`` failure this cadence replaces), and no
        other cadence may carry one (it would be meaningless — those rows
        self-page on their own age). The *referential* half (the anchor
        exists and is itself non-``event_driven``) is registry-global, so it
        is enforced in :func:`build_dependency_graph`, not here — a single
        spec cannot see its siblings.
        """
        if self.cadence == "event_driven":
            if not self.liveness_via or not isinstance(self.liveness_via, str):
                raise ValueError(
                    f"ArtifactSpec.cadence='event_driven' ({self.artifact_id!r}) "
                    "requires liveness_via=<anchor artifact_id> — an "
                    "event_driven row without a liveness anchor is silently "
                    "blind (the grace_period_cycles=999 failure this cadence "
                    "exists to replace)"
                )
            if self.liveness_via == self.artifact_id:
                raise ValueError(
                    f"ArtifactSpec.liveness_via for {self.artifact_id!r} "
                    "references itself — an event_driven row cannot be its "
                    "own liveness anchor"
                )
        elif self.liveness_via is not None:
            raise ValueError(
                f"ArtifactSpec.liveness_via is only valid for "
                f"cadence='event_driven' (got cadence={self.cadence!r} on "
                f"{self.artifact_id!r})"
            )

    def _validate_lineage(self) -> None:
        """Validate (and normalize) the ``produces`` / ``depends_on`` DAG edges.

        Both arrive from YAML as lists; coerce each to a tuple of unique,
        order-preserving, non-empty ``artifact_id`` strings on the frozen
        dataclass so the spec stays hashable and the edges are total.
        Referential integrity (every ``depends_on`` id exists in the
        registry) and acyclicity are *registry-global* properties, so they
        are enforced by :func:`build_dependency_graph`, not here — a single
        spec cannot see its siblings.
        """
        for attr in ("produces", "depends_on"):
            raw = getattr(self, attr)
            if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
                raise ValueError(
                    f"ArtifactSpec.{attr} must be a list/tuple of artifact_id "
                    f"strings, got {raw!r}"
                )
            seen: set[str] = set()
            edges: list[str] = []
            for item in raw:
                if not isinstance(item, str) or not item:
                    raise ValueError(
                        f"ArtifactSpec.{attr} entries must be non-empty "
                        f"strings, got {item!r}"
                    )
                if item == self.artifact_id and attr == "depends_on":
                    raise ValueError(
                        f"ArtifactSpec.depends_on for {self.artifact_id!r} "
                        f"references itself — a self-edge is a 1-cycle"
                    )
                if item not in seen:
                    seen.add(item)
                    edges.append(item)
            object.__setattr__(self, attr, tuple(edges))

    def _validate_active_window(self) -> None:
        """Validate (and normalize) the continuous calendar fields.

        ``run_calendar`` and the (deprecated) active-window bounds are all
        ``continuous``-only — the SF cadences carry their own calendar gate.
        ``active_hours_utc`` arrives from YAML as a list; coerce it to a
        ``(start, end)`` int tuple on the frozen dataclass so downstream
        comparisons are total.
        """
        if self.run_calendar is not None:
            if self.run_calendar not in RUN_CALENDAR_SYMBOLS:
                raise ValueError(
                    f"ArtifactSpec.run_calendar={self.run_calendar!r} not in "
                    f"{sorted(RUN_CALENDAR_SYMBOLS)}"
                )
            if self.cadence != "continuous":
                raise ValueError(
                    "ArtifactSpec.run_calendar is only valid for "
                    f"cadence='continuous' (got cadence={self.cadence!r})"
                )
        # market_hours requires the session-window bounds; resolve the
        # effective calendar (explicit field OR legacy-boolean fallback) so
        # the requirement holds however it was declared.
        if _resolve_run_calendar(self) == "market_hours" and (
            self.active_hours_utc is None
        ):
            raise ValueError(
                "ArtifactSpec.run_calendar='market_hours' requires "
                "active_hours_utc=(start, end)"
            )
        if self.active_hours_utc is None:
            return
        if self.cadence != "continuous":
            raise ValueError(
                "ArtifactSpec.active_hours_utc is only valid for "
                f"cadence='continuous' (got cadence={self.cadence!r})"
            )
        if not isinstance(self.active_hours_utc, (list, tuple)):
            raise ValueError(
                "ArtifactSpec.active_hours_utc must be a (start, end) pair, "
                f"got {self.active_hours_utc!r}"
            )
        hours = tuple(self.active_hours_utc)
        if len(hours) != 2 or not all(isinstance(h, int) for h in hours):
            raise ValueError(
                "ArtifactSpec.active_hours_utc must be a 2-tuple of ints, "
                f"got {self.active_hours_utc!r}"
            )
        start, end = hours
        if not (0 <= start < end <= 24):
            raise ValueError(
                "ArtifactSpec.active_hours_utc must satisfy "
                f"0 <= start < end <= 24, got {self.active_hours_utc!r}"
            )
        object.__setattr__(self, "active_hours_utc", (start, end))


# ── Result ──────────────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    """One ``check_freshness`` outcome.

    Attributes:
        state: Outcome class. ``fresh`` ⇒ the most recent instance's
            ``last_modified`` is at or after the freshness floor (or a
            recovery instance is). ``stale`` ⇒ an instance exists but its
            newest ``last_modified`` predates the floor (this cycle's
            artifact genuinely missing; off-cycle/early production within
            the floor reads fresh). ``missing`` ⇒ no instance at all.
            ``probe_failed`` ⇒ S3 client error on the canonical probe
            other than 404 (the monitor itself is broken).
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
    semantics use :func:`nousergon_lib.trading_calendar.last_closed_trading_day`
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
    :func:`nousergon_lib.trading_calendar.last_closed_trading_day`;
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

    if spec.cadence == "event_driven":
        # Event-driven rows have no cron schedule — their absence is correct
        # and they never self-page (see check_freshness step 0). This label
        # is only consumed by resolve_dedup_key / canonical_key reporting, so
        # a stable per-UTC-day window suffices to collapse enrichment retries.
        day = now_utc.date()
        tick = datetime(
            day.year, day.month, day.day, 0, 0, tzinfo=timezone.utc,
        )
        return tick, f"event_driven_{day.isoformat()}"

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
    :func:`nousergon_lib.alerts.publish`.

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


# ── Recency-window resolution (the "most recent / fresh" model) ──────────────
#
# Freshness is judged by the RECENCY of the newest existing instance, NOT by
# whether an artifact exists under one exact cron-date key. The prior model
# HEADed `key.format(date=cron_tick)` and called a present object "stale" iff
# `last_modified < cron_tick` — which false-alarmed whenever the producer
# wrote a day off the cron (an off-cycle Friday run, a {trading_day} that
# resolves to the prior close, a Friday-morning run anchoring to Thursday).
# A monitor that can't tolerate a one-day anchor shift isn't doing its job.
#
# Instead: find the most recent instance (LIST the prefix for date-templated
# keys; HEAD for fixed keys) and check its age against a cadence-derived
# freshness floor. Off-cycle / off-by-one / early production all land within
# the floor and read FRESH; a genuinely missed cycle ages past the floor and
# reads STALE; nothing at all reads MISSING.


def _is_templated(template: str) -> bool:
    """True when the key carries a per-cycle ``{...}`` placeholder."""
    return "{" in template


def _listable_prefix(template: str) -> str:
    """The fixed prefix of a templated key — everything before the first
    ``{`` placeholder. ``signals/{trading_day}/signals.json`` -> ``signals/``;
    ``market_data/weekly/{date}/manifest.json`` -> ``market_data/weekly/``.
    """
    return template.split("{", 1)[0]


def _key_suffix(template: str) -> str:
    """The fixed suffix after the last ``}`` placeholder — used to filter
    listed objects to the artifact (not its siblings under the same prefix).
    ``signals/{trading_day}/signals.json`` -> ``/signals.json``;
    ``predictor/predictions/{trading_day}.json`` -> ``.json``. Empty string
    when the placeholder is terminal (rare).
    """
    return template.rsplit("}", 1)[-1]


def _newest_under_prefix(
    s3_client: Any,
    bucket: str,
    prefix: str,
    suffix: str,
    *,
    cap_pages: int = 8,
) -> tuple[str, datetime | None, str | None]:
    """Return ``(newest_key, newest_last_modified, probe_error)`` for the
    most-recently-modified object under ``prefix`` whose key ends with
    ``suffix``.

    Pure w.r.t. side effects beyond ``s3_client`` LIST calls. Paginates up
    to ``cap_pages`` pages (8 × 1000 = 8000 objects) — every freshness-
    tracked date-templated prefix is far smaller; the cap is a runaway
    backstop. A LIST client error (403 / network) returns
    ``("", None, reason)`` so the caller can surface ``probe_failed`` rather
    than mis-reporting ``missing``.
    """
    newest_lm: datetime | None = None
    newest_key = ""
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        for page_idx, page in enumerate(pages):
            if page_idx >= cap_pages:
                break
            for obj in page.get("Contents", []) or []:
                key = obj.get("Key", "")
                if suffix and not key.endswith(suffix):
                    continue
                lm = obj.get("LastModified")
                if lm is None:
                    continue
                if lm.tzinfo is None:
                    lm = lm.replace(tzinfo=timezone.utc)
                if newest_lm is None or lm > newest_lm:
                    newest_lm, newest_key = lm, key
    except Exception as err:  # noqa: BLE001 — duck-typed boto error classification
        state, reason = _classify_client_error(err)
        # A LIST that 404s is nonsensical (prefix-level); treat any LIST
        # error as a probe failure — the monitor can't see the bucket.
        if state == "missing":
            reason = f"S3 LIST returned 404-class for prefix {prefix!r}: {reason}"
        return ("", None, reason)
    return (newest_key, newest_lm, None)


# Max-age staleness window for the saturday weekly cadence, counted in
# CALENDAR days from ``now``. The freshness floor is ``now − 10 days``: the
# newest existing instance is FRESH iff its ``last_modified`` is within the
# last 10 calendar days, STALE otherwise. (config#1297, Brian's directive
# 2026-06-27.)
#
# Why 10 calendar days anchored to ``now`` (not the prior trading-day slack
# anchored to ``cycle_tick``):
#
#   1. **No Saturday-SF-start burst.** The prior model derived the floor from
#      ``cycle_tick`` (the most-recent Saturday 09:00 UTC tick). At the instant
#      the Saturday cron ticked, the "current cycle" flipped to this week, but
#      this week's multi-hour SF had not yet produced its artifacts — so last
#      week's ~7-day-old instances were judged against a freshly-advanced floor
#      and flipped STALE, paging ~22 false alerts at 2am PT (2026-06-27).
#      Anchoring the window to ``now`` and sizing it at 10 days means last
#      week's run (≤7 days old) is comfortably inside the window regardless of
#      whether this Saturday's run has started — the burst is structurally
#      impossible.
#   2. **Run-day jitter tolerated.** A weekly run that lands on Friday and the
#      next on the following Sunday is ≈9 calendar days apart — normal jitter,
#      must NOT alert. 9 ≤ 10, so it reads FRESH.
#   3. **Genuine misses still caught.** A genuinely-skipped week ages past 10
#      calendar days and reads STALE — the absence backstop. (Real-time SF
#      failure is separately caught by the Saturday-SF Watch agent.)
#
# A per-spec ``stale_after_days`` override may be threaded later; for now the
# constant is the single source of the weekly window.
_SATURDAY_SF_STALE_DAYS: Final[int] = 10


def _freshness_floor(
    spec: ArtifactSpec, now_utc: datetime, cycle_tick: datetime
) -> datetime:
    """The oldest ``last_modified`` that still counts as FRESH for ``spec``.

    - ``saturday_sf``: ``now`` minus :data:`_SATURDAY_SF_STALE_DAYS` (10)
      CALENDAR days — a max-age recency window anchored to ``now``, NOT to
      ``cycle_tick``. The newest instance is FRESH iff modified within the
      last 10 days. Anchoring to ``now`` (rather than the Saturday cron tick)
      is what kills the Saturday-SF-start false-positive burst: last week's
      ~7-day-old artifacts stay inside the window at the instant the Saturday
      cron ticks, before this week's multi-hour SF has produced replacements.
      A Fri→Sun run-day-jitter gap (≈9d) also stays inside; a genuinely-missed
      week (>10d) ages out and reads STALE.
    - ``weekday_sf`` / ``eod_sf``: the start of ``previous_trading_day(
      last_closed_trading_day(now))`` — calendar-aware, so the newest
      instance must be from within the last ~2 trading days (tolerates the
      {trading_day} anchor + overnight + weekends/holidays).
    - ``continuous``: depends on the resolved run-calendar
      (:func:`_resolve_run_calendar`):

      * ``all_days`` — wall-clock ``now - (interval + sla)``. Correct only
        for a producer that genuinely runs every calendar day.
      * ``trading_days`` / ``market_hours`` with a daily-or-longer interval
        (``>= 1440`` min) — the TRADING-DAY floor ``previous_trading_day(
        last_closed_trading_day(now))`` (same as ``weekday_sf``). This is
        what fixes the Monday-morning false positive: the wall-clock
        ``now - 1440 - sla`` window reaches back across the weekend and
        flags Friday/Saturday writes stale before Monday's run; the
        trading-day floor counts only session days, so the weekend gap
        isn't held against the producer. The non-trading-day case is
        already short-circuited to ``fresh`` upstream by the idle gate.
      * ``trading_days`` / ``market_hours`` with a SUB-daily interval
        (``< 1440`` min) — the rolling ``now - (interval + sla)`` window.
        A sub-day lookback inside the active window never spans a
        non-session gap, so the wall-clock window is already correct; the
        idle gate handles everything outside the window.
    """
    if spec.cadence == "saturday_sf":
        return now_utc - timedelta(days=_SATURDAY_SF_STALE_DAYS)
    if spec.cadence in ("weekday_sf", "eod_sf"):
        floor_day = previous_trading_day(last_closed_trading_day(now_utc))
        return datetime(
            floor_day.year, floor_day.month, floor_day.day,
            tzinfo=timezone.utc,
        )
    if spec.cadence == "continuous":
        assert spec.interval_minutes is not None
        rc = _resolve_run_calendar(spec)
        if rc != "all_days" and spec.interval_minutes >= 1440:
            # Daily-or-longer trading-day producer: count the lookback in
            # trading days, not wall-clock, so a weekend/holiday gap before
            # the current session is not counted against the producer.
            floor_day = previous_trading_day(last_closed_trading_day(now_utc))
            return datetime(
                floor_day.year, floor_day.month, floor_day.day,
                tzinfo=timezone.utc,
            )
        return now_utc - (
            timedelta(minutes=spec.interval_minutes)
            + timedelta(minutes=spec.sla_minutes_after_cron)
        )
    if spec.cadence == "event_driven":
        # Never classified by an age floor — check_freshness short-circuits
        # event_driven to ``fresh`` before this is consulted. Return the
        # unix epoch as a defensive far-past floor so that, on any path that
        # does reach here, the newest instance (if one exists) is trivially
        # ``>= floor`` ⇒ fresh, never stale on age.
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    raise ValueError(f"unknown cadence {spec.cadence!r}")


def _resolve_run_calendar(spec: ArtifactSpec) -> str:
    """Resolve the effective run-calendar for a ``continuous`` spec.

    Precedence (S3-contract-safe migration off the deprecated boolean):

    1. Explicit ``spec.run_calendar`` wins.
    2. Else ``active_hours_utc`` set ⇒ ``market_hours`` (the session-window
       producer's legacy declaration).
    3. Else ``all_days`` (the conservative default — preserves the prior
       24/7 wall-clock behavior for any continuous spec that declares
       nothing, so the migration never silently flips an un-migrated spec).

    Non-continuous cadences carry their own calendar gate; this is only
    consulted on the ``continuous`` path.
    """
    if spec.run_calendar is not None:
        return spec.run_calendar
    if spec.active_hours_utc is not None:
        return "market_hours"
    return "all_days"


def _continuous_idle_reason(spec: ArtifactSpec, now_utc: datetime) -> str | None:
    """Return a human reason when a ``continuous`` spec is OUTSIDE its
    active production window (so artifact absence is correct), else ``None``.

    Driven by the resolved :func:`_resolve_run_calendar`:

    - ``all_days``: never idle (24/7 producer).
    - ``trading_days``: idle on weekends + NYSE holidays.
    - ``market_hours``: idle on non-session days AND outside the
      ``active_hours_utc`` ``[start, end)`` UTC window.

    This gate only suppresses the structural off-window false positive
    (a market-hours-only daemon judged against the 24/7 ``now - interval -
    sla`` floor). INSIDE the window the normal recency floor still applies,
    so a producer that dies mid-window is caught as ``stale``.
    """
    rc = _resolve_run_calendar(spec)
    if rc == "all_days":
        return None
    if not is_trading_day(now_utc.date()):
        return (
            f"non-trading day {now_utc.date().isoformat()} — continuous "
            f"producer (run_calendar={rc}) idle, absence is correct"
        )
    if rc == "market_hours" and spec.active_hours_utc is not None:
        start, end = spec.active_hours_utc
        if not (start <= now_utc.hour < end):
            return (
                f"{now_utc.hour:02d}:xx UTC outside active production window "
                f"[{start:02d}:00,{end:02d}:00) UTC — producer idle, "
                "absence is correct"
            )
    return None


# ── check_freshness — the core public function ──────────────────────────────


def check_freshness(
    s3_client: Any, spec: ArtifactSpec, now: datetime
) -> CheckResult:
    """Probe ``spec`` and return the classified outcome.

    Pure with respect to side effects beyond the ``s3_client.head_object``
    call (no logging, no alerting, no DDB / S3 marker writes). The
    Lambda is responsible for routing the result to
    :func:`nousergon_lib.alerts.publish` with
    ``dedup_key=resolve_dedup_key(spec, now)``.

    Freshness is judged by the RECENCY of the most recent existing
    instance — NOT by whether an artifact exists under one exact
    cron-date key. The prior model HEADed ``key.format(date=cron_tick)``
    and called a present object stale iff ``last_modified < cron_tick``,
    which false-alarmed whenever the producer wrote a day off the cron
    (an off-cycle ``run_weekly_offcycle.sh full`` Friday run, a
    ``{trading_day}`` that resolves to the prior close, a Friday-morning
    run anchoring to Thursday). A monitor that can't tolerate a one-day
    anchor shift isn't doing its job. The probe walks four steps:

    1. **Grace-period gate.** If ``(now - spec.created_at)`` is shorter
       than ``spec.grace_period_cycles`` cycles, return
       ``state="grace_period"`` — newly-onboarded producers don't
       false-alarm on their first emissions.
    2. **Calendar-holiday gate.** When ``spec.calendar_aware`` and the
       resolved cycle's date is NOT a trading day, return
       ``state="fresh"`` with a holiday reason — the cron didn't fire,
       so absence is correct.
    3. **Find the most recent instance.** For date-templated keys, LIST
       the prefix and take the newest object matching the template
       suffix; for fixed keys (latest-pointers / manifests), HEAD once.
       A canonical-probe error (403 / network) is AUTHORITATIVE →
       ``state="probe_failed"`` (the monitor is blind; don't mask it).
       The recovery template, if any, folds in as a best-effort second
       source.
    4. **Classify by recency.** Compare the newest instance's
       ``last_modified`` against :func:`_freshness_floor` — a max-age
       window (``now`` minus 10 calendar days for ``saturday_sf``;
       ~2 trading days for ``weekday_sf`` / ``eod_sf``; ``interval + sla``
       for ``continuous``). ``>= floor`` ⇒ ``fresh``; older ⇒ ``stale``;
       no instance at all ⇒ ``missing``. The 10-day ``saturday_sf`` window
       is anchored to ``now`` (not the Saturday cron tick) so last week's
       artifacts stay fresh at the instant this Saturday's SF starts —
       killing the Saturday-SF-start false-positive burst (config#1297).

    ``sla_violated_by_minutes`` reports how far the breach is past the
    SLA/floor; clipped at zero so the field is always non-negative.
    """
    now_utc = _utc(now)
    cycle_tick, cycle_label = resolve_current_cycle(spec, now_utc)

    # ── 0. Event-driven short-circuit ───────────────────────────────────
    # An event_driven artifact is written only when a gated producer decides
    # to write; its absence is CORRECT, so it must never self-page on age.
    # Its producer-liveness rides an independent freshness-monitored proxy
    # (``spec.liveness_via``, validated real + non-event_driven in
    # build_dependency_graph) — the proxy going stale is what pages. So the
    # row's own freshness is always ``fresh`` here: liveness is not lost, it
    # is relocated to a signal that CAN go stale (config#1718/#1726). This
    # replaces the ``grace_period_cycles: 999`` blind-spot with an explicit,
    # enforced contract.
    if spec.cadence == "event_driven":
        return CheckResult(
            state="fresh",
            reason=(
                f"event_driven — absence is correct for gated writes; "
                f"producer-liveness delegated to freshness proxy "
                f"{spec.liveness_via!r}"
            ),
            canonical_key=_format_key(
                spec.s3_key_template, cycle_label, cycle_tick,
            ),
        )

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

    # ── 2b. Continuous active-window short-circuit ──────────────────────
    # A continuous producer that only runs in a bounded window — e.g. the
    # executor daemon, which writes trades/open_orders/latest.json each tick
    # ONLY while paper-trading on an NYSE session day — is correctly idle
    # overnight, on weekends, and on holidays. Judging it against the 24/7
    # continuous floor (now - interval - sla) false-alarms every interval
    # outside that window (the 2026-06-26 open_orders overnight alert storm).
    # Short-circuit to fresh when the producer is idle by design; inside the
    # window the floor below still applies, so a daemon that dies mid-session
    # is still caught.
    if spec.cadence == "continuous":
        idle_reason = _continuous_idle_reason(spec, now_utc)
        if idle_reason is not None:
            return CheckResult(
                state="fresh",
                reason=idle_reason,
                canonical_key=_format_key(
                    spec.s3_key_template, cycle_label, cycle_tick,
                ),
            )

    # ── 3. Find the MOST RECENT instance (recency model) ────────────────
    # No exact-cron-date matching: for date-templated keys we LIST the
    # prefix and take the newest object matching the template suffix; for
    # fixed keys (latest-pointers / manifests) we HEAD once. The recovery
    # template, if any, folds into the same "newest" search. This is robust
    # to off-cycle runs, the {trading_day} anchor, and early/late writes —
    # whatever the producer actually wrote, wherever it landed, we find the
    # freshest one. ``expected_key`` is kept only as a reporting hint.
    expected_key = _format_key(spec.s3_key_template, cycle_label, cycle_tick)
    floor = _freshness_floor(spec, now_utc, cycle_tick)

    def _probe(tmpl: str) -> tuple[str, datetime | None, str | None]:
        """Return ``(newest_key, newest_last_modified, probe_error)`` for one
        template — LIST the prefix (date-templated) or HEAD (fixed key)."""
        if _is_templated(tmpl):
            return _newest_under_prefix(
                s3_client, spec.s3_bucket,
                _listable_prefix(tmpl), _key_suffix(tmpl),
            )
        state, lm, reason = _head_object(s3_client, spec.s3_bucket, tmpl)
        if state == "probe_failed":
            return ("", None, reason)
        return (tmpl if state == "present" else "", lm if state == "present" else None, None)

    # Canonical probe is AUTHORITATIVE: if the monitor can't read the
    # canonical location, surface probe_failed (don't mask it with a
    # recovery hit — the operator needs to know the monitor is blind).
    newest_key, newest_lm, canonical_error = _probe(spec.s3_key_template)
    if canonical_error is not None:
        return CheckResult(
            state="probe_failed",
            reason=canonical_error,
            canonical_key=expected_key,
        )

    # Recovery is best-effort: fold its newest instance in (recovery probe
    # errors are non-authoritative and ignored).
    recovery_substituted = False
    if spec.recovery_key_template is not None:
        rec_key, rec_lm, _rec_err = _probe(spec.recovery_key_template)
        if rec_lm is not None and (newest_lm is None or rec_lm > newest_lm):
            newest_lm, newest_key = rec_lm, rec_key
            recovery_substituted = True

    # ── 4. Classify by recency vs the freshness floor ───────────────────
    if newest_lm is None:
        # Canonical probe already succeeded (probe_failed returns early
        # above) and found nothing; recovery, if any, also empty.
        sla_deadline = cycle_tick + timedelta(
            minutes=spec.sla_minutes_after_cron,
        )
        return CheckResult(
            state="missing",
            sla_violated_by_minutes=int(
                max(0, (now_utc - sla_deadline).total_seconds() // 60)
            ),
            reason=(
                f"no instance found under "
                f"{_listable_prefix(spec.s3_key_template)!r} "
                f"(expected ~{expected_key})"
            ),
            canonical_key=expected_key,
        )

    age_min = int((now_utc - newest_lm).total_seconds() // 60)
    if newest_lm >= floor:
        return CheckResult(
            state="fresh",
            last_modified=newest_lm,
            reason=(
                f"freshest instance {newest_key} "
                f"last_modified={newest_lm.isoformat()} (age {age_min}min) "
                f">= freshness floor {floor.isoformat()}"
            ),
            canonical_key=expected_key,
            recovery_substituted=recovery_substituted,
        )

    return CheckResult(
        state="stale",
        last_modified=newest_lm,
        sla_violated_by_minutes=int(
            max(0, (floor - newest_lm).total_seconds() // 60)
        ),
        reason=(
            f"freshest instance {newest_key} "
            f"last_modified={newest_lm.isoformat()} (age {age_min}min) "
            f"older than freshness floor {floor.isoformat()}"
        ),
        canonical_key=expected_key,
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
    if spec.cadence == "event_driven":
        # Coarse cold-start unit only (event_driven short-circuits the
        # grace-period arithmetic in check_freshness); a nominal day keeps
        # this non-raising for any incidental caller.
        return 24 * 3600
    raise ValueError(f"unknown cadence {spec.cadence!r}")


# ── Per-cycle completion rollup ───────────────────────────────────────────────


CycleState = Literal["complete", "incomplete", "indeterminate"]


@dataclass
class CycleCompletion:
    """Per-cycle completion verdict — the artifact-union judgment.

    Aggregates the per-artifact :class:`CheckResult` rows for one
    execution cycle into a single verdict over the *required* set
    (the ``severity="critical"`` rows). Answers the question the
    raw orchestrator status cannot on a recovery-stitched run: *did
    this cycle actually deliver every load-bearing artifact?*

    Recovery substitution is already folded in upstream — a
    canonical-missing artifact rescued by its ``recovery_key_template``
    arrives here as ``state="fresh"``. So this rollup judges the
    execution UNION without re-HEADing anything.

    Attributes:
        state: ``"complete"`` ⇒ every required artifact is present +
            valid (``fresh``, or suppressed by ``grace_period``).
            ``"incomplete"`` ⇒ at least one required artifact is
            ``missing`` / ``stale`` (a real delivery gap).
            ``"indeterminate"`` ⇒ no real gap, but at least one probe
            ``probe_failed`` (the monitor itself is broken, so the
            cycle can't be confirmed). A real gap outranks an
            indeterminate probe.
        complete: ``True`` iff ``state == "complete"``.
        cycle_label: The cycle's window label (e.g. ``"2026-W22"``),
            for reporting. Informational — the caller passes it.
        n_required: Count of ``severity="critical"`` artifacts judged.
        n_satisfied: Count present + valid (``fresh`` + ``grace_period``).
        missing / stale / probe_failed / grace_period: ``artifact_id``
            localization lists — which artifacts landed in each state.
        reason: Human-readable summary; routed to the report surface.
    """

    state: CycleState
    complete: bool
    cycle_label: str | None = None
    n_required: int = 0
    n_satisfied: int = 0
    missing: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    probe_failed: list[str] = field(default_factory=list)
    grace_period: list[str] = field(default_factory=list)
    reason: str = ""


def cycle_completion(
    spec_results: Iterable[tuple[ArtifactSpec, CheckResult]],
    *,
    cycle_label: str | None = None,
) -> CycleCompletion:
    """Roll per-artifact freshness results up into one cycle verdict.

    ``cycle_completion(C) = ∀ required artifact a: present(a@C) ∧ valid(a@C)``
    over the execution UNION, where the required set is the
    ``severity="critical"`` rows. Non-critical (``warning``) artifacts
    are excluded — they inform per-artifact alerting but never gate the
    cycle verdict.

    Pure: consumes already-computed :class:`CheckResult` rows (as
    ``(spec, result)`` pairs so there's no positional-pairing hazard)
    and performs no I/O. Recovery substitution and the calendar-holiday
    short-circuit are already reflected in each ``result.state`` by
    :func:`check_freshness`, so a holiday cycle or a recovery-rescued
    artifact both count as satisfied here.

    State precedence: a real delivery gap (``missing`` / ``stale``)
    outranks a broken probe (``probe_failed``) — a confirmed miss is
    more actionable than an unconfirmable one. ``grace_period`` counts
    as satisfied (the producer is newly onboarded; suppressed by design)
    but is surfaced in its own list so the caller can see it.

    An empty required set returns ``state="complete"`` (vacuous truth) —
    a cycle with no critical artifacts cannot be incomplete.
    """
    required = [(s, r) for s, r in spec_results if s.severity == "critical"]

    missing: list[str] = []
    stale: list[str] = []
    probe_failed: list[str] = []
    grace_period: list[str] = []
    satisfied = 0

    for spec, res in required:
        if res.state == "fresh":
            satisfied += 1
        elif res.state == "grace_period":
            satisfied += 1
            grace_period.append(spec.artifact_id)
        elif res.state == "stale":
            stale.append(spec.artifact_id)
        elif res.state == "missing":
            missing.append(spec.artifact_id)
        elif res.state == "probe_failed":
            probe_failed.append(spec.artifact_id)

    n_required = len(required)

    if missing or stale:
        gaps = []
        if missing:
            gaps.append(f"missing={missing}")
        if stale:
            gaps.append(f"stale={stale}")
        return CycleCompletion(
            state="incomplete",
            complete=False,
            cycle_label=cycle_label,
            n_required=n_required,
            n_satisfied=satisfied,
            missing=missing,
            stale=stale,
            probe_failed=probe_failed,
            grace_period=grace_period,
            reason=(
                f"cycle incomplete: {satisfied}/{n_required} critical artifacts "
                f"present+valid; " + "; ".join(gaps)
            ),
        )

    if probe_failed:
        return CycleCompletion(
            state="indeterminate",
            complete=False,
            cycle_label=cycle_label,
            n_required=n_required,
            n_satisfied=satisfied,
            probe_failed=probe_failed,
            grace_period=grace_period,
            reason=(
                f"cycle indeterminate: monitor probe failed for {probe_failed} — "
                f"cannot confirm cycle ({satisfied}/{n_required} confirmed fresh)"
            ),
        )

    grace_note = f" ({len(grace_period)} in grace period)" if grace_period else ""
    return CycleCompletion(
        state="complete",
        complete=True,
        cycle_label=cycle_label,
        n_required=n_required,
        n_satisfied=satisfied,
        grace_period=grace_period,
        reason=(
            f"cycle complete: all {n_required} critical artifacts present+valid"
            + grace_note
        ),
    )


# ── Dependency DAG + root-cause localization (Phase 2) ────────────────────────
#
# The freshness substrate above judges each artifact *independently*. That is
# correct for freshness but wrong for *alerting*: on a recovery-stitched cycle a
# single upstream miss (say DataPhase1) makes every downstream artifact that
# reads it also read "missing", so the operator gets a cascade of alerts whose
# real root is one stage. Two symptoms this closes (design doc
# `artifact-completion-monitoring-design-260529.md` §4 Move 2):
#
#   1. Root-cause localization — page the *first missing upstream* (the actual
#      failed stage), not the downstream deliverables it starved.
#   2. Leaf-only / healed-downstream suppression — if the terminal deliverable
#      landed anyway (a recovery run produced it directly), the intermediate
#      upstream miss is moot and must NOT fire.
#
# The unit is `depends_on` (artifact → the upstream artifacts its producer
# reads). An artifact with no `depends_on` is a DAG *root*: its miss is its own
# root cause and pages exactly as it does today, so the flat (unedged) registry
# is behaviour-preserving — edges only ever *suppress* redundant pages, never
# add new ones.

# A delivered artifact — neither pages nor blocks its downstream. `grace_period`
# is suppressed-by-design (newly onboarded producer), counted satisfied here
# exactly as `cycle_completion` does.
_SATISFIED_STATES: Final[frozenset[str]] = frozenset({"fresh", "grace_period"})
# A *confirmed* delivery gap — the only state that can explain (block) a
# downstream artifact's own miss. `probe_failed` is deliberately excluded: an
# unconfirmable upstream must not suppress a real downstream miss (mirrors
# cycle_completion's "a real gap outranks an unconfirmable probe").
_GAP_STATES: Final[frozenset[str]] = frozenset({"missing", "stale"})

# Per-artifact routing verdict.
#   ``ok``       — satisfied (fresh / grace_period); never pages.
#   ``failed``   — a confirmed gap (missing/stale) at a DAG root-of-gap position
#                  (its own producer is at fault). Pages iff *consequential*.
#   ``blocked``  — a gap / probe_failed explained by a confirmed upstream gap
#                  (not its own fault); the upstream root pages, this stays
#                  silent.
#   ``degraded`` — probe_failed at a root-of-gap position (the monitor could not
#                  confirm). Preserves today's page-on-probe_failed, iff
#                  consequential.
ArtifactClass = Literal["ok", "failed", "blocked", "degraded"]


@dataclass(frozen=True)
class DependencyGraph:
    """Immutable adjacency view over a registry's ``depends_on`` edges.

    Built by :func:`build_dependency_graph`, which enforces the two
    registry-global invariants a single :class:`ArtifactSpec` cannot:
    referential integrity (every ``depends_on`` target is a real
    ``artifact_id``) and acyclicity.

    Attributes:
        depends_on: ``artifact_id`` → the upstream ids its producer reads.
        dependents: reverse edges — ``artifact_id`` → the downstream ids
            that read it.
        roots: ids with no ``depends_on`` (they read nothing tracked).
        leaves: ids nothing depends on (terminal deliverables).
    """

    depends_on: dict[str, tuple[str, ...]]
    dependents: dict[str, tuple[str, ...]]
    roots: tuple[str, ...]
    leaves: tuple[str, ...]

    def upstream(self, artifact_id: str) -> tuple[str, ...]:
        """Direct upstream ids ``artifact_id``'s producer reads."""
        return self.depends_on.get(artifact_id, ())

    def downstream(self, artifact_id: str) -> tuple[str, ...]:
        """Direct downstream ids that read ``artifact_id``."""
        return self.dependents.get(artifact_id, ())

    def reachable_leaves(self, artifact_id: str) -> tuple[str, ...]:
        """Terminal deliverables reachable downstream of ``artifact_id``.

        A node with no dependents is itself a leaf ⇒ ``(artifact_id,)``.
        Cycle-safe via a visited set (the graph is acyclic by
        construction, but the guard keeps a hand-built graph from looping).
        """
        seen: set[str] = set()
        out: list[str] = []
        stack = [artifact_id]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            kids = self.dependents.get(node, ())
            if not kids:
                out.append(node)
            else:
                stack.extend(kids)
        return tuple(out)


@dataclass(frozen=True)
class AlertDecision:
    """Whether (and why) one artifact should page, after DAG resolution.

    Attributes:
        artifact_id: The artifact this verdict is for.
        classification: :data:`ArtifactClass` — ``ok`` / ``failed`` /
            ``blocked`` / ``degraded``.
        should_page: Whether the Lambda should route an alert. ``False``
            for satisfied, blocked-by-upstream, and healed-downstream nodes.
        root_cause_ids: The upstream root(s) explaining this node's gap —
            ``(artifact_id,)`` when the node is itself the root, the
            confirmed-gap ancestors otherwise (a diamond can have several).
            ``()`` for ``ok``.
        reason: Human-readable diagnostic routed into the alert body.
    """

    artifact_id: str
    classification: ArtifactClass
    should_page: bool
    root_cause_ids: tuple[str, ...] = ()
    reason: str = ""


def build_dependency_graph(specs: Iterable[ArtifactSpec]) -> DependencyGraph:
    """Assemble a :class:`DependencyGraph` from registry specs.

    Enforces the two registry-global lineage invariants:

    * **Referential integrity** — every ``depends_on`` target is a real
      ``artifact_id`` in ``specs``. A dangling edge is a typo that would
      silently break localization, so it raises.
    * **Acyclicity** — the ``depends_on`` relation is a DAG. A cycle would
      make "walk to the first missing upstream" non-terminating and has no
      real-pipeline meaning, so it raises with the offending cycle.

    Raises:
        ValueError: on a duplicate ``artifact_id``, a dangling ``depends_on``
            target, or a dependency cycle.
    """
    specs = list(specs)
    depends_on: dict[str, tuple[str, ...]] = {}
    for spec in specs:
        if spec.artifact_id in depends_on:
            raise ValueError(
                f"duplicate artifact_id {spec.artifact_id!r} in dependency graph"
            )
        depends_on[spec.artifact_id] = tuple(spec.depends_on)

    all_ids = set(depends_on)
    dependents: dict[str, list[str]] = {aid: [] for aid in all_ids}
    for aid, ups in depends_on.items():
        for up in ups:
            if up not in all_ids:
                raise ValueError(
                    f"artifact {aid!r} depends_on unknown artifact {up!r} — "
                    f"dangling lineage edge (not an artifact_id in the registry)"
                )
            dependents[up].append(aid)

    # ── event_driven liveness-anchor referential integrity ──────────────
    # Every event_driven row delegates its liveness to a proxy named by
    # ``liveness_via``. The per-spec validator (_validate_liveness_via)
    # guarantees the field is present and non-self; the registry-global
    # invariants — enforceable only with all specs in hand — are:
    #   (1) the anchor is a real artifact_id in the registry (a dangling
    #       anchor silently blinds the row, the failure this cadence exists
    #       to prevent); and
    #   (2) the anchor is NOT itself event_driven (liveness must not chain
    #       into another never-paging row, or the whole chain is blind).
    cadence_by_id = {spec.artifact_id: spec.cadence for spec in specs}
    for spec in specs:
        if spec.cadence != "event_driven":
            continue
        anchor = spec.liveness_via
        if anchor not in all_ids:
            raise ValueError(
                f"event_driven artifact {spec.artifact_id!r} names "
                f"liveness_via={anchor!r} which is not an artifact_id in the "
                f"registry — dangling liveness anchor (the row would be "
                f"silently blind)"
            )
        if cadence_by_id[anchor] == "event_driven":
            raise ValueError(
                f"event_driven artifact {spec.artifact_id!r} names "
                f"liveness_via={anchor!r} which is itself event_driven — "
                f"liveness cannot chain into another never-paging row; the "
                f"anchor must be an independently freshness-monitored "
                f"(non-event_driven) deliverable"
            )

    _raise_on_cycle(depends_on)

    roots = tuple(sorted(a for a, ups in depends_on.items() if not ups))
    leaves = tuple(sorted(a for a in all_ids if not dependents[a]))
    return DependencyGraph(
        depends_on=depends_on,
        dependents={a: tuple(v) for a, v in dependents.items()},
        roots=roots,
        leaves=leaves,
    )


def _raise_on_cycle(depends_on: dict[str, tuple[str, ...]]) -> None:
    """DFS three-colouring; raise ValueError naming a cycle if one exists."""
    WHITE, GREY, BLACK = 0, 1, 2
    colour = {a: WHITE for a in depends_on}

    def visit(node: str, path: list[str]) -> None:
        colour[node] = GREY
        path.append(node)
        for up in depends_on.get(node, ()):
            if colour[up] == GREY:
                cyc = path[path.index(up):] + [up]
                raise ValueError(
                    "dependency cycle in ARTIFACT_REGISTRY depends_on: "
                    + " -> ".join(cyc)
                )
            if colour[up] == WHITE:
                visit(up, path)
        path.pop()
        colour[node] = BLACK

    for aid in depends_on:
        if colour[aid] == WHITE:
            visit(aid, [])


def leaf_alert_decisions(
    spec_results: Iterable[tuple[ArtifactSpec, CheckResult]],
    *,
    graph: DependencyGraph | None = None,
) -> dict[str, AlertDecision]:
    """Resolve per-artifact freshness results into DAG-aware page decisions.

    Given the same ``(spec, CheckResult)`` pairs the Lambda already computes,
    return one :class:`AlertDecision` per artifact so the Lambda can route
    **leaf-only, root-cause-localized** alerts instead of one page per
    independently-missing artifact.

    An artifact pages iff ALL of:

    1. it has a gap of its own — ``state`` ∈ {missing, stale, probe_failed}
       (i.e. not satisfied); AND
    2. it is the *root* of that gap — no direct upstream is a confirmed gap
       (missing/stale); a node blocked by a confirmed upstream gap stays
       silent and points at that upstream; AND
    3. the gap is *consequential* — at least one terminal deliverable
       reachable downstream did not land. If every downstream leaf is
       satisfied (a recovery run produced it directly), the upstream miss was
       healed and must not fire.

    Unedged specs (no ``depends_on`` and no dependents) are simultaneously
    root and leaf, so this reduces to "page a confirmed/unconfirmable miss" —
    exactly today's behaviour. Edges only suppress redundant pages.

    Pure: consumes already-computed :class:`CheckResult` rows and performs no
    I/O. Builds the graph from ``specs`` when ``graph`` is not supplied.
    """
    pairs = list(spec_results)
    if graph is None:
        graph = build_dependency_graph(s for s, _ in pairs)
    state_by_id = {s.artifact_id: r.state for s, r in pairs}
    severity_by_id = {s.artifact_id: s.severity for s, _ in pairs}

    def is_confirmed_gap(aid: str) -> bool:
        return state_by_id.get(aid) in _GAP_STATES

    def blocking_upstreams(aid: str) -> list[str]:
        return [u for u in graph.upstream(aid) if is_confirmed_gap(u)]

    decisions: dict[str, AlertDecision] = {}
    for aid, state in state_by_id.items():
        if state in _SATISFIED_STATES:
            decisions[aid] = AlertDecision(aid, "ok", False, (), reason=f"{state}")
            continue

        blockers = blocking_upstreams(aid)
        if blockers:
            roots = _root_causes(aid, graph, is_confirmed_gap)
            decisions[aid] = AlertDecision(
                aid,
                "blocked",
                should_page=False,
                root_cause_ids=roots,
                reason=(
                    f"{state}: blocked on upstream gap "
                    f"{sorted(roots) or sorted(blockers)} — suppressed "
                    f"(root pages)"
                ),
            )
            continue

        # Root of its own gap. Does it actually cost a deliverable?
        leaves = graph.reachable_leaves(aid)
        consequential = any(
            state_by_id.get(leaf) not in _SATISFIED_STATES for leaf in leaves
        )
        classification: ArtifactClass = (
            "degraded" if state == "probe_failed" else "failed"
        )
        if not consequential:
            landed = [leaf for leaf in leaves if state_by_id.get(leaf) in _SATISFIED_STATES]
            decisions[aid] = AlertDecision(
                aid,
                classification,
                should_page=False,
                root_cause_ids=(aid,),
                reason=(
                    f"{state}: healed downstream — deliverable(s) "
                    f"{sorted(landed)} landed; suppressed"
                ),
            )
            continue

        lost = [leaf for leaf in leaves if state_by_id.get(leaf) not in _SATISFIED_STATES]
        note = (
            "root-cause miss"
            if classification == "failed"
            else "unconfirmable probe"
        )
        decisions[aid] = AlertDecision(
            aid,
            classification,
            should_page=True,
            root_cause_ids=(aid,),
            reason=(
                f"{state}: {note} (severity={severity_by_id.get(aid)}); "
                f"starves deliverable(s) {sorted(lost)}"
                if lost != [aid]
                else f"{state}: {note} (severity={severity_by_id.get(aid)})"
            ),
        )
    return decisions


def _root_causes(
    artifact_id: str,
    graph: DependencyGraph,
    is_confirmed_gap,
) -> tuple[str, ...]:
    """Walk ``depends_on`` up from ``artifact_id`` to the confirmed-gap root(s).

    A root cause is a confirmed-gap ancestor that is *itself* not blocked by a
    further confirmed-gap upstream — i.e. the first missing stage on each
    branch. Diamonds can yield several. Cycle-safe via a visited set.
    """
    seen: set[str] = set()
    roots: list[str] = []
    stack = [u for u in graph.upstream(artifact_id) if is_confirmed_gap(u)]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        deeper = [u for u in graph.upstream(node) if is_confirmed_gap(u)]
        if deeper:
            stack.extend(deeper)
        else:
            roots.append(node)
    return tuple(sorted(set(roots)))


def localize_root_causes(
    spec_results: Iterable[tuple[ArtifactSpec, CheckResult]],
    *,
    graph: DependencyGraph | None = None,
) -> dict[str, tuple[str, ...]]:
    """Map each *gapped* artifact_id → its upstream confirmed-gap root(s).

    A thin projection of :func:`leaf_alert_decisions` for callers that only
    want localization (e.g. a report surface): satisfied artifacts are
    omitted; a root-of-gap artifact maps to ``(itself,)``; a blocked
    downstream maps to the upstream root(s) that explain it.
    """
    return {
        aid: d.root_cause_ids
        for aid, d in leaf_alert_decisions(spec_results, graph=graph).items()
        if d.classification != "ok"
    }
