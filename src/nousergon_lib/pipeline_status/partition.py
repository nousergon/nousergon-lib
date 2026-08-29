"""Which S3 date partition a weekly artifact belongs to — and the deadline.

``alpha-engine-config-I8809``.

## The defect

One weekly cycle was written across TWO date partitions. The weekly state
machine's ``InitializeInput`` stamped ``run_date = date($$.Execution.StartTime)``
— the CALENDAR date — and each consumer then decided for itself whether to
normalise it to the trading day. Measured on the 2026-08-22 cycle: **28
``_stage_coverage`` verdicts landed under ``2026-08-21`` and 11 under
``2026-08-22``, from the same cycle.** ``weekly-coverage-sweep`` reads one
partition, so its first production firing would have reported 28 stages
``absent`` — the single state this module's sibling
:mod:`~.coverage` documents as the serious one, from the surface built so a
thin cycle could not read as complete.

## The fix, and why it is dated

The state machine now emits BOTH fields — ``run_date`` (the trading day, via
the idempotent :func:`krepis.dates.resolve_trading_day`) and ``calendar_date``
— so every keying site reads the field it means. But history is NOT rewritten
(Brian ruling 2026-08-27), and the not-yet-converged writers keep landing in
the calendar partition for one cycle. So the reader unions both families
during a migration window and cuts over to the trading-day family alone at
:data:`CUTOVER_DATE`.

**The window is a deadline, not a preference.** A compatibility fallback with
no expiry is indistinguishable from the defect it was added to hide: once both
partitions are read forever, a writer that silently reverts to the calendar
family produces no signal at all.
``tests/test_pipeline_status_partition.py`` FAILS on and after
:data:`CUTOVER_DATE` while :data:`CUTOVER_DATE` is still in the future of
nothing — i.e. the test is the thing that cannot be forgotten, and clearing it
requires deleting the fallback rather than moving the date.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Final

__all__ = [
    "CUTOVER_DATE",
    "PARTITION_FAMILIES",
    "cycle_keys",
    "dual_partition_active",
    "partition_dates",
]

#: After this date the calendar-partition fallback is GONE and the trading-day
#: family is the only one read. Not a soft target: the partition test fails
#: from this date onward while the fallback still exists.
CUTOVER_DATE: Final[date] = date(2026, 9, 5)

#: The closed vocabulary ``ARTIFACT_REGISTRY.yaml``'s ``partition_family``
#: field may take. Mirrored here so a consumer can validate without the
#: private registry.
PARTITION_FAMILIES: Final[frozenset[str]] = frozenset({"trading_day", "calendar_date"})


def _today(today: date | None = None) -> date:
    if today is not None:
        return today
    return datetime.now(timezone.utc).date()


def dual_partition_active(today: date | None = None) -> bool:
    """Is the migration window still open?

    ``True`` strictly BEFORE :data:`CUTOVER_DATE`. On the cutover date itself
    the window is shut — a window that includes its own end date is one more
    day of ambiguity for no stated reason.
    """
    return _today(today) < CUTOVER_DATE


def partition_dates(
    run_date: str,
    calendar_date: str | None = None,
    *,
    today: date | None = None,
) -> tuple[str, ...]:
    """The date partitions to read for one cycle, canonical FIRST.

    The canonical partition is always ``run_date`` (the trading day). While
    :func:`dual_partition_active`, ``calendar_date`` is appended when it is
    present and different. Order is load-bearing: a verdict found in the
    canonical partition WINS, so a stage that has converged is never reported
    from the legacy one.

    Returns a tuple with no duplicates and no empty entries.
    """
    out: list[str] = []
    for candidate in (run_date, calendar_date if dual_partition_active(today) else None):
        text = (candidate or "").strip()
        if text and text not in out:
            out.append(text)
    return tuple(out)


def cycle_keys(run_date: str, calendar_date: str | None = None) -> tuple[str, ...]:
    """The execution-identity keys belonging to one cycle, canonical FIRST.

    **This is NOT :func:`partition_dates`, and it does not expire.**
    ``partition_dates`` answers *which S3 prefixes hold this cycle's
    artifacts* — a migration question, closed at :data:`CUTOVER_DATE`.
    This answers *which Step Functions executions ARE this cycle*, and the
    answer is permanently two-valued for a trading-day-keyed cycle:

    ``nousergon_lib.pipeline_status.read._extract_run_date`` resolves an
    execution's cycle key from ``input.run_date`` → the execution name →
    ``startDate``. A scheduled weekly execution carries no ``run_date`` in its
    input and an opaque UUID name, so its key always falls through to
    ``startDate`` — the execution's WALL-CLOCK day. That derivation is
    calendar-family by construction and cannot be made trading-day-family:
    Step Functions stamps the start time, and no NYSE calendar reaches it.

    Since ``alpha-engine-config-I8809`` the weekly graph keys its artifacts on
    the TRADING day, so on every Saturday run the two disagree — the cycle is
    ``2026-08-28`` and its own execution's identity is ``2026-08-29``.
    Measured 2026-08-29 against the live state machine:
    ``read_cycle_shape(arn, "2026-08-28")`` returned ``skipped
    (declared_skip), 0/16 stages`` — the FRIDAY gate-skip execution — while
    the Saturday execution that did the week's work was not a contributor at
    all. Two consequences, and the second is the serious one:

    1. the coverage denominator becomes the skip run's entered set, so almost
       every real stage reads ``not_entered`` and ``ABSENT`` — the one state
       :mod:`~.coverage` calls serious — becomes structurally unreachable;
    2. :func:`~.completion_marker.augment_marker` stamps ``cycle.verdict:
       skipped`` onto the completion marker of a cycle that fully completed,
       which is ``alpha-engine-config-I8186`` re-created with the sign
       flipped, on the object every ``gate:*`` ``Verified-when:`` predicate
       reads.

    Admitting both keys is safe in the direction that matters: a cycle's
    coverage is a UNION (:func:`~.cycle_shape.build_cycle_shape`), so an extra
    contributor can only ADD entered stages, and a declared-skip tick of the
    same weekly cadence adds none. Under-reading drops the run that did the
    work; over-reading cannot manufacture work that no execution performed.

    Returns a tuple with no duplicates and no empty entries.
    """
    out: list[str] = []
    for candidate in (run_date, calendar_date):
        text = (candidate or "").strip()
        if text and text not in out:
            out.append(text)
    return tuple(out)
