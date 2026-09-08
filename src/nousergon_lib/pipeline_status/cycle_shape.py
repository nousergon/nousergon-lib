"""A cadence CYCLE's real shape — every execution that contributed to it.

``work.py`` answers *did this EXECUTION do the work*. That is not the same
question as *did the CYCLE complete*, and on ``ne-weekly-freshness-pipeline``
the two routinely disagree in the direction nothing detects.

## The defect this module exists to close (``alpha-engine-config-I8186``)

``s3://alpha-engine-research/_sf_completion/ne-weekly-freshness-pipeline/2026-08-22.json``
reads ``status: SUCCEEDED`` and records **one** ``execution_arn``. That
execution entered 14 states and dispatched ``Director``,
``ScannerLeaderboard`` and two health checks; the other ~21 stages of the
weekly graph were skipped because the *scheduled* execution had already
completed them hours earlier.

**The cycle genuinely completed.** That is what a mechanical recovery IS, and
the marker is not lying about the cycle. What it cannot express is that the
cycle completed across *two* executions — so:

- no consumer can tell a full run from a recovery tail,
- ``sf_success_rate`` counts one execution where two ran,
- and the ``gate:*`` ``Verified-when:`` predicates written against the
  marker's mere existence clear on whichever execution happened to reach
  ``WriteCompletionMarker``.

The fix is not to withhold the marker from a recovery — that would call a
legitimate recovery a failure, which is the overcorrection
``alpha-engine-config-I8186`` explicitly forbids. The fix is to make the
marker carry the cycle's real shape: **which executions contributed, what
each entered, and what the union of them adds up to.**

## Union semantics, and why they are not "most recent wins"

A cycle's coverage is the UNION of its contributing executions' entered
stages, because a recovery rerun deliberately skips what the scheduled run
already did. Reading only the last execution reports a 1-of-16 tail; reading
only the first reports a failure that was subsequently repaired. Neither is
the cycle.

The union is taken over **cadence and recovery roles only**
(:data:`~.roles.CADENCE_ROLES` | :data:`~.roles.RECOVERY_ROLES`). An exercise
run or a smoke test writes real artifacts but is not the cycle's deliverable,
and folding one into the union would let a Tuesday debugging run mark the
week's belief refresh complete — the separation ``roles.py`` calls a binding
invariant.

## The four cycle verdicts

:class:`CycleVerdict` is closed and has no fall-through:

- :attr:`CycleVerdict.COMPLETED` — the union of contributing executions
  covers the declared spine. Reached in one execution or in five; the count
  is reported, never the discriminator.
- :attr:`CycleVerdict.SKIPPED` — every contributing execution reached a
  declared skip terminal. No work was due.
- :attr:`CycleVerdict.INCOMPLETE` — the union does not cover the spine.
- :attr:`CycleVerdict.IN_FLIGHT` — at least one contributor is still running
  and the union is not yet complete. Never collapsed into INCOMPLETE: a run
  that has not finished has not failed.

A cycle with **no** contributing executions is :attr:`CycleVerdict.INCOMPLETE`
with reason ``no_executions`` — never SKIPPED, never absent from the surface.
A cadence tick that produced no execution at all is the missed-run class, and
rendering it as "nothing to say" is how a never-fired schedule stays invisible
(``observability-policy.md`` §8.3: *no data* is never green).

Public surface:

- :class:`CycleVerdict` / :class:`CycleExecution` / :class:`CycleShape`
- :func:`build_cycle_shape` — pure ``(outcomes) -> CycleShape``.
- :func:`read_cycle_shape` — the boto3 front door for one ``run_date``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from .partition import cycle_keys
from .read import (
    RunStatus,
    _extract_pipeline_role,
    _paged_execution_history,
    _parse_ts,
    _raise_for_boto_error,
    _sfn_client,
)
from .registry import stage_order_for
from .roles import CADENCE_ROLES, RECOVERY_ROLES
from .work import WorkOutcome, WorkVerdict, classify_work, entered_states_from_history

if TYPE_CHECKING:  # pragma: no cover
    from mypy_boto3_stepfunctions.client import SFNClient
else:  # pragma: no cover
    SFNClient = Any

logger = logging.getLogger(__name__)

__all__ = [
    "CONTRIBUTING_ROLES",
    "cycle_key_for",
    "CycleExecution",
    "CycleShape",
    "CycleVerdict",
    "build_cycle_shape",
    "read_cycle_shape",
]

#: Roles whose executions count toward a cycle's coverage union. Cadence runs
#: ARE the cycle; recovery runs legitimately complete it after a failure.
#: Exercise and ad-hoc roles are deliberately excluded — see the module
#: docstring and ``roles.py``'s own invariant.
CONTRIBUTING_ROLES = CADENCE_ROLES | RECOVERY_ROLES

#: How far back ``read_cycle_shape`` will walk ``ListExecutions`` looking for
#: a cycle's contributors. The weekly SF fires THU-SAT and reruns are same-day,
#: so a cycle's executions are always within the most recent few dozen. A cap
#: rather than an unbounded walk: an unbounded scan of a long-lived state
#: machine is an unpriced API bill, and a cycle whose contributors fall off
#: the end is reported as ``walk_exhausted`` rather than silently truncated.
DEFAULT_WALK_CAP: int = 60

#: How far BEFORE a cycle's own date ``read_cycle_shape`` keeps walking.
#: ``ListExecutions`` returns newest-first by ``startDate``, and no execution
#: that started before this window can belong to the cycle: an execution's
#: cycle key resolves from its input ``run_date``, its name, or its
#: ``startDate`` — all three of which are on or after the cycle's own day for
#: a contributor, never a fortnight before it. So the walk has a CORRECT
#: termination condition and does not need to rely on the count cap.
#:
#: ``alpha-engine-config-I10161``. Measured 2026-09-08: every weekly sweep
#: since the mechanism shipped reported ``walk_exhausted: true`` — 60 was
#: smaller than the number of executions this state machine accumulates in a
#: week, so the *documented* downgrade ("an INCOMPLETE verdict under an
#: exhausted walk is not trustworthy") applied on EVERY run and was
#: implemented by no caller. A date bound makes exhaustion exceptional again,
#: which is the precondition for the downgrade meaning anything.
DEFAULT_LOOKBACK_DAYS: int = 14


def cycle_key_for(describe_resp: Mapping[str, Any]) -> str | None:
    """The cycle key of one execution — delegates to :func:`~.read._extract_run_date`.

    THIS IS A THIN ALIAS AND MUST STAY ONE. The three-source resolution
    (``input.run_date`` → execution name → ``startDate`` normalised to UTC)
    lives in ``read._extract_run_date`` and nowhere else.

    It briefly lived here as a second implementation, because ``read``'s
    resolver had only two sources and dropped every scheduled execution
    (``alpha-engine-config-I8216``). Two resolvers for one key is how the cycle
    surface and the reliability window start disagreeing about which
    executions belong to which cycle — the defect one layer up from the one
    the third source fixes. The fix was to lift the third source into
    ``read``, not to keep a better copy beside it.
    """
    from .read import _extract_run_date  # local: avoids a cycle at import time

    return _extract_run_date(describe_resp)


class CycleVerdict(str, Enum):
    """Did the CYCLE complete? Closed, exhaustive, no fall-through."""

    COMPLETED = "completed"
    SKIPPED = "skipped"
    INCOMPLETE = "incomplete"
    IN_FLIGHT = "in_flight"


@dataclass(frozen=True)
class CycleExecution:
    """One execution's contribution to a cycle."""

    execution_arn: str
    execution_name: str
    pipeline_role: str | None
    status: str
    verdict: str
    reason: str
    duration_sec: float | None
    #: Declared spine stages THIS execution entered, spine order.
    stages_entered: tuple[str, ...] = ()
    #: Every state name this execution entered, in order — the substrate the
    #: coverage sweep intersects against the artifact registry's stage set.
    #: Kept out of ``repr`` because it runs to a few hundred names.
    all_states_entered: tuple[str, ...] = field(default=(), repr=False)
    #: This execution is the one PERFORMING the observation — the sweep runs
    #: as a state inside it. Its entered states are real and count toward the
    #: union; its RUNNING status does not, because it cannot terminate before
    #: the thing it is running. See :func:`build_cycle_shape`.
    is_observer: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_arn": self.execution_arn,
            "execution_name": self.execution_name,
            "pipeline_role": self.pipeline_role,
            "status": self.status,
            "verdict": self.verdict,
            "reason": self.reason,
            "duration_sec": self.duration_sec,
            "stages_entered": list(self.stages_entered),
            "stages_entered_count": len(self.stages_entered),
            "is_observer": self.is_observer,
        }


@dataclass(frozen=True)
class CycleShape:
    """The cycle's verdict plus every execution it was derived from."""

    pipeline: str
    run_date: str
    verdict: CycleVerdict
    #: Closed vocabulary: ``full_cycle`` · ``recovered_across_executions`` ·
    #: ``declared_skip`` · ``no_executions`` · ``partial_cycle`` ·
    #: ``still_running``.
    reason: str
    executions: tuple[CycleExecution, ...] = ()
    stage_spine: tuple[str, ...] = field(default=(), repr=False)
    stages_entered: tuple[str, ...] = ()
    stages_missing: tuple[str, ...] = ()
    #: True when the ``ListExecutions`` walk hit its cap before running out of
    #: executions — the contributor set may be incomplete, so a COMPLETED
    #: verdict is still trustworthy (the union only grows) but an INCOMPLETE
    #: one is not, and is downgraded to a stated uncertainty by the caller.
    walk_exhausted: bool = False
    #: The execution the sweep itself is running inside, when the observer was
    #: named. Empty when the cycle was read from outside any of its own
    #: executions (the CLI, a backfill, an out-of-band re-sweep).
    observer_execution_arn: str = ""

    @property
    def execution_count(self) -> int:
        return len(self.executions)

    @property
    def is_recovery_tail(self) -> bool:
        """The cycle completed, but not within a single execution.

        The fact ``alpha-engine-config-I8186`` says the old marker could not
        express. It is NOT a failure and must never be rendered as one — it is
        the difference between a clean scheduled run and a repaired week, and
        a reliability surface that cannot see it reports both as one green.
        """
        return self.verdict is CycleVerdict.COMPLETED and self.execution_count > 1

    @property
    def stage_coverage(self) -> str:
        return f"{len(self.stages_entered)}/{len(self.stage_spine)}"

    @property
    def did_work(self) -> bool:
        return self.verdict is CycleVerdict.COMPLETED

    @property
    def is_terminal(self) -> bool:
        """The cycle has reached a verdict that will not change by waiting.

        ``IN_FLIGHT`` is the only non-terminal verdict. A consumer that grades
        a non-terminal cycle is grading a partial fact and reporting it as a
        whole one — which is what produced the 2026-09-04 false page (13
        stages called ``absent`` on a cycle the same artifact declared
        ``in_flight``).
        """
        return self.verdict is not CycleVerdict.IN_FLIGHT

    @property
    def verdict_trustworthy(self) -> bool:
        """Is this verdict safe to act on?

        The downgrade :attr:`walk_exhausted` documents and no caller applied.
        The union only GROWS with more contributors, so a ``COMPLETED``
        verdict survives a truncated walk; ``INCOMPLETE`` and its
        ``stages_missing`` do not — an unseen contributor is
        indistinguishable from a stage that never ran.
        """
        if not self.walk_exhausted:
            return True
        return self.verdict in (CycleVerdict.COMPLETED, CycleVerdict.SKIPPED)

    @property
    def untrustworthy_reason(self) -> str:
        """Why :attr:`verdict_trustworthy` is False, or ``""``."""
        if self.verdict_trustworthy:
            return ""
        return (
            f"the ListExecutions walk hit its cap before the cycle window closed, so the "
            f"contributor set may be incomplete and a {self.verdict.value} verdict "
            f"(missing {', '.join(self.stages_missing) or 'nothing'}) cannot be distinguished "
            f"from an unseen contributor that entered those stages"
        )

    def explain(self) -> str:
        head = f"{self.pipeline} cycle {self.run_date}: {self.verdict.value} ({self.reason})"
        across = (
            f" across {self.execution_count} executions"
            if self.execution_count != 1
            else " in 1 execution"
        )
        tail = f", stages {self.stage_coverage}{across}"
        if self.stages_missing:
            tail += f", missing {', '.join(self.stages_missing)}"
        if self.walk_exhausted:
            tail += " [ListExecutions walk cap reached — contributor set may be incomplete]"
        return head + tail

    def to_dict(self) -> dict[str, Any]:
        """Primitive projection — this IS the completion marker's payload."""
        return {
            "pipeline": self.pipeline,
            "run_date": self.run_date,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "did_work": self.did_work,
            "is_recovery_tail": self.is_recovery_tail,
            "execution_count": self.execution_count,
            "executions": [e.to_dict() for e in self.executions],
            "stages_entered": list(self.stages_entered),
            "stages_missing": list(self.stages_missing),
            "stage_coverage": self.stage_coverage,
            "walk_exhausted": self.walk_exhausted,
            "observer_execution_arn": self.observer_execution_arn,
            "is_terminal": self.is_terminal,
            "verdict_trustworthy": self.verdict_trustworthy,
            "explain": self.explain(),
        }


def build_cycle_shape(
    *,
    pipeline: str,
    run_date: str,
    outcomes: Sequence[tuple[WorkOutcome, str | None, Sequence[str]]],
    stage_spine: Sequence[str] | None = None,
    walk_exhausted: bool = False,
    observer_execution_arn: str | None = None,
) -> CycleShape:
    """Fold one cycle's executions into a single verdict. Pure.

    ``outcomes`` is ``[(work_outcome, pipeline_role, all_entered_states)]``,
    most-recent-first or oldest-first — order is irrelevant to the union and
    the executions are re-sorted oldest-first for the record.

    Only :data:`CONTRIBUTING_ROLES` are folded in. A contributor with **no**
    role is included: an untagged manual run is legitimate and common
    (``roles.py``), and excluding it would drop real work from the union.

    ## The observer (``alpha-engine-config-I10161``)

    ``observer_execution_arn`` names the execution the CALLER is running
    inside. Its entered states still count toward the union — they are real
    work — but its ``RUNNING`` status does **not** make the cycle
    ``IN_FLIGHT``, because it cannot possibly have terminated: the caller is
    a state within it.

    Without this, a sweep invoked as a tail state of its own pipeline reads
    ``in_flight (still_running)`` on **every single run**, forever. Measured
    on the live 2026-09-04 cycle: the sweep at ``21:39:18Z`` graded a cycle it
    declared in-flight because ``watch-rerun-2026-09-04-4`` was RUNNING — and
    that execution terminated one second later, at ``21:39:19Z``, on the
    Succeed state two hops past the sweep. The same shape appears on the
    2026-08-28 cycle's artifact, so this was chronic and not an incident. Two
    consequences, and the second is the one that reaches other systems:

    1. ``IN_FLIGHT`` is structurally unreachable as a *signal* — a cycle that
       is genuinely still running is indistinguishable from every healthy one;
    2. :func:`~.completion_marker.augment_marker` stamps ``cycle_verdict:
       in_flight`` and ``did_work: false`` onto the completion marker of a
       cycle that COMPLETED — the object every ``gate:*`` ``Verified-when:``
       predicate reads. That is ``alpha-engine-config-I8186`` re-created with
       the sign flipped, which is exactly what :func:`~.partition.cycle_keys`
       warned about for the mirror-image bug.

    Excluding only the observer is safe in the direction that matters: any
    OTHER running execution of the same cycle still yields ``IN_FLIGHT``, so
    the honest "not yet gradeable" answer is preserved for the case that
    actually means it.
    """
    spine = tuple(stage_spine) if stage_spine is not None else stage_order_for(pipeline)

    contributors: list[tuple[WorkOutcome, str | None, Sequence[str]]] = []
    for outcome, role, states in outcomes:
        if role and role not in CONTRIBUTING_ROLES:
            # Real work, real writes, but not this cycle's deliverable.
            continue
        contributors.append((outcome, role, states))

    observer = (observer_execution_arn or "").strip()

    executions = tuple(
        CycleExecution(
            execution_arn=outcome.execution_arn or "",
            execution_name=outcome.execution_name or "",
            pipeline_role=role,
            status=outcome.status.value,
            verdict=outcome.verdict.value,
            reason=outcome.reason,
            duration_sec=outcome.duration_sec,
            stages_entered=outcome.stages_entered,
            all_states_entered=tuple(states),
            is_observer=bool(observer) and (outcome.execution_arn or "") == observer,
        )
        for outcome, role, states in contributors
    )

    union: set[str] = set()
    for outcome, _role, _states in contributors:
        union.update(outcome.stages_entered)
    entered = tuple(s for s in spine if s in union)
    missing = tuple(s for s in spine if s not in union)

    common: dict[str, Any] = {
        "pipeline": pipeline,
        "run_date": run_date,
        "executions": executions,
        "stage_spine": spine,
        "stages_entered": entered,
        "stages_missing": missing,
        "walk_exhausted": walk_exhausted,
        "observer_execution_arn": observer,
    }

    if not contributors:
        # No execution at all is the missed-run class, and it is INCOMPLETE.
        # Rendering it as "nothing to report" is how a schedule that never
        # fired stays invisible for a month.
        return CycleShape(verdict=CycleVerdict.INCOMPLETE, reason="no_executions", **common)

    if not missing:
        reason = "recovered_across_executions" if len(contributors) > 1 else "full_cycle"
        return CycleShape(verdict=CycleVerdict.COMPLETED, reason=reason, **common)

    if all(o.verdict is WorkVerdict.SKIPPED for o, _r, _s in contributors):
        return CycleShape(verdict=CycleVerdict.SKIPPED, reason="declared_skip", **common)

    if any(
        o.verdict is WorkVerdict.IN_FLIGHT
        and not (observer and (o.execution_arn or "") == observer)
        for o, _r, _s in contributors
    ):
        # Not yet complete and something OTHER THAN THE OBSERVER is still
        # running. A run that has not finished has not failed — but the
        # execution the caller is running inside can never have finished, and
        # counting it here made IN_FLIGHT the permanent answer.
        return CycleShape(verdict=CycleVerdict.IN_FLIGHT, reason="still_running", **common)

    return CycleShape(verdict=CycleVerdict.INCOMPLETE, reason="partial_cycle", **common)


def read_cycle_shape(
    state_machine_arn: str,
    run_date: str,
    *,
    calendar_date: str | None = None,
    client: SFNClient | None = None,
    walk_cap: int = DEFAULT_WALK_CAP,
    stage_spine: Sequence[str] | None = None,
    observer_execution_arn: str | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> CycleShape:
    """Read every execution of ``run_date`` and fold them into one verdict.

    Walks ``ListExecutions`` newest-first up to ``walk_cap``, keeping the
    executions whose ``run_date`` (input field, else execution name, else
    ``startDate`` in UTC — the three sources :func:`~.read._extract_run_date`
    resolves) is one of this cycle's keys.

    ``calendar_date`` is the execution's own wall-clock day, and on a
    trading-day-keyed cycle it is a SECOND legitimate identity for the same
    cycle — see :func:`~.partition.cycle_keys` for the measurement and for
    why this is permanent rather than a migration fallback. The returned
    :attr:`CycleShape.run_date` is always the CANONICAL ``run_date``, so the
    completion marker this shape augments keeps its canonical key.

    The walk terminates on a DATE BOUND, not on the count cap: executions come
    back newest-first by ``startDate``, and once one started more than
    ``lookback_days`` before the cycle's own day, no later page can hold a
    contributor. :data:`DEFAULT_WALK_CAP` remains as a hard stop and is still
    reported as :attr:`CycleShape.walk_exhausted`, but reaching it is now the
    exception rather than — as measured on every weekly sweep since the
    mechanism shipped — the norm (``alpha-engine-config-I10161``).

    ``observer_execution_arn`` is passed straight to :func:`build_cycle_shape`;
    see its docstring for why the execution a sweep runs inside must not make
    its own cycle ``IN_FLIGHT``.

    Raises :class:`~.read.SFNAccessDenied` / :class:`~.read.SFNThrottled`
    rather than returning an empty cycle: a transport or authorization
    failure rendered as "no executions" is a verdict manufactured from a
    denial, and this fleet has shipped that bug six times in a week.
    """
    if client is None:  # pragma: no cover — production path
        client = _sfn_client(state_machine_arn)

    pipeline = state_machine_arn.rsplit(":", 1)[-1]
    keys = set(cycle_keys(run_date, calendar_date))
    inspected = 0
    next_token: str | None = None
    collected: list[tuple[WorkOutcome, str | None, list[str]]] = []
    exhausted = False
    floor = _walk_floor(keys, lookback_days)
    before_floor = False

    while inspected < walk_cap:
        kwargs: dict[str, Any] = {
            "stateMachineArn": state_machine_arn,
            "maxResults": min(100, walk_cap - inspected),
        }
        if next_token:
            kwargs["nextToken"] = next_token
        try:
            page = client.list_executions(**kwargs)
        except Exception as exc:  # noqa: BLE001 — narrowed + re-raised
            _raise_for_boto_error(exc, "ListExecutions")

        rows = page.get("executions") or []
        if not rows:
            break

        for row in rows:
            inspected += 1
            arn = str(row.get("executionArn") or "")
            if not arn:
                continue
            if floor is not None:
                started = _parse_ts(row.get("startDate"))
                if started is not None and started < floor:
                    # Newest-first ordering: everything from here back is
                    # older still, so the cycle's contributor set is CLOSED.
                    # A correct termination condition, not a budget.
                    before_floor = True
                    break
            try:
                desc = client.describe_execution(executionArn=arn)
            except Exception as exc:  # noqa: BLE001
                _raise_for_boto_error(exc, "DescribeExecution")
            if cycle_key_for(desc) not in keys:
                continue

            start = _parse_ts(desc.get("startDate"))
            stop = _parse_ts(desc.get("stopDate"))
            events = _paged_execution_history(client, arn)
            states = entered_states_from_history(events)
            outcome = classify_work(
                state_machine_name=pipeline,
                status=RunStatus(str(desc.get("status"))),
                entered_states=states,
                duration_sec=(stop - start).total_seconds() if start and stop else None,
                execution_arn=arn,
                execution_name=str(desc.get("name") or "") or None,
                stage_spine=stage_spine,
            )
            collected.append((outcome, _extract_pipeline_role(desc), states))

        if before_floor:
            break

        next_token = page.get("nextToken")
        if not next_token:
            break
    else:
        exhausted = True

    collected.reverse()  # oldest-first for the record
    return build_cycle_shape(
        pipeline=pipeline,
        run_date=run_date,
        outcomes=collected,
        stage_spine=stage_spine,
        walk_exhausted=exhausted,
        observer_execution_arn=observer_execution_arn,
    )


def _walk_floor(keys: set[str], lookback_days: int) -> datetime | None:
    """The oldest ``startDate`` that could still belong to this cycle.

    ``None`` when no key parses as a date — the walk then falls back to the
    count cap alone rather than guessing a bound, because a wrong floor would
    silently DROP contributors, which is the one direction that manufactures
    absences.
    """
    if lookback_days < 0:
        return None
    parsed: list[date] = []
    for key in keys:
        try:
            parsed.append(date.fromisoformat(key))
        except (TypeError, ValueError):
            continue
    if not parsed:
        return None
    return datetime.combine(
        min(parsed) - timedelta(days=lookback_days), time.min, tzinfo=timezone.utc
    )


def _main(argv: Sequence[str] | None = None) -> int:
    """``python -m nousergon_lib.pipeline_status.cycle_shape``.

    Exit code IS the verdict, matching ``work.py``'s convention so a shell
    caller gates on the same numbers: 0 completed, 0 skipped, 2 incomplete,
    3 in flight.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="nousergon_lib.pipeline_status.cycle_shape")
    parser.add_argument("--state-machine-arn", required=True)
    parser.add_argument("--run-date", required=True)
    parser.add_argument(
        "--calendar-date",
        default=None,
        help=(
            "the execution's own wall-clock day — a second identity for the same "
            "cycle whenever the cycle is keyed on the trading day "
            "(alpha-engine-config-I8809)"
        ),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    shape = read_cycle_shape(
        args.state_machine_arn, args.run_date, calendar_date=args.calendar_date
    )
    print(json.dumps(shape.to_dict(), indent=2) if args.json else shape.explain())
    return {
        CycleVerdict.COMPLETED: 0,
        CycleVerdict.SKIPPED: 0,
        CycleVerdict.INCOMPLETE: 2,
        CycleVerdict.IN_FLIGHT: 3,
    }[shape.verdict]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
