"""Did the pipeline actually do the work? — the three-way verdict.

``read.py`` answers *what the execution's status was*. That is not the same
question as *did the work happen*, and on ``ne-weekly-freshness-pipeline``
the two answers routinely disagree.

## The defect this module exists to close (alpha-engine-config-I8045)

The weekly SF fires THU-SAT and self-selects the single correct run day
behind ``WeeklyRunDayGate``. On the other two days the execution enters
``InitializeInput`` -> ``CheckWeeklyRunDayGate`` -> ``WeeklyRunDayGate`` ->
``WeeklyRunDayGateChoice`` -> ``WeeklyRunDaySkip`` and reports **SUCCEEDED**
in a few seconds, having deliberately done nothing. That behaviour is
CORRECT — the state's own comment calls it a designed no-op.

What is not correct is every reader that saw ``SUCCEEDED`` and concluded the
pipeline was healthy. Measured 2026-08-21 against the live state machine:
across the trailing 30 executions every SUCCEEDED scheduled run lasted 2.7
to 5.7 seconds and entered exactly those five states. The one longer success,
``watch-rerun-2026-08-16-4``, ran 8m40s, wrote a completion marker, and
entered **zero** of the sixteen declared substantive stages. The genuine
scheduled run on 2026-08-15 FAILED at ``PredictorTraining``-depth after 3h53m.
Not one full weekly graph completed in that window, and two independent
agent sessions read ``list-executions`` and reported the pipeline as
succeeding.

## Three outcomes, never two

A boolean cannot carry this. ``SUCCEEDED``/``FAILED`` collapses a correct
skip into the same bucket as a real run, and any fix that instead reported
the skip as a failure would page a human through working-as-designed
behaviour. So :class:`WorkVerdict` has three terminal values:

- :attr:`WorkVerdict.COMPLETED` — succeeded and entered every stage on the
  pipeline's declared substantive spine. This is the only value that counts
  as a clean cycle.
- :attr:`WorkVerdict.SKIPPED` — succeeded by reaching a **declared** skip
  terminal (``WeeklyRunDaySkip``, ``ShellRunComplete``, ``NotifyHolidaySkip``).
  Correct behaviour. Never alerts, and never counts as a cycle either — a
  surface that folds skips into a success rate reports a rate over a
  denominator that did no work.
- :attr:`WorkVerdict.INCOMPLETE` — the work did not happen. Three reasons,
  named rather than merged: ``execution_failed``, ``vacuous_success``
  (succeeded, not via a declared skip terminal, entered no substantive
  stage) and ``partial_success`` (succeeded, entered some but not all).

plus :attr:`WorkVerdict.IN_FLIGHT` for a run that has not reached a terminal
state. That is not a fourth *work* outcome; it is the refusal to invent one,
matching :attr:`cycles.CycleReliability.first_attempt_succeeded` returning
``None`` rather than ``False`` for a running attempt.

## Why a declared list and not a duration threshold

The gate-out is legible by its duration — 3 seconds against a 4-hour graph —
and a threshold would have caught these four executions. It is still the
wrong instrument. A threshold has to be re-tuned every time the graph grows
or a stage moves to spot, it cannot tell a fast skip from a fast failure, and
it silently reclassifies when the pipeline gets faster. ``DISABLED`` /
``SKIPPED`` are **declared, never inferred** (``observability-policy.md``
§8.3), so the verdict rests on two declarations in :mod:`.registry` —
:data:`~.registry.SKIP_TERMINALS` and :data:`~.registry.PIPELINE_STAGE_ORDER`
— and duration is reported alongside for the reader, deciding nothing.

A state machine with no declared spine cannot be judged: :func:`classify_work`
raises rather than returning a green it has no basis for. *No data* is never
rendered as a pass (``principles.md`` §2.7).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from .read import (
    RunStatus,
    _paged_execution_history,
    _parse_ts,
    _raise_for_boto_error,
    _sfn_client,
)
from .registry import skip_terminals_for, stage_order_for

if TYPE_CHECKING:  # pragma: no cover
    from mypy_boto3_stepfunctions.client import SFNClient
else:  # pragma: no cover
    SFNClient = Any

__all__ = [
    "UndeclaredPipeline",
    "WorkOutcome",
    "WorkVerdict",
    "classify_work",
    "entered_states_from_history",
    "read_work_outcome",
]

#: Statuses that end an execution without success.
_FAILED_STATUSES = frozenset({RunStatus.FAILED, RunStatus.TIMED_OUT, RunStatus.ABORTED})


class WorkVerdict(str, Enum):
    """Did the pipeline do the work? Closed, exhaustive, no fall-through."""

    COMPLETED = "completed"
    SKIPPED = "skipped"
    INCOMPLETE = "incomplete"
    IN_FLIGHT = "in_flight"


class UndeclaredPipeline(ValueError):
    """No substantive-stage spine is declared for this state machine.

    Raised rather than defaulted. A pipeline with no declared spine has no
    basis on which any execution of it can be called complete, and returning
    :attr:`WorkVerdict.COMPLETED` for it would be exactly the silent green
    this module exists to remove.
    """


@dataclass(frozen=True)
class WorkOutcome:
    """The three-way verdict plus every fact it was derived from."""

    verdict: WorkVerdict
    #: Closed vocabulary. One of: ``full_run`` · ``declared_skip`` ·
    #: ``execution_failed`` · ``vacuous_success`` · ``partial_success`` ·
    #: ``still_running``.
    reason: str
    state_machine_name: str
    status: RunStatus
    terminal_state: str | None = None
    duration_sec: float | None = None
    execution_arn: str | None = None
    execution_name: str | None = None
    #: Declared substantive stages this execution entered, spine order.
    stages_entered: tuple[str, ...] = ()
    #: Declared substantive stages it did not enter, spine order.
    stages_missing: tuple[str, ...] = ()
    stage_spine: tuple[str, ...] = field(default=(), repr=False)

    @property
    def did_work(self) -> bool:
        """True only for a run that entered the whole declared spine."""
        return self.verdict is WorkVerdict.COMPLETED

    @property
    def counts_as_cycle(self) -> bool:
        """Whether this execution belongs in a success-rate denominator.

        A declared skip does not: it is the pipeline correctly declining to
        run, and counting it inflates every rate computed over it. Neither
        does an in-flight run, which has no outcome yet.
        """
        return self.verdict in (WorkVerdict.COMPLETED, WorkVerdict.INCOMPLETE)

    @property
    def should_alert(self) -> bool:
        """Only INCOMPLETE pages. A skip is correct behaviour, not an event."""
        return self.verdict is WorkVerdict.INCOMPLETE

    @property
    def stage_coverage(self) -> str:
        return f"{len(self.stages_entered)}/{len(self.stage_spine)}"

    def explain(self) -> str:
        """One line naming the verdict, the reason and the evidence.

        Every surface renders this verbatim so an operator reading an alert,
        a console cell and a CI log sees the same sentence.
        """
        dur = "unknown duration" if self.duration_sec is None else f"{self.duration_sec:.1f}s"
        head = f"{self.state_machine_name}: {self.verdict.value} ({self.reason})"
        if self.verdict is WorkVerdict.SKIPPED:
            return f"{head} — reached declared skip terminal {self.terminal_state!r} after {dur}; no work was due"
        if self.verdict is WorkVerdict.IN_FLIGHT:
            return f"{head} — no terminal state yet"
        return (
            f"{head} — status {self.status.value}, {dur}, "
            f"terminal state {self.terminal_state!r}, "
            f"stages {self.stage_coverage}"
            + (f", missing {', '.join(self.stages_missing)}" if self.stages_missing else "")
        )

    def to_dict(self) -> dict[str, Any]:
        """Primitive projection — for JSON emission and cached surfaces."""
        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "state_machine_name": self.state_machine_name,
            "status": self.status.value,
            "terminal_state": self.terminal_state,
            "duration_sec": self.duration_sec,
            "execution_arn": self.execution_arn,
            "execution_name": self.execution_name,
            "stages_entered": list(self.stages_entered),
            "stages_missing": list(self.stages_missing),
            "stage_coverage": self.stage_coverage,
            "did_work": self.did_work,
            "counts_as_cycle": self.counts_as_cycle,
            "should_alert": self.should_alert,
            "explain": self.explain(),
        }


def entered_states_from_history(history_events: Sequence[Mapping[str, Any]]) -> list[str]:
    """Ordered, de-duplicated state names this execution entered.

    Consecutive duplicates are collapsed (a poll loop re-enters its Wait
    companion hundreds of times); a state re-entered after a different one
    is kept, because the ORDER is what makes the last element the terminal.
    """
    names: list[str] = []
    for event in history_events:
        details = event.get("stateEnteredEventDetails")
        if not details:
            continue
        name = details.get("name")
        if not name:
            continue
        if names and names[-1] == name:
            continue
        names.append(str(name))
    return names


def classify_work(
    *,
    state_machine_name: str,
    status: RunStatus,
    entered_states: Sequence[str],
    duration_sec: float | None = None,
    execution_arn: str | None = None,
    execution_name: str | None = None,
    stage_spine: Sequence[str] | None = None,
    skip_terminals: frozenset[str] | None = None,
) -> WorkOutcome:
    """Project one execution onto the three-way work verdict.

    Pure — no boto3, no clock. ``entered_states`` must be in entry order and
    COMPLETE: a truncated history makes the last element the wrong terminal
    and turns a real run into a ``vacuous_success``. :func:`read_work_outcome`
    pages ``GetExecutionHistory`` to the end for exactly this reason.
    """
    spine = tuple(stage_spine) if stage_spine is not None else stage_order_for(state_machine_name)
    if not spine:
        raise UndeclaredPipeline(
            f"no substantive-stage spine declared for {state_machine_name!r}; "
            "add one to nousergon_lib.pipeline_status.registry.PIPELINE_STAGE_ORDER "
            "before any surface reports on it"
        )
    skips = skip_terminals if skip_terminals is not None else skip_terminals_for(state_machine_name)

    terminal = entered_states[-1] if entered_states else None
    entered = set(entered_states)
    hit = tuple(s for s in spine if s in entered)
    missing = tuple(s for s in spine if s not in entered)

    common = {
        "state_machine_name": state_machine_name,
        "status": status,
        "terminal_state": terminal,
        "duration_sec": duration_sec,
        "execution_arn": execution_arn,
        "execution_name": execution_name,
        "stages_entered": hit,
        "stages_missing": missing,
        "stage_spine": spine,
    }

    if status is RunStatus.RUNNING:
        return WorkOutcome(verdict=WorkVerdict.IN_FLIGHT, reason="still_running", **common)

    if status in _FAILED_STATUSES:
        return WorkOutcome(verdict=WorkVerdict.INCOMPLETE, reason="execution_failed", **common)

    # SUCCEEDED (or NOT_RUN, which has no states and no spine coverage).
    if terminal is not None and terminal in skips:
        return WorkOutcome(verdict=WorkVerdict.SKIPPED, reason="declared_skip", **common)
    if not hit:
        # Succeeded without touching one declared stage and without a
        # declared skip terminal. This is the shape of watch-rerun-2026-08-16-4,
        # which even wrote a completion marker — so an artifact-freshness
        # detector reading that marker sees a clean cycle too.
        return WorkOutcome(verdict=WorkVerdict.INCOMPLETE, reason="vacuous_success", **common)
    if missing:
        return WorkOutcome(verdict=WorkVerdict.INCOMPLETE, reason="partial_success", **common)
    return WorkOutcome(verdict=WorkVerdict.COMPLETED, reason="full_run", **common)


def read_work_outcome(
    execution_arn: str,
    *,
    client: SFNClient | None = None,
) -> WorkOutcome:
    """Read one execution and return its three-way work verdict.

    Two API calls plus history pages: ``DescribeExecution`` for status and
    timings, ``GetExecutionHistory`` (paged) for the entered-state sequence.
    """
    sfn = client or _sfn_client(execution_arn)
    try:
        desc = sfn.describe_execution(executionArn=execution_arn)
    except Exception as exc:  # noqa: BLE001
        _raise_for_boto_error(exc, "DescribeExecution")

    sm_arn = str(desc.get("stateMachineArn") or "")
    sm_name = sm_arn.rsplit(":", 1)[-1] if sm_arn else execution_arn.split(":")[6]
    start = _parse_ts(desc.get("startDate"))
    stop = _parse_ts(desc.get("stopDate"))
    duration = (stop - start).total_seconds() if start and stop else None

    events = _paged_execution_history(sfn, execution_arn)
    return classify_work(
        state_machine_name=sm_name,
        status=RunStatus(str(desc.get("status"))),
        entered_states=entered_states_from_history(events),
        duration_sec=duration,
        execution_arn=execution_arn,
        execution_name=str(desc.get("name") or "") or None,
    )


def _main(argv: Sequence[str] | None = None) -> int:
    """``python -m nousergon_lib.pipeline_status.work`` — the shell surface.

    Shell and workflow consumers get the SAME predicate rather than a
    re-implemented ``grep SUCCEEDED``. Exit code IS the verdict, so a caller
    can gate on it without parsing: 0 completed, 0 skipped (correct
    behaviour never fails a gate), 2 incomplete, 3 in flight.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="nousergon_lib.pipeline_status.work")
    parser.add_argument("--execution-arn", required=True)
    parser.add_argument("--json", action="store_true", help="emit the full outcome as JSON")
    args = parser.parse_args(argv)

    outcome = read_work_outcome(args.execution_arn)
    print(json.dumps(outcome.to_dict(), indent=2) if args.json else outcome.explain())
    return {
        WorkVerdict.COMPLETED: 0,
        WorkVerdict.SKIPPED: 0,
        WorkVerdict.INCOMPLETE: 2,
        WorkVerdict.IN_FLIGHT: 3,
    }[outcome.verdict]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
