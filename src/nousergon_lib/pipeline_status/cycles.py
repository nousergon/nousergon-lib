"""Cycle-level reliability projection — is the pipeline getting better?

`read.py` projects ONE execution. This module projects a *cycle*: a
scheduled run plus every operator rerun chasing the same run date, which is
the unit an operator actually experiences and the unit
`sf-pipeline-policy.md` §1 states its revisit trigger in ("any run requiring
>1 operator rerun to reach a complete terminal state").

**Why this exists** (alpha-engine-config-I6919). Asked on 2026-08-11 whether
the weekly-SF work was making progress or looping, nobody could answer from
a surface — it took a session of hand-reading `list-executions` over 26 days.
The reconstruction showed 16 attempts on 07-26 falling to first-attempt
success on four consecutive cycles (08-04 … 08-08), then a regression on
08-10 coincident with the per-stage spot cutover. That is a legible story and
none of it was visible; every cycle rendered as a red alert of equal weight.

## The three signals, and why red/green is none of them

1. **Attempts-to-success.** A cycle that succeeds on rerun 6 and one that
   succeeds first try both read "SUCCEEDED" in every existing surface. The
   difference between them is the entire question.
2. **Stage depth.** Within a bad stretch, a run dying LATER than the last one
   is progress — each fix reveals the next defect. Measured 2026-08-10/11:
   13m, 13m, 3m, 7m (all at MorningEnrich) then 3h34m (at PredictorTraining).
   Five red alerts; four fixes landed.
3. **Whether the causes are NEW.** This is the one that answers "are we
   looping". A cycle whose failures are all first-seen fingerprints is the
   onion being peeled. A cycle repeating a fingerprint from an earlier cycle
   is a regression or an incomplete fix, and it is the only one of the three
   that distinguishes progress from motion.

## The honesty rule this module is built around

An attempt whose failing state cannot be determined gets a fingerprint that
is **unique to that attempt** and is excluded from repeat detection. The
tempting alternative — one shared `UNKNOWN` bucket — would make two
unrelated undiagnosable failures look like the same cause recurring, which
manufactures exactly the "we are looping" verdict this module exists to
establish honestly. `principles.md` §2.7: *no data* is never rendered as a
finding. :attr:`CycleReliability.unresolved_attempts` reports how many, so
the absence is visible rather than inferred.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from .read import PipelineExecutionSummary, RunStatus

__all__ = [
    "AttemptOutcome",
    "CycleReliability",
    "ReliabilityWindow",
    "build_reliability_window",
    "fingerprint",
]


# Statuses that end an attempt without success. RUNNING is deliberately not
# here: an in-flight attempt is neither a failure nor a success, and folding
# it into either is how a cycle currently mid-recovery reads as settled.
_FAILED_STATUSES = frozenset({RunStatus.FAILED, RunStatus.TIMED_OUT, RunStatus.ABORTED})


def fingerprint(failing_state: str | None, error: str | None, *, attempt_id: str) -> str:
    """Stable identity for a failure cause, or a unique token when unknown.

    ``(failing_state, error)`` is the coarsest pair that separates the
    defects seen in practice: `MorningEnrich`/`States.TaskFailed` and
    `PredictorTraining`/`States.TaskFailed` are different work, and
    `Scanner`/`Sandbox.Timedout` and `Scanner`/`States.Runtime` are different
    bugs in the same stage.

    It is deliberately NOT the error *message*. Messages carry instance ids,
    timestamps and byte counts, so message-level fingerprints never repeat —
    which would report perfect progress forever, the failure mode inverse to
    the one this guards.

    When the failing state cannot be determined, the fingerprint folds in
    ``attempt_id`` and is therefore unique. Such an attempt can never match
    an earlier one, so it is counted as unresolved rather than silently
    joining a shared UNKNOWN bucket and inventing a recurrence.
    """
    if not failing_state:
        return f"unresolved:{attempt_id}"
    return f"{failing_state}:{error or 'unspecified'}"


@dataclass(frozen=True)
class AttemptOutcome:
    """One execution inside a cycle."""

    name: str
    execution_arn: str
    status: RunStatus
    start_utc: datetime
    duration_sec: float | None
    failing_state: str | None
    error: str | None
    #: Index of the deepest declared stage this attempt reached, or None when
    #: the stage order does not name any state it entered. Higher is further.
    depth_index: int | None
    depth_stage: str | None

    @property
    def succeeded(self) -> bool:
        return self.status == RunStatus.SUCCEEDED

    @property
    def failed(self) -> bool:
        return self.status in _FAILED_STATUSES

    @property
    def fingerprint(self) -> str | None:
        """None for a non-failed attempt — a success has no cause."""
        if not self.failed:
            return None
        return fingerprint(self.failing_state, self.error, attempt_id=self.execution_arn)

    @property
    def cause_is_unresolved(self) -> bool:
        return self.failed and not self.failing_state


@dataclass
class CycleReliability:
    """A scheduled run plus every rerun chasing the same run date."""

    cycle_key: str
    attempts: list[AttemptOutcome] = field(default_factory=list)
    #: Fingerprints first seen in THIS cycle, in first-seen order.
    new_causes: list[str] = field(default_factory=list)
    #: Fingerprints already seen in an EARLIER cycle. Non-empty means a
    #: previously-encountered cause came back — the loop signal.
    repeat_causes: list[str] = field(default_factory=list)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def first_attempt(self) -> AttemptOutcome | None:
        return self.attempts[0] if self.attempts else None

    @property
    def first_attempt_succeeded(self) -> bool | None:
        """None when the first attempt is still running — not False.

        The distinction matters on the surface: a cycle whose first attempt
        has not finished has no first-attempt verdict yet, and rendering it
        as a failure is the same error as rendering absence as green.
        """
        first = self.first_attempt
        if first is None or first.status == RunStatus.RUNNING:
            return None
        return first.succeeded

    @property
    def attempts_to_success(self) -> int | None:
        """1-based index of the first succeeding attempt, or None.

        None means the cycle never reached a success — including a cycle
        still in flight. Callers that want to distinguish those two read
        :attr:`settled`.
        """
        for i, attempt in enumerate(self.attempts, start=1):
            if attempt.succeeded:
                return i
        return None

    @property
    def settled(self) -> bool:
        """No attempt is still running."""
        return all(a.status != RunStatus.RUNNING for a in self.attempts)

    @property
    def recovered(self) -> bool:
        """Succeeded, but not on the first attempt — the operator-cost case."""
        n = self.attempts_to_success
        return n is not None and n > 1

    @property
    def depth_index(self) -> int | None:
        """Deepest declared stage reached by ANY attempt in the cycle."""
        seen = [a.depth_index for a in self.attempts if a.depth_index is not None]
        return max(seen) if seen else None

    @property
    def depth_stage(self) -> str | None:
        best: AttemptOutcome | None = None
        for a in self.attempts:
            if a.depth_index is None:
                continue
            if best is None or a.depth_index > (best.depth_index or -1):
                best = a
        return best.depth_stage if best else None

    @property
    def wall_clock_sec(self) -> float:
        """Summed attempt duration — the compute the cycle actually cost."""
        return sum(a.duration_sec or 0.0 for a in self.attempts)

    @property
    def unresolved_attempts(self) -> int:
        """Failed attempts whose cause could not be identified.

        Reported rather than hidden: a window with many of these has a weak
        loop verdict, and the reader must be able to see that.
        """
        return sum(1 for a in self.attempts if a.cause_is_unresolved)


@dataclass
class ReliabilityWindow:
    """Cycles over a trailing window, oldest first."""

    cycles: list[CycleReliability] = field(default_factory=list)

    @property
    def clean_streak(self) -> int:
        """Trailing settled cycles that succeeded on the first attempt.

        The headline. On 2026-08-08 this was 4 and nothing rendered it; on
        2026-08-10 it went to 0 and nothing rendered that either.
        """
        streak = 0
        for cycle in reversed(self.cycles):
            if not cycle.settled:
                continue
            if cycle.first_attempt_succeeded:
                streak += 1
            else:
                break
        return streak

    @property
    def repeat_cause_cycles(self) -> list[CycleReliability]:
        """Cycles that re-encountered a cause seen in an earlier cycle."""
        return [c for c in self.cycles if c.repeat_causes]

    @property
    def looping(self) -> bool | None:
        """Whether the most-recent settled cycle repeated an earlier cause.

        None when there is no settled cycle to judge — never False. "We have
        not seen a cycle finish" and "the last cycle introduced only new
        causes" are different states, and collapsing them answers Brian's
        question with a confidence the data does not support.
        """
        for cycle in reversed(self.cycles):
            if cycle.settled and cycle.attempts:
                return bool(cycle.repeat_causes)
        return None

    @property
    def depth_trend(self) -> list[tuple[str, int | None]]:
        """``(cycle_key, depth_index)`` oldest first — the getting-further signal."""
        return [(c.cycle_key, c.depth_index) for c in self.cycles]

    @property
    def unresolved_attempts(self) -> int:
        return sum(c.unresolved_attempts for c in self.cycles)

    def cause_frequency(self) -> Counter[str]:
        """How often each identified cause appeared, across the window.

        Unresolved fingerprints are excluded — they are unique by
        construction, so counting them would pad the table with N entries of
        1 and bury the causes that actually recur.
        """
        counts: Counter[str] = Counter()
        for cycle in self.cycles:
            for attempt in cycle.attempts:
                fp = attempt.fingerprint
                if fp and not attempt.cause_is_unresolved:
                    counts[fp] += 1
        return counts


def _depth_of(entered_states: Iterable[str], stage_order: Sequence[str]) -> tuple[int | None, str | None]:
    """Deepest ``stage_order`` position among the states this attempt entered.

    States absent from ``stage_order`` are ignored rather than ranked last:
    the order is a declared spine of the pipeline's substantive stages, and a
    poll or gate state entering does not mean the run got that far.
    """
    best_index: int | None = None
    best_name: str | None = None
    positions = {name: i for i, name in enumerate(stage_order)}
    for state in entered_states:
        i = positions.get(state)
        if i is None:
            continue
        if best_index is None or i > best_index:
            best_index, best_name = i, state
    return best_index, best_name


def build_reliability_window(
    summaries: Sequence[PipelineExecutionSummary],
    *,
    cycle_key_of: Callable[[PipelineExecutionSummary], str | None],
    failure_of: Callable[[PipelineExecutionSummary], tuple[str | None, str | None]],
    entered_states_of: Callable[[PipelineExecutionSummary], Iterable[str]],
    stage_order: Sequence[str],
    max_cycles: int | None = None,
) -> ReliabilityWindow:
    """Group executions into cycles and label each cause new or repeat.

    The three callables are injected rather than called here so this module
    stays free of boto3 and testable on fixtures — the caller decides whether
    a cycle key comes from the execution input's ``run_date``, from the
    execution name, or from the completion marker, and pays for the
    ``GetExecutionHistory`` calls the other two need.

    ``summaries`` may arrive in any order; cycles are built in chronological
    order of their first attempt, because "seen in an EARLIER cycle" is only
    meaningful against a stable ordering.
    """
    ordered = sorted(summaries, key=lambda s: s.start_utc)

    grouped: dict[str, list[PipelineExecutionSummary]] = {}
    for summary in ordered:
        key = cycle_key_of(summary)
        if key is None:
            # An execution we cannot attribute to a cycle is DROPPED, not
            # given its own — a synthetic single-attempt cycle would dilute
            # attempts-to-success and the clean streak with runs that were
            # never part of a cadence.
            continue
        grouped.setdefault(key, []).append(summary)

    seen_causes: set[str] = set()
    cycles: list[CycleReliability] = []

    for key, group in sorted(grouped.items(), key=lambda kv: kv[1][0].start_utc):
        cycle = CycleReliability(cycle_key=key)
        for summary in group:
            failing_state, error = failure_of(summary) if summary.status in _FAILED_STATUSES else (None, None)
            depth_index, depth_stage = _depth_of(entered_states_of(summary), stage_order)
            cycle.attempts.append(
                AttemptOutcome(
                    name=summary.name,
                    execution_arn=summary.execution_arn,
                    status=summary.status,
                    start_utc=summary.start_utc,
                    duration_sec=summary.duration_sec,
                    failing_state=failing_state,
                    error=error,
                    depth_index=depth_index,
                    depth_stage=depth_stage,
                )
            )

        # Cause labelling is per CYCLE, not per attempt: the same cause
        # hitting rerun 1 and rerun 2 of one cycle is one unfixed defect
        # being retried, not a recurrence. Only a cause crossing a cycle
        # boundary is the loop signal.
        this_cycle: list[str] = []
        for attempt in cycle.attempts:
            fp = attempt.fingerprint
            if fp is None or attempt.cause_is_unresolved:
                continue
            if fp in this_cycle:
                continue
            this_cycle.append(fp)
            if fp in seen_causes:
                cycle.repeat_causes.append(fp)
            else:
                cycle.new_causes.append(fp)
        seen_causes.update(this_cycle)
        cycles.append(cycle)

    if max_cycles is not None:
        # Trim from the OLD end after labelling, so a cause first seen
        # outside the window still reads as a repeat inside it. Trimming
        # first would relabel old recurrences as new and report progress
        # that did not happen.
        cycles = cycles[-max_cycles:]

    return ReliabilityWindow(cycles=cycles)
