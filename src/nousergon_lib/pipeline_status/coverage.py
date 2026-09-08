"""The reader of the stage-coverage surface — expected set versus what landed.

``krepis.stage_coverage`` makes every stage record its own verdict. Nothing
read the resulting surface. This is that reader.

## The defect this module exists to close (``alpha-engine-config-I8154``)

``krepis/src/krepis/stage_coverage.py`` documents its own metric as *"the
absence of the datapoint is itself visible as a gap — ``no data`` is never
rendered as green"*. Measured 2026-08-22, there was **no alarm on it**::

    $ aws cloudwatch describe-alarms \\
        --query 'MetricAlarms[?MetricName==`StageCoverage`].AlarmName'
    (empty)

and no consumer anywhere compared the expected stage set against the verdicts
that actually landed. All three things reading the mechanism read a *single
stage's own* verdict, in observe mode:

- ``crucible-dashboard/health_checker.py::_assert_stage_coverage`` — per-stage,
  self-reported, explicitly "can never fail the stage".
- ``nousergon-data/infrastructure/spot_*.sh`` — each box stage asserting itself.
- ``alpha-engine-config/scripts/check_stage_coverage_drift.py`` — compares the
  registry's ``pipeline_stages`` against the SF *definition*. Both sides are
  static files; it never reads S3.

So nothing asked *"did every stage of run X report, and what did they say?"*
— the only question that catches either of the two classes that blindness hid:

1. **Four stages had never recorded a verdict at all** (``Director``,
   ``ReportCard``, ``EvaluatorDeployDriftCheck``,
   ``EvaluatorDirectorDeployDriftCheck``) — denied by IAM since the prefix was
   created, for eight days (``alpha-engine-config-I8152``). Their CloudWatch
   datapoints published normally, so the metric surface and the S3 surface
   disagreed and nothing compared them.
2. **Three stages reported ``MISSING``/``is_finding: true`` and nothing paged**
   — ``Backtester`` (8 of 14 declared artifacts absent), ``Scanner`` (1),
   ``RegimeSubstrate`` (1, on two consecutive runs).

## Why this reads S3 and not the metric

**The metric cannot distinguish "denied" from "never entered."** A stage whose
role lacks ``s3:PutObject`` publishes its CloudWatch datapoint normally and
writes no verdict object; a stage the graph never entered publishes nothing
and writes nothing. On the metric surface the first looks healthy and the
second looks like a gap; on the S3 surface both are the same absence — and it
is the *expected set* that tells them apart. So the sweep reads the objects.

## The three counts, and why ABSENT is the serious one

:class:`CoverageSweep` reports ``covered`` / ``findings`` / ``absent``.

``absent`` — expected, no verdict object — is the most serious of the three
and pages **at all**, with no threshold. It is the one state that looks
identical to a stage that never ran, and it is the state the recording
mechanism itself produces when it is broken. A findings count has a
distribution; an absent count must be zero by construction.

``findings`` pages above a declared threshold (:data:`DEFAULT_FINDING_THRESHOLD`,
zero — any finding pages). The threshold exists so it can be *raised* with a
written reason, never so the default is permissive.

## The denominator, and the one thing that must not be guessed

Expected = the artifact registry's ``pipeline_stages`` INTERSECTED with the
states the cycle's contributing executions actually entered
(:func:`~.cycle_shape.read_cycle_shape`). Both halves are load-bearing:

- **Registry alone** over-counts. The weekly graph has ~49 declared stages and
  a given run enters a subset — conditional parity stages, the first-Saturday
  judge submit, the degraded-path twins. Reporting 38 absent on a healthy run
  is the noise that trains a reader to ignore the surface.
- **Entered set alone** under-counts, and worse, it cannot see the
  ``I8152`` class at all: a stage that entered, ran, and was denied its write
  is exactly a stage in the entered set with no verdict.

When the entered set cannot be read, the denominator falls back to the
declared set and :attr:`CoverageSweep.denominator_source` says so. That is
**not** a silent degrade: a sweep running on the declared-only denominator is
itself reported and pages, because a coverage reader that cannot establish its
own denominator is unobserved, not healthy.

Public surface:

- :class:`RowState` / :class:`StageRow` / :class:`CoverageSweep`
- :func:`sweep_coverage` — the pure core.
- :func:`read_coverage_sweep` — the boto3 front door.
- :func:`publish_sweep` — metrics + the S3 console artifact.
- :func:`render_rows` — the console/CLI table.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from .cycle_shape import CycleShape, read_cycle_shape
from .partition import partition_dates

if TYPE_CHECKING:  # pragma: no cover
    from mypy_boto3_stepfunctions.client import SFNClient
else:  # pragma: no cover
    SFNClient = Any

logger = logging.getLogger(__name__)

__all__ = [
    "COVERED_STATUSES",
    "DEFAULT_FINDING_THRESHOLD",
    "METRIC_NAMESPACE",
    "SWEEP_ARTIFACT_PREFIX",
    "VERDICT_PREFIX",
    "CoverageSweep",
    "RowState",
    "StageRow",
    "publish_sweep",
    "read_coverage_sweep",
    "render_rows",
    "sweep_coverage",
]

#: Prefix ``krepis.stage_coverage`` writes per-stage verdicts under.
VERDICT_PREFIX = "_stage_coverage"

#: Prefix this sweep's own per-run artifact lands under — the console's row
#: source and the durable record of what the sweep saw.
SWEEP_ARTIFACT_PREFIX = "_stage_coverage/_sweep"

#: CloudWatch namespace. Same as the per-stage metric so one dashboard covers
#: the mechanism and its reader.
METRIC_NAMESPACE = "AlphaEngine"

#: Verdict statuses that are a pass. ``COVERED_NO_OUTPUT`` is a pass by
#: POSITIVE declaration — the stage said it writes nothing — which is the
#: distinction the registry's stage section exists to draw.
COVERED_STATUSES = frozenset({"COVERED", "COVERED_NO_OUTPUT"})

#: Findings above this count page. Zero: any finding pages. Raising it is a
#: declaration with a written reason, never a default.
DEFAULT_FINDING_THRESHOLD = 0


class RowState(str, Enum):
    """What the sweep concluded about one expected stage. Closed, no default."""

    #: A verdict landed and it was a pass.
    COVERED = "covered"
    #: A verdict landed and it declared itself a finding.
    FINDING = "finding"
    #: The stage was expected and NO verdict object exists. The serious one.
    ABSENT = "absent"
    #: A verdict landed but establishes no claim — the module's own
    #: ``UNMEASURED``, or a status that is neither a pass nor a declared
    #: finding (a ``STALE`` still carrying ``is_finding: false``). Never a
    #: pass: not-covered-and-not-declared-a-finding is an absence of evidence.
    UNMEASURED = "unmeasured"
    #: Declared in the registry, not entered by any contributing execution.
    #: Outside the denominator, reported so the reader can see the shape of
    #: the run rather than a silently smaller world.
    NOT_ENTERED = "not_entered"
    #: Declared, not entered, and the CALLER declared this cycle deliberately
    #: does not run it (a cadence declaration, a recorded operator flag).
    #: Outside the denominator and outside every not-entered count — a
    #: declared skip and a stage nobody ran must never produce the same
    #: number, or the exclusion mechanism becomes a way to quiet a real
    #: absence (``alpha-engine-config-I10199``, ``-I10175``).
    DECLARED_SKIP = "declared_skip"
    #: The sweep's OWN stage. It is executing at the moment it grades, so it
    #: has not completed and cannot have written a verdict about its own
    #: completion. Outside the denominator, reported — never ``ABSENT``, which
    #: it would otherwise be on every single run by construction
    #: (``alpha-engine-config-I10161``).
    SELF = "self"


@dataclass(frozen=True)
class StageRow:
    """One declared stage, as the sweep found it."""

    stage: str
    state: RowState
    stage_class: str = ""
    declared_output: str = ""
    verdict_status: str | None = None
    reason: str = ""
    covered_artifacts: tuple[str, ...] = ()
    missing_artifacts: tuple[str, ...] = ()
    stale_artifacts: tuple[str, ...] = ()
    unmeasured_artifacts: tuple[str, ...] = ()
    verdict_key: str | None = None
    recorded_at: str = ""
    #: The date partition the verdict was actually found under. Equal to the
    #: sweep's ``run_date`` for a converged stage; the calendar partition for
    #: one still writing to the legacy family (alpha-engine-config-I8809).
    partition_date: str = ""

    @property
    def is_finding(self) -> bool:
        return self.state is RowState.FINDING

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "state": self.state.value,
            "stage_class": self.stage_class,
            "declared_output": self.declared_output,
            "verdict_status": self.verdict_status,
            "reason": self.reason,
            "covered_artifacts": list(self.covered_artifacts),
            "missing_artifacts": list(self.missing_artifacts),
            "stale_artifacts": list(self.stale_artifacts),
            "unmeasured_artifacts": list(self.unmeasured_artifacts),
            "verdict_key": self.verdict_key,
            "recorded_at": self.recorded_at,
            "partition_date": self.partition_date,
        }


@dataclass(frozen=True)
class CoverageSweep:
    """The per-run answer to "did every stage report, and what did it say?"."""

    pipeline: str
    run_date: str
    rows: tuple[StageRow, ...]
    #: ``entered_states`` (registry ∩ what the cycle entered) or
    #: ``declared_only`` (the entered set could not be read).
    denominator_source: str
    denominator_reason: str = ""
    #: Every date partition this sweep unioned, canonical first. One entry once
    #: the 2026-09-05 cutover lands; two during the migration window
    #: (alpha-engine-config-I8809). Reported so a reader can never mistake a
    #: union for a single-partition result.
    partitions_read: tuple[str, ...] = ()
    cycle: CycleShape | None = field(default=None, repr=False)
    finding_threshold: int = DEFAULT_FINDING_THRESHOLD
    swept_at: str = ""
    #: Stages the CALLER declared this cycle deliberately does not run. A
    #: declaration, never an inference: this module has no reader for
    #: ``run_scope.json`` and must not grow one (``nousergon-lib-PR392``).
    declared_skips: tuple[str, ...] = ()
    #: Declared skips the graph nevertheless ENTERED. The declaration and the
    #: run disagree; the sweep says so rather than reconciling them.
    declared_skips_entered: tuple[str, ...] = ()
    #: Declared skips naming no stage the registry declares — a typo or a
    #: rename. Without this, ``declared_skips`` is a mechanism for making a
    #: real absence invisible by misspelling it.
    declared_skips_unknown: tuple[str, ...] = ()

    #: States that are reported but sit OUTSIDE the coverage denominator.
    _OUT_OF_DENOMINATOR = (RowState.NOT_ENTERED, RowState.SELF, RowState.DECLARED_SKIP)

    def _in_denominator(self) -> tuple[StageRow, ...]:
        return tuple(r for r in self.rows if r.state not in self._OUT_OF_DENOMINATOR)

    @property
    def expected(self) -> tuple[str, ...]:
        return tuple(r.stage for r in self._in_denominator())

    @property
    def covered(self) -> int:
        return sum(1 for r in self.rows if r.state is RowState.COVERED)

    @property
    def findings(self) -> int:
        return sum(1 for r in self.rows if r.state is RowState.FINDING)

    @property
    def absent(self) -> int:
        return sum(1 for r in self.rows if r.state is RowState.ABSENT)

    @property
    def unmeasured(self) -> int:
        return sum(1 for r in self.rows if r.state is RowState.UNMEASURED)

    @property
    def not_entered(self) -> int:
        return sum(1 for r in self.rows if r.state is RowState.NOT_ENTERED)

    @property
    def declared_skip(self) -> int:
        return sum(1 for r in self.rows if r.state is RowState.DECLARED_SKIP)

    @property
    def self_stage(self) -> int:
        return sum(1 for r in self.rows if r.state is RowState.SELF)

    @property
    def not_entered_stages(self) -> tuple[str, ...]:
        return tuple(r.stage for r in self.rows if r.state is RowState.NOT_ENTERED)

    @property
    def spine(self) -> tuple[str, ...]:
        """The cycle's EFFECTIVE substantive spine, or ``()``.

        Taken from the cycle rather than from the registry, so a caller that
        excluded deliberately-skipped stages (``nousergon-lib-PR392``) is
        honoured here too. ``()`` when no cycle could be read — a sweep in
        that state already pages under ``denominator_unestablished`` and must
        not also assert a spine claim it has no spine for.
        """
        return self.cycle.stage_spine if self.cycle else ()

    @property
    def spine_not_entered_stages(self) -> tuple[str, ...]:
        """Not-entered stages that are DECLARED SUBSTANTIVE for this pipeline.

        The discriminator that makes ``not_entered`` alertable at all.
        ``not_entered`` in the raw is 9-12 rows on a perfectly healthy weekly
        run — the monthly judge-submit fork, the parity branch, every
        degraded-path twin — so paging on the raw count would page on every
        healthy week, which is the chronic-false-positive class this fleet
        already carries an incident register for. The spine is the standing
        declaration of which stages' entry is what "the pipeline ran" MEANS,
        so a spine stage nobody entered is work that was due and was not done.
        """
        spine = set(self.spine)
        return tuple(s for s in self.not_entered_stages if s in spine)

    @property
    def spine_not_entered(self) -> int:
        return len(self.spine_not_entered_stages)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "expected": len(self.expected),
            "covered": self.covered,
            "findings": self.findings,
            "absent": self.absent,
            "unmeasured": self.unmeasured,
            "not_entered": self.not_entered,
            "declared_skip": self.declared_skip,
            "self": self.self_stage,
        }

    @property
    def legacy_partition_rows(self) -> int:
        """Verdicts found OUTSIDE the canonical partition.

        Never an alert condition on its own — during the migration window it
        is the expected shape, and paging on it would page on the very thing
        the window exists to tolerate. It is reported because at the cutover
        it becomes the number that must be zero.
        """
        return sum(
            1
            for r in self.rows
            if r.partition_date and r.partition_date != self.run_date
        )

    @property
    def deferral_reason(self) -> str:
        """Why an ABSENCE cannot yet be asserted about this cycle, or ``""``.

        ``alpha-engine-config-I10161``. ``ABSENT`` means *"expected, entered,
        and recorded nothing"* — a claim only a cycle that has stopped
        changing can support. Two conditions make the claim unsupportable,
        and before this property existed the sweep paged on both anyway:

        1. **the cycle is still IN_FLIGHT** — a stage that has not finished is
           indistinguishable from one that finished and wrote nothing;
        2. **the contributor walk was truncated on a non-COMPLETED cycle** —
           :attr:`CycleShape.verdict_trustworthy`, the downgrade
           ``cycle_shape`` documents and no caller applied.

        Deferral is NOT silence and never renders green: the sweep still
        pages (see :attr:`alert_conditions`), the artifact still names every
        would-be-absent stage in ``rows``, and ``publish_sweep`` emits
        ``StageCoverageSweepDeferred=1`` while WITHHOLDING the absent count —
        an absent datapoint the alarm would page on is a number the sweep
        cannot stand behind, and publishing one anyway is what produced the
        2026-09-04 false page.
        """
        if self.cycle is None:
            return ""
        if not self.cycle.is_terminal:
            return (
                f"the cycle's own verdict is {self.cycle.verdict.value} "
                f"({self.cycle.reason}) — a stage that has not finished cannot be "
                "distinguished from one that finished and recorded nothing"
            )
        if not self.cycle.verdict_trustworthy:
            return self.cycle.untrustworthy_reason
        return ""

    @property
    def coverage_established(self) -> bool:
        """Can this sweep stand behind its absent count? Inverse of deferral.

        ``True`` when there is no cycle to check against: a sweep run without
        a state machine ARN already pages under ``denominator_unestablished``,
        and adding a second reason for the same fact would double-report it.
        """
        return not self.deferral_reason

    @property
    def deferred(self) -> bool:
        return not self.coverage_established

    @property
    def absent_stages(self) -> tuple[str, ...]:
        return tuple(r.stage for r in self.rows if r.state is RowState.ABSENT)

    @property
    def alert_conditions(self) -> tuple[str, ...]:
        """Every reason this sweep pages, named. Empty ⇒ it does not page."""
        conditions: list[str] = []
        if self.denominator_source != "entered_states":
            conditions.append(
                f"denominator_unestablished: {self.denominator_reason or 'entered set unreadable'}"
            )
        if self.deferred:
            # Loud, and about the RIGHT thing. The would-be-absent stages are
            # named so the page is still diagnosable, but they are named as
            # UNESTABLISHED rather than asserted as absences the sweep cannot
            # support.
            names = ", ".join(self.absent_stages) or "none"
            conditions.append(
                f"coverage_deferred: {self.deferral_reason}; the absent count is "
                f"WITHHELD, not zero — {len(self.absent_stages)} stage(s) had no verdict "
                f"at sweep time ({names}). Re-sweep this cycle once it is terminal: "
                "invoke alpha-engine-weekly-coverage-sweep with this run_date and "
                "state_machine_arn."
            )
        elif self.absent:
            names = ", ".join(self.absent_stages)
            conditions.append(
                f"absent_verdicts={self.absent} ({names}) — expected, entered, and no "
                "verdict object; indistinguishable from a stage that never ran"
            )
        if not self.deferred and self.spine_not_entered_stages:
            # alpha-engine-config-I10199. Declared substantive, entered by no
            # contributing execution, and covered by no declared skip: the
            # cycle did not do that work, and before this nothing said so —
            # NOT_ENTERED was outside the denominator, published no metric and
            # listed no condition. Deferred cycles are excluded because a
            # stage that has not been entered YET is not a stage that was
            # never entered; the deferral already pages, naming the cycle.
            names = ", ".join(self.spine_not_entered_stages)
            conditions.append(
                f"spine_not_entered={self.spine_not_entered} ({names}) — declared "
                "substantive stage(s) no contributing execution entered and no declared "
                "skip covers; the cycle did not do that work"
            )
        if self.declared_skips_unknown:
            names = ", ".join(self.declared_skips_unknown)
            conditions.append(
                f"declared_skip_unknown: {names} — the caller declared a skip for "
                "stage(s) the registry does not declare; a misspelled skip silently "
                "suppresses nothing, it pages"
            )
        if self.declared_skips_entered:
            names = ", ".join(self.declared_skips_entered)
            conditions.append(
                f"declared_skip_entered: {names} — declared a deliberate skip and "
                "ENTERED anyway; the caller's declaration and the graph disagree"
            )
        if self.findings > self.finding_threshold:
            names = ", ".join(r.stage for r in self.rows if r.state is RowState.FINDING)
            conditions.append(
                f"findings={self.findings} > threshold {self.finding_threshold} ({names})"
            )
        return tuple(conditions)

    @property
    def should_alert(self) -> bool:
        return bool(self.alert_conditions)

    def explain(self) -> str:
        c = self.counts
        head = (
            f"{self.pipeline} coverage {self.run_date}: "
            f"{c['covered']} covered / {c['findings']} findings / {c['absent']} absent "
            f"of {c['expected']} expected"
        )
        extra = f" ({c['unmeasured']} unmeasured, {c['not_entered']} not entered"
        if c["declared_skip"]:
            extra += f", {c['declared_skip']} declared skip"
        extra += ")"
        if len(self.partitions_read) > 1:
            extra += (
                f" [partitions unioned: {', '.join(self.partitions_read)}; "
                f"{self.legacy_partition_rows} row(s) from a non-canonical partition]"
            )
        if self.denominator_source != "entered_states":
            extra += f" [denominator: {self.denominator_source}]"
        if not self.should_alert:
            return head + extra + " — no finding"
        return head + extra + " — PAGES: " + " | ".join(self.alert_conditions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "run_date": self.run_date,
            "swept_at": self.swept_at,
            "denominator_source": self.denominator_source,
            "denominator_reason": self.denominator_reason,
            "partitions_read": list(self.partitions_read),
            "legacy_partition_rows": self.legacy_partition_rows,
            "finding_threshold": self.finding_threshold,
            "counts": self.counts,
            "coverage_established": self.coverage_established,
            "deferral_reason": self.deferral_reason,
            "absent_stages": list(self.absent_stages),
            "not_entered_stages": list(self.not_entered_stages),
            "spine_not_entered_stages": list(self.spine_not_entered_stages),
            "declared_skips": list(self.declared_skips),
            "declared_skips_entered": list(self.declared_skips_entered),
            "declared_skips_unknown": list(self.declared_skips_unknown),
            "should_alert": self.should_alert,
            "alert_conditions": list(self.alert_conditions),
            "rows": [r.to_dict() for r in self.rows],
            "cycle": self.cycle.to_dict() if self.cycle else None,
            "explain": self.explain(),
        }


def _strs(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    out: list[str] = []
    for item in value:
        if isinstance(item, (list, tuple)) and item:
            out.append(str(item[0]))
        else:
            out.append(str(item))
    return tuple(out)


def sweep_coverage(
    *,
    pipeline: str,
    run_date: str,
    registry: Mapping[str, Any],
    verdicts: Mapping[str, Mapping[str, Any]],
    entered_states: Iterable[str] | None,
    entered_reason: str = "",
    partitions_read: Sequence[str] | None = None,
    cycle: CycleShape | None = None,
    finding_threshold: int = DEFAULT_FINDING_THRESHOLD,
    observer_stage: str | None = None,
    declared_skips: Iterable[str] | None = None,
    now: datetime | None = None,
) -> CoverageSweep:
    """Compare the declared stage set against the verdicts that landed. Pure.

    ``verdicts`` maps stage name to the parsed verdict object. A stage in
    ``verdicts`` but absent from the registry is **reported**, not dropped —
    a verdict for a stage nobody declared means the two declarations have
    drifted, and silently ignoring it is how the drift stays invisible.

    ``observer_stage`` names the pipeline state the sweep is ITSELF running
    as. It is executing while it grades, so it can never have written its own
    verdict — reporting it ``ABSENT`` is a false positive guaranteed on every
    run, and it was one of the 13 on 2026-09-04. It gets
    :attr:`RowState.SELF`, outside the denominator and visible in the rows.
    A verdict that DOES exist for it (written by an earlier execution of the
    same cycle) is graded normally: the carve-out covers the impossible case
    only, never a real absence.

    ``declared_skips`` names the stages the CALLER declared this cycle
    deliberately does not run — a cadence declaration or a recorded operator
    flag, read from ``run_scope.json`` by the caller, never inferred here
    (``alpha-engine-config-I10175``). They get :attr:`RowState.DECLARED_SKIP`
    rather than :attr:`RowState.NOT_ENTERED`, so a deliberate exclusion and a
    stage nobody ran are never the same number
    (``alpha-engine-config-I10199``). Two things keep the mechanism from
    becoming a way to quiet a real absence, and both PAGE: a declared skip
    naming no declared stage, and a declared skip the graph entered anyway.
    """
    now = now or datetime.now(timezone.utc)
    partitions = tuple(partitions_read) if partitions_read else (run_date,)
    declared = [row for row in (registry.get("pipeline_stages") or []) if row.get("stage")]

    if entered_states is None:
        entered: set[str] | None = None
        denominator_source = "declared_only"
        denominator_reason = entered_reason or "entered-state set could not be read"
    else:
        entered = {str(s) for s in entered_states}
        denominator_source = "entered_states"
        denominator_reason = ""

    skips = {str(s) for s in (declared_skips or ())}

    rows: list[StageRow] = []
    for row in declared:
        stage = str(row["stage"])
        stage_class = str(row.get("stage_class", ""))
        declared_output = str(row.get("output", ""))
        verdict = verdicts.get(stage)

        if verdict is None:
            if observer_stage and stage == observer_stage:
                rows.append(
                    StageRow(
                        stage=stage,
                        state=RowState.SELF,
                        stage_class=stage_class,
                        declared_output=declared_output,
                        reason=(
                            "the sweep's own stage — it is executing while it grades and "
                            "cannot have recorded a verdict about its own completion"
                        ),
                    )
                )
            elif entered is not None and stage not in entered and stage in skips:
                rows.append(
                    StageRow(
                        stage=stage,
                        state=RowState.DECLARED_SKIP,
                        stage_class=stage_class,
                        declared_output=declared_output,
                        reason=(
                            f"declared, not entered, and the caller declared {stage} a "
                            "deliberate skip for this cycle — outside the denominator "
                            "and outside the not-entered count"
                        ),
                    )
                )
            elif entered is not None and stage not in entered:
                rows.append(
                    StageRow(
                        stage=stage,
                        state=RowState.NOT_ENTERED,
                        stage_class=stage_class,
                        declared_output=declared_output,
                        reason="declared, not entered by any contributing execution",
                    )
                )
            else:
                rows.append(
                    StageRow(
                        stage=stage,
                        state=RowState.ABSENT,
                        stage_class=stage_class,
                        declared_output=declared_output,
                        reason=(
                            "no verdict object under "
                            + " or ".join(f"{VERDICT_PREFIX}/{p}/" for p in partitions)
                            + " — the stage "
                            + (
                                "entered and recorded nothing"
                                if entered is not None
                                else "may have entered and recorded nothing"
                            )
                            + (
                                " — AND the caller declared it a deliberate skip, "
                                "which the graph contradicts"
                                if stage in skips
                                else ""
                            )
                        ),
                    )
                )
            continue

        status = str(verdict.get("status", "")) or None
        if status in COVERED_STATUSES:
            state = RowState.COVERED
        elif bool(verdict.get("is_finding")):
            state = RowState.FINDING
        else:
            state = RowState.UNMEASURED

        rows.append(
            StageRow(
                stage=stage,
                state=state,
                stage_class=stage_class or str(verdict.get("stage_class", "")),
                declared_output=declared_output or str(verdict.get("declared_output", "")),
                verdict_status=status,
                reason=str(verdict.get("reason", "")),
                covered_artifacts=_strs(verdict.get("covered")),
                missing_artifacts=_strs(verdict.get("missing")),
                stale_artifacts=_strs(verdict.get("stale")),
                unmeasured_artifacts=_strs(verdict.get("unmeasured")),
                verdict_key=(
                    f"{VERDICT_PREFIX}/"
                    f"{str(verdict.get('_partition_date') or run_date)}/{stage}.json"
                ),
                recorded_at=str(verdict.get("recorded_at", "")),
                partition_date=str(verdict.get("_partition_date") or run_date),
            )
        )

    declared_names = {str(r["stage"]) for r in declared}
    for stage in sorted(set(verdicts) - declared_names):
        verdict = verdicts[stage]
        rows.append(
            StageRow(
                stage=stage,
                state=RowState.UNMEASURED,
                verdict_status=str(verdict.get("status", "")) or None,
                reason=(
                    "a verdict landed for a stage with no pipeline_stages row — the "
                    "registry and the state machine have drifted; see "
                    "alpha-engine-config/scripts/check_stage_coverage_drift.py"
                ),
                verdict_key=(
                    f"{VERDICT_PREFIX}/"
                    f"{str(verdict.get('_partition_date') or run_date)}/{stage}.json"
                ),
                recorded_at=str(verdict.get("recorded_at", "")),
                partition_date=str(verdict.get("_partition_date") or run_date),
            )
        )

    return CoverageSweep(
        pipeline=pipeline,
        run_date=run_date,
        rows=tuple(rows),
        declared_skips=tuple(sorted(skips)),
        declared_skips_entered=tuple(
            sorted(s for s in skips if entered is not None and s in entered)
        ),
        declared_skips_unknown=tuple(sorted(s for s in skips if s not in declared_names)),
        denominator_source=denominator_source,
        denominator_reason=denominator_reason,
        partitions_read=partitions,
        cycle=cycle,
        finding_threshold=finding_threshold,
        swept_at=now.astimezone(timezone.utc).isoformat(),
    )


# ── S3 front door ────────────────────────────────────────────────────────────


def _load_verdicts(
    s3_client: Any,
    *,
    bucket: str,
    run_date: str,
    partitions: Sequence[str] | None = None,
    prefix: str = VERDICT_PREFIX,
) -> dict[str, dict[str, Any]]:
    """Read every verdict object under each ``<prefix>/<date>/`` partition.

    ``partitions`` is the ordered family list from
    :func:`~.partition.partition_dates`, canonical FIRST. A stage found in an
    earlier partition is never overwritten by a later one, so a converged
    stage is always reported from the canonical family and the legacy family
    only FILLS GAPS (``alpha-engine-config-I8809``). Each verdict carries the
    partition it came from under ``_partition_date`` so the row can say so.

    A key that lists but will not parse is kept with an explicit
    ``status: UNREADABLE``: dropping it would make a corrupt verdict
    indistinguishable from an absent one, and only the second is the
    ``I8152`` class.
    """
    out: dict[str, dict[str, Any]] = {}
    for partition in partitions or (run_date,):
        _load_verdicts_one(s3_client, bucket=bucket, date_key=partition, prefix=prefix, out=out)
    return out


def _load_verdicts_one(
    s3_client: Any,
    *,
    bucket: str,
    date_key: str,
    prefix: str,
    out: dict[str, dict[str, Any]],
) -> None:
    token: str | None = None
    base = f"{prefix}/{date_key}/"
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": base}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3_client.list_objects_v2(**kwargs)
        for obj in page.get("Contents") or []:
            key = str(obj.get("Key") or "")
            if not key.endswith(".json"):
                continue
            stage = key[len(base) : -len(".json")]
            if not stage or "/" in stage:
                continue
            if stage in out:
                # Canonical partition wins. A duplicate in a later family is
                # the migration's expected shape, not a conflict to resolve.
                continue
            try:
                body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
                parsed = json.loads(body)
                if isinstance(parsed, str):  # defensively unwrap double-encoding
                    parsed = json.loads(parsed)
            except Exception as exc:  # noqa: BLE001 — recorded, never dropped
                logger.error("coverage sweep: verdict %s unreadable", key, exc_info=True)
                out[stage] = {
                    "status": "UNREADABLE",
                    "is_finding": False,
                    "reason": f"verdict object unreadable: {type(exc).__name__}: {exc}",
                    "_partition_date": date_key,
                }
                continue
            if isinstance(parsed, dict):
                parsed["_partition_date"] = date_key
                out[stage] = parsed
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
        if not token:
            break


def read_coverage_sweep(
    *,
    pipeline: str,
    run_date: str,
    calendar_date: str | None = None,
    state_machine_arn: str | None = None,
    registry: Mapping[str, Any] | None = None,
    bucket: str = "alpha-engine-research",
    s3_client: Any = None,
    sfn_client: SFNClient | None = None,
    finding_threshold: int = DEFAULT_FINDING_THRESHOLD,
    observer_execution_arn: str | None = None,
    observer_stage: str | None = None,
    stage_spine: Sequence[str] | None = None,
    declared_skips: Iterable[str] | None = None,
    now: datetime | None = None,
) -> CoverageSweep:
    """Read the registry, the verdicts and the cycle, and sweep them.

    ``run_date`` is the CANONICAL partition — the cycle's trading day.
    ``calendar_date`` is the legacy family, unioned in only while the
    ``alpha-engine-config-I8809`` migration window is open
    (:func:`~.partition.dual_partition_active`). Passing it after the cutover
    is a no-op rather than an error: the window is closed by
    :data:`~.partition.CUTOVER_DATE`, not by every caller remembering.

    ``observer_execution_arn`` and ``observer_stage`` are the caller's answer
    to "where am I running from?" — the execution and the pipeline state the
    sweep is itself inside. Both default to ``None``, the correct answer for
    an out-of-band re-sweep, a backfill or the CLI: those observe from
    OUTSIDE the cycle and have no self to exclude
    (``alpha-engine-config-I10161``).

    ``stage_spine`` overrides the declared spine
    (:func:`~.registry.stage_order_for`) forwarded to
    :func:`~.cycle_shape.read_cycle_shape` verbatim — ``None`` (the default)
    uses the full declared spine. This is the caller's answer to "does this
    CYCLE intend to run every declared stage" — a question the registry
    cannot answer alone: :data:`~.registry.PIPELINE_STAGE_ORDER` names
    ``ParityParallel`` / ``PitParityCompare`` unconditionally, but a run with
    ``skip_parity: true`` (a recorded operator flag, not a defect) never
    intends to enter them. Without an exclusion, such a cycle reads
    ``INCOMPLETE`` forever, on every run for as long as the flag is set —
    ``alpha-engine-config-I10175``, measured live on the 2026-09-04 cycle.
    The caller (not this function) is responsible for deriving which stages
    a given cycle deliberately excludes — this module has no reader for
    ``run_scope.json`` and must not grow one just to answer this.

    ``declared_skips`` is the same declaration expressed on the COVERAGE
    surface: those stages are reported ``declared_skip`` rather than
    ``not_entered``, so a deliberate exclusion never lands in the count that
    pages (``alpha-engine-config-I10199``). Pass both, from the same source,
    or the two surfaces will disagree about the same cycle.
    """
    if s3_client is None:  # pragma: no cover — production path
        import boto3
        from krepis.aws_region import resolve_region

        s3_client = boto3.client("s3", region_name=resolve_region())

    if registry is None:
        from krepis import stage_coverage as sc

        registry = sc.load_registry(s3_client, bucket=bucket)

    partitions = partition_dates(run_date, calendar_date)
    verdicts = _load_verdicts(
        s3_client, bucket=bucket, run_date=run_date, partitions=partitions
    )

    cycle: CycleShape | None = None
    entered: list[str] | None = None
    entered_reason = ""
    if state_machine_arn:
        try:
            # The cycle's identity keys are NOT the artifact partitions:
            # partition_dates expires at the I8809 cutover, cycle_keys does
            # not. A scheduled execution's key falls through to startDate —
            # its wall-clock day — so a trading-day-keyed cycle must admit
            # the calendar key forever, or the Saturday run that did the
            # week's work is not a contributor to its own cycle.
            cycle = read_cycle_shape(
                state_machine_arn,
                run_date,
                calendar_date=calendar_date,
                client=sfn_client,
                observer_execution_arn=observer_execution_arn,
                stage_spine=stage_spine,
            )
        except Exception as exc:  # noqa: BLE001 — degrades LOUDLY, never silently
            entered_reason = f"{type(exc).__name__}: {exc}"
            logger.error(
                "coverage sweep: could not read the cycle's entered states for %s %s "
                "— falling back to the DECLARED denominator, which pages",
                pipeline,
                run_date,
                exc_info=True,
            )
        else:
            seen: list[str] = []
            for execution in cycle.executions:
                seen.extend(execution.all_states_entered)
            entered = seen
    else:
        entered_reason = "no state machine ARN supplied — cannot read the entered-state set"

    return sweep_coverage(
        pipeline=pipeline,
        run_date=run_date,
        registry=registry,
        verdicts=verdicts,
        entered_states=entered,
        entered_reason=entered_reason,
        partitions_read=partitions,
        cycle=cycle,
        finding_threshold=finding_threshold,
        observer_stage=observer_stage,
        declared_skips=declared_skips,
        now=now,
    )


# ── Emission ─────────────────────────────────────────────────────────────────


def sweep_artifact_key(sweep: CoverageSweep, *, prefix: str = SWEEP_ARTIFACT_PREFIX) -> str:
    return f"{prefix}/{sweep.pipeline}/{sweep.run_date}.json"


def publish_sweep(
    sweep: CoverageSweep,
    *,
    s3_client: Any = None,
    cloudwatch_client: Any = None,
    bucket: str = "alpha-engine-research",
    prefix: str = SWEEP_ARTIFACT_PREFIX,
) -> None:
    """Persist the sweep and publish its counts. Never raises.

    ``StageCoverageSweepRan`` is published unconditionally and FIRST. The
    surface has to be observable itself (``observability-policy.md`` §9): a
    sweep that stops running publishes no count at all, and an alarm on the
    absence of *that* datapoint is the only thing that can tell a clean week
    from a dead reader.
    """
    if cloudwatch_client is not None:
        dims = [
            {"Name": "Pipeline", "Value": sweep.pipeline},
        ]
        data = [{"MetricName": "StageCoverageSweepRan", "Dimensions": dims, "Value": 1.0, "Unit": "None"}]
        emitted: list[tuple[str, float]] = [
            ("StageCoverageSweepExpected", sweep.counts["expected"]),
            ("StageCoverageSweepCovered", sweep.covered),
            ("StageCoverageSweepFindings", sweep.findings),
            ("StageCoverageSweepUnmeasured", sweep.unmeasured),
            # 1 exactly when the absent count below is WITHHELD. Published on
            # EVERY sweep, so the alarm on it separates "not deferred" from
            # "the sweep stopped running" — that second fact belongs to
            # StageCoverageSweepRan, and neither is ever inferred from the
            # other's silence (alpha-engine-config-I10161, principles.md 2.7).
            ("StageCoverageSweepDeferred", 1.0 if sweep.deferred else 0.0),
            # The caller's DECLARATION, not a claim about the cycle's shape —
            # nothing about a still-running cycle makes it unsupportable, so
            # it publishes on every sweep. It is the denominator against which
            # a rising NotEntered is read: a week that moved a stage from
            # "declared skip" to "nobody ran it" shows as one falling and the
            # other rising, which neither number says alone.
            ("StageCoverageSweepDeclaredSkip", sweep.declared_skip),
        ]
        if sweep.coverage_established:
            # WITHHELD, not zeroed, when the cycle cannot support the claim.
            # Publishing 0 would render an unestablished surface green;
            # publishing the count anyway is what produced the 2026-09-04
            # false page of 13 absences on a cycle the same artifact declared
            # in_flight. The deferred metric above is what stays loud.
            emitted.append(("StageCoverageSweepAbsent", sweep.absent))
            # alpha-engine-config-I10199. Withheld on the same condition and
            # for the same reason as the absent count: a stage that has not
            # been entered YET is not a stage that was never entered, and a 0
            # published on an unestablished sweep renders the surface green.
            # StageCoverageSweepDeferred (above, every sweep) and
            # StageCoverageSweepRan (below, every sweep) are what separate
            # "clean" from "withheld" from "the reader is dead" — no data is
            # never inferred from another metric's silence (principles.md 2.7).
            emitted.append(("StageCoverageSweepNotEntered", sweep.not_entered))
            emitted.append(("StageCoverageSweepSpineNotEntered", sweep.spine_not_entered))
        for name, value in emitted:
            data.append(
                {"MetricName": name, "Dimensions": dims, "Value": float(value), "Unit": "Count"}
            )
        try:
            cloudwatch_client.put_metric_data(Namespace=METRIC_NAMESPACE, MetricData=data)
        except Exception:  # noqa: BLE001 — fail-soft, recorded at ERROR
            logger.error("coverage sweep: FAILED to publish sweep metrics", exc_info=True)

    if s3_client is not None:
        key = sweep_artifact_key(sweep, prefix=prefix)
        try:
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(sweep.to_dict(), indent=2, default=str).encode(),
                ContentType="application/json",
            )
        except Exception:  # noqa: BLE001 — fail-soft, recorded at ERROR
            logger.error(
                "coverage sweep: FAILED to write the sweep artifact to s3://%s/%s",
                bucket,
                key,
                exc_info=True,
            )


_STATE_GLYPH = {
    RowState.COVERED: "OK  ",
    RowState.FINDING: "FIND",
    RowState.ABSENT: "GONE",
    RowState.UNMEASURED: "????",
    RowState.NOT_ENTERED: "----",
    RowState.DECLARED_SKIP: "SKIP",
    RowState.SELF: "SELF",
}


def render_rows(sweep: CoverageSweep, *, include_not_entered: bool = True) -> str:
    """Per-stage table — the console surface and the CLI's stdout.

    ``NOT_ENTERED`` rows are rendered by default. A surface that hides the
    stages a run did not reach reports a true number about a smaller world
    than its name implies.
    """
    order = {
        RowState.ABSENT: 0,
        RowState.FINDING: 1,
        RowState.UNMEASURED: 2,
        RowState.COVERED: 3,
        RowState.NOT_ENTERED: 4,
        RowState.DECLARED_SKIP: 5,
        RowState.SELF: 6,
    }
    rows = [r for r in sweep.rows if include_not_entered or r.state is not RowState.NOT_ENTERED]
    rows.sort(key=lambda r: (order[r.state], r.stage))
    width = max((len(r.stage) for r in rows), default=5)
    lines = [sweep.explain(), ""]
    for row in rows:
        status = row.verdict_status or "-"
        lines.append(f"  {_STATE_GLYPH[row.state]}  {row.stage:<{width}}  {status:<18}  {row.reason}")
    return "\n".join(lines)


def _main(argv: Sequence[str] | None = None) -> int:
    """``python -m nousergon_lib.pipeline_status.coverage``.

    Exit codes: 0 clean · 2 the sweep pages · 3 the sweep itself could not run.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="nousergon_lib.pipeline_status.coverage")
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--run-date", required=True, help="the cycle's TRADING day — the canonical partition")
    parser.add_argument(
        "--calendar-date",
        default=None,
        help=(
            "the execution's calendar date — the legacy partition, unioned in "
            "only while the alpha-engine-config-I8809 migration window is open"
        ),
    )
    parser.add_argument("--state-machine-arn", default=None)
    parser.add_argument("--bucket", default="alpha-engine-research")
    parser.add_argument("--registry-path", default=None, help="read the registry from disk")
    parser.add_argument("--finding-threshold", type=int, default=DEFAULT_FINDING_THRESHOLD)
    parser.add_argument(
        "--observer-execution-arn",
        default=None,
        help=(
            "the execution this sweep is running INSIDE, when it is running inside one. "
            "Its RUNNING status does not make its own cycle in_flight "
            "(alpha-engine-config-I10161). An out-of-band re-sweep leaves this unset."
        ),
    )
    parser.add_argument(
        "--observer-stage",
        default=None,
        help=(
            "the pipeline state this sweep IS. It cannot have recorded a verdict about "
            "its own completion, so it is reported 'self' rather than 'absent'."
        ),
    )
    parser.add_argument(
        "--declared-skip",
        action="append",
        default=None,
        metavar="STAGE",
        dest="declared_skips",
        help=(
            "a stage this cycle DELIBERATELY does not run (repeatable). Reported "
            "'declared_skip', outside the denominator and outside the not-entered "
            "count that pages. A skip naming no declared stage, or one the graph "
            "entered anyway, PAGES (alpha-engine-config-I10199)."
        ),
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--publish", action="store_true", help="write the sweep artifact and publish its metrics"
    )
    parser.add_argument(
        "--alert", action="store_true", help="publish an alert when the sweep pages"
    )
    parser.add_argument(
        "--augment-marker",
        action="store_true",
        help=(
            "merge the cycle's real shape into the SF completion marker "
            "(alpha-engine-config-I8186) — the sweep already read the cycle to "
            "establish its own denominator, so this costs one GET and one PUT"
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import boto3
    from krepis.aws_region import resolve_region

    region = resolve_region()
    s3_client = boto3.client("s3", region_name=region)

    registry = None
    if args.registry_path:
        from krepis import stage_coverage as sc

        registry = sc.load_registry(None, local_path=args.registry_path)

    try:
        sweep = read_coverage_sweep(
            pipeline=args.pipeline,
            run_date=args.run_date,
            calendar_date=args.calendar_date,
            state_machine_arn=args.state_machine_arn,
            registry=registry,
            bucket=args.bucket,
            s3_client=s3_client,
            finding_threshold=args.finding_threshold,
            observer_execution_arn=args.observer_execution_arn,
            observer_stage=args.observer_stage,
            declared_skips=args.declared_skips,
        )
    except Exception as exc:  # noqa: BLE001 — a sweep that cannot run says so
        print(
            f"ERROR: coverage sweep {args.pipeline} {args.run_date} COULD NOT RUN — "
            f"{type(exc).__name__}: {exc}. This is not a clean result.",
            file=__import__("sys").stderr,
        )
        return 3

    print(json.dumps(sweep.to_dict(), indent=2, default=str) if args.json else render_rows(sweep))

    if args.publish:
        publish_sweep(
            sweep,
            s3_client=s3_client,
            cloudwatch_client=boto3.client("cloudwatch", region_name=region),
            bucket=args.bucket,
        )

    if args.augment_marker:
        if sweep.cycle is None:
            print(
                "WARNING: --augment-marker was asked for but the cycle could not be "
                "read, so the marker keeps its bare envelope claim. A marker with no "
                "cycle block resolves to UNKNOWN, never to a pass.",
                file=__import__("sys").stderr,
            )
        else:
            from .completion_marker import augment_marker

            augment_marker(
                sweep.cycle,
                s3_client=s3_client,
                bucket=args.bucket,
                also_dates=sweep.partitions_read,
            )

    if args.alert and sweep.should_alert:
        from krepis import alerts

        alerts.publish(
            sweep.explain(),
            severity="error",
            source=f"stage-coverage-sweep/{args.pipeline}",
            dedup_key=f"stage-coverage-sweep/{args.pipeline}/{args.run_date}",
        )

    return 2 if sweep.should_alert else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
