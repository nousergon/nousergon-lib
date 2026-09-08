"""A week nobody judged must not read green — the spine and the sweep halves.

``alpha-engine-config-I10199`` deliverables 2 and 3. Two structural
blindnesses, both measured 2026-09-08 against the live
``ne-weekly-freshness-pipeline``:

1. ``PIPELINE_STAGE_ORDER["ne-weekly-freshness-pipeline"]`` declared **no**
   eval-judge stage, so :func:`~.cycle_shape.build_cycle_shape` was
   structurally incapable of returning ``incomplete`` for a week whose judge
   chain never ran, however many contributing executions skipped it.
2. :class:`~.coverage.CoverageSweep` excluded ``NOT_ENTERED`` from its
   denominator, published no metric for it and listed no alert condition for
   it — so a declared stage no contributing execution entered was, by
   construction, silent on every metric and every alarm.

## The recorded shape this file replays

``tests/fixtures/sf_substatus/weekly_2026-08-15_scheduled.json`` is the SAME
artifact ``nous-ergon-ops-PR1119`` froze for clause
``SFP-2.3b-stage-status-is-the-worst-substatus`` — the 76 stage results of the
2026-08-15 scheduled execution
(``54acfc69-3bbf-10bd-8238-2859fede685b_f1036888-ffe2-528a-a421-327ccd785a8c``),
read with FULL pagination. Both repos assert against one artifact rather than
against two inventions of the same week.

**What that week was and was not.** 2026-08-15 **entered** ``EvalRollingMean``
— the judge chain ran and returned ``Payload.status: OK`` over
``agent_quality {status: ERROR, error: "'str' object has no attribute
'isoformat'"}``. That is the sub-status swallow, fixed at its source in
``crucible-research`` and given its clause by ``nous-ergon-ops-PR1119``; the
spine cannot catch it and this file does not claim it can. The spine catches
the OTHER shape — the eval chain never entered at all (the self-perpetuating
skip ratchet of ``alpha-engine-config-I10199`` deliverable 1, a branch that
fails to enter, a cadence that silently drops it) — which today produces a
``completed`` cycle verdict with nothing anywhere saying otherwise. 2026-08-15
is therefore the POSITIVE control here, and it is used as one.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from nousergon_lib.pipeline_status.coverage import (
    RowState,
    publish_sweep,
    sweep_coverage,
)
from nousergon_lib.pipeline_status.cycle_shape import (
    CycleVerdict,
    build_cycle_shape,
)
from nousergon_lib.pipeline_status.read import RunStatus
from nousergon_lib.pipeline_status.registry import (
    PIPELINE_STAGE_ORDER,
    stage_order_for,
)
from nousergon_lib.pipeline_status.work import classify_work

WEEKLY = "ne-weekly-freshness-pipeline"
RUN_DATE = "2026-08-15"
NOW = datetime(2026, 8, 15, 23, 0, tzinfo=timezone.utc)

#: The one state every non-skip route through the eval-judge branch converges
#: on. See :data:`~.registry.PIPELINE_STAGE_ORDER`'s own note for the walk.
EVAL_STAGE = "EvalRollingMean"

FIXTURE = (
    Path(__file__).parent / "fixtures" / "sf_substatus" / "weekly_2026-08-15_scheduled.json"
)


@pytest.fixture(scope="module")
def recorded_states() -> list[str]:
    """The states the 2026-08-15 scheduled execution actually entered."""
    return [row["state"] for row in json.loads(FIXTURE.read_text())]


def _outcome(status, entered, *, name="e", arn="arn", spine=None):
    return classify_work(
        state_machine_name=WEEKLY,
        status=status,
        entered_states=list(entered),
        execution_arn=arn,
        execution_name=name,
        stage_spine=spine if spine is not None else stage_order_for(WEEKLY),
        skip_terminals=frozenset({"WeeklyRunDaySkip"}),
    )


def _cycle(entered, *, spine=None, status=RunStatus.SUCCEEDED):
    return build_cycle_shape(
        pipeline=WEEKLY,
        run_date=RUN_DATE,
        outcomes=[(_outcome(status, entered, spine=spine), "weekly", list(entered))],
        stage_spine=spine,
    )


# ── Deliverable 2 — the spine declares the eval stage ────────────────────────


def test_the_weekly_spine_declares_the_eval_judge_stage():
    """The blindness, executable: before this, no eval state was declared."""
    spine = stage_order_for(WEEKLY)
    assert EVAL_STAGE in spine, (
        "a spine with no eval-judge stage cannot express 'nobody judged this week'"
    )


def test_the_spine_declares_the_convergence_state_and_not_the_forks():
    """Which name, and why only that one.

    Walked from the live ``nousergon-data/infrastructure/step_function.json``
    on 2026-09-08 (the eval chain is branch 0 of ``ResearchPredictorParallel``):

    - ``EvalJudgeSubmitWeekly`` / ``EvalJudgeSubmitFirstSaturday`` are a
      monthly-cadence FORK — declaring either makes every week on the other
      arm permanently ``incomplete``;
    - ``EvalJudgeProcess`` is skipped by two legitimate routes,
      ``EvalJudgeEmptyPlan -> EvalRollingMean`` (a contract-declared
      non-degraded outcome) and ``MarkEvalJudgeDegraded -> EvalRollingMean``;
    - ``EvalRollingMean`` is entered by ALL THREE of those routes and by no
      skip route — ``CheckSkipEvalJudge``'s skip branch goes straight to
      ``CheckSkipRationaleClustering``, past it.

    So its non-entry means exactly one thing: the eval chain did not run.
    """
    spine = stage_order_for(WEEKLY)
    forks = {
        "EvalJudgeSubmitWeekly",
        "EvalJudgeSubmitFirstSaturday",
        "EvalJudgeProcess",
        "EvalJudgeEmptyPlan",
        "MarkEvalJudgeDegraded",
    }
    assert forks.isdisjoint(spine), (
        "a conditional or degraded-path state on the spine makes the cycle "
        "permanently incomplete — the alpha-engine-config-I10175 failure"
    )


def test_the_eval_stage_sits_inside_the_parallel_it_belongs_to():
    """Ordered deepest-last: ``cycles._depth_of`` reads position as progress.

    The eval chain is inside ``ResearchPredictorParallel``, which wholly
    precedes the top-level ``Backtester``.
    """
    spine = PIPELINE_STAGE_ORDER[WEEKLY]
    assert spine.index("DataPhase2") < spine.index(EVAL_STAGE) < spine.index("Backtester")


def test_the_recorded_2026_08_15_execution_did_enter_the_eval_stage(recorded_states):
    """Positive control, and an honesty guard on the claim above.

    2026-08-15 produced zero eval artifacts, and this test asserts the spine
    would NOT have caught it: the chain entered. That week's defect was the
    swallowed ``agent_quality`` sub-status (``nous-ergon-ops-PR1119``
    ``SFP-2.3b``, repaired at source in ``crucible-research``). Deleting this
    test to make the spine look like the fix for 2026-08-15 would be the
    overclaim it exists to prevent.
    """
    assert EVAL_STAGE in recorded_states
    cycle = _cycle(recorded_states)
    assert EVAL_STAGE not in cycle.stages_missing


def test_a_week_whose_eval_chain_never_ran_is_incomplete(recorded_states):
    """The closes-when: ``cycle_shape`` can now reach ``incomplete`` for it."""
    full = set(recorded_states) | set(stage_order_for(WEEKLY))
    unjudged = [s for s in full if not s.startswith("Eval") or s.startswith("Evaluator")]

    cycle = _cycle(unjudged)
    assert cycle.verdict is CycleVerdict.INCOMPLETE
    assert cycle.reason == "partial_cycle"
    assert cycle.stages_missing == (EVAL_STAGE,), (
        "the eval stage must be the ONLY thing this cycle is missing, or the "
        "test is passing for an unrelated reason"
    )


def test_a_judged_week_with_every_other_stage_is_completed(recorded_states):
    """The other polarity — the spine addition must not turn every week red."""
    full = sorted(set(recorded_states) | set(stage_order_for(WEEKLY)))
    cycle = _cycle(full)
    assert cycle.verdict is CycleVerdict.COMPLETED
    assert cycle.stages_missing == ()


def test_a_deliberately_skipped_eval_stage_is_excluded_by_the_caller_not_here():
    """``nousergon-lib-PR392``'s seam is what makes a real skip expressible.

    A cadence-declared or operator-declared skip of the judge is legitimate.
    The caller — which alone can read ``run_scope.json`` — drops the stage
    from the spine it passes, and the cycle reads ``completed`` rather than
    being permanently ``incomplete`` for as long as the flag holds
    (``alpha-engine-config-I10175``). This module never infers that itself.
    """
    reduced = tuple(s for s in stage_order_for(WEEKLY) if s != EVAL_STAGE)
    cycle = _cycle(reduced, spine=reduced)
    assert cycle.verdict is CycleVerdict.COMPLETED
    assert EVAL_STAGE not in cycle.stage_spine


# ── Deliverable 3 — not_entered becomes visible ──────────────────────────────

REGISTRY = {
    "pipeline_stages": [
        {"stage": "Scanner", "stage_class": "product", "output": "signals"},
        {"stage": EVAL_STAGE, "stage_class": "product", "output": "eval rolling mean"},
        {"stage": "EvalJudgeSubmitFirstSaturday", "stage_class": "product", "output": "plan"},
    ]
}
COVERED = {"status": "COVERED", "is_finding": False}


def _sweep(entered, *, declared_skips=None, verdicts=None, cycle_spine=None, status=RunStatus.SUCCEEDED):
    """A sweep whose cycle carries a real spine, so the spine filter is live."""
    spine = cycle_spine if cycle_spine is not None else stage_order_for(WEEKLY)
    return sweep_coverage(
        pipeline=WEEKLY,
        run_date=RUN_DATE,
        registry=REGISTRY,
        verdicts=verdicts if verdicts is not None else {"Scanner": COVERED},
        entered_states=entered,
        declared_skips=declared_skips,
        cycle=_cycle(entered, spine=spine, status=status),
        now=NOW,
    )


def test_a_spine_stage_nobody_entered_is_not_entered_and_pages():
    """The blindness, executable. Before this it was a row and nothing else."""
    sweep = _sweep(["Scanner"])

    row = next(r for r in sweep.rows if r.stage == EVAL_STAGE)
    assert row.state is RowState.NOT_ENTERED
    # Two rows are not entered — the spine stage and the monthly fork — and
    # only the first of them is a claim about work that was due.
    assert sweep.not_entered == 2
    assert sweep.spine_not_entered == 1
    assert sweep.should_alert
    conditions = " ".join(sweep.alert_conditions)
    assert "spine_not_entered" in conditions
    assert EVAL_STAGE in conditions


def test_a_conditional_stage_nobody_entered_is_reported_and_does_not_page():
    """``EvalJudgeSubmitFirstSaturday`` is not entered on 11 weeks in 12.

    Paging on every ``not_entered`` row would page on the monthly fork, the
    parity branch and every degraded-path twin on every healthy run — the
    chronic-false-positive class this fleet already carries an incident
    register for. The spine is the discriminator: it is the declaration of
    which stages' entry is what "the pipeline ran" MEANS.
    """
    sweep = _sweep(["Scanner", EVAL_STAGE], verdicts={"Scanner": COVERED, EVAL_STAGE: COVERED})

    row = next(r for r in sweep.rows if r.stage == "EvalJudgeSubmitFirstSaturday")
    assert row.state is RowState.NOT_ENTERED
    assert sweep.not_entered == 1
    assert sweep.spine_not_entered == 0
    assert not sweep.should_alert


def test_a_declared_skip_and_a_stage_nobody_ran_are_different_numbers():
    """``alpha-engine-config-I10175``'s carve-out, kept distinguishable.

    A deliberate skip is a DECLARATION by the caller. It must never land in
    the same count as a stage nobody ran, or the exclusion mechanism becomes
    a way to make a real absence quiet.
    """
    skipped = _sweep(["Scanner"], declared_skips=[EVAL_STAGE])

    row = next(r for r in skipped.rows if r.stage == EVAL_STAGE)
    assert row.state is RowState.DECLARED_SKIP
    assert EVAL_STAGE in row.reason
    assert skipped.declared_skip == 1
    assert skipped.spine_not_entered == 0
    assert skipped.not_entered == 1, "only the monthly fork remains not_entered"
    assert not skipped.should_alert

    unskipped = _sweep(["Scanner"])
    assert unskipped.declared_skip == 0
    assert unskipped.not_entered == 2 and unskipped.spine_not_entered == 1
    assert skipped.counts["declared_skip"] != unskipped.counts["declared_skip"]
    assert skipped.counts["not_entered"] != unskipped.counts["not_entered"]


def test_a_declared_skip_is_outside_the_denominator_like_not_entered():
    sweep = _sweep(["Scanner"], declared_skips=[EVAL_STAGE])
    assert EVAL_STAGE not in sweep.expected
    assert sweep.counts["expected"] + sweep.not_entered + sweep.declared_skip == len(sweep.rows)


def test_a_declared_skip_the_graph_actually_entered_pages():
    """The declaration and the graph disagree. Loud, never reconciled here."""
    sweep = _sweep(
        ["Scanner", EVAL_STAGE],
        declared_skips=[EVAL_STAGE],
        verdicts={"Scanner": COVERED, EVAL_STAGE: COVERED},
    )
    assert sweep.declared_skips_entered == (EVAL_STAGE,)
    assert sweep.should_alert
    assert "declared_skip_entered" in " ".join(sweep.alert_conditions)


def test_a_declared_skip_naming_no_declared_stage_pages():
    """A typo'd or renamed skip silently suppresses nothing — it pages.

    Without this, ``declared_skips`` is a mechanism for making a real absence
    invisible by misspelling it.
    """
    sweep = _sweep(["Scanner", EVAL_STAGE], declared_skips=["EvalRollingMeen"])
    assert sweep.declared_skips_unknown == ("EvalRollingMeen",)
    assert sweep.should_alert
    assert "declared_skip_unknown" in " ".join(sweep.alert_conditions)


# ── What reaches CloudWatch ──────────────────────────────────────────────────


class _CW:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def put_metric_data(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def _metrics(sweep) -> dict[str, float]:
    cw = _CW()
    publish_sweep(sweep, cloudwatch_client=cw, s3_client=None)
    return {m["MetricName"]: m["Value"] for m in cw.calls[0]["MetricData"]}


def test_an_established_sweep_publishes_the_not_entered_counts():
    metrics = _metrics(_sweep(["Scanner"]))
    assert metrics["StageCoverageSweepNotEntered"] == 2.0
    assert metrics["StageCoverageSweepSpineNotEntered"] == 1.0
    assert metrics["StageCoverageSweepDeclaredSkip"] == 0.0


def test_a_deferred_sweep_withholds_the_not_entered_datapoints():
    """Mirrors ``StageCoverageSweepAbsent``: a zero on an unestablished sweep
    renders green, and a stage that has not been entered YET is not a stage
    that was never entered. ``StageCoverageSweepDeferred`` carries the fact,
    and ``StageCoverageSweepRan`` separates it from a dead reader
    (``principles.md`` 2.7 — no data is never green)."""
    metrics = _metrics(_sweep(["Scanner"], status=RunStatus.RUNNING))
    assert "StageCoverageSweepNotEntered" not in metrics
    assert "StageCoverageSweepSpineNotEntered" not in metrics
    assert metrics["StageCoverageSweepDeferred"] == 1.0
    assert metrics["StageCoverageSweepRan"] == 1.0


def test_a_deferred_sweep_does_not_page_a_not_yet_entered_spine_stage():
    """The would-be page is the deferral, which already names the cycle."""
    sweep = _sweep(["Scanner"], status=RunStatus.RUNNING)
    assert sweep.deferred
    assert "spine_not_entered" not in " ".join(sweep.alert_conditions)
    assert any("coverage_deferred" in c for c in sweep.alert_conditions)


def test_the_declared_skip_count_publishes_even_when_deferred():
    """It is the CALLER's declaration, not a claim about the cycle's shape,
    so nothing about a running cycle makes it unsupportable."""
    metrics = _metrics(_sweep(["Scanner"], declared_skips=[EVAL_STAGE], status=RunStatus.RUNNING))
    assert metrics["StageCoverageSweepDeclaredSkip"] == 1.0


def test_the_artifact_records_every_new_field():
    """The console and any later reader see the same split the alarm sees."""
    payload = _sweep(["Scanner"], declared_skips=["EvalRollingMeen"]).to_dict()
    assert payload["counts"]["not_entered"] == 2
    assert payload["counts"]["declared_skip"] == 0
    assert payload["not_entered_stages"] == [EVAL_STAGE, "EvalJudgeSubmitFirstSaturday"]
    assert payload["spine_not_entered_stages"] == [EVAL_STAGE]
    assert payload["declared_skips_unknown"] == ["EvalRollingMeen"]
