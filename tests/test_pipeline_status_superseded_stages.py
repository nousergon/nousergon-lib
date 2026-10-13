"""A spine stage that REPLACED earlier stages is not missing for an execution
whose walk entered one of those predecessors (alpha-engine-config-I12271).

Every grader judges an execution against today's spine. The I11269 decoupled
data cutover replaced the v1 data legs with ``WaitForCollectionManifests``,
so every execution of the pre-cutover definition read as short a stage that
did not exist yet: the evaluator's 28-day substrate window graded its
pre-cutover preopen cycles 2/19. ``SUPERSEDED_STAGES`` maps the successor to
its predecessors; entering any predecessor is walk EVIDENCE of the old
definition. A walk that entered neither is still short the successor —
absence never excuses anything.
"""

from __future__ import annotations

import pytest

from nousergon_lib.pipeline_status import (
    PIPELINE_STAGE_ORDER,
    STATE_TO_ARCHIVE_PAGE,
    SUPERSEDED_STAGES,
    RunStatus,
    WorkVerdict,
    classify_work,
    retiring_definition_stages_for,
    superseded_stages_served,
)
from nousergon_lib.pipeline_status.cycle_shape import CycleVerdict, build_cycle_shape

WEEKLY = "ne-weekly-freshness-pipeline"
PREOPEN = "ne-preopen-trading-pipeline"
SUCCESSOR = "WaitForCollectionManifests"

#: The two entries this suite pins. Deleting them (Re-exam: 2026-11-02, once
#: no pre-cutover cycle can sit in a 28-day window) deletes these cases.
EXPECTED = {
    WEEKLY: {SUCCESSOR: ("MorningEnrich", "DataPhase1")},
    PREOPEN: {SUCCESSOR: ("LaunchMorningEnrichSpot", "LaunchMorningArcticAppendSpot")},
}
ROLE = {WEEKLY: "weekly", PREOPEN: "daily"}


def _pre_cutover_walk(pipeline, *, predecessors=None, without=()):
    """A run of the OLD definition: the predecessors where the successor now
    sits, every other spine stage entered unless named in ``without``."""
    preds = EXPECTED[pipeline][SUCCESSOR] if predecessors is None else predecessors
    states = ["InitializeInput"]
    for stage in PIPELINE_STAGE_ORDER[pipeline]:
        if stage == SUCCESSOR:
            states.extend(preds)
            continue
        if stage in without:
            continue
        states.append(stage)
    states.append("WriteCompletionMarker")
    return states


def _run(pipeline, states, status=RunStatus.SUCCEEDED):
    return classify_work(state_machine_name=pipeline, status=status, entered_states=states)


# ── the registry ─────────────────────────────────────────────────────────


def test_the_weekly_and_preopen_maps_are_exactly_the_cutover_lineage():
    assert SUPERSEDED_STAGES[WEEKLY] == EXPECTED[WEEKLY]
    assert SUPERSEDED_STAGES[PREOPEN] == EXPECTED[PREOPEN]


def test_every_pipeline_has_a_map_and_no_other_pipeline_carries_an_entry():
    assert set(SUPERSEDED_STAGES) == set(PIPELINE_STAGE_ORDER)
    for pipeline, table in SUPERSEDED_STAGES.items():
        if pipeline not in EXPECTED:
            assert table == {}, pipeline


@pytest.mark.parametrize("pipeline", sorted(SUPERSEDED_STAGES))
def test_every_successor_is_a_spine_stage_and_every_predecessor_is_not(pipeline):
    """A predecessor still on the spine would be graded twice; a successor
    off the spine would excuse nothing and hide a stale entry."""
    spine = set(PIPELINE_STAGE_ORDER[pipeline])
    for successor, preds in SUPERSEDED_STAGES[pipeline].items():
        assert successor in spine
        assert preds, f"{pipeline}:{successor} names no predecessor"
        for pred in preds:
            assert pred not in spine
            # Kept for history, so pre-change executions still render.
            assert pred in STATE_TO_ARCHIVE_PAGE


def test_arn_and_bare_name_resolve_the_same_served_set():
    arn = f"arn:aws:states:us-east-1:1:stateMachine:{PREOPEN}"
    walk = ["LaunchMorningEnrichSpot"]
    assert superseded_stages_served(arn, walk) == superseded_stages_served(PREOPEN, walk)
    assert superseded_stages_served(PREOPEN, walk) == {SUCCESSOR}


def test_nothing_is_served_without_a_predecessor_in_the_walk():
    assert superseded_stages_served(PREOPEN, ["StartExecutorEC2", "RunDaemon"]) == frozenset()
    assert superseded_stages_served("ne-postclose-trading-pipeline", ["CaptureSnapshot"]) == frozenset()


# ── one execution ────────────────────────────────────────────────────────


@pytest.mark.parametrize("pipeline", sorted(EXPECTED))
def test_a_pre_cutover_run_that_entered_the_predecessors_is_a_full_run(pipeline):
    outcome = _run(pipeline, _pre_cutover_walk(pipeline))
    assert outcome.verdict is WorkVerdict.COMPLETED and outcome.reason == "full_run"
    assert SUCCESSOR not in outcome.stages_missing
    # Excused, not credited: the successor was never entered.
    assert SUCCESSOR not in outcome.stages_entered
    n = len(PIPELINE_STAGE_ORDER[pipeline])
    assert outcome.stage_coverage == f"{n - 1}/{n}"


@pytest.mark.parametrize("pipeline", sorted(EXPECTED))
def test_one_predecessor_is_enough_evidence_of_the_old_definition(pipeline):
    first = EXPECTED[pipeline][SUCCESSOR][:1]
    outcome = _run(pipeline, _pre_cutover_walk(pipeline, predecessors=first))
    assert outcome.verdict is WorkVerdict.COMPLETED


@pytest.mark.parametrize("pipeline", sorted(EXPECTED))
def test_a_walk_with_neither_predecessor_nor_successor_is_still_short(pipeline):
    outcome = _run(pipeline, _pre_cutover_walk(pipeline, predecessors=()))
    assert outcome.verdict is WorkVerdict.INCOMPLETE
    assert outcome.reason == "partial_success"
    assert outcome.stages_missing == (SUCCESSOR,)


@pytest.mark.parametrize("pipeline", sorted(EXPECTED))
def test_a_predecessor_excuses_no_other_stage(pipeline):
    # Skip stages another marker already excuses (weekly RAGIngestion /
    # DataPhase2 are retiring), so the assertion isolates this mechanism.
    retiring = retiring_definition_stages_for(pipeline)
    other = next(
        s for s in PIPELINE_STAGE_ORDER[pipeline] if s != SUCCESSOR and s not in retiring
    )
    outcome = _run(pipeline, _pre_cutover_walk(pipeline, without=(other,)))
    assert outcome.verdict is WorkVerdict.INCOMPLETE
    assert outcome.stages_missing == (other,)


@pytest.mark.parametrize("pipeline", sorted(EXPECTED))
def test_a_post_cutover_run_that_entered_the_successor_is_unchanged(pipeline):
    walk = ["InitializeInput", *PIPELINE_STAGE_ORDER[pipeline], "WriteCompletionMarker"]
    outcome = _run(pipeline, walk)
    assert outcome.verdict is WorkVerdict.COMPLETED
    assert SUCCESSOR in outcome.stages_entered


# ── one cycle ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pipeline", sorted(EXPECTED))
def test_a_pre_cutover_cycle_is_complete(pipeline):
    walk = _pre_cutover_walk(pipeline)
    shape = build_cycle_shape(
        pipeline=pipeline, run_date="2026-09-25",
        outcomes=[(_run(pipeline, walk), ROLE[pipeline], walk)],
    )
    assert shape.verdict is CycleVerdict.COMPLETED
    assert SUCCESSOR not in shape.stages_missing


@pytest.mark.parametrize("pipeline", sorted(EXPECTED))
def test_a_cycle_with_neither_predecessor_nor_successor_is_not_complete(pipeline):
    walk = _pre_cutover_walk(pipeline, predecessors=())
    shape = build_cycle_shape(
        pipeline=pipeline, run_date="2026-09-25",
        outcomes=[(_run(pipeline, walk), ROLE[pipeline], walk)],
    )
    assert shape.verdict is not CycleVerdict.COMPLETED
    assert shape.stages_missing == (SUCCESSOR,)


def test_a_pre_cutover_cycle_recovered_across_executions_is_complete():
    """The scheduled run entered the predecessors and died later; a same-cycle
    recovery entered the rest. The walk evidence comes from the cycle's
    executions together, as the stage union does."""
    full = _pre_cutover_walk(PREOPEN)
    cut = full.index("PredictorInference")
    first = [*full[: cut + 1], "FailExecution"]
    rest = ["InitializeInput", *full[cut + 1 :]]
    shape = build_cycle_shape(
        pipeline=PREOPEN, run_date="2026-09-25",
        outcomes=[
            (_run(PREOPEN, first, RunStatus.FAILED), "daily", first),
            (_run(PREOPEN, rest), "recovery", rest),
        ],
    )
    assert shape.verdict is CycleVerdict.COMPLETED
    assert shape.stages_missing == ()
