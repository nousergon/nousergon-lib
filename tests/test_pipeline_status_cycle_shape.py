"""A cadence cycle's real shape — replayed against 2026-08-22.

``alpha-engine-config-I8186``. The fixture is a verbatim capture of the four
executions that contributed to the 2026-08-22 weekly cycle, read live from
``ListExecutions`` + ``GetExecutionHistory`` on that date.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from nousergon_lib.pipeline_status.cycle_shape import (
    CONTRIBUTING_ROLES,
    CycleVerdict,
    build_cycle_shape,
    cycle_key_for,
    read_cycle_shape,
)
from nousergon_lib.pipeline_status.read import RunStatus, SFNAccessDenied
from nousergon_lib.pipeline_status.work import WorkVerdict, classify_work

FIXTURES = Path(__file__).parent / "fixtures" / "coverage_sweep_260822"
PIPELINE = "ne-weekly-freshness-pipeline"
SPINE = ("A", "B", "C")


def _outcome(status, entered, *, name="e", arn="arn", spine=SPINE):
    return classify_work(
        state_machine_name=PIPELINE,
        status=status,
        entered_states=list(entered),
        execution_arn=arn,
        execution_name=name,
        stage_spine=spine,
        skip_terminals=frozenset({"WeeklyRunDaySkip"}),
    )


# ── The cycle key ────────────────────────────────────────────────────────────


def test_a_scheduled_execution_gets_its_cycle_key_from_its_start_date():
    """The class defect: scheduled weekly runs carry NO run_date in their input.

    Measured 2026-08-22 against the live state machine — every
    ``pipeline_role: weekly`` execution has ``run_date: None`` in its
    DescribeExecution input and an opaque UUID name, because the state
    machine's own ``InitializeInput`` stamps ``run_date`` from
    ``$$.Execution.StartTime``. Reading only input+name drops every scheduled
    run out of its own cycle: on 2026-08-22 that left the cycle looking like
    three ``watch-rerun`` executions at 1/16 stages, hiding the 02:00
    scheduled run that reached 14/16.
    """
    desc = {
        "input": json.dumps({"pipeline_role": "weekly"}),
        "name": "1ed4d68f-574b-7496-6196-cf052d7178ba_ab3ec94",
        "startDate": datetime(2026, 8, 22, 2, 0, 49, tzinfo=timezone.utc),
    }
    assert cycle_key_for(desc) == "2026-08-22"


def test_an_explicit_run_date_still_wins_over_the_start_date():
    desc = {
        "input": json.dumps({"run_date": "2026-08-15"}),
        "name": "watch-rerun-2026-08-16-4",
        "startDate": datetime(2026, 8, 15, 20, 21, tzinfo=timezone.utc),
    }
    assert cycle_key_for(desc) == "2026-08-15"


def test_no_start_date_and_no_run_date_is_none_never_a_guess():
    assert cycle_key_for({"input": "{}", "name": "opaque"}) is None


# ── The union ────────────────────────────────────────────────────────────────


def test_a_cycle_completed_across_two_executions_is_completed_not_failed():
    """The overcorrection ``I8186`` forbids: a legitimate recovery is not a failure."""
    shape = build_cycle_shape(
        pipeline=PIPELINE,
        run_date="2026-08-22",
        outcomes=[
            (_outcome(RunStatus.FAILED, ["A", "B"], name="scheduled"), "weekly", ["A", "B"]),
            (_outcome(RunStatus.SUCCEEDED, ["C"], name="rerun"), "watch-rerun", ["C"]),
        ],
        stage_spine=SPINE,
    )
    assert shape.verdict is CycleVerdict.COMPLETED
    assert shape.reason == "recovered_across_executions"
    assert shape.is_recovery_tail is True
    assert shape.execution_count == 2
    assert shape.did_work is True


def test_a_single_full_execution_is_completed_and_not_a_recovery_tail():
    shape = build_cycle_shape(
        pipeline=PIPELINE,
        run_date="2026-08-22",
        outcomes=[(_outcome(RunStatus.SUCCEEDED, list(SPINE)), "weekly", list(SPINE))],
        stage_spine=SPINE,
    )
    assert shape.verdict is CycleVerdict.COMPLETED
    assert shape.reason == "full_cycle"
    assert shape.is_recovery_tail is False


def test_a_partial_cycle_is_incomplete_and_names_what_is_missing():
    shape = build_cycle_shape(
        pipeline=PIPELINE,
        run_date="2026-08-22",
        outcomes=[(_outcome(RunStatus.SUCCEEDED, ["A"]), "weekly", ["A"])],
        stage_spine=SPINE,
    )
    assert shape.verdict is CycleVerdict.INCOMPLETE
    assert shape.reason == "partial_cycle"
    assert shape.stages_missing == ("B", "C")
    assert "missing B, C" in shape.explain()


def test_an_exercise_run_never_completes_a_cadence_cycle():
    """``roles.py``'s binding invariant, enforced at the cycle level."""
    shape = build_cycle_shape(
        pipeline=PIPELINE,
        run_date="2026-08-22",
        outcomes=[(_outcome(RunStatus.SUCCEEDED, list(SPINE)), "exercise", list(SPINE))],
        stage_spine=SPINE,
    )
    assert shape.verdict is CycleVerdict.INCOMPLETE
    assert shape.reason == "no_executions"
    assert shape.execution_count == 0
    assert "exercise" not in CONTRIBUTING_ROLES


def test_an_untagged_manual_run_does_contribute():
    shape = build_cycle_shape(
        pipeline=PIPELINE,
        run_date="2026-08-22",
        outcomes=[(_outcome(RunStatus.SUCCEEDED, list(SPINE)), None, list(SPINE))],
        stage_spine=SPINE,
    )
    assert shape.verdict is CycleVerdict.COMPLETED


def test_a_cadence_tick_with_no_execution_is_incomplete_never_silent():
    shape = build_cycle_shape(
        pipeline=PIPELINE, run_date="2026-08-22", outcomes=[], stage_spine=SPINE
    )
    assert shape.verdict is CycleVerdict.INCOMPLETE
    assert shape.reason == "no_executions"


def test_every_contributor_skipping_is_a_skip_not_a_failure():
    outcome = _outcome(RunStatus.SUCCEEDED, ["WeeklyRunDaySkip"])
    assert outcome.verdict is WorkVerdict.SKIPPED
    shape = build_cycle_shape(
        pipeline=PIPELINE,
        run_date="2026-08-22",
        outcomes=[(outcome, "weekly", ["WeeklyRunDaySkip"])],
        stage_spine=SPINE,
    )
    assert shape.verdict is CycleVerdict.SKIPPED


def test_an_in_flight_contributor_is_never_collapsed_into_incomplete():
    shape = build_cycle_shape(
        pipeline=PIPELINE,
        run_date="2026-08-22",
        outcomes=[
            (_outcome(RunStatus.FAILED, ["A"], name="a"), "weekly", ["A"]),
            (_outcome(RunStatus.RUNNING, ["B"], name="b"), "watch-rerun", ["B"]),
        ],
        stage_spine=SPINE,
    )
    assert shape.verdict is CycleVerdict.IN_FLIGHT
    assert shape.reason == "still_running"


# ── The 2026-08-22 replay ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def recorded() -> dict[str, Any]:
    return json.loads((FIXTURES / "cycle_shape_expected.json").read_text())


def test_the_2026_08_22_cycle_is_incomplete_across_four_executions(recorded):
    """The marker said SUCCEEDED and named ONE execution. Four contributed.

    The completion marker for this date records ``status: SUCCEEDED`` and a
    single ``execution_arn`` — ``watch-rerun-2026-08-22-3``, which entered one
    declared stage. The cycle it belongs to had four contributing executions
    and reached 14 of 16 stages. Both facts are true; only the second is a
    verdict, and the marker could carry neither.
    """
    assert recorded["execution_count"] == 4
    assert recorded["verdict"] == "incomplete"
    assert recorded["reason"] == "partial_cycle"
    assert recorded["stage_coverage"] == "14/16"
    assert recorded["stages_missing"] == ["ParityParallel", "PitParityCompare"]

    roles = [e["pipeline_role"] for e in recorded["executions"]]
    assert roles == ["weekly", "watch-rerun", "watch-rerun", "watch-rerun"]
    marker_execution = next(
        e for e in recorded["executions"] if e["execution_name"] == "watch-rerun-2026-08-22-3"
    )
    assert marker_execution["status"] == "SUCCEEDED"
    assert marker_execution["stages_entered_count"] == 1


def test_the_marker_payload_carries_the_shape_the_old_one_could_not():
    shape = build_cycle_shape(
        pipeline=PIPELINE,
        run_date="2026-08-22",
        outcomes=[
            (_outcome(RunStatus.FAILED, ["A", "B"], name="scheduled", arn="arn:a"), "weekly", ["A", "B"]),
            (_outcome(RunStatus.SUCCEEDED, ["C"], name="rerun", arn="arn:b"), "watch-rerun", ["C"]),
        ],
        stage_spine=SPINE,
    )
    payload = shape.to_dict()
    assert payload["execution_count"] == 2
    assert payload["is_recovery_tail"] is True
    assert payload["stage_coverage"] == "3/3"
    assert [e["execution_arn"] for e in payload["executions"]] == ["arn:a", "arn:b"]
    assert payload["verdict"] == "completed"


# ── Transport failures are never a verdict ───────────────────────────────────


class _DeniedSFN:
    def list_executions(self, **_: Any) -> dict[str, Any]:
        class _Denied(Exception):
            response = {"Error": {"Code": "AccessDeniedException"}}

        raise _Denied("denied")


def test_a_denial_raises_rather_than_reporting_an_empty_cycle():
    """A transport or authorization failure rendered as "no executions" is a
    verdict manufactured from a denial — six instances of that class in one
    week, one of which failed OPEN on a live trading gate."""
    with pytest.raises(SFNAccessDenied):
        read_cycle_shape(
            "arn:aws:states:us-east-1:1:stateMachine:x",
            "2026-08-22",
            client=_DeniedSFN(),
            stage_spine=SPINE,
        )


def test_walk_exhaustion_is_stated_not_swallowed():
    shape = build_cycle_shape(
        pipeline=PIPELINE,
        run_date="2026-08-22",
        outcomes=[(_outcome(RunStatus.SUCCEEDED, ["A"]), "weekly", ["A"])],
        stage_spine=SPINE,
        walk_exhausted=True,
    )
    assert shape.walk_exhausted is True
    assert "walk cap reached" in shape.explain()


# ── Cadence-declared skips (alpha-engine-config-I11170) ──────────────────────

PARITY_SPINE = ("A", "ParityParallel", "PitParityCompare", "Director")
PARITY_MARKER = "MarkParityVerdictUnknownByCadence"


def _parity_outcome(status, entered, *, name="e", arn="arn"):
    return classify_work(
        state_machine_name=PIPELINE,
        status=status,
        entered_states=list(entered),
        execution_arn=arn,
        execution_name=name,
        stage_spine=PARITY_SPINE,
        skip_terminals=frozenset({"WeeklyRunDaySkip"}),
    )


def test_a_cadence_declared_skip_is_not_a_missing_stage():
    """Measured 2026-09-18: `skip_parity: true` on the live Saturday trigger.

    The cycle read ``incomplete (partial_cycle), 15/16, missing
    PitParityCompare`` on a run that SUCCEEDED with
    ``degraded_summary.degraded: false``. With the marker state in the walk,
    the two parity stages are excused rather than missing.
    """
    states = ["A", PARITY_MARKER, "Director"]
    shape = build_cycle_shape(
        pipeline=PIPELINE,
        run_date="2026-09-18",
        outcomes=[(_parity_outcome(RunStatus.SUCCEEDED, states), "weekly", states)],
        stage_spine=PARITY_SPINE,
    )
    assert shape.verdict is CycleVerdict.COMPLETED
    assert shape.stages_missing == ()
    assert shape.stages_declared_skipped == ("ParityParallel", "PitParityCompare")
    assert shape.has_declared_skips is True


def test_the_reason_says_the_cycle_was_allowed_to_skip():
    """A bare ``full_cycle`` would make an excused cycle read as an exhaustive one."""
    states = ["A", PARITY_MARKER, "Director"]
    shape = build_cycle_shape(
        pipeline=PIPELINE,
        run_date="2026-09-18",
        outcomes=[(_parity_outcome(RunStatus.SUCCEEDED, states), "weekly", states)],
        stage_spine=PARITY_SPINE,
    )
    assert shape.reason == "full_cycle_with_declared_skips"
    assert "declared-skipped ParityParallel, PitParityCompare" in shape.explain()


def test_a_recovery_that_completes_a_skipped_cycle_says_both_things():
    states_a = ["A"]
    states_b = [PARITY_MARKER, "Director"]
    shape = build_cycle_shape(
        pipeline=PIPELINE,
        run_date="2026-09-18",
        outcomes=[
            (_parity_outcome(RunStatus.FAILED, states_a, name="sched", arn="arn:a"), "weekly", states_a),
            (
                _parity_outcome(RunStatus.SUCCEEDED, states_b, name="rerun", arn="arn:b"),
                "watch-rerun",
                states_b,
            ),
        ],
        stage_spine=PARITY_SPINE,
    )
    assert shape.verdict is CycleVerdict.COMPLETED
    assert shape.reason == "recovered_across_executions_with_declared_skips"
    assert shape.is_recovery_tail is True


def test_a_complete_cycle_with_no_skip_marker_still_reads_full_cycle():
    """The unchanged case: everything ran, so nothing is excused and nothing says so."""
    states = ["A", "ParityParallel", "PitParityCompare", "Director"]
    shape = build_cycle_shape(
        pipeline=PIPELINE,
        run_date="2026-09-18",
        outcomes=[(_parity_outcome(RunStatus.SUCCEEDED, states), "weekly", states)],
        stage_spine=PARITY_SPINE,
    )
    assert shape.verdict is CycleVerdict.COMPLETED
    assert shape.reason == "full_cycle"
    assert shape.stages_declared_skipped == ()
    assert "declared-skipped" not in shape.explain()


def test_absence_without_a_marker_stays_missing():
    """``sf-pipeline-policy.md`` §2.3a — absence is UNKNOWN, never a pass.

    Inferring the skip from the stage's own absence is the tautology that
    would forgive every missing stage on every pipeline.
    """
    states = ["A", "Director"]
    shape = build_cycle_shape(
        pipeline=PIPELINE,
        run_date="2026-09-18",
        outcomes=[(_parity_outcome(RunStatus.SUCCEEDED, states), "weekly", states)],
        stage_spine=PARITY_SPINE,
    )
    assert shape.verdict is CycleVerdict.INCOMPLETE
    assert shape.reason == "partial_cycle"
    assert shape.stages_missing == ("ParityParallel", "PitParityCompare")
    assert shape.stages_declared_skipped == ()


def test_a_marker_excuses_only_the_stages_it_names():
    """One genuinely missing stage keeps the cycle INCOMPLETE."""
    spine = PARITY_SPINE + ("ReportCard",)
    outcome = classify_work(
        state_machine_name=PIPELINE,
        status=RunStatus.SUCCEEDED,
        entered_states=["A", PARITY_MARKER, "Director"],
        execution_arn="arn",
        execution_name="e",
        stage_spine=spine,
    )
    shape = build_cycle_shape(
        pipeline=PIPELINE,
        run_date="2026-09-18",
        outcomes=[(outcome, "weekly", ["A", PARITY_MARKER, "Director"])],
        stage_spine=spine,
    )
    assert shape.verdict is CycleVerdict.INCOMPLETE
    assert shape.stages_missing == ("ReportCard",)
    assert shape.stages_declared_skipped == ("ParityParallel", "PitParityCompare")


def test_a_declared_skip_stage_that_ran_anyway_counts_as_entered():
    """The declaration never erases observed work — entered wins over excused."""
    states = ["A", PARITY_MARKER, "ParityParallel", "PitParityCompare", "Director"]
    shape = build_cycle_shape(
        pipeline=PIPELINE,
        run_date="2026-09-18",
        outcomes=[(_parity_outcome(RunStatus.SUCCEEDED, states), "weekly", states)],
        stage_spine=PARITY_SPINE,
    )
    assert shape.stages_declared_skipped == ()
    assert shape.reason == "full_cycle"
    assert shape.stages_entered == PARITY_SPINE


def test_a_skip_terminal_cycle_is_still_SKIPPED_not_completed():
    """A cadence marker must never turn a cycle in which nothing ran into COMPLETED."""
    states = [PARITY_MARKER, "WeeklyRunDaySkip"]
    shape = build_cycle_shape(
        pipeline=PIPELINE,
        run_date="2026-09-18",
        outcomes=[(_parity_outcome(RunStatus.SUCCEEDED, states), "weekly", states)],
        stage_spine=PARITY_SPINE,
    )
    assert shape.verdict is CycleVerdict.SKIPPED
    assert shape.reason == "declared_skip"


def test_the_declared_skips_reach_the_completion_marker_payload():
    """``merge_cycle_shape`` stamps ``to_dict()``, so the field must be in it."""
    from nousergon_lib.pipeline_status.completion_marker import merge_cycle_shape

    states = ["A", PARITY_MARKER, "Director"]
    shape = build_cycle_shape(
        pipeline=PIPELINE,
        run_date="2026-09-18",
        outcomes=[(_parity_outcome(RunStatus.SUCCEEDED, states), "weekly", states)],
        stage_spine=PARITY_SPINE,
    )
    payload = shape.to_dict()
    assert payload["stages_declared_skipped"] == ["ParityParallel", "PitParityCompare"]
    merged = merge_cycle_shape({"status": "SUCCEEDED"}, shape)
    assert merged["cycle"]["stages_declared_skipped"] == [
        "ParityParallel",
        "PitParityCompare",
    ]
    assert merged["cycle_verdict"] == "completed"


def test_an_unknown_state_name_declares_nothing():
    from nousergon_lib.pipeline_status.registry import declared_skip_stages_for

    assert declared_skip_stages_for(PIPELINE, []) == frozenset()
    assert declared_skip_stages_for(PIPELINE, ["SomeOtherState"]) == frozenset()
    assert declared_skip_stages_for("ne-preopen-trading-pipeline", [PARITY_MARKER]) == frozenset()
    assert declared_skip_stages_for("no-such-pipeline", [PARITY_MARKER]) == frozenset()


def test_the_accessor_resolves_an_arn_like_the_other_registry_readers():
    from nousergon_lib.pipeline_status.registry import declared_skip_stages_for

    arn = f"arn:aws:states:us-east-1:711398986525:stateMachine:{PIPELINE}"
    assert declared_skip_stages_for(arn, [PARITY_MARKER]) == frozenset(
        {"ParityParallel", "PitParityCompare"}
    )
