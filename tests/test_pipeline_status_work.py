"""The three-way work verdict, proved against REAL executions.

Every fixture under ``tests/fixtures/pipeline_status/`` was captured live
from ``ne-weekly-freshness-pipeline`` on 2026-08-21 (``describe-execution``
plus a fully-paged ``get-execution-history``); each carries its provenance
inline. They are the four shapes the sweep in alpha-engine-config-I8045
found, and the point of the suite is that a status-only reader calls three
of the four green:

===============================  =========  =========  ==============================
execution                        status     duration   what actually happened
===============================  =========  =========  ==============================
scheduled 2026-08-21             SUCCEEDED      5.7s   run-day gate-out, 0/16 stages
scheduled 2026-08-20             SUCCEEDED      3.5s   run-day gate-out, 0/16 stages
watch-rerun-2026-08-16-4         SUCCEEDED     8m40s   0/16 stages, wrote a marker
scheduled 2026-08-15             FAILED        3h53m   10/16 stages, died in Parity
===============================  =========  =========  ==============================

:func:`test_status_only_reader_calls_three_of_four_green` is the RED proof:
it asserts the OLD predicate's verdict on these files, so the reason each
fix is needed is executable rather than described.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nousergon_lib.pipeline_status import (
    PIPELINE_STAGE_ORDER,
    RunStatus,
    UndeclaredPipeline,
    WorkVerdict,
    classify_work,
    entered_states_from_history,
    skip_terminals_for,
    stage_order_for,
)

FIXTURES = Path(__file__).parent / "fixtures" / "pipeline_status"
WEEKLY = "ne-weekly-freshness-pipeline"

GATEOUT_0821 = "weekly_gateout_2026_08_21"
GATEOUT_0820 = "weekly_gateout_2026_08_20"
VACUOUS = "weekly_vacuous_success_watch_rerun_2026_08_16_4"
FAILED = "weekly_failed_2026_08_15"
ALL_FIXTURES = (GATEOUT_0821, GATEOUT_0820, VACUOUS, FAILED)


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def outcome_for(name: str):
    rec = load(name)
    return classify_work(
        state_machine_name=WEEKLY,
        status=RunStatus(rec["status"]),
        entered_states=rec["entered_states"],
        duration_sec=rec["duration_sec"],
        execution_arn=rec["execution_arn"],
        execution_name=rec["name"],
    )


# ── RED first: what the predicate being replaced says about these files ───


def test_status_only_reader_calls_three_of_four_green():
    """The defect, executable.

    Three of the four real executions report SUCCEEDED. Not one of them ran
    the weekly graph. This test does not assert the new behaviour — it pins
    the old one, so that if someone later reintroduces a status-only
    predicate the reason it is wrong is right here.
    """
    green = [n for n in ALL_FIXTURES if load(n)["status"] == "SUCCEEDED"]
    assert green == [GATEOUT_0821, GATEOUT_0820, VACUOUS]
    for name in green:
        rec = load(name)
        entered = set(rec["entered_states"])
        ran = [s for s in PIPELINE_STAGE_ORDER[WEEKLY] if s in entered]
        assert ran == [], f"{name} was expected to have run nothing, ran {ran}"


# ── The three-way verdict ────────────────────────────────────────────────


@pytest.mark.parametrize("name", [GATEOUT_0821, GATEOUT_0820])
def test_run_day_gate_out_is_skipped_not_success_and_not_failure(name):
    o = outcome_for(name)
    assert o.verdict is WorkVerdict.SKIPPED
    assert o.reason == "declared_skip"
    assert o.terminal_state == "WeeklyRunDaySkip"
    # The two failure modes this verdict exists between.
    assert not o.did_work, "a gate-out must never read as a completed cycle"
    assert not o.should_alert, "a gate-out must never page — it is correct behaviour"
    assert not o.counts_as_cycle, "a gate-out must not sit in a success-rate denominator"
    assert "no work was due" in o.explain()


def test_succeeded_having_dispatched_nothing_is_incomplete():
    """``watch-rerun-2026-08-16-4``: SUCCEEDED, 8m40s, zero stages, and it
    wrote a completion marker — so the artifact detectors saw a clean cycle
    too. Duration alone would not have caught this one."""
    o = outcome_for(VACUOUS)
    assert o.verdict is WorkVerdict.INCOMPLETE
    assert o.reason == "vacuous_success"
    assert o.status is RunStatus.SUCCEEDED
    assert o.duration_sec is not None and o.duration_sec > 500
    assert o.terminal_state == "WriteCompletionMarker"
    assert o.terminal_state not in skip_terminals_for(WEEKLY)
    assert o.should_alert and not o.did_work


def test_real_failure_is_incomplete_and_names_the_stages_it_reached():
    o = outcome_for(FAILED)
    assert o.verdict is WorkVerdict.INCOMPLETE
    assert o.reason == "execution_failed"
    assert o.should_alert
    assert o.stage_coverage == "10/16"
    assert "ParityParallel" in o.stages_missing
    assert "MorningEnrich" in o.stages_entered


def test_a_full_run_is_the_only_completed_verdict():
    """Synthesised from the declared spine — no full weekly graph completed
    in the 30-execution window, which is itself the I7176 finding."""
    o = classify_work(
        state_machine_name=WEEKLY,
        status=RunStatus.SUCCEEDED,
        entered_states=["InitializeInput", *PIPELINE_STAGE_ORDER[WEEKLY], "WriteCompletionMarker"],
        duration_sec=14000.0,
    )
    assert o.verdict is WorkVerdict.COMPLETED
    assert o.reason == "full_run"
    assert o.did_work and o.counts_as_cycle and not o.should_alert
    assert o.stages_missing == ()


def test_partial_success_is_incomplete_not_completed():
    spine = PIPELINE_STAGE_ORDER[WEEKLY]
    o = classify_work(
        state_machine_name=WEEKLY,
        status=RunStatus.SUCCEEDED,
        entered_states=list(spine[:-2]),
        duration_sec=9000.0,
    )
    assert o.verdict is WorkVerdict.INCOMPLETE
    assert o.reason == "partial_success"
    assert o.stages_missing == spine[-2:]


def test_running_is_in_flight_never_a_work_verdict():
    o = classify_work(
        state_machine_name=WEEKLY,
        status=RunStatus.RUNNING,
        entered_states=["InitializeInput"],
        duration_sec=None,
    )
    assert o.verdict is WorkVerdict.IN_FLIGHT
    assert not o.did_work and not o.should_alert and not o.counts_as_cycle


# ── Declarations, not inference ──────────────────────────────────────────


def test_undeclared_pipeline_raises_rather_than_passing():
    with pytest.raises(UndeclaredPipeline):
        classify_work(
            state_machine_name="ne-some-new-pipeline",
            status=RunStatus.SUCCEEDED,
            entered_states=["Whatever"],
        )


def test_all_three_live_pipelines_have_a_declared_spine():
    for name in ("ne-weekly-freshness-pipeline", "ne-preopen-trading-pipeline", "ne-postclose-trading-pipeline"):
        assert stage_order_for(name), f"{name} has no declared substantive spine"
        assert stage_order_for(f"arn:aws:states:us-east-1:1:stateMachine:{name}") == stage_order_for(name)


def test_preopen_holiday_skip_is_a_task_terminal_not_a_succeed_state():
    """``NotifyHolidaySkip`` is a ``Task`` with ``End: true``. A reader that
    enumerated ``Succeed``-typed states to find skips would miss it, which is
    why the declaration is a name list."""
    assert "NotifyHolidaySkip" in skip_terminals_for("ne-preopen-trading-pipeline")
    o = classify_work(
        state_machine_name="ne-preopen-trading-pipeline",
        status=RunStatus.SUCCEEDED,
        entered_states=["InitializeInput", "MarketHoursGate", "NotifyHolidaySkip"],
        duration_sec=4.0,
    )
    assert o.verdict is WorkVerdict.SKIPPED


def test_postclose_declares_no_skip_terminal():
    assert skip_terminals_for("ne-postclose-trading-pipeline") == frozenset()


# ── History handling ─────────────────────────────────────────────────────


def test_entered_states_from_raw_history_matches_the_captured_sequence():
    raw = json.loads((FIXTURES / f"{GATEOUT_0821}_raw_history.json").read_text())
    assert entered_states_from_history(raw["events"]) == load(GATEOUT_0821)["entered_states"]


def test_a_truncated_history_would_have_reported_the_wrong_terminal():
    """Why ``_paged_execution_history`` exists.

    The 2026-08-15 execution carries 6715 events; the first 1000-event page
    ends deep inside ``MorningEnrich``'s poll loop. Truncated, the terminal
    state is a Wait companion and most of the spine looks unentered.
    """
    rec = load(FAILED)
    assert rec["history_event_count"] > 1000
    full = rec["entered_states"]
    assert full[-1] == "FailExecution"
    # A prefix standing in for the first page names something else entirely.
    assert full[len(full) // 4] != "FailExecution"


# ── Cycle-level: the clean streak that reported health it did not have ───


def test_clean_streak_no_longer_counts_run_day_gate_outs():
    """The headline surface, on the real week.

    Before I8045 the two gate-outs of 08-20 and 08-21 each incremented the
    clean streak, so the console reported a 2-cycle clean run in a week
    whose only real scheduled execution failed after 3h53m.
    """
    from datetime import datetime, timezone

    from nousergon_lib.pipeline_status import (
        AttemptOutcome,
        CycleReliability,
        ReliabilityWindow,
    )

    def attempt(name: str, cycle_start: datetime) -> AttemptOutcome:
        rec = load(name)
        o = outcome_for(name)
        return AttemptOutcome(
            name=rec["name"],
            execution_arn=rec["execution_arn"],
            status=RunStatus(rec["status"]),
            start_utc=cycle_start,
            duration_sec=rec["duration_sec"],
            failing_state="FailExecution" if rec["status"] == "FAILED" else None,
            error=None,
            depth_index=None,
            depth_stage=None,
            work=o,
        )

    d = lambda day: datetime(2026, 8, day, tzinfo=timezone.utc)  # noqa: E731
    window = ReliabilityWindow(
        cycles=[
            CycleReliability(cycle_key="2026-08-15", attempts=[attempt(FAILED, d(15))]),
            CycleReliability(cycle_key="2026-08-20", attempts=[attempt(GATEOUT_0820, d(20))]),
            CycleReliability(cycle_key="2026-08-21", attempts=[attempt(GATEOUT_0821, d(21))]),
        ]
    )
    assert window.cycles[1].skip_only and window.cycles[2].skip_only
    assert not window.cycles[0].skip_only
    # Skip-only cycles neither extend the streak nor break it; the last real
    # cycle failed, so the streak is 0 — not the 2 the old predicate gave.
    assert window.clean_streak == 0


def test_a_vacuous_success_does_not_extend_the_clean_streak_either():
    from datetime import datetime, timezone

    from nousergon_lib.pipeline_status import (
        AttemptOutcome,
        CycleReliability,
        ReliabilityWindow,
    )

    rec = load(VACUOUS)
    a = AttemptOutcome(
        name=rec["name"],
        execution_arn=rec["execution_arn"],
        status=RunStatus.SUCCEEDED,
        start_utc=datetime(2026, 8, 16, tzinfo=timezone.utc),
        duration_sec=rec["duration_sec"],
        failing_state=None,
        error=None,
        depth_index=None,
        depth_stage=None,
        work=outcome_for(VACUOUS),
    )
    w = ReliabilityWindow(cycles=[CycleReliability(cycle_key="2026-08-16", attempts=[a])])
    assert not a.succeeded, "SUCCEEDED with zero substantive stages is not a success"
    assert not a.skipped, "it reached no DECLARED skip terminal — it is incomplete"
    assert w.clean_streak == 0
