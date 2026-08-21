"""Cycle-level reliability projection (alpha-engine-config-I6919).

The question these tests hold: *"we have spent so many tokens on fixing the
weekly sf and it's unclear whether we are actually making progress. how can I
tell?"* — Brian, 2026-08-11. Red/green cannot answer it. Attempts-to-success,
stage depth, and new-vs-repeat causes can.

The last test replays the real `ne-weekly-freshness-pipeline` history for
2026-07-25 → 08-11 and asserts the shape that had to be reconstructed by hand
to answer him: four consecutive first-attempt successes, then a regression.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from nousergon_lib.pipeline_status import (
    PipelineExecutionSummary,
    RunStatus,
    build_reliability_window,
    fingerprint,
)

_STAGES = (
    "MorningEnrich",
    "DataPhase1",
    "RAGIngestion",
    "PredictorTraining",
    "Backtester",
    "EvaluatorDiagnostics",
    "EvaluatorOptimize",
    "ReportCard",
)

_T0 = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)


def _summary(name: str, status: RunStatus, *, minutes: int, dur: float = 600.0):
    return PipelineExecutionSummary(
        execution_arn=f"arn:aws:states:us-east-1:1:execution:sf:{name}",
        name=name,
        status=status,
        start_utc=_T0 + timedelta(minutes=minutes),
        duration_sec=dur,
    )


def _window(specs, *, max_cycles=None):
    """specs: (name, status, cycle_key, minutes, failing_state, error, entered)."""
    summaries = [_summary(s[0], s[1], minutes=s[3]) for s in specs]
    by_name = {s[0]: s for s in specs}
    return build_reliability_window(
        summaries,
        cycle_key_of=lambda s: by_name[s.name][2],
        failure_of=lambda s: (by_name[s.name][4], by_name[s.name][5]),
        entered_states_of=lambda s: by_name[s.name][6],
        stage_order=_STAGES,
        max_cycles=max_cycles,
    )


# ── Fingerprints ─────────────────────────────────────────────────────────


def test_the_same_stage_and_error_is_one_cause():
    assert fingerprint("MorningEnrich", "States.TaskFailed", attempt_id="a") == fingerprint(
        "MorningEnrich", "States.TaskFailed", attempt_id="b"
    )


def test_the_same_stage_with_a_different_error_is_a_different_cause():
    """Sandbox.Timedout and States.Runtime in one stage are different bugs."""
    assert fingerprint("Scanner", "Sandbox.Timedout", attempt_id="a") != fingerprint(
        "Scanner", "States.Runtime", attempt_id="a"
    )


def test_an_undiagnosable_failure_never_matches_another_one():
    """The honesty rule.

    A shared UNKNOWN bucket would make two unrelated undiagnosable failures
    read as one cause recurring — manufacturing the "we are looping" verdict
    this module exists to establish honestly.
    """
    assert fingerprint(None, "x", attempt_id="a") != fingerprint(None, "x", attempt_id="b")


# ── Attempts-to-success ──────────────────────────────────────────────────


def test_a_cycle_that_needed_six_reruns_is_distinguishable_from_a_clean_one():
    """Both render SUCCEEDED everywhere today. That is the whole problem."""
    w = _window(
        [
            *[(f"r{i}", RunStatus.FAILED, "2026-08-01", i, "MorningEnrich", "E", ["MorningEnrich"]) for i in range(6)],
            ("r6", RunStatus.SUCCEEDED, "2026-08-01", 6, None, None, list(_STAGES)),
            ("clean", RunStatus.SUCCEEDED, "2026-08-04", 100, None, None, list(_STAGES)),
        ]
    )
    hard, easy = w.cycles
    assert hard.attempts_to_success == 7
    assert hard.recovered is True
    assert easy.attempts_to_success == 1
    assert easy.recovered is False


def test_an_unfinished_first_attempt_has_no_verdict_rather_than_a_failing_one():
    w = _window([("run", RunStatus.RUNNING, "2026-08-11", 0, None, None, ["MorningEnrich"])])
    assert w.cycles[0].first_attempt_succeeded is None
    assert w.cycles[0].settled is False


# ── Stage depth ──────────────────────────────────────────────────────────


def test_dying_later_registers_as_deeper():
    """13m at MorningEnrich then 3h34m at PredictorTraining — measured 08-10/11."""
    w = _window(
        [
            ("a", RunStatus.FAILED, "c1", 0, "MorningEnrich", "E", ["MorningEnrich"]),
            (
                "b",
                RunStatus.FAILED,
                "c2",
                10,
                "PredictorTraining",
                "E",
                ["MorningEnrich", "DataPhase1", "RAGIngestion", "PredictorTraining"],
            ),
        ]
    )
    shallow, deep = w.cycles
    assert deep.depth_index > shallow.depth_index
    assert deep.depth_stage == "PredictorTraining"


def test_states_outside_the_declared_order_do_not_count_as_depth():
    """A poll or gate entering is not the run getting further."""
    w = _window([("a", RunStatus.FAILED, "c1", 0, "X", "E", ["PollMorningEnrichSpot", "CheckSkipFoo"])])
    assert w.cycles[0].depth_index is None


# ── New vs repeat: the loop detector ─────────────────────────────────────


def test_a_cause_returning_in_a_later_cycle_is_a_repeat():
    w = _window(
        [
            ("a", RunStatus.FAILED, "c1", 0, "MorningEnrich", "E", ["MorningEnrich"]),
            ("b", RunStatus.FAILED, "c2", 10, "MorningEnrich", "E", ["MorningEnrich"]),
        ]
    )
    assert w.cycles[0].new_causes and not w.cycles[0].repeat_causes
    assert w.cycles[1].repeat_causes and not w.cycles[1].new_causes
    assert w.looping is True


def test_a_cause_hit_twice_inside_one_cycle_is_not_a_repeat():
    """Rerun 1 and rerun 2 failing identically is ONE unfixed defect retried.

    Counting it as a recurrence would report looping on every cycle that
    took more than one attempt, which is most of them.
    """
    w = _window(
        [
            ("a", RunStatus.FAILED, "c1", 0, "MorningEnrich", "E", ["MorningEnrich"]),
            ("b", RunStatus.FAILED, "c1", 5, "MorningEnrich", "E", ["MorningEnrich"]),
        ]
    )
    assert w.cycles[0].repeat_causes == []
    assert w.looping is False


def test_peeling_the_onion_reads_as_progress_not_looping():
    """Each cycle failing on a NEW cause is the good shape, and it is red."""
    w = _window(
        [
            ("a", RunStatus.FAILED, "c1", 0, "MorningEnrich", "E1", ["MorningEnrich"]),
            ("b", RunStatus.FAILED, "c2", 10, "DataPhase1", "E2", ["MorningEnrich", "DataPhase1"]),
            (
                "c",
                RunStatus.FAILED,
                "c3",
                20,
                "PredictorTraining",
                "E3",
                ["MorningEnrich", "DataPhase1", "RAGIngestion", "PredictorTraining"],
            ),
        ]
    )
    assert all(not c.repeat_causes for c in w.cycles)
    assert w.looping is False
    assert [d for _, d in w.depth_trend] == [0, 1, 3]


def test_undiagnosable_failures_are_counted_not_folded_into_the_verdict():
    w = _window(
        [
            ("a", RunStatus.FAILED, "c1", 0, None, "E", ["MorningEnrich"]),
            ("b", RunStatus.FAILED, "c2", 10, None, "E", ["MorningEnrich"]),
        ]
    )
    assert w.unresolved_attempts == 2
    assert w.looping is False, "two unknown causes are not one cause recurring"
    assert w.cause_frequency() == {}


def test_looping_is_none_when_nothing_has_settled():
    """Not False — 'no finished cycle' and 'no repeat' are different states."""
    w = _window([("a", RunStatus.RUNNING, "c1", 0, None, None, ["MorningEnrich"])])
    assert w.looping is None


# ── Clean streak ─────────────────────────────────────────────────────────


def test_clean_streak_counts_trailing_first_attempt_successes():
    w = _window(
        [
            ("bad", RunStatus.FAILED, "c0", 0, "MorningEnrich", "E", ["MorningEnrich"]),
            ("g1", RunStatus.SUCCEEDED, "c1", 10, None, None, list(_STAGES)),
            ("g2", RunStatus.SUCCEEDED, "c2", 20, None, None, list(_STAGES)),
            ("g3", RunStatus.SUCCEEDED, "c3", 30, None, None, list(_STAGES)),
        ]
    )
    assert w.clean_streak == 3


def test_a_recovered_cycle_breaks_the_clean_streak():
    """Succeeded-on-rerun is not a clean cycle. That is the point of the metric."""
    w = _window(
        [
            ("g1", RunStatus.SUCCEEDED, "c1", 0, None, None, list(_STAGES)),
            ("f", RunStatus.FAILED, "c2", 10, "MorningEnrich", "E", ["MorningEnrich"]),
            ("r", RunStatus.SUCCEEDED, "c2", 20, None, None, list(_STAGES)),
        ]
    )
    assert w.clean_streak == 0


# ── Window trimming ──────────────────────────────────────────────────────


def test_trimming_happens_after_labelling_so_old_causes_still_read_as_repeats():
    """Trimming first would relabel a recurrence as new and invent progress."""
    w = _window(
        [
            ("a", RunStatus.FAILED, "c1", 0, "MorningEnrich", "E", ["MorningEnrich"]),
            ("b", RunStatus.SUCCEEDED, "c2", 10, None, None, list(_STAGES)),
            ("c", RunStatus.FAILED, "c3", 20, "MorningEnrich", "E", ["MorningEnrich"]),
        ],
        max_cycles=2,
    )
    assert [c.cycle_key for c in w.cycles] == ["c2", "c3"]
    assert w.cycles[-1].repeat_causes, "the c1 occurrence is outside the window but still counts"


def test_an_execution_with_no_cycle_key_is_dropped_not_given_its_own():
    """A director-verify or smoke run must not dilute attempts-to-success."""
    w = _window(
        [
            ("real", RunStatus.SUCCEEDED, "c1", 0, None, None, list(_STAGES)),
            ("smoke", RunStatus.FAILED, None, 5, "X", "E", []),
        ]
    )
    assert [c.cycle_key for c in w.cycles] == ["c1"]
    assert w.clean_streak == 1


# ── The real history ─────────────────────────────────────────────────────


def test_the_2026_08_weekly_history_renders_the_story_that_took_a_day_to_find():
    """Replays `ne-weekly-freshness-pipeline`, 2026-07-25 → 08-11.

    Attempt counts measured from `aws stepfunctions list-executions`. The
    shape this asserts is exactly what had to be reconstructed by hand to
    answer "are we making progress": a long bad stretch, four consecutive
    first-attempt successes, then a regression coincident with the
    2026-08-09 per-stage spot cutover.
    """
    # (cycle_key, attempts, first_attempt_succeeded, any_success)
    history = [
        ("2026-07-25", 11, False, False),
        ("2026-07-26", 16, False, True),
        ("2026-07-27", 5, False, True),
        ("2026-07-29", 1, False, False),
        ("2026-07-30", 4, True, True),
        ("2026-07-31", 4, True, True),
        ("2026-08-01", 7, False, True),
        ("2026-08-03", 2, False, True),
        ("2026-08-04", 1, True, True),
        ("2026-08-06", 1, True, True),
        ("2026-08-07", 2, True, True),
        ("2026-08-08", 1, True, True),
        ("2026-08-10", 5, False, False),
    ]
    specs = []
    minute = 0
    for key, attempts, first_ok, any_ok in history:
        for i in range(attempts):
            first = i == 0
            last = i == attempts - 1
            ok = (first and first_ok) or (last and any_ok)
            specs.append(
                (
                    f"{key}-{i}",
                    RunStatus.SUCCEEDED if ok else RunStatus.FAILED,
                    key,
                    minute,
                    None if ok else "MorningEnrich",
                    None if ok else f"E-{key}",
                    list(_STAGES) if ok else ["MorningEnrich"],
                )
            )
            minute += 1

    w = _window(specs)
    by_key = {c.cycle_key: c for c in w.cycles}

    # The four-cycle clean run nothing rendered.
    for key in ("2026-08-04", "2026-08-06", "2026-08-07", "2026-08-08"):
        assert by_key[key].first_attempt_succeeded is True, key
        assert by_key[key].attempts_to_success == 1, key

    # The regression that followed the 08-09 cutover.
    assert by_key["2026-08-10"].first_attempt_succeeded is False
    assert by_key["2026-08-10"].attempts_to_success is None
    assert by_key["2026-08-10"].attempt_count == 5

    # The streak is 0 as of the last cycle, and WAS 4 one cycle earlier.
    assert w.clean_streak == 0
    # Re-projected without the regression cycle. `entered_states_of` must
    # keep using each attempt's own state list: alpha-engine-config-I8045
    # made "succeeded" mean "entered the whole declared spine", so a stub
    # returning one stage for every attempt would report the four clean
    # cycles as partial — the same blindness, inverted.
    earlier_specs = [s for s in specs if s[2] != "2026-08-10"]
    earlier_by_name = {s[0]: s for s in earlier_specs}
    earlier = build_reliability_window(
        [_summary(s[0], s[1], minutes=s[3]) for s in earlier_specs],
        cycle_key_of=lambda s: s.name.rsplit("-", 1)[0],
        failure_of=lambda s: (earlier_by_name[s.name][4], earlier_by_name[s.name][5]),
        entered_states_of=lambda s: earlier_by_name[s.name][6],
        stage_order=_STAGES,
    )
    assert earlier.clean_streak == 4

    # Worst cycle by operator cost is 07-26 at 16 attempts — the number that
    # makes the improvement legible.
    assert max(w.cycles, key=lambda c: c.attempt_count).cycle_key == "2026-07-26"


# ── The boto-backed adapter ──────────────────────────────────────────────


class _FakeSFN:
    """Minimal stepfunctions double: list + describe + history."""

    def __init__(self, executions, inputs, histories):
        self._executions = executions
        self._inputs = inputs
        self._histories = histories
        self.history_calls = 0

    def list_executions(self, **kwargs):
        return {"executions": self._executions}

    def describe_execution(self, *, executionArn):
        name = executionArn.rsplit(":", 1)[-1]
        return {
            "executionArn": executionArn,
            "name": name,
            "status": next(e["status"] for e in self._executions if e["executionArn"] == executionArn),
            "startDate": next(e["startDate"] for e in self._executions if e["executionArn"] == executionArn),
            "input": self._inputs.get(name, "{}"),
        }

    def get_execution_history(self, *, executionArn, **kwargs):
        self.history_calls += 1
        return {"events": self._histories.get(executionArn.rsplit(":", 1)[-1], [])}


def _entered(name):
    return {"type": "TaskStateEntered", "taskStateEnteredEventDetails": {"name": name}}


def _make_client():
    arn = "arn:aws:states:us-east-1:1:execution:sf:"
    executions = [
        {
            "executionArn": arn + "sched-uuid",
            "name": "sched-uuid",
            "status": "FAILED",
            "startDate": _T0,
        },
        {
            "executionArn": arn + "watch-rerun-2026-08-10-1",
            "name": "watch-rerun-2026-08-10-1",
            "status": "SUCCEEDED",
            "startDate": _T0 + timedelta(hours=1),
        },
    ]
    inputs = {"sched-uuid": '{"pipeline_role":"weekly","run_date":"2026-08-10"}'}
    histories = {
        "sched-uuid": [
            _entered("MorningEnrich"),
            {"type": "TaskFailed", "taskFailedEventDetails": {"error": "States.TaskFailed"}},
        ],
        # A SUCCEEDED rerun must enter the whole declared spine to count as
        # one (I8045) — a rerun that succeeded two stages in did not recover
        # the cycle, and before I8045 it read as if it had.
        "watch-rerun-2026-08-10-1": [_entered(name) for name in _STAGES],
    }
    return _FakeSFN(executions, inputs, histories)


def test_the_adapter_groups_a_scheduled_run_and_its_rerun_into_one_cycle():
    """The scheduled run carries run_date in its input; the rerun in its NAME.

    Reading only the input would drop every operator rerun into "no cycle"
    and destroy attempts-per-cycle, which is the metric.
    """
    from nousergon_lib.pipeline_status import read_reliability_window

    client = _make_client()
    w = read_reliability_window("arn:aws:states:us-east-1:1:stateMachine:sf", stage_order=_STAGES, client=client)
    assert [c.cycle_key for c in w.cycles] == ["2026-08-10"]
    cycle = w.cycles[0]
    assert cycle.attempt_count == 2
    assert cycle.attempts_to_success == 2
    assert cycle.first_attempt_succeeded is False
    assert cycle.depth_stage == _STAGES[-1]


def test_the_adapter_reads_each_history_once():
    """Depth and failure-cause both need the history; paying twice per
    execution doubles an already O(scan_limit) call budget."""
    from nousergon_lib.pipeline_status import read_reliability_window

    client = _make_client()
    read_reliability_window("arn:aws:states:us-east-1:1:stateMachine:sf", stage_order=_STAGES, client=client)
    assert client.history_calls == 2, "one per execution, not one per accessor"


def test_run_date_is_read_from_the_input_before_the_name():
    from nousergon_lib.pipeline_status.read import _extract_run_date

    assert _extract_run_date({"input": '{"run_date":"2026-08-10"}', "name": "watch-rerun-2026-08-11-1"}) == "2026-08-10"
    assert _extract_run_date({"name": "watch-rerun-2026-08-11-1"}) == "2026-08-11"
    assert _extract_run_date({"name": "director-verify-uuid"}) is None
