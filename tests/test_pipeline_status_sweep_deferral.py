"""The sweep may not assert an absence a cycle cannot support.

``alpha-engine-config-I10161``. Every number in this file is a VERBATIM
measurement of the live 2026-09-04 weekly cycle, taken 2026-09-08:

- ``s3://alpha-engine-research/_stage_coverage/_sweep/ne-weekly-freshness-pipeline/2026-09-04.json``
  recorded ``counts {expected: 43, covered: 26, findings: 3, absent: 13,
  unmeasured: 1, not_entered: 9}``, ``should_alert: true``, and — in the SAME
  object — ``cycle.verdict: in_flight`` / ``cycle.reason: still_running`` with
  ``cycle.walk_exhausted: true``.
- ``swept_at`` was ``2026-09-05T21:39:18Z``. ``DescribeExecution`` on
  ``watch-rerun-2026-09-04-4`` — the execution the sweep reported RUNNING —
  returns ``startDate 2026-09-05T21:34:07Z``, ``stopDate 2026-09-05T21:39:19Z``,
  ``status SUCCEEDED``. The sweep is a state INSIDE that execution, so it
  graded the cycle ONE SECOND before the cycle's last execution terminated,
  and could never have done otherwise.
- The 2026-08-28 cycle's artifact carries the same ``in_flight`` /
  ``walk_exhausted: true`` pair, so this is the mechanism's normal state
  rather than an incident.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from nousergon_lib.pipeline_status.coverage import (
    RowState,
    publish_sweep,
    sweep_coverage,
)
from nousergon_lib.pipeline_status.cycle_shape import (
    CycleVerdict,
    build_cycle_shape,
    read_cycle_shape,
)
from nousergon_lib.pipeline_status.read import RunStatus
from nousergon_lib.pipeline_status.work import classify_work

PIPELINE = "ne-weekly-freshness-pipeline"
RUN_DATE = "2026-09-04"
SPINE = ("A", "B", "C")
OBSERVER = (
    "arn:aws:states:us-east-1:711398986525:execution:"
    "ne-weekly-freshness-pipeline:watch-rerun-2026-09-04-4"
)
NOW = datetime(2026, 9, 5, 21, 39, 18, tzinfo=timezone.utc)


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


# ── The observer ─────────────────────────────────────────────────────────────


def test_the_execution_the_sweep_runs_inside_does_not_make_its_cycle_in_flight():
    """THE headline defect, in one assertion.

    ``WeeklyCoverageSweep`` is a state of ``ne-weekly-freshness-pipeline``,
    placed immediately after ``WriteCompletionMarker``. Its own execution is
    therefore RUNNING at the instant it grades, on every run, forever — so
    without naming the observer the answer is ``in_flight`` every week and
    ``IN_FLIGHT`` can never mean what it says.
    """
    outcomes = [
        (_outcome(RunStatus.FAILED, ["A"], name="earlier", arn="arn:earlier"), "weekly", ["A"]),
        (_outcome(RunStatus.RUNNING, ["B"], name="rerun-4", arn=OBSERVER), "watch-rerun", ["B"]),
    ]
    without = build_cycle_shape(
        pipeline=PIPELINE, run_date=RUN_DATE, outcomes=outcomes, stage_spine=SPINE
    )
    assert without.verdict is CycleVerdict.IN_FLIGHT
    assert without.is_terminal is False

    with_observer = build_cycle_shape(
        pipeline=PIPELINE,
        run_date=RUN_DATE,
        outcomes=outcomes,
        stage_spine=SPINE,
        observer_execution_arn=OBSERVER,
    )
    assert with_observer.verdict is not CycleVerdict.IN_FLIGHT
    assert with_observer.is_terminal is True
    assert [e.is_observer for e in with_observer.executions] == [False, True]


def test_the_observers_entered_states_still_count():
    """Excluding the observer's STATUS never excludes its work."""
    shape = build_cycle_shape(
        pipeline=PIPELINE,
        run_date=RUN_DATE,
        outcomes=[
            (_outcome(RunStatus.RUNNING, ["A", "B", "C"], arn=OBSERVER), "watch-rerun", ["A", "B", "C"]),
        ],
        stage_spine=SPINE,
        observer_execution_arn=OBSERVER,
    )
    assert shape.stages_entered == ("A", "B", "C")
    assert shape.verdict is CycleVerdict.COMPLETED


def test_another_running_execution_still_yields_in_flight():
    """The carve-out is one ARN wide. A cycle genuinely still running must
    still say so, or the fix has bought a false green for a false red."""
    shape = build_cycle_shape(
        pipeline=PIPELINE,
        run_date=RUN_DATE,
        outcomes=[
            (_outcome(RunStatus.RUNNING, ["A"], arn=OBSERVER), "watch-rerun", ["A"]),
            (_outcome(RunStatus.RUNNING, ["B"], arn="arn:other"), "watch-rerun", ["B"]),
        ],
        stage_spine=SPINE,
        observer_execution_arn=OBSERVER,
    )
    assert shape.verdict is CycleVerdict.IN_FLIGHT
    assert shape.is_terminal is False


# ── The sweep's own stage ────────────────────────────────────────────────────


REGISTRY = {
    "pipeline_stages": [
        {"stage": "A", "stage_class": "product", "output": "registered"},
        {"stage": "WeeklyCoverageSweep", "stage_class": "control", "output": "registered"},
    ]
}


def test_the_sweeps_own_stage_is_never_absent():
    """``WeeklyCoverageSweep`` was 1 of the 13 absences on 2026-09-04. It is
    executing while it grades and has written no verdict BY CONSTRUCTION —
    an absence guaranteed on every run is not evidence of anything."""
    sweep = sweep_coverage(
        pipeline=PIPELINE,
        run_date=RUN_DATE,
        registry=REGISTRY,
        verdicts={"A": {"status": "COVERED"}},
        entered_states=["A", "WeeklyCoverageSweep"],
        observer_stage="WeeklyCoverageSweep",
        now=NOW,
    )
    states = {r.stage: r.state for r in sweep.rows}
    assert states["WeeklyCoverageSweep"] is RowState.SELF
    assert sweep.absent == 0
    assert sweep.counts["expected"] == 1  # SELF is outside the denominator
    assert sweep.should_alert is False


def test_a_real_verdict_for_the_observer_stage_is_graded_normally():
    """The carve-out covers the impossible case only. An earlier execution of
    the same cycle CAN have written the sweep's verdict, and that verdict is
    evidence like any other."""
    sweep = sweep_coverage(
        pipeline=PIPELINE,
        run_date=RUN_DATE,
        registry=REGISTRY,
        verdicts={
            "A": {"status": "COVERED"},
            "WeeklyCoverageSweep": {"status": "MISSING", "is_finding": True, "missing": ["x"]},
        },
        entered_states=["A", "WeeklyCoverageSweep"],
        observer_stage="WeeklyCoverageSweep",
        now=NOW,
    )
    states = {r.stage: r.state for r in sweep.rows}
    assert states["WeeklyCoverageSweep"] is RowState.FINDING
    assert sweep.findings == 1


# ── Deferral ─────────────────────────────────────────────────────────────────


def _sweep_with_cycle(cycle):
    return sweep_coverage(
        pipeline=PIPELINE,
        run_date=RUN_DATE,
        registry=REGISTRY,
        verdicts={},
        entered_states=["A"],
        cycle=cycle,
        observer_stage="WeeklyCoverageSweep",
        now=NOW,
    )


def _cycle(status, *, walk_exhausted=False, entered=("A",), observer=None):
    return build_cycle_shape(
        pipeline=PIPELINE,
        run_date=RUN_DATE,
        outcomes=[(_outcome(status, list(entered), arn="arn:x"), "weekly", list(entered))],
        stage_spine=SPINE,
        walk_exhausted=walk_exhausted,
        observer_execution_arn=observer,
    )


def test_an_in_flight_cycle_defers_the_absent_claim_but_still_pages():
    """The 2026-09-04 shape. ABSENT means "expected, entered, recorded
    nothing" — a claim only a cycle that has stopped changing can support.
    Deferral is NOT silence: the sweep still pages, and it names every
    would-be-absent stage so the page stays diagnosable."""
    sweep = _sweep_with_cycle(_cycle(RunStatus.RUNNING))
    assert sweep.deferred is True
    assert sweep.coverage_established is False
    assert "in_flight" in sweep.deferral_reason
    assert sweep.absent_stages == ("A",)
    assert sweep.should_alert is True
    joined = " ".join(sweep.alert_conditions)
    assert "coverage_deferred" in joined
    assert "WITHHELD, not zero" in joined
    assert "absent_verdicts=" not in joined


def test_a_truncated_walk_on_a_non_completed_cycle_defers():
    """The downgrade ``cycle_shape`` documents and no caller applied: an
    unseen contributor is indistinguishable from a stage that never ran."""
    sweep = _sweep_with_cycle(_cycle(RunStatus.FAILED, walk_exhausted=True))
    assert sweep.cycle.verdict_trustworthy is False
    assert sweep.deferred is True
    assert "walk hit its cap" in sweep.deferral_reason


def test_a_truncated_walk_on_a_completed_cycle_does_not_defer():
    """The union only GROWS with more contributors, so COMPLETED survives a
    truncated walk. Deferring on it would make ABSENT unreachable again."""
    sweep = _sweep_with_cycle(
        _cycle(RunStatus.SUCCEEDED, walk_exhausted=True, entered=("A", "B", "C"))
    )
    assert sweep.cycle.verdict is CycleVerdict.COMPLETED
    assert sweep.deferred is False


def test_a_terminal_cycle_still_pages_absences_loudly():
    """The detector must not have been weakened into silence."""
    sweep = _sweep_with_cycle(_cycle(RunStatus.SUCCEEDED, entered=("A", "B", "C")))
    assert sweep.deferred is False
    assert sweep.absent == 1
    assert sweep.should_alert is True
    assert "absent_verdicts=1 (A)" in " ".join(sweep.alert_conditions)


def test_no_cycle_at_all_does_not_defer():
    """A sweep with no state machine ARN already pages under
    ``denominator_unestablished``; a second reason would double-report it."""
    sweep = _sweep_with_cycle(None)
    assert sweep.deferred is False


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


def test_a_deferred_sweep_withholds_the_absent_datapoint_and_says_so():
    """Not zeroed — WITHHELD. A 0 would render an unestablished surface
    green, and the count the alarm pages on is one the sweep cannot stand
    behind. ``StageCoverageSweepDeferred`` carries the fact instead."""
    metrics = _metrics(_sweep_with_cycle(_cycle(RunStatus.RUNNING)))
    assert "StageCoverageSweepAbsent" not in metrics
    assert metrics["StageCoverageSweepDeferred"] == 1.0
    assert metrics["StageCoverageSweepRan"] == 1.0


def test_an_established_sweep_publishes_absent_and_a_zero_deferred():
    metrics = _metrics(_sweep_with_cycle(_cycle(RunStatus.SUCCEEDED, entered=("A", "B", "C"))))
    assert metrics["StageCoverageSweepAbsent"] == 1.0
    assert metrics["StageCoverageSweepDeferred"] == 0.0


# ── The walk's termination condition ─────────────────────────────────────────


class _StubSFN:
    """Newest-first pages, the ordering ``ListExecutions`` guarantees."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.listed = 0
        self.described = 0

    def list_executions(self, **kwargs: Any) -> dict[str, Any]:
        start = int(kwargs.get("nextToken") or 0)
        n = int(kwargs.get("maxResults") or 100)
        page = self.rows[start : start + n]
        self.listed += len(page)
        nxt = start + n
        out: dict[str, Any] = {"executions": page}
        if nxt < len(self.rows):
            out["nextToken"] = str(nxt)
        return out

    def describe_execution(self, executionArn: str) -> dict[str, Any]:  # noqa: N803
        self.described += 1
        day = executionArn.rsplit(":", 1)[-1]
        return {
            "executionArn": executionArn,
            "name": day,
            "status": "SUCCEEDED",
            "startDate": datetime.fromisoformat(day + "T02:00:00+00:00"),
            "stopDate": datetime.fromisoformat(day + "T03:00:00+00:00"),
            "input": "{}",
        }

    def get_execution_history(self, **_: Any) -> dict[str, Any]:
        return {"events": []}


def _rows(days: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "executionArn": f"arn:aws:states:us-east-1:1:execution:{PIPELINE}:{d}",
            "startDate": datetime.fromisoformat(d + "T02:00:00+00:00"),
        }
        for d in days
    ]


def test_the_walk_stops_on_a_date_bound_not_on_the_count_cap():
    """``walk_exhausted: true`` appeared on EVERY weekly sweep because 60 was
    smaller than a week's execution count. Executions come back newest-first,
    so once one started before the cycle window no later page can hold a
    contributor — a correct termination condition, not a budget."""
    days = ["2026-09-04"] + [f"2026-08-{d:02d}" for d in range(31, 0, -1)]
    stub = _StubSFN(_rows(days))
    shape = read_cycle_shape(
        f"arn:aws:states:us-east-1:1:stateMachine:{PIPELINE}",
        "2026-09-04",
        client=stub,
        stage_spine=SPINE,
        lookback_days=3,
    )
    assert shape.walk_exhausted is False
    # 2026-09-04, 09-03..09-01 do not exist; the first row before the floor
    # (2026-08-22) stops the walk, so nothing older is even described.
    assert stub.described < len(days)


def test_the_count_cap_survives_as_a_hard_stop_and_is_still_reported():
    """A date bound that cannot be computed must not become an unbounded
    walk. The cap stays, and reaching it is still stated."""
    days = [f"2026-09-{d:02d}" for d in range(4, 0, -1)] * 40
    stub = _StubSFN(_rows(days))
    shape = read_cycle_shape(
        f"arn:aws:states:us-east-1:1:stateMachine:{PIPELINE}",
        "2026-09-04",
        client=stub,
        stage_spine=SPINE,
        walk_cap=10,
        lookback_days=-1,
    )
    assert shape.walk_exhausted is True
    assert shape.verdict_trustworthy is False


def test_a_completed_verdict_survives_a_truncated_walk() -> None:
    shape = build_cycle_shape(
        pipeline=PIPELINE,
        run_date=RUN_DATE,
        outcomes=[(_outcome(RunStatus.SUCCEEDED, ["A", "B", "C"], arn="arn:x"), "weekly", ["A", "B", "C"])],
        stage_spine=SPINE,
        walk_exhausted=True,
    )
    assert shape.verdict is CycleVerdict.COMPLETED
    assert shape.verdict_trustworthy is True
    assert shape.untrustworthy_reason == ""


def test_a_declared_skip_survives_a_truncated_walk() -> None:
    """A skip tick enters no spine stage, so an unseen contributor cannot
    contradict it: there is nothing for the union to grow into."""
    shape = build_cycle_shape(
        pipeline=PIPELINE,
        run_date=RUN_DATE,
        outcomes=[
            (
                _outcome(RunStatus.SUCCEEDED, ["WeeklyRunDaySkip"], name="skip", arn="arn:x"),
                "weekly",
                ["WeeklyRunDaySkip"],
            )
        ],
        stage_spine=SPINE,
        walk_exhausted=True,
    )
    assert shape.verdict is CycleVerdict.SKIPPED
    assert shape.verdict_trustworthy is True


# ── The closed maps stay total ───────────────────────────────────────────────


def test_every_row_state_has_a_glyph_and_a_render_order():
    """``RowState`` is closed, and two module-level dicts key on it with no
    fall-through. Adding ``SELF`` crashed ``render_rows`` with a ``KeyError``
    on the first live invocation — a closed map is only safe while something
    proves it total."""
    from nousergon_lib.pipeline_status import coverage as cov

    for state in RowState:
        assert state in cov._STATE_GLYPH, state
    rendered = cov.render_rows(
        sweep_coverage(
            pipeline=PIPELINE,
            run_date=RUN_DATE,
            registry=REGISTRY,
            verdicts={"A": {"status": "COVERED"}},
            entered_states=["A", "WeeklyCoverageSweep"],
            observer_stage="WeeklyCoverageSweep",
            now=NOW,
        )
    )
    assert "WeeklyCoverageSweep" in rendered
