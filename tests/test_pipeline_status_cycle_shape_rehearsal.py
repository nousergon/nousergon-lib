"""Rehearsal executions never join a production cycle — ``alpha-engine-config-I12382``.

Measured read-only 2026-10-17 08:00Z on lib v0.124.172::

    read_cycle_shape(ne-weekly-freshness-pipeline, "2026-10-16", calendar_date="2026-10-17")
    -> incomplete (partial_cycle), stages 8/16 across 3 executions
       contributors: Fri gate-skip (weekly, skipped),
                     rehearsal-2026-10-16-1 (watch-rerun, FAILED),
                     rehearsal-2026-10-16-2 (watch-rerun, SUCCEEDED)

Every entered stage came from the two rehearsals. They got in twice over:
their cycle key came from the NAME's date (their input carried
``calendar_date 2026-10-15``), and their ``watch-rerun`` role is a recovery
role the union admits. The fake SFN below replays that shape.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from nousergon_lib.pipeline_status.cycle_shape import (
    REHEARSAL_EXECUTION_PREFIX,
    CycleVerdict,
    build_cycle_shape,
    is_rehearsal_execution,
    read_cycle_shape,
)
from nousergon_lib.pipeline_status.read import RunStatus
from nousergon_lib.pipeline_status.work import classify_work

ARN = "arn:aws:states:us-east-1:1:stateMachine:ne-weekly-freshness-pipeline"
PIPELINE = "ne-weekly-freshness-pipeline"
SPINE = ["RAGIngestion", "Scanner", "Backtester", "ReportCard"]

_REHEARSAL_INPUT = {
    "pipeline_role": "watch-rerun",
    "calendar_date": "2026-10-15",
    "rehearsal": "2026-10-16-agent-launched",
    "rehearsal_deadline": "2026-10-17T07:00:00Z",
    "skip_parity": True,
}


def _exec_arn(name: str) -> str:
    return f"arn:aws:states:us-east-1:1:execution:{PIPELINE}:{name}"


class _WeekSFN:
    """The 2026-10-16 cycle: a Friday gate tick plus whatever extra runs a test adds.

    ``fri-gate`` has an opaque name and no ``run_date``, so its key is its
    start date, 2026-10-16. Each extra execution is ``(name, status, input,
    entered)``. ``history_reads`` records which executions' histories were
    paged, so a test can prove a rehearsal was dropped BEFORE the history read.
    """

    def __init__(self, extra: list[tuple[str, str, dict[str, Any], list[str]]]):
        self._rows: dict[str, dict[str, Any]] = {}
        self._history: dict[str, list[str]] = {}
        # Newest first, as ListExecutions returns them.
        for i, (name, status, payload, entered) in enumerate(extra):
            self._rows[name] = {
                "status": status,
                "startDate": datetime(2026, 10, 16, 23 - i, 0, tzinfo=timezone.utc),
                "input": json.dumps(payload),
            }
            self._history[name] = entered
        self._rows["fri-gate"] = {
            "status": "SUCCEEDED",
            "startDate": datetime(2026, 10, 16, 9, 0, tzinfo=timezone.utc),
            "input": json.dumps({"pipeline_role": "weekly"}),
        }
        self._history["fri-gate"] = ["WeeklyRunDayGate", "WeeklyRunDaySkip"]
        self.history_reads: list[str] = []

    def list_executions(self, **_: Any) -> dict[str, Any]:
        return {
            "executions": [
                {"executionArn": _exec_arn(n), "name": n, **{k: r[k] for k in ("status", "startDate")}}
                for n, r in self._rows.items()
            ]
        }

    def describe_execution(self, *, executionArn: str) -> dict[str, Any]:
        name = executionArn.rsplit(":", 1)[-1]
        return {"executionArn": executionArn, "name": name, **self._rows[name]}

    def get_execution_history(self, *, executionArn: str, **_: Any) -> dict[str, Any]:
        name = executionArn.rsplit(":", 1)[-1]
        self.history_reads.append(name)
        return {
            "events": [
                {"type": "TaskStateEntered", "stateEnteredEventDetails": {"name": s}} for s in self._history[name]
            ]
        }


def _replay_20261016() -> _WeekSFN:
    return _WeekSFN(
        [
            ("rehearsal-2026-10-16-2", "SUCCEEDED", _REHEARSAL_INPUT, list(SPINE)),
            ("rehearsal-2026-10-16-1", "FAILED", _REHEARSAL_INPUT, ["RAGIngestion", "Scanner"]),
        ]
    )


def test_the_2026_10_16_rehearsals_are_not_contributors_to_the_production_cycle():
    sfn = _replay_20261016()
    shape = read_cycle_shape(ARN, "2026-10-16", calendar_date="2026-10-17", client=sfn, stage_spine=SPINE)

    assert [e.execution_name for e in shape.executions] == ["fri-gate"]
    # Nothing the rehearsals entered is credited to the production cycle.
    assert shape.stages_entered == ()
    assert shape.stage_coverage == "0/4"
    assert shape.verdict is not CycleVerdict.COMPLETED
    assert not shape.is_recovery_tail
    # Dropped visibly, oldest-first, and before paying for their history.
    assert shape.rehearsals_excluded == ("rehearsal-2026-10-16-1", "rehearsal-2026-10-16-2")
    assert sfn.history_reads == ["fri-gate"]


def test_a_rehearsal_cannot_complete_a_production_cycle_that_skipped_a_stage():
    """The impact the issue names: a Saturday run that skipped a stage reading
    COMPLETED because a rehearsal had entered it."""
    sfn = _WeekSFN(
        [
            ("rehearsal-2026-10-16-1", "SUCCEEDED", _REHEARSAL_INPUT, ["ReportCard"]),
            (
                "sat-scheduled",
                "SUCCEEDED",
                {"pipeline_role": "weekly", "run_date": "2026-10-16"},
                ["RAGIngestion", "Scanner", "Backtester"],
            ),
        ]
    )
    shape = read_cycle_shape(ARN, "2026-10-16", client=sfn, stage_spine=SPINE)

    assert shape.verdict is CycleVerdict.INCOMPLETE
    assert shape.stages_missing == ("ReportCard",)
    assert {e.execution_name for e in shape.executions} == {"fri-gate", "sat-scheduled"}


def test_an_input_flagged_rehearsal_is_excluded_even_without_the_name_prefix():
    payload = {**_REHEARSAL_INPUT}
    sfn = _WeekSFN([("watch-rerun-2026-10-16-9", "SUCCEEDED", payload, list(SPINE))])
    shape = read_cycle_shape(ARN, "2026-10-16", client=sfn, stage_spine=SPINE)

    assert [e.execution_name for e in shape.executions] == ["fri-gate"]
    assert shape.rehearsals_excluded == ("watch-rerun-2026-10-16-9",)


def test_a_run_mode_rehearsal_is_excluded():
    payload = {"pipeline_role": "recovery", "run_mode": "rehearsal"}
    sfn = _WeekSFN([("manual-2026-10-16", "SUCCEEDED", payload, list(SPINE))])
    shape = read_cycle_shape(ARN, "2026-10-16", client=sfn, stage_spine=SPINE)

    assert shape.stages_entered == ()
    assert shape.rehearsals_excluded == ("manual-2026-10-16",)


def test_a_genuine_recovery_rerun_still_completes_the_cycle():
    """The exclusion must not cost a real recovery its place in the union."""
    sfn = _WeekSFN(
        [
            (
                "watch-rerun-2026-10-16-1",
                "SUCCEEDED",
                {"pipeline_role": "watch-rerun", "run_date": "2026-10-16", "rehearsal": False},
                list(SPINE),
            )
        ]
    )
    shape = read_cycle_shape(ARN, "2026-10-16", client=sfn, stage_spine=SPINE)

    assert shape.verdict is CycleVerdict.COMPLETED
    assert shape.rehearsals_excluded == ()


def test_a_sweep_running_inside_a_rehearsal_still_sees_its_own_row():
    """The weekly SF's ``CheckExecutionDidWork`` reads the observer's row; a
    rehearsal's own sweep must keep it. Other rehearsals stay out."""
    sfn = _replay_20261016()
    observer = _exec_arn("rehearsal-2026-10-16-2")
    shape = read_cycle_shape(ARN, "2026-10-16", client=sfn, stage_spine=SPINE, observer_execution_arn=observer)

    names = [e.execution_name for e in shape.executions]
    assert "rehearsal-2026-10-16-2" in names
    assert "rehearsal-2026-10-16-1" not in names
    row = next(e for e in shape.executions if e.execution_name == "rehearsal-2026-10-16-2")
    assert row.is_observer is True
    assert shape.rehearsals_excluded == ("rehearsal-2026-10-16-1",)


def test_the_pure_fold_drops_a_rehearsal_by_name():
    def outcome(name: str, entered: list[str]):
        return classify_work(
            state_machine_name=PIPELINE,
            status=RunStatus.SUCCEEDED,
            entered_states=entered,
            execution_arn=_exec_arn(name),
            execution_name=name,
            stage_spine=SPINE,
        )

    shape = build_cycle_shape(
        pipeline=PIPELINE,
        run_date="2026-10-16",
        outcomes=[
            (outcome("sat", SPINE[:3]), "weekly", SPINE[:3]),
            (outcome("rehearsal-2026-10-16-1", SPINE), "watch-rerun", SPINE),
        ],
        stage_spine=SPINE,
    )
    assert shape.verdict is CycleVerdict.INCOMPLETE
    assert shape.rehearsals_excluded == ("rehearsal-2026-10-16-1",)


def test_the_exclusion_reaches_the_marker_payload_and_the_explanation():
    shape = read_cycle_shape(ARN, "2026-10-16", client=_replay_20261016(), stage_spine=SPINE)
    payload = shape.to_dict()

    assert payload["rehearsals_excluded"] == ["rehearsal-2026-10-16-1", "rehearsal-2026-10-16-2"]
    assert payload["execution_count"] == 1
    assert "2 rehearsal execution(s) excluded" in shape.explain()


@pytest.mark.parametrize(
    ("desc", "expected"),
    [
        ({"name": "rehearsal-2026-10-16-1"}, True),
        ({"executionArn": _exec_arn("rehearsal-2026-10-16-1")}, True),
        ({"name": "x", "input": json.dumps({"rehearsal": "2026-10-16-agent-launched"})}, True),
        ({"name": "x", "input": json.dumps({"rehearsal": True})}, True),
        ({"name": "x", "input": json.dumps({"run_mode": "Rehearsal"})}, True),
        ({"name": "x", "input": json.dumps({"rehearsal": False})}, False),
        ({"name": "x", "input": json.dumps({"run_mode": "production"})}, False),
        ({"name": "x", "input": "{not json"}, False),
        ({"name": "x", "input": json.dumps(["rehearsal"])}, False),
        ({"name": "watch-rerun-2026-10-16-1", "input": json.dumps({"pipeline_role": "watch-rerun"})}, False),
    ],
)
def test_is_rehearsal_execution(desc, expected):
    assert is_rehearsal_execution(desc) is expected


def test_the_prefix_matches_the_phase_registry_literal():
    pytest.importorskip("yaml")
    from nousergon_lib.phase_registry import REHEARSAL_EXECUTION_PREFIX as PHASE_PREFIX

    assert REHEARSAL_EXECUTION_PREFIX == PHASE_PREFIX
