"""Unit tests for ``alpha_engine_lib.pipeline_status.read``.

Mocks the boto3 ``stepfunctions`` client and exercises every documented
code path:

- Happy path — SUCCEEDED execution with a representative substantive-state
  set materializes into the expected TaskRow list (filter + Wait-grouping
  applied; durations measured parent_entry → wait_exit).
- FAILED execution — failure_cause extraction matches sf-telegram-notifier
  byte-for-byte; failing_state populated.
- RUNNING execution — no end_utc; tasks with no exit event get TaskStatus.RUNNING.
- Exception paths — AccessDenied / Throttling / no-executions all raise
  typed exceptions.
- Pretty-label resolution + the NOT_RUN sentinel never being returned
  from the live path (it would have raised SFNNoExecutions instead).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from alpha_engine_lib.pipeline_status import (
    PipelineRun,
    RunStatus,
    SFNAccessDenied,
    SFNNoExecutions,
    SFNThrottled,
    TaskRow,
    TaskStatus,
    read_pipeline_state,
)
from alpha_engine_lib.pipeline_status.read import (
    PipelineStatusError,
    _failure_cause_from,
    _materialize_tasks,
    _parse_ts,
    _region_from_arn,
)
from alpha_engine_lib.pipeline_status.registry import (
    ArchivePageRef,
    ArtifactReason,
)


SATURDAY_ARN = (
    "arn:aws:states:us-east-1:711398986525:stateMachine:alpha-engine-saturday-pipeline"
)
EXECUTION_ARN = (
    "arn:aws:states:us-east-1:711398986525:execution:"
    "alpha-engine-saturday-pipeline:test-exec-1"
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_sfn_mock(
    *,
    list_response=None,
    describe_response=None,
    history_response=None,
    list_exc=None,
    describe_exc=None,
    history_exc=None,
):
    """Build a boto3 stepfunctions mock with configurable responses + errors."""
    client = MagicMock()

    if list_exc is not None:
        client.list_executions.side_effect = list_exc
    else:
        client.list_executions.return_value = list_response or {
            "executions": [
                {"executionArn": EXECUTION_ARN, "name": "test-exec-1"}
            ]
        }

    if describe_exc is not None:
        client.describe_execution.side_effect = describe_exc
    else:
        client.describe_execution.return_value = describe_response or {
            "status": "SUCCEEDED",
            "startDate": datetime(2026, 5, 24, 9, 0, tzinfo=timezone.utc),
            "stopDate": datetime(2026, 5, 24, 11, 30, tzinfo=timezone.utc),
        }

    if history_exc is not None:
        client.get_execution_history.side_effect = history_exc
    else:
        client.get_execution_history.return_value = history_response or {"events": []}

    return client


def _boto_client_error(code: str) -> Exception:
    """Build an exception shaped like botocore.exceptions.ClientError."""
    exc = Exception(f"boto3 simulated {code}")
    exc.response = {"Error": {"Code": code, "Message": "test"}}  # type: ignore[attr-defined]
    return exc


# ── Helpers ───────────────────────────────────────────────────────────────


def _entered(state: str, ts: datetime) -> dict:
    return {
        "type": "TaskStateEntered",
        "timestamp": ts,
        "stateEnteredEventDetails": {"name": state},
    }


def _exited(state: str, ts: datetime) -> dict:
    return {
        "type": "TaskStateExited",
        "timestamp": ts,
        "stateExitedEventDetails": {"name": state},
    }


def _failed(error: str, cause: str) -> dict:
    return {
        "type": "TaskFailed",
        "timestamp": datetime(2026, 5, 24, 10, 0, tzinfo=timezone.utc),
        "taskFailedEventDetails": {"error": error, "cause": cause},
    }


# ── _parse_ts ─────────────────────────────────────────────────────────────


def test_parse_ts_handles_none():
    assert _parse_ts(None) is None


def test_parse_ts_passes_through_utc_aware_datetime():
    dt = datetime(2026, 5, 24, 9, 0, tzinfo=timezone.utc)
    assert _parse_ts(dt) == dt


def test_parse_ts_coerces_naive_datetime_to_utc():
    dt = datetime(2026, 5, 24, 9, 0)
    out = _parse_ts(dt)
    assert out is not None
    assert out.tzinfo == timezone.utc


def test_parse_ts_rejects_non_datetime():
    assert _parse_ts("2026-05-24") is None
    assert _parse_ts(12345) is None


# ── _failure_cause_from ───────────────────────────────────────────────────


def test_failure_cause_from_concatenates_error_and_cause():
    out = _failure_cause_from({"error": "States.TaskFailed", "cause": "exit 1"})
    assert out == "States.TaskFailed: exit 1"


def test_failure_cause_from_handles_error_only():
    assert _failure_cause_from({"error": "BOOM", "cause": ""}) == "BOOM"


def test_failure_cause_from_handles_cause_only():
    assert _failure_cause_from({"error": "", "cause": "bad"}) == "bad"


def test_failure_cause_from_handles_empty():
    assert _failure_cause_from({}) == ""
    assert _failure_cause_from({"error": "", "cause": ""}) == ""


def test_failure_cause_from_truncates_at_280():
    long_cause = "A" * 500
    out = _failure_cause_from({"error": "", "cause": long_cause})
    assert len(out) == 280
    assert out.endswith("…")


# ── _materialize_tasks: substantive-state filter + Wait-grouping ──────────


def test_materialize_tasks_filters_to_registry_entries():
    """States not in the registry are filtered out entirely (Choice / Pass /
    Succeed control-flow plumbing)."""
    base = datetime(2026, 5, 24, 9, 0, tzinfo=timezone.utc)
    events = [
        _entered("CheckTradingDayChoice", base),  # not in registry → filtered
        _exited("CheckTradingDayChoice", base),
        _entered("Research", base),
        _exited("Research", base),
    ]
    rows = _materialize_tasks(events)
    names = [r.state_name for r in rows]
    assert "CheckTradingDayChoice" not in names
    assert "Research" in names


def test_materialize_tasks_absorbs_wait_companion_into_parent():
    """WaitForDataPhase1 rolls into DataPhase1; duration spans
    parent.entered → wait.exited."""
    t0 = datetime(2026, 5, 24, 9, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 5, 24, 9, 0, 30, tzinfo=timezone.utc)  # parent exit @ 30s
    t2 = datetime(2026, 5, 24, 9, 0, 35, tzinfo=timezone.utc)  # wait entered @ 35s
    t3 = datetime(2026, 5, 24, 9, 40, tzinfo=timezone.utc)  # wait exited @ 40m later

    events = [
        _entered("DataPhase1", t0),
        _exited("DataPhase1", t1),
        _entered("WaitForDataPhase1", t2),
        _exited("WaitForDataPhase1", t3),
    ]
    rows = _materialize_tasks(events)

    # Only one row — the parent
    assert len(rows) == 1
    row = rows[0]
    assert row.state_name == "DataPhase1"
    # Duration is t0 → t3 (parent entered to wait exited) = ~2400s
    assert row.duration_sec is not None
    assert row.duration_sec == pytest.approx(2400.0, abs=1.0)


def test_materialize_tasks_running_state_has_no_end():
    """A state with TaskStateEntered but no TaskStateExited renders as
    TaskStatus.RUNNING with no duration."""
    t0 = datetime(2026, 5, 24, 9, 0, tzinfo=timezone.utc)
    events = [_entered("Research", t0)]
    rows = _materialize_tasks(events)
    assert len(rows) == 1
    assert rows[0].status == TaskStatus.RUNNING
    assert rows[0].end_utc is None
    assert rows[0].duration_sec is None


def test_materialize_tasks_failed_state_carries_cause():
    """A TaskFailed event attaches to the most-recent still-open state."""
    t0 = datetime(2026, 5, 24, 9, 0, tzinfo=timezone.utc)
    events = [
        _entered("Research", t0),
        _failed("States.TaskFailed", "exit 1: ArcticDB E_NO_SUCH_VERSION"),
    ]
    rows = _materialize_tasks(events)
    assert len(rows) == 1
    assert rows[0].status == TaskStatus.FAILED
    assert "ArcticDB E_NO_SUCH_VERSION" in (rows[0].failure_cause or "")


def test_materialize_tasks_attaches_registry_entry_to_each_row():
    """Every materialized row carries the registry entry for its state name."""
    t0 = datetime(2026, 5, 24, 9, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 5, 24, 9, 30, tzinfo=timezone.utc)
    events = [
        _entered("Research", t0),
        _exited("Research", t1),
        _entered("DataPhase1", t0),
        _exited("DataPhase1", t1),
    ]
    rows = {r.state_name: r for r in _materialize_tasks(events)}
    assert isinstance(rows["Research"].archive, ArchivePageRef)
    assert rows["Research"].archive.page == "17_Research_Briefing_Archive"
    assert isinstance(rows["DataPhase1"].archive, ArtifactReason)


# ── read_pipeline_state happy path ────────────────────────────────────────


def test_read_pipeline_state_succeeded_happy_path():
    """SUCCEEDED execution with two substantive states materializes
    correctly: top-level fields + tasks list + no failure fields populated."""
    t0 = datetime(2026, 5, 24, 9, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 5, 24, 9, 30, tzinfo=timezone.utc)
    t2 = datetime(2026, 5, 24, 11, 30, tzinfo=timezone.utc)

    client = _make_sfn_mock(
        history_response={
            "events": [
                _entered("DataPhase1", t0),
                _exited("DataPhase1", t1),
                _entered("Research", t1),
                _exited("Research", t2),
            ]
        }
    )
    run = read_pipeline_state(SATURDAY_ARN, client=client)

    assert isinstance(run, PipelineRun)
    assert run.state_machine_arn == SATURDAY_ARN
    assert run.pretty_label == "Saturday SF"
    assert run.execution_arn == EXECUTION_ARN
    assert run.execution_name == "test-exec-1"
    assert run.status == RunStatus.SUCCEEDED
    assert run.start_utc == datetime(2026, 5, 24, 9, 0, tzinfo=timezone.utc)
    assert run.end_utc == datetime(2026, 5, 24, 11, 30, tzinfo=timezone.utc)
    assert run.duration_sec == pytest.approx(2.5 * 3600, abs=1.0)
    assert run.failing_state is None
    assert run.failure_cause is None
    task_names = {t.state_name for t in run.tasks}
    assert task_names == {"DataPhase1", "Research"}


def test_read_pipeline_state_failed_populates_failing_state_and_cause():
    """FAILED execution populates failing_state + failure_cause from the
    DescribeExecution response + history."""
    t0 = datetime(2026, 5, 24, 9, 0, tzinfo=timezone.utc)

    client = _make_sfn_mock(
        describe_response={
            "status": "FAILED",
            "startDate": t0,
            "stopDate": datetime(2026, 5, 24, 9, 30, tzinfo=timezone.utc),
            "error": "States.TaskFailed",
            "cause": "Research Lambda exited 1: ArcticDB error",
        },
        history_response={
            "events": [
                _entered("DataPhase1", t0),
                _exited(
                    "DataPhase1", datetime(2026, 5, 24, 9, 5, tzinfo=timezone.utc)
                ),
                _entered("Research", datetime(2026, 5, 24, 9, 5, tzinfo=timezone.utc)),
                _failed("States.TaskFailed", "Research Lambda exited 1"),
            ]
        },
    )
    run = read_pipeline_state(SATURDAY_ARN, client=client)

    assert run.status == RunStatus.FAILED
    assert run.failing_state == "Research"
    assert run.failure_cause == "States.TaskFailed: Research Lambda exited 1: ArcticDB error"


def test_read_pipeline_state_running_execution():
    """RUNNING execution has stopDate=None → no end_utc + no duration."""
    t0 = datetime(2026, 5, 24, 13, 0, tzinfo=timezone.utc)
    client = _make_sfn_mock(
        describe_response={"status": "RUNNING", "startDate": t0, "stopDate": None},
        history_response={"events": [_entered("MorningEnrich", t0)]},
    )
    run = read_pipeline_state(SATURDAY_ARN, client=client)
    assert run.status == RunStatus.RUNNING
    assert run.end_utc is None
    assert run.duration_sec is None
    # The substantive state is still RUNNING (no exit event)
    me = next(t for t in run.tasks if t.state_name == "MorningEnrich")
    assert me.status == TaskStatus.RUNNING


# ── Exception paths ───────────────────────────────────────────────────────


def test_read_pipeline_state_no_executions_raises():
    client = _make_sfn_mock(list_response={"executions": []})
    with pytest.raises(SFNNoExecutions):
        read_pipeline_state(SATURDAY_ARN, client=client)


def test_read_pipeline_state_list_access_denied_raises_typed():
    exc = _boto_client_error("AccessDeniedException")
    client = _make_sfn_mock(list_exc=exc)
    with pytest.raises(SFNAccessDenied) as exc_info:
        read_pipeline_state(SATURDAY_ARN, client=client)
    assert "ListExecutions" in str(exc_info.value)


def test_read_pipeline_state_describe_throttled_raises_typed():
    exc = _boto_client_error("ThrottlingException")
    client = _make_sfn_mock(describe_exc=exc)
    with pytest.raises(SFNThrottled) as exc_info:
        read_pipeline_state(SATURDAY_ARN, client=client)
    assert "DescribeExecution" in str(exc_info.value)


def test_read_pipeline_state_history_access_denied_raises_typed():
    exc = _boto_client_error("AccessDeniedException")
    client = _make_sfn_mock(history_exc=exc)
    with pytest.raises(SFNAccessDenied) as exc_info:
        read_pipeline_state(SATURDAY_ARN, client=client)
    assert "GetExecutionHistory" in str(exc_info.value)


def test_read_pipeline_state_unknown_boto_error_raises_pipeline_status_error():
    """Unknown boto3 error code → re-raised as PipelineStatusError (NOT
    swallowed silently per feedback_no_silent_fails)."""
    exc = _boto_client_error("WeirdNewError")
    client = _make_sfn_mock(list_exc=exc)
    with pytest.raises(PipelineStatusError) as exc_info:
        read_pipeline_state(SATURDAY_ARN, client=client)
    # PipelineStatusError is the base class; the typed subclasses inherit
    # from it, so we check we don't get a more-specific subclass mistakenly.
    assert type(exc_info.value) is PipelineStatusError


def test_read_pipeline_state_unknown_status_string_raises():
    """boto3 returning a status the lib doesn't recognize MUST fail loud
    (forward-compatibility check — would surface if AWS adds a new status)."""
    client = _make_sfn_mock(
        describe_response={
            "status": "WEIRD_NEW_STATUS",
            "startDate": datetime(2026, 5, 24, 9, 0, tzinfo=timezone.utc),
            "stopDate": None,
        }
    )
    with pytest.raises(PipelineStatusError) as exc_info:
        read_pipeline_state(SATURDAY_ARN, client=client)
    assert "WEIRD_NEW_STATUS" in str(exc_info.value)


# ── Pretty-label resolution ───────────────────────────────────────────────


def test_pretty_label_for_each_known_pipeline():
    for arn, expected in [
        (
            "arn:aws:states:us-east-1:711398986525:stateMachine:alpha-engine-saturday-pipeline",
            "Saturday SF",
        ),
        (
            "arn:aws:states:us-east-1:711398986525:stateMachine:alpha-engine-weekday-pipeline",
            "Weekday SF",
        ),
        (
            "arn:aws:states:us-east-1:711398986525:stateMachine:alpha-engine-eod-pipeline",
            "EOD SF",
        ),
    ]:
        client = _make_sfn_mock()
        run = read_pipeline_state(arn, client=client)
        assert run.pretty_label == expected


def test_pretty_label_falls_back_to_arn_suffix_for_unknown_sf():
    """Unknown SF ARN renders the bare state-machine name (not a generic
    'Unknown SF' string) so an operator can still tell what pipeline they're
    looking at if a new SF appears before the registry is updated."""
    client = _make_sfn_mock()
    arn = "arn:aws:states:us-east-1:711398986525:stateMachine:future-pipeline-name"
    run = read_pipeline_state(arn, client=client)
    assert run.pretty_label == "future-pipeline-name"


# ── Region extraction ─────────────────────────────────────────────────────


def test_region_from_arn_extracts_us_east_1():
    assert _region_from_arn(SATURDAY_ARN) == "us-east-1"


def test_region_from_arn_extracts_non_us_east_1():
    arn = "arn:aws:states:eu-west-2:123456789012:stateMachine:some-pipeline"
    assert _region_from_arn(arn) == "eu-west-2"


def test_region_from_arn_returns_none_for_malformed():
    """Permissive on bad input — boto3 will raise NoRegionError downstream,
    which surfaces as a typed PipelineStatusError per _raise_for_boto_error."""
    assert _region_from_arn("") is None
    assert _region_from_arn("not-an-arn") is None
    assert _region_from_arn("arn:aws:states") is None
    assert _region_from_arn("arn:aws:states::123:stateMachine:x") is None
