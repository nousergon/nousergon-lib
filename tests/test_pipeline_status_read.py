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
    PipelineExecutionSummary,
    PipelineRun,
    RunStatus,
    SFNAccessDenied,
    SFNNoExecutions,
    SFNThrottled,
    TaskRow,
    TaskStatus,
    list_recent_pipeline_runs,
    read_pipeline_state,
)
from alpha_engine_lib.pipeline_status.read import (
    PipelineStatusError,
    _extract_pipeline_role,
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


# ── Archive-union JSON round-trip (regression for "registry drift" false positive) ──


def _entered_and_succeeded(name: str, t0: datetime, t1: datetime) -> list[dict]:
    return [
        _entered(name, t0),
        {
            "type": "TaskStateExited",
            "timestamp": t1,
            "stateExitedEventDetails": {"name": name},
        },
    ]


def test_task_row_archive_round_trips_through_json_for_archive_page_ref():
    """The dashboard's st.cache_data wraps read_pipeline_state by doing
    ``model_dump(mode="json")`` → cache → ``model_validate(dict)``. Before
    this regression-guard, ``TaskRow.archive`` was typed ``Optional[Any]``,
    so the JSON round-trip flattened ArchivePageRef instances to plain
    dicts; page-25's ``isinstance(archive, ArchivePageRef)`` then misfired
    and rendered ``⚠️ Registry drift`` for every state with a valid
    registry entry. The discriminated-union typing
    (``Annotated[Union[...], Field(discriminator='kind')]``) reconstructs
    the typed instance on validate; this test guards the contract."""
    t0 = datetime(2026, 5, 24, 9, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 5, 24, 9, 5, tzinfo=timezone.utc)
    client = _make_sfn_mock(
        history_response={"events": _entered_and_succeeded("Research", t0, t1)}
    )

    run = read_pipeline_state(SATURDAY_ARN, client=client)
    research_task = next(t for t in run.tasks if t.state_name == "Research")
    assert isinstance(research_task.archive, ArchivePageRef)

    # The actual code path that broke production: JSON round-trip.
    round_tripped = PipelineRun.model_validate(run.model_dump(mode="json"))
    round_tripped_task = next(
        t for t in round_tripped.tasks if t.state_name == "Research"
    )
    assert isinstance(round_tripped_task.archive, ArchivePageRef), (
        "TaskRow.archive must reconstruct as ArchivePageRef on JSON "
        "round-trip — otherwise page 25's isinstance check falls through "
        "to the registry-drift sentinel for every state."
    )
    assert round_tripped_task.archive.page == "17_Research_Briefing_Archive"


def test_task_row_archive_round_trips_through_json_for_artifact_reason():
    """Mirrors the ArchivePageRef round-trip but for the ArtifactReason
    variant (substrate-only states like NotifyComplete + Scanner). Both
    variants must reconstruct correctly on JSON round-trip; the
    discriminated union differentiates them via the ``kind`` field."""
    t0 = datetime(2026, 5, 24, 9, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 5, 24, 9, 0, 1, tzinfo=timezone.utc)
    client = _make_sfn_mock(
        history_response={"events": _entered_and_succeeded("NotifyComplete", t0, t1)}
    )

    run = read_pipeline_state(SATURDAY_ARN, client=client)
    notify_task = next(t for t in run.tasks if t.state_name == "NotifyComplete")
    assert isinstance(notify_task.archive, ArtifactReason)

    round_tripped = PipelineRun.model_validate(run.model_dump(mode="json"))
    round_tripped_task = next(
        t for t in round_tripped.tasks if t.state_name == "NotifyComplete"
    )
    assert isinstance(round_tripped_task.archive, ArtifactReason), (
        "TaskRow.archive must reconstruct as ArtifactReason on JSON "
        "round-trip — same regression class as the ArchivePageRef test."
    )
    assert "Terminal success" in round_tripped_task.archive.reason


# ── pipeline_role extraction (Option-D substrate) ─────────────────────────


def test_extract_pipeline_role_happy_path():
    """Standard EventBridge cron payload with pipeline_role set."""
    describe = {
        "input": '{"pipeline_role": "weekly", "run_date": "2026-05-30"}',
    }
    assert _extract_pipeline_role(describe) == "weekly"


def test_extract_pipeline_role_missing_field():
    """Pre-Option-D execution input (no pipeline_role key) returns None."""
    describe = {"input": '{"run_date": "2026-05-30"}'}
    assert _extract_pipeline_role(describe) is None


def test_extract_pipeline_role_missing_input_field():
    """DescribeExecution may omit the input field entirely on terminal
    states (rare but possible) — degrade to None, not crash."""
    assert _extract_pipeline_role({}) is None
    assert _extract_pipeline_role({"input": None}) is None
    assert _extract_pipeline_role({"input": ""}) is None


def test_extract_pipeline_role_malformed_json():
    """Malformed input JSON — WARN-and-return-None per the lib's
    permissive parse policy. Recording surface is the WARN log."""
    describe = {"input": "{not valid json"}
    assert _extract_pipeline_role(describe) is None


def test_extract_pipeline_role_input_is_array_not_object():
    """SF allows array-shaped input; defensively handle it (return None
    rather than raise) — pipeline_role is a top-level field on object
    inputs only."""
    describe = {"input": '["weekly"]'}
    assert _extract_pipeline_role(describe) is None


def test_extract_pipeline_role_empty_string_returns_none():
    """An explicit empty string in pipeline_role is treated as 'not set'
    so the dashboard renders 'role: unknown' instead of '': empty cells
    are operator-noise."""
    describe = {"input": '{"pipeline_role": ""}'}
    assert _extract_pipeline_role(describe) is None


# ── Role filter + execution_arn paths in read_pipeline_state ──────────────


def _make_describe_response(*, status="SUCCEEDED", role: Optional[str] = None) -> dict:
    """Build a DescribeExecution response carrying an optional
    pipeline_role on the input JSON. Default times preserved."""
    body: dict = {
        "status": status,
        "startDate": datetime(2026, 5, 24, 9, 0, tzinfo=timezone.utc),
        "stopDate": datetime(2026, 5, 24, 11, 30, tzinfo=timezone.utc),
    }
    if role is not None:
        body["input"] = f'{{"pipeline_role": "{role}", "run_date": "2026-05-24"}}'
    else:
        body["input"] = '{"run_date": "2026-05-24"}'
    return body


def _make_multi_execution_mock(
    *,
    executions: list[dict],
    describe_by_arn: dict[str, dict],
) -> MagicMock:
    """Build an SFN mock where ListExecutions returns a list and
    DescribeExecution dispatches by executionArn to the right response."""
    client = MagicMock()
    client.list_executions.return_value = {"executions": executions}

    def _dispatch(executionArn: str, **_kwargs):
        return describe_by_arn[executionArn]

    client.describe_execution.side_effect = _dispatch
    client.get_execution_history.return_value = {"events": []}
    return client


def test_read_pipeline_state_default_returns_most_recent_unchanged():
    """No role_filter, no execution_arn — same as pre-Option-D: most-recent
    execution per ListExecutions maxResults=1."""
    client = _make_sfn_mock()
    run = read_pipeline_state(SATURDAY_ARN, client=client)
    assert run.status == RunStatus.SUCCEEDED
    # ListExecutions was called with maxResults=1 (default path).
    client.list_executions.assert_called_once()
    call_kwargs = client.list_executions.call_args.kwargs
    assert call_kwargs.get("maxResults") == 1


def test_read_pipeline_state_with_role_filter_finds_first_match():
    """Three executions in history: smoke / weekly / smoke. Filter to
    'weekly' — picks the middle one."""
    smoke1_arn = EXECUTION_ARN + "-smoke1"
    weekly_arn = EXECUTION_ARN + "-weekly"
    smoke2_arn = EXECUTION_ARN + "-smoke2"
    client = _make_multi_execution_mock(
        executions=[
            {"executionArn": smoke1_arn, "name": "smoke-l1995"},
            {"executionArn": weekly_arn, "name": "weekly-20260524T090000"},
            {"executionArn": smoke2_arn, "name": "smoke-debug"},
        ],
        describe_by_arn={
            smoke1_arn: _make_describe_response(role="smoke"),
            weekly_arn: _make_describe_response(role="weekly"),
            smoke2_arn: _make_describe_response(role="smoke"),
        },
    )
    run = read_pipeline_state(SATURDAY_ARN, role_filter={"weekly"}, client=client)
    assert run.execution_arn == weekly_arn
    assert run.pipeline_role == "weekly"


def test_read_pipeline_state_with_role_filter_no_match_raises():
    """Three smoke executions, filter to 'weekly' — raises
    SFNNoExecutions naming the filter so the caller can render an
    operator-actionable banner."""
    client = _make_multi_execution_mock(
        executions=[
            {"executionArn": EXECUTION_ARN + f"-{i}", "name": f"smoke-{i}"}
            for i in range(3)
        ],
        describe_by_arn={
            EXECUTION_ARN + f"-{i}": _make_describe_response(role="smoke")
            for i in range(3)
        },
    )
    with pytest.raises(SFNNoExecutions) as exc_info:
        read_pipeline_state(
            SATURDAY_ARN, role_filter={"weekly"}, search_limit=10, client=client
        )
    assert "weekly" in str(exc_info.value)


def test_read_pipeline_state_with_role_filter_treats_missing_role_as_no_match():
    """Pre-Option-D executions lack pipeline_role; role_filter must NOT
    match those (otherwise the filter is no filter at all). The walk
    keeps going until an explicitly-tagged execution turns up."""
    untagged_arn = EXECUTION_ARN + "-untagged"
    weekly_arn = EXECUTION_ARN + "-weekly"
    client = _make_multi_execution_mock(
        executions=[
            {"executionArn": untagged_arn, "name": "old-pre-option-d"},
            {"executionArn": weekly_arn, "name": "weekly-20260524T090000"},
        ],
        describe_by_arn={
            untagged_arn: _make_describe_response(role=None),
            weekly_arn: _make_describe_response(role="weekly"),
        },
    )
    run = read_pipeline_state(SATURDAY_ARN, role_filter={"weekly"}, client=client)
    assert run.execution_arn == weekly_arn


def test_read_pipeline_state_with_execution_arn_fetches_specific_execution():
    """Dropdown-click path: when execution_arn is set, the function fetches
    that specific execution directly (bypasses ListExecutions). role_filter
    and search_limit are ignored on this path."""
    target_arn = EXECUTION_ARN + "-specific"
    client = _make_multi_execution_mock(
        executions=[],  # ListExecutions intentionally empty — proves it's not called
        describe_by_arn={target_arn: _make_describe_response(role="smoke")},
    )
    run = read_pipeline_state(SATURDAY_ARN, execution_arn=target_arn, client=client)
    assert run.execution_arn == target_arn
    assert run.pipeline_role == "smoke"
    # ListExecutions must NOT have been called on the execution_arn path.
    client.list_executions.assert_not_called()


def test_read_pipeline_state_carries_pipeline_role_to_returned_run():
    """The pipeline_role field on PipelineRun is populated from input JSON
    even when no role_filter is applied (default path) — the dashboard's
    section header shows it regardless of how the execution was picked."""
    client = _make_sfn_mock(
        describe_response=_make_describe_response(role="weekly"),
    )
    run = read_pipeline_state(SATURDAY_ARN, client=client)
    assert run.pipeline_role == "weekly"


def test_read_pipeline_state_pipeline_role_none_when_input_lacks_role():
    """No pipeline_role in input → PipelineRun.pipeline_role is None
    (rendered as 'role: unknown' on the dashboard)."""
    client = _make_sfn_mock(
        describe_response=_make_describe_response(role=None),
    )
    run = read_pipeline_state(SATURDAY_ARN, client=client)
    assert run.pipeline_role is None


# ── list_recent_pipeline_runs ─────────────────────────────────────────────


def test_list_recent_pipeline_runs_returns_summaries_with_roles():
    """Returns last N executions, each carrying its pipeline_role for the
    operator dropdown's at-a-glance smoke-vs-weekly distinction."""
    arns = [EXECUTION_ARN + f"-{i}" for i in range(5)]
    roles = ["smoke", "weekly", "smoke", "weekly", "recovery"]
    client = _make_multi_execution_mock(
        executions=[
            {"executionArn": a, "name": f"exec-{i}"} for i, a in enumerate(arns)
        ],
        describe_by_arn={a: _make_describe_response(role=r) for a, r in zip(arns, roles)},
    )
    summaries = list_recent_pipeline_runs(SATURDAY_ARN, limit=5, client=client)
    assert len(summaries) == 5
    assert all(isinstance(s, PipelineExecutionSummary) for s in summaries)
    assert [s.pipeline_role for s in summaries] == roles


def test_list_recent_pipeline_runs_role_filter_pre_filters():
    """When role_filter is set, only matching executions are returned —
    the operator's "show me weekly runs only" view."""
    arns = [EXECUTION_ARN + f"-{i}" for i in range(6)]
    roles = ["smoke", "weekly", "smoke", "weekly", "recovery", "weekly"]
    client = _make_multi_execution_mock(
        executions=[
            {"executionArn": a, "name": f"exec-{i}"} for i, a in enumerate(arns)
        ],
        describe_by_arn={a: _make_describe_response(role=r) for a, r in zip(arns, roles)},
    )
    summaries = list_recent_pipeline_runs(
        SATURDAY_ARN, limit=10, role_filter={"weekly"}, client=client
    )
    assert len(summaries) == 3
    assert all(s.pipeline_role == "weekly" for s in summaries)


def test_list_recent_pipeline_runs_empty_returns_empty_list():
    """Zero executions → empty list (NOT SFNNoExecutions). The dropdown
    just renders 'no executions yet' inline; the page-25 section banner
    is the load-bearing error surface, not this lighter-weight API."""
    client = MagicMock()
    client.list_executions.return_value = {"executions": []}
    summaries = list_recent_pipeline_runs(SATURDAY_ARN, limit=5, client=client)
    assert summaries == []
