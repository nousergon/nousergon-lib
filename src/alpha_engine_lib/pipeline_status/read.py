"""SF-state projection — ``read_pipeline_state`` and shapes.

Reads the most-recent execution of a Step Function and projects it onto
the typed :class:`PipelineRun` shape the dashboard page 25 (and any future
Slack/CLI subscriber) consumes.

Three API calls per invocation:

1. ``states:ListExecutions(maxResults=1)`` — find the latest execution arn.
2. ``states:DescribeExecution(executionArn=...)`` — top-level status +
   start/stop timestamps + failure cause when applicable.
3. ``states:GetExecutionHistory(executionArn=..., maxResults=1000)`` —
   per-state entry/exit events. Substantive-state filter + Wait-grouping
   applied in :func:`_materialize_tasks`.

**Exception contract** — every documented error path raises a typed
subclass of :class:`PipelineStatusError` so the dashboard page can switch
on the cause and render the appropriate banner state:

- :class:`SFNAccessDenied` — IAM missing one of states:Describe / Get / List.
  Page renders a red banner naming the missing action.
- :class:`SFNThrottled` — SF API rate-limited. Page falls back to the
  ``pipeline_status_cache.json`` last-good cache with a yellow banner.
- :class:`SFNNoExecutions` — SF has never been executed. Page renders an
  empty section with a "no executions yet" note (NOT an error).

**Never silently degrades** per ``feedback_no_silent_fails`` — unknown
boto3 errors are re-raised as :class:`PipelineStatusError` so the page's
red banner always names a specific cause.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .registry import (
    PIPELINE_LABELS,
    SUBSTANTIVE_RESOURCES,
    WAIT_GROUPING,
    ArchivePageRef,
    ArtifactReason,
    lookup_registry,
)

if TYPE_CHECKING:  # pragma: no cover — type-only import
    from mypy_boto3_stepfunctions.client import SFNClient

logger = logging.getLogger(__name__)

# Failure-cause truncation — kept verbatim with sf-telegram-notifier
# (alpha-engine-data/infrastructure/lambdas/sf-telegram-notifier/index.py
# line 69) so the email + Telegram + page channels render byte-identical
# cause snippets.
_CAUSE_MAX_CHARS = 280

# Bounds the ``getExecutionHistory`` page size. The Saturday SF emits
# ~600-800 history events on a clean run; the page-25 SLA is one round-trip
# per poll, so we want enough headroom not to paginate but not so much we
# load 100k events for a one-state-failed execution. 1000 is what
# states:GetExecutionHistory's MaxResults caps at anyway.
_HISTORY_PAGE_SIZE = 1000


# ── Status enums ──────────────────────────────────────────────────────────


class RunStatus(str, Enum):
    """Terminal status of a Step Functions execution.

    Mirrors the AWS Step Functions API ``status`` field verbatim. The
    ``NOT-RUN`` sentinel is lib-internal — returned by
    :func:`read_pipeline_state` when the SF has never been executed (vs
    raising), so the page can render "no executions yet" cleanly.

    ``str`` mixin lets ``RunStatus.SUCCEEDED == "SUCCEEDED"`` compare True,
    which the dashboard's existing component patterns rely on.
    """

    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    ABORTED = "ABORTED"
    NOT_RUN = "NOT-RUN"


class TaskStatus(str, Enum):
    """Per-state status as projected from the execution history.

    Adds ``SKIPPED`` (state was reached but a Choice branched past it
    without entering) + ``NOT_RUN`` (state exists in the SF JSON but was
    never reached this execution — e.g. ``ReinvokePredictor`` on a clean
    run) to the AWS status vocabulary. The Choice/Pass control flow is
    rolled up upstream (filter in :func:`_materialize_tasks`) so these two
    extras only ever apply to genuinely-substantive states.
    """

    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    ABORTED = "ABORTED"
    SKIPPED = "SKIPPED"
    NOT_RUN = "NOT-RUN"


# ── Exception types ───────────────────────────────────────────────────────


class PipelineStatusError(Exception):
    """Base class for all read-side errors."""


class SFNAccessDenied(PipelineStatusError):
    """IAM missing one of states:DescribeExecution / GetExecutionHistory /
    ListExecutions. Caller (page 25) renders a red banner naming the action.
    """


class SFNThrottled(PipelineStatusError):
    """SF API rate-limited. Caller falls back to the last-good cache."""


class SFNNoExecutions(PipelineStatusError):
    """SF has never been executed. Caller renders 'no executions yet'.

    Distinct from a generic empty response — this is the ``executions: []``
    branch from ``ListExecutions`` and means the SF exists but has zero
    history. Most often surfaced in dev/test environments; production
    SFs all have history.
    """


# ── Output shapes (Pydantic v2) ───────────────────────────────────────────


# ``model_config = ConfigDict(extra="forbid")`` because every field is
# strictly defined; an unknown key indicates a schema drift that should
# fail loud (per feedback_no_silent_fails) rather than silently widen.
_STRICT_CONFIG: ConfigDict = ConfigDict(extra="forbid", arbitrary_types_allowed=False)


class TaskRow(BaseModel):
    """One row on the page-25 per-pipeline state table."""

    model_config = _STRICT_CONFIG

    state_name: str
    status: TaskStatus
    start_utc: Optional[datetime] = None
    end_utc: Optional[datetime] = None
    duration_sec: Optional[float] = None
    # Either an ArchivePageRef (deep-link) OR an ArtifactReason (explicit
    # substrate-only reason). ``None`` here means "state name not in the
    # registry" and is a CI-time bug — the consumer should treat it as a
    # registry-drift signal, not a renderable placeholder.
    archive: Optional[Any] = None  # ArchivePageRef | ArtifactReason | None
    failure_cause: Optional[str] = None  # populated only when status == FAILED


class PipelineRun(BaseModel):
    """Top-level shape returned by :func:`read_pipeline_state`."""

    model_config = _STRICT_CONFIG

    state_machine_arn: str
    pretty_label: str  # "Saturday SF" / "Weekday SF" / "EOD SF" — from registry
    execution_arn: Optional[str] = None  # None iff status == NOT_RUN
    execution_name: Optional[str] = None  # human-readable execution id
    status: RunStatus
    start_utc: Optional[datetime] = None
    end_utc: Optional[datetime] = None
    duration_sec: Optional[float] = None
    tasks: list[TaskRow] = Field(default_factory=list)
    failing_state: Optional[str] = None  # populated only when status == FAILED
    failure_cause: Optional[str] = None  # populated only when status == FAILED


# ── Helpers ───────────────────────────────────────────────────────────────


def _label_for_arn(state_machine_arn: str) -> str:
    """Mirror sf-telegram-notifier's ``_label_for_arn`` semantics."""
    sm_name = state_machine_arn.rsplit(":", 1)[-1] if state_machine_arn else ""
    return PIPELINE_LABELS.get(sm_name, sm_name or "Unknown SF")


def _failure_cause_from(describe_resp: dict) -> str:
    """Extract + truncate the failure cause from DescribeExecution response.

    Mirrors sf-telegram-notifier's ``_failure_cause_from`` (lines 125-136)
    BYTE-FOR-BYTE so the email body + Telegram message + page cell all
    render the same snippet. Truncation policy: 280 chars max, ``…``
    appended on overflow.
    """
    error = (describe_resp.get("error") or "").strip()
    cause = (describe_resp.get("cause") or "").strip()
    if error and cause:
        snippet = f"{error}: {cause}"
    else:
        snippet = error or cause
    if len(snippet) > _CAUSE_MAX_CHARS:
        snippet = snippet[: _CAUSE_MAX_CHARS - 1] + "…"
    return snippet


def _parse_ts(value: Any) -> Optional[datetime]:
    """Normalize boto3's datetime values to UTC.

    boto3 returns ``datetime`` objects with offset-aware ``tzinfo`` (usually
    ``tzutc()``); we coerce to ``timezone.utc`` for round-trip consistency.
    Returns None for falsy input.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return None


def _is_substantive_event(event: dict) -> bool:
    """Filter for events that name a substantive state.

    Used to scan the history for ``TaskStateEntered`` / ``TaskStateExited``
    events whose state corresponds to a Task with a substantive Resource.
    The history's event-type taxonomy is one filter axis; the registry's
    ``SUBSTANTIVE_RESOURCES`` is the second (applied via the state
    definition lookup, which the SF JSON walk handles upstream).

    The history events don't carry the Resource ARN directly — we rely on
    the registry membership check at materialization time instead.
    """
    return event.get("type", "").startswith(("TaskStateEntered", "TaskStateExited"))


def _absorb_wait_companion(state_name: str) -> str:
    """Return the parent state name if ``state_name`` is a Wait companion.

    Per §3.2 of the plan doc, ``WaitForDataPhase1`` is rolled up into
    ``DataPhase1`` for display — operators think in terms of "DataPhase1
    took 40 min" not "DataPhase1 took 200ms; WaitForDataPhase1 took 39m
    59s 800ms". The duration math in :func:`_materialize_tasks` measures
    parent-entered → wait-exited when both exist.
    """
    return WAIT_GROUPING.get(state_name, state_name)


def _materialize_tasks(history_events: list[dict]) -> list[TaskRow]:
    """Walk the execution history and produce one TaskRow per substantive state.

    Algorithm:

    1. Walk every event; for each ``TaskStateEntered`` / ``TaskStateExited``,
       record the state name + timestamp + outcome.
    2. Absorb Wait companions: a ``WaitForX`` entered event extends ``X``'s
       end timestamp; the Wait state never becomes its own row.
    3. Filter to states that are in the registry (registry membership IS
       the substantive-state filter at this layer).
    4. Render each surviving state as a TaskRow with status + duration +
       registry entry attached.
    """
    # state_name → {"start": dt | None, "end": dt | None, "status": TaskStatus,
    #               "cause": str | None}
    by_state: dict[str, dict[str, Any]] = {}

    for event in history_events:
        etype = event.get("type", "")
        details_key = None
        if etype == "TaskStateEntered":
            details_key = "stateEnteredEventDetails"
        elif etype == "TaskStateExited":
            details_key = "stateExitedEventDetails"
        elif etype == "TaskFailed":
            # TaskFailed carries cause+error on TaskFailedEventDetails; we
            # attach to the most recently-entered state. boto3's history
            # iteration preserves chronological order, so the "last
            # entered" state at TaskFailed time is the failing state.
            cause = (event.get("taskFailedEventDetails") or {}).get("cause", "")
            error = (event.get("taskFailedEventDetails") or {}).get("error", "")
            snippet = f"{error}: {cause}" if (error and cause) else (error or cause)
            if len(snippet) > _CAUSE_MAX_CHARS:
                snippet = snippet[: _CAUSE_MAX_CHARS - 1] + "…"
            # Attach to the most-recent state that entered without exiting.
            for sn, rec in reversed(list(by_state.items())):
                if rec.get("end") is None:
                    rec["status"] = TaskStatus.FAILED
                    rec["cause"] = snippet
                    break
            continue
        else:
            continue

        details = event.get(details_key) or {}
        state_name = details.get("name")
        if not state_name:
            continue

        # Roll Wait companions into their parent.
        parent_name = _absorb_wait_companion(state_name)

        rec = by_state.setdefault(
            parent_name,
            {"start": None, "end": None, "status": TaskStatus.SUCCEEDED, "cause": None},
        )

        ts = _parse_ts(event.get("timestamp"))
        if etype == "TaskStateEntered":
            # Only set start if this is the FIRST entered event for this
            # parent (parent entered first, Wait companion entered after).
            if rec["start"] is None:
                rec["start"] = ts
        elif etype == "TaskStateExited":
            # The LATEST exited event wins (Wait companion exits last).
            rec["end"] = ts

    rows: list[TaskRow] = []
    for state_name, rec in by_state.items():
        # Filter: only render states that are in the registry. Anything
        # else is control-flow (Choice / Pass / Succeed) that we don't
        # surface on the page.
        archive_entry = lookup_registry(state_name)
        if archive_entry is None:
            continue

        start = rec["start"]
        end = rec["end"]
        duration: Optional[float] = None
        if start is not None and end is not None:
            duration = (end - start).total_seconds()

        # If the state was entered but never exited, status is RUNNING.
        status: TaskStatus = rec["status"]
        if end is None and start is not None and status == TaskStatus.SUCCEEDED:
            status = TaskStatus.RUNNING

        rows.append(
            TaskRow(
                state_name=state_name,
                status=status,
                start_utc=start,
                end_utc=end,
                duration_sec=duration,
                archive=archive_entry,
                failure_cause=rec["cause"],
            )
        )
    return rows


def _failing_state_from_history(history_events: list[dict]) -> Optional[str]:
    """Identify the state that emitted TaskFailed (or ExecutionFailed) first."""
    for event in history_events:
        etype = event.get("type", "")
        if etype == "TaskFailed":
            # Walk backwards through prior events to find the most-recent
            # TaskStateEntered without a matching TaskStateExited.
            idx = history_events.index(event)
            for prior in reversed(history_events[:idx]):
                if prior.get("type") == "TaskStateEntered":
                    name = (prior.get("stateEnteredEventDetails") or {}).get("name")
                    return _absorb_wait_companion(name) if name else None
        if etype == "ExecutionFailed":
            cause_details = event.get("executionFailedEventDetails") or {}
            # ExecutionFailed doesn't directly carry state name; walk back.
            idx = history_events.index(event)
            for prior in reversed(history_events[:idx]):
                if prior.get("type") == "TaskStateEntered":
                    name = (prior.get("stateEnteredEventDetails") or {}).get("name")
                    return _absorb_wait_companion(name) if name else None
            # Fallback: synthesize from the cause if no entered event found.
            return (cause_details.get("error") or None)
    return None


# ── Public entry point ────────────────────────────────────────────────────


def read_pipeline_state(
    state_machine_arn: str,
    *,
    client: Optional["SFNClient"] = None,
) -> PipelineRun:
    """Project the most-recent execution of ``state_machine_arn`` onto a
    typed :class:`PipelineRun`.

    Calls (in order):

    1. ``states:ListExecutions(stateMachineArn=..., maxResults=1)`` — finds
       the latest execution arn. If the SF has zero executions, raises
       :class:`SFNNoExecutions`.
    2. ``states:DescribeExecution(executionArn=...)`` — top-level status +
       start/stop + failure cause.
    3. ``states:GetExecutionHistory(executionArn=..., maxResults=1000)`` —
       per-state events for the Task row table.

    Parameters
    ----------
    state_machine_arn:
        Full SF ARN, e.g. ``arn:aws:states:us-east-1:711398986525:stateMachine:alpha-engine-saturday-pipeline``.
    client:
        Optional boto3 ``stepfunctions`` client. Tests pass a mock here;
        production passes None and gets a fresh client per call (cheap;
        boto3 caches under the hood).

    Returns
    -------
    PipelineRun
        Fully populated except when ``status == NOT_RUN`` (only
        ``state_machine_arn`` + ``pretty_label`` + ``status`` set).

    Raises
    ------
    SFNAccessDenied
        IAM denial on any of the three required actions.
    SFNThrottled
        Rate-limit on any of the three.
    SFNNoExecutions
        SF exists but has zero executions ever.
    PipelineStatusError
        Any other unexpected error path — the caller renders a red banner.
    """
    if client is None:  # pragma: no cover — production path
        import boto3

        client = boto3.client("stepfunctions")

    label = _label_for_arn(state_machine_arn)

    # 1. ListExecutions
    try:
        list_resp = client.list_executions(
            stateMachineArn=state_machine_arn,
            maxResults=1,
        )
    except Exception as exc:  # noqa: BLE001 — narrow + re-raise
        _raise_for_boto_error(exc, "ListExecutions")

    executions = list_resp.get("executions") or []
    if not executions:
        raise SFNNoExecutions(
            f"State machine {state_machine_arn} has no executions yet."
        )

    latest = executions[0]
    execution_arn = latest.get("executionArn")
    execution_name = latest.get("name")

    # 2. DescribeExecution
    try:
        describe_resp = client.describe_execution(executionArn=execution_arn)
    except Exception as exc:  # noqa: BLE001 — narrow + re-raise
        _raise_for_boto_error(exc, "DescribeExecution")

    status_str = describe_resp.get("status", "RUNNING")
    try:
        run_status = RunStatus(status_str)
    except ValueError:
        # Unknown status string from boto3 (forward-compatibility) — fail
        # loud rather than silently mis-render.
        raise PipelineStatusError(
            f"Unknown SF execution status {status_str!r} from boto3 for {execution_arn}"
        )

    start_utc = _parse_ts(describe_resp.get("startDate"))
    end_utc = _parse_ts(describe_resp.get("stopDate"))
    duration: Optional[float] = None
    if start_utc is not None and end_utc is not None:
        duration = (end_utc - start_utc).total_seconds()

    failure_cause = (
        _failure_cause_from(describe_resp) if run_status == RunStatus.FAILED else None
    )

    # 3. GetExecutionHistory
    try:
        history_resp = client.get_execution_history(
            executionArn=execution_arn,
            maxResults=_HISTORY_PAGE_SIZE,
            reverseOrder=False,
        )
    except Exception as exc:  # noqa: BLE001 — narrow + re-raise
        _raise_for_boto_error(exc, "GetExecutionHistory")

    events = history_resp.get("events") or []
    tasks = _materialize_tasks(events)
    failing_state = (
        _failing_state_from_history(events) if run_status == RunStatus.FAILED else None
    )

    return PipelineRun(
        state_machine_arn=state_machine_arn,
        pretty_label=label,
        execution_arn=execution_arn,
        execution_name=execution_name,
        status=run_status,
        start_utc=start_utc,
        end_utc=end_utc,
        duration_sec=duration,
        tasks=tasks,
        failing_state=failing_state,
        failure_cause=failure_cause,
    )


def _raise_for_boto_error(exc: Exception, action: str) -> None:
    """Translate a boto3 exception into a typed PipelineStatusError.

    Inspects the ``ClientError.response["Error"]["Code"]`` for the common
    cases (AccessDenied / Throttling) and re-raises the matching typed
    exception. Unknown error codes re-raise as :class:`PipelineStatusError`
    with the boto3 cause attached.
    """
    code = ""
    response = getattr(exc, "response", None) or {}
    error_dict = response.get("Error") or {}
    code = error_dict.get("Code", "")

    if code in ("AccessDeniedException", "AccessDenied"):
        raise SFNAccessDenied(
            f"states:{action} denied — add the action to the dashboard "
            f"EC2 role's inline policy. Boto3 detail: {exc}"
        ) from exc
    if code in ("ThrottlingException", "Throttling", "TooManyRequestsException"):
        raise SFNThrottled(
            f"states:{action} rate-limited; page falls back to last-good cache."
        ) from exc
    if code in ("StateMachineDoesNotExist", "ExecutionDoesNotExist"):
        raise SFNNoExecutions(
            f"states:{action} returned {code}: {exc}"
        ) from exc
    raise PipelineStatusError(
        f"Unexpected boto3 error on states:{action}: {code or type(exc).__name__}: {exc}"
    ) from exc
