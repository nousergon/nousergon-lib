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

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Mapping, NoReturn, Optional, Sequence, Union, cast

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

# Bounds the ``getExecutionHistory`` page size. The Weekly Freshness SF emits
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
    #
    # The Annotated[Union[...], Field(discriminator="kind")] form is the
    # SOTA tagged-union pattern for Pydantic V2: ``model_dump(mode="json")``
    # serializes the ``kind`` field on each variant, and ``model_validate``
    # routes dict input to the right class via that tag. Prior to this
    # tagging, ``archive`` was typed ``Optional[Any]``, so a JSON round-trip
    # left it as a plain dict — page-25's ``isinstance`` checks then
    # misfired and rendered "Registry drift" for every state, even those
    # with valid registry entries.
    archive: Optional[
        Annotated[
            Union[ArchivePageRef, ArtifactReason],
            Field(discriminator="kind"),
        ]
    ] = None
    failure_cause: Optional[str] = None  # populated only when status == FAILED


class PipelineRun(BaseModel):
    """Top-level shape returned by :func:`read_pipeline_state`."""

    model_config = _STRICT_CONFIG

    state_machine_arn: str
    pretty_label: str  # "Weekly Freshness SF" / "Pre-open Trading SF" / "Post-close Trading SF" — from registry
    execution_arn: Optional[str] = None  # None iff status == NOT_RUN
    execution_name: Optional[str] = None  # human-readable execution id
    status: RunStatus
    start_utc: Optional[datetime] = None
    end_utc: Optional[datetime] = None
    duration_sec: Optional[float] = None
    tasks: list[TaskRow] = Field(default_factory=list)
    failing_state: Optional[str] = None  # populated only when status == FAILED
    failure_cause: Optional[str] = None  # populated only when status == FAILED
    # The ``pipeline_role`` carried on this execution's input JSON
    # (e.g. "weekly" / "daily" / "eod" / "smoke" / "recovery" /
    # "shell-run" / "backfill" / "operator-replay"). None when the input
    # JSON doesn't carry the field — typical of pre-Option-D executions
    # and ad-hoc operator launches that haven't adopted the convention.
    # The dashboard exposes this in the section header so the operator
    # always knows whether they're looking at the canonical cadence run
    # or a smoke / recovery overlay.
    pipeline_role: Optional[str] = None


class PipelineExecutionSummary(BaseModel):
    """Lightweight per-execution summary for the operator dropdown.

    Returned by :func:`list_recent_pipeline_runs`. Does NOT carry the
    full per-state task table (that lives on :class:`PipelineRun`) — the
    dropdown's job is to let the operator pick one execution to inspect
    in detail, at which point :func:`read_pipeline_state` returns the
    full run for the chosen ARN.

    ``pipeline_role`` is parsed from the execution's input JSON via the
    DescribeExecution call; None when the input lacks the field.
    """

    model_config = _STRICT_CONFIG

    execution_arn: str
    name: str
    status: RunStatus
    start_utc: datetime
    end_utc: Optional[datetime] = None
    duration_sec: Optional[float] = None
    pipeline_role: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────


def _label_for_arn(state_machine_arn: str) -> str:
    """Mirror sf-telegram-notifier's ``_label_for_arn`` semantics."""
    sm_name = state_machine_arn.rsplit(":", 1)[-1] if state_machine_arn else ""
    return PIPELINE_LABELS.get(sm_name, sm_name or "Unknown SF")


def _region_from_arn(state_machine_arn: str) -> Optional[str]:
    """Extract the AWS region from a Step Functions ARN.

    ARN shape: ``arn:aws:states:<region>:<account>:stateMachine:<name>``.
    Returns the region segment, or None if the ARN doesn't parse — in
    which case the boto3 client falls back to its normal region resolution
    (env vars / config / instance metadata). The lib is permissive on
    malformed input here because the downstream boto3 call will fail
    loud with a typed error that surfaces via ``_raise_for_boto_error``.

    Why this exists: Step Functions is a regional service and boto3
    raises ``NoRegionError`` if no region is discoverable. Streamlit
    systemd environments on EC2 may not have ``AWS_REGION`` set, but the
    ARN ALWAYS carries the region — extracting it eliminates a class of
    "missing region" failures at the lib chokepoint.
    """
    if not state_machine_arn or not state_machine_arn.startswith("arn:"):
        return None
    parts = state_machine_arn.split(":")
    if len(parts) < 4 or not parts[3]:
        return None
    return parts[3]


def _failure_cause_from(describe_resp: Mapping[str, Any]) -> str:
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


def _materialize_tasks(history_events: Sequence[Mapping[str, Any]]) -> list[TaskRow]:
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


def _failing_state_from_history(history_events: Sequence[Mapping[str, Any]]) -> Optional[str]:
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


# ── Role-filter helpers (Option-D execution-picker substrate) ─────────────


# Bounds the ListExecutions walk when a role filter is set — we page
# backwards through history looking for the first execution whose
# input.pipeline_role matches the filter. 50 is enough to span ~6 months
# of weekly cadence even if every intervening execution is a smoke /
# recovery overlay; raise it only if smoke-density is genuinely that high.
_DEFAULT_ROLE_SEARCH_LIMIT = 50

# ListExecutions page size — boto3 caps at 1000 but we keep pages small
# so a typical "find the most-recent weekly within the last 50" walk only
# hits the API once or twice.
_LIST_EXECUTIONS_PAGE_SIZE = 25


def _extract_pipeline_role(describe_resp: Mapping[str, Any]) -> Optional[str]:
    """Parse ``input.pipeline_role`` from a DescribeExecution response.

    DescribeExecution returns ``input`` as a JSON-encoded string. The
    Option-D convention is that all cron-triggered executions carry a
    ``pipeline_role`` field at top level (``{"pipeline_role": "weekly",
    ...}``) and ad-hoc operator launches set it explicitly (smoke /
    recovery / operator-replay / etc).

    Returns None on:
    - missing ``input`` field
    - malformed JSON (logged at WARN; the page renders "role: unknown")
    - JSON parses but ``pipeline_role`` is absent

    Permissive on parse failures (warn + return None rather than raise)
    because input-shape is operator-controlled and we'd rather show the
    execution with role=None than blackhole the whole page on a malformed
    input JSON. Per ``feedback_no_silent_fails`` the WARN log is the
    recording surface.
    """
    raw_input = describe_resp.get("input")
    if not raw_input or not isinstance(raw_input, str):
        return None
    try:
        parsed = json.loads(raw_input)
    except (ValueError, TypeError) as exc:
        logger.warning(
            "Could not parse SF execution input JSON; pipeline_role=None: %s", exc
        )
        return None
    if not isinstance(parsed, dict):
        return None
    role = parsed.get("pipeline_role")
    return role if isinstance(role, str) and role else None


def _build_pipeline_run_from_execution_arn(
    execution_arn: str,
    state_machine_arn: str,
    *,
    client: "SFNClient",
) -> PipelineRun:
    """Project a known execution ARN onto a typed :class:`PipelineRun`.

    Helper that holds the DescribeExecution + GetExecutionHistory +
    materialize-tasks pipeline. Callers responsible for the execution
    name (passed in via the ARN — derived if not supplied separately).

    Used by :func:`read_pipeline_state` after the role-filter walk picks
    the target execution, AND directly when an operator clicks a specific
    execution in the dropdown.
    """
    label = _label_for_arn(state_machine_arn)
    # Derive execution_name from ARN — the ARN tail is
    # ``execution:<sm-name>:<execution-name>``.
    execution_name = execution_arn.rsplit(":", 1)[-1] if execution_arn else None

    try:
        describe_resp = client.describe_execution(executionArn=execution_arn)
    except Exception as exc:  # noqa: BLE001 — narrow + re-raise
        _raise_for_boto_error(exc, "DescribeExecution")

    status_str = describe_resp.get("status", "RUNNING")
    try:
        run_status = RunStatus(status_str)
    except ValueError:
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
    pipeline_role = _extract_pipeline_role(describe_resp)

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
        pipeline_role=pipeline_role,
    )


def _find_execution_matching_role(
    state_machine_arn: str,
    role_filter: set[str],
    *,
    client: "SFNClient",
    search_limit: int,
) -> Optional[tuple[str, Optional[str]]]:
    """Walk ListExecutions pages until finding an execution whose
    ``input.pipeline_role`` ∈ ``role_filter``, or until ``search_limit``
    executions have been inspected.

    Returns ``(execution_arn, role)`` on hit, ``None`` on exhaustion.
    The N+1 DescribeExecution calls are the cost of the role filter;
    typical cron-cadence SFs find a match within the first 1-3 executions
    so the cost is bounded in practice. Smoke-heavy windows pay more but
    the ``search_limit`` cap bounds worst case.

    Caller is responsible for translating None into the right outcome —
    either SFNNoExecutions (when ListExecutions was empty in the first
    page) or a "no execution matches filter" fallback signal.
    """
    inspected = 0
    next_token: Optional[str] = None
    while inspected < search_limit:
        kwargs: dict[str, Any] = {
            "stateMachineArn": state_machine_arn,
            "maxResults": min(_LIST_EXECUTIONS_PAGE_SIZE, search_limit - inspected),
        }
        if next_token:
            kwargs["nextToken"] = next_token
        try:
            list_resp = client.list_executions(**kwargs)
        except Exception as exc:  # noqa: BLE001 — narrow + re-raise
            _raise_for_boto_error(exc, "ListExecutions")

        executions = list_resp.get("executions") or []
        if not executions:
            return None
        for ex in executions:
            inspected += 1
            execution_arn = ex.get("executionArn")
            if not execution_arn:
                continue
            try:
                describe_resp = client.describe_execution(executionArn=execution_arn)
            except Exception as exc:  # noqa: BLE001 — narrow + re-raise
                _raise_for_boto_error(exc, "DescribeExecution")
            role = _extract_pipeline_role(describe_resp)
            if role is not None and role in role_filter:
                return execution_arn, role

        next_token = list_resp.get("nextToken")
        if not next_token:
            return None

    return None


# ── Public entry point ────────────────────────────────────────────────────


def read_pipeline_state(
    state_machine_arn: str,
    *,
    role_filter: Optional[set[str]] = None,
    search_limit: int = _DEFAULT_ROLE_SEARCH_LIMIT,
    execution_arn: Optional[str] = None,
    client: Optional["SFNClient"] = None,
) -> PipelineRun:
    """Project the chosen execution of ``state_machine_arn`` onto a typed
    :class:`PipelineRun`.

    Default behavior (no ``role_filter``, no ``execution_arn``) is
    backwards-compatible: returns the most-recent execution per
    ``ListExecutions maxResults=1``, same as pre-Option-D.

    Option-D execution-picker semantics:

    - When ``execution_arn`` is set, fetches that specific execution
      directly (bypasses ListExecutions). Used by the dashboard's
      dropdown "click a row to inspect this execution" path.
    - When ``role_filter`` is set, walks ListExecutions pages until
      finding the most-recent execution whose ``input.pipeline_role``
      is in the filter set. If none match within ``search_limit``
      executions, raises :class:`SFNNoExecutions` with a message naming
      the filter — the caller (page 25) renders a banner like "No
      'weekly' execution in the last 50 runs; click 'View other recent
      executions' to inspect what's actually been running."

    Parameters
    ----------
    state_machine_arn:
        Full SF ARN.
    role_filter:
        Optional set of ``pipeline_role`` values to filter executions by
        (e.g. ``{"weekly"}`` for the Saturday-SF cadence run, ``{"daily"}``
        for the Weekday-SF cadence run). ``None`` = no filter (most-recent
        regardless of role — current behavior).
    search_limit:
        Bounds the role-filter walk. Default 50 — see
        :data:`_DEFAULT_ROLE_SEARCH_LIMIT`. Ignored when ``role_filter``
        is None.
    execution_arn:
        Optional specific execution ARN to fetch. When set, both
        ``role_filter`` and ``search_limit`` are ignored.
    client:
        Optional boto3 ``stepfunctions`` client. Tests pass a mock here;
        production passes None.

    Raises
    ------
    SFNAccessDenied
        IAM denial on any of the three required actions.
    SFNThrottled
        Rate-limit on any of the three.
    SFNNoExecutions
        SF has zero executions, OR ``role_filter`` is set and no
        execution within the search window matches.
    PipelineStatusError
        Any other unexpected error path.
    """
    if client is None:  # pragma: no cover — production path
        import boto3

        # mypy-boto3-stepfunctions is a stub-only package (no runtime
        # overloads for boto3.client's service-name dispatch), so the cast
        # tells pyright what boto3 actually hands back at runtime.
        client = cast(
            "SFNClient",
            boto3.client("stepfunctions", region_name=_region_from_arn(state_machine_arn)),
        )

    # Path 1: explicit execution_arn — fetch directly.
    if execution_arn is not None:
        return _build_pipeline_run_from_execution_arn(
            execution_arn, state_machine_arn, client=client
        )

    # Path 2: role_filter — walk ListExecutions until match.
    if role_filter:
        match = _find_execution_matching_role(
            state_machine_arn, role_filter, client=client, search_limit=search_limit
        )
        if match is None:
            raise SFNNoExecutions(
                f"No execution with pipeline_role in {sorted(role_filter)!r} "
                f"found within last {search_limit} executions of {state_machine_arn}."
            )
        matched_arn, _matched_role = match
        return _build_pipeline_run_from_execution_arn(
            matched_arn, state_machine_arn, client=client
        )

    # Path 3 (default): most-recent execution regardless of role —
    # backwards-compatible with pre-Option-D callers.
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
    return _build_pipeline_run_from_execution_arn(
        latest.get("executionArn"), state_machine_arn, client=client
    )


def list_recent_pipeline_runs(
    state_machine_arn: str,
    *,
    limit: int = 10,
    role_filter: Optional[set[str]] = None,
    client: Optional["SFNClient"] = None,
) -> list[PipelineExecutionSummary]:
    """Return lightweight summaries of the most-recent N executions.

    Backs the page-25 "View other recent executions" disclosure: shows
    the operator what's been running on this SF, ranked most-recent
    first, with the ``pipeline_role`` of each so smoke vs. weekly vs.
    recovery is visible at a glance.

    Each summary requires one ``DescribeExecution`` call (to extract
    ``pipeline_role`` from the input JSON) on top of one
    ``ListExecutions`` call, so this is O(limit) API calls. Default
    ``limit=10`` puts the dashboard's "show me last N" view at ~11
    SF API calls per page render — well within the 25-TPS soft limit
    states:DescribeExecution applies.

    Parameters
    ----------
    state_machine_arn:
        Full SF ARN.
    limit:
        Max number of executions to return. Default 10.
    role_filter:
        Optional pre-filter (returns only executions whose
        ``pipeline_role`` ∈ ``role_filter``). When set, the API call
        budget grows because we may have to walk past role-mismatched
        executions; bounded by an internal walk cap of ``limit * 5``.
    client:
        Optional boto3 ``stepfunctions`` client.
    """
    if client is None:  # pragma: no cover — production path
        import boto3

        # mypy-boto3-stepfunctions is a stub-only package (no runtime
        # overloads for boto3.client's service-name dispatch), so the cast
        # tells pyright what boto3 actually hands back at runtime.
        client = cast(
            "SFNClient",
            boto3.client("stepfunctions", region_name=_region_from_arn(state_machine_arn)),
        )

    walk_cap = limit if role_filter is None else min(limit * 5, _DEFAULT_ROLE_SEARCH_LIMIT)
    summaries: list[PipelineExecutionSummary] = []
    inspected = 0
    next_token: Optional[str] = None

    while len(summaries) < limit and inspected < walk_cap:
        kwargs: dict[str, Any] = {
            "stateMachineArn": state_machine_arn,
            "maxResults": min(_LIST_EXECUTIONS_PAGE_SIZE, walk_cap - inspected),
        }
        if next_token:
            kwargs["nextToken"] = next_token
        try:
            list_resp = client.list_executions(**kwargs)
        except Exception as exc:  # noqa: BLE001 — narrow + re-raise
            _raise_for_boto_error(exc, "ListExecutions")

        executions = list_resp.get("executions") or []
        if not executions:
            break
        for ex in executions:
            inspected += 1
            execution_arn = ex.get("executionArn")
            if not execution_arn:
                continue
            try:
                describe_resp = client.describe_execution(executionArn=execution_arn)
            except Exception as exc:  # noqa: BLE001 — narrow + re-raise
                _raise_for_boto_error(exc, "DescribeExecution")
            role = _extract_pipeline_role(describe_resp)
            if role_filter is not None and role not in role_filter:
                continue
            status_str = describe_resp.get("status", "RUNNING")
            try:
                status = RunStatus(status_str)
            except ValueError:
                raise PipelineStatusError(
                    f"Unknown SF execution status {status_str!r} from boto3 for {execution_arn}"
                )
            start_utc = _parse_ts(describe_resp.get("startDate"))
            end_utc = _parse_ts(describe_resp.get("stopDate"))
            duration: Optional[float] = None
            if start_utc is not None and end_utc is not None:
                duration = (end_utc - start_utc).total_seconds()
            if start_utc is None:
                # An execution without a start time is degenerate; skip
                # rather than fail the whole list.
                continue
            summaries.append(
                PipelineExecutionSummary(
                    execution_arn=execution_arn,
                    name=ex.get("name") or execution_arn.rsplit(":", 1)[-1],
                    status=status,
                    start_utc=start_utc,
                    end_utc=end_utc,
                    duration_sec=duration,
                    pipeline_role=role,
                )
            )
            if len(summaries) >= limit:
                break

        next_token = list_resp.get("nextToken")
        if not next_token:
            break

    return summaries


def _raise_for_boto_error(exc: Exception, action: str) -> NoReturn:
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
