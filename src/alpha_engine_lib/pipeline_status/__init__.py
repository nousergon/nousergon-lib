"""Pipeline-status projection of the three Alpha Engine Step Functions.

Substrate for the pipeline-reporting-revamp arc (ROADMAP L3050, plan doc
``alpha-engine-docs/private/pipeline-reporting-revamp-260524.md``). Projects
``states:DescribeExecution`` + ``states:GetExecutionHistory`` onto a typed
:class:`PipelineRun` so the dashboard page 25 (and any future Slack/CLI
subscriber) renders SF state without rebuilding the projection logic per
consumer.

**Public surface:**

- :func:`read_pipeline_state` — projection entry point. Returns a
  :class:`PipelineRun` for the most-recent execution of the given SF ARN.
- :class:`PipelineRun` / :class:`TaskRow` / :class:`RunStatus` — typed shape.
- :data:`STATE_TO_ARCHIVE_PAGE` — registry mapping every substantive Task
  state to either an :class:`ArchivePageRef` deep-link OR a non-generic
  :class:`ArtifactReason` string (per ``feedback_no_silent_fails`` — no
  generic "no artifact" placeholders).
- :func:`format_success_message` / :func:`format_failure_message` — verbatim
  Python parity for the ``States.Format`` templates baked into the SF JSON.
  Lets future non-SF consumers render byte-identical message bodies without
  duplicating the template.

**Why this lives in lib (not in alpha-engine-dashboard):** second adoption
is anticipated — the same projection is the natural backing for a Slack
subscriber + a CLI ``ae pipeline status`` command. Per the SOTA / institutional
sub-sub-rule in ``~/Development/CLAUDE.md`` item 9, the lift goes upstream
on first build, not after the second consumer arrives.
"""

from __future__ import annotations

from .read import (
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
from .registry import (
    PIPELINE_LABELS,
    STATE_TO_ARCHIVE_PAGE,
    SUBSTANTIVE_RESOURCES,
    WAIT_GROUPING,
    ArchivePageRef,
    ArtifactReason,
)
from .templates import format_failure_message, format_success_message

__all__ = [
    "ArchivePageRef",
    "ArtifactReason",
    "PIPELINE_LABELS",
    "PipelineExecutionSummary",
    "PipelineRun",
    "RunStatus",
    "SFNAccessDenied",
    "SFNNoExecutions",
    "SFNThrottled",
    "STATE_TO_ARCHIVE_PAGE",
    "SUBSTANTIVE_RESOURCES",
    "TaskRow",
    "TaskStatus",
    "WAIT_GROUPING",
    "format_failure_message",
    "format_success_message",
    "list_recent_pipeline_runs",
    "read_pipeline_state",
]
