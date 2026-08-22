"""Pipeline-status projection of the three Alpha Engine Step Functions.

Substrate for the pipeline-reporting-revamp arc (ROADMAP L3050, plan doc
``alpha-engine-docs/private/pipeline-reporting-revamp-260524.md``). Projects
``states:DescribeExecution`` + ``states:GetExecutionHistory`` onto a typed
:class:`PipelineRun` so the dashboard page 25 (and any future Slack/CLI
subscriber) renders SF state without rebuilding the projection logic per
consumer.

**Public surface:**

- :func:`build_reliability_window` — CYCLE-level projection: attempts per
  cycle, attempts-to-success, stage depth reached, and whether each failure
  cause is new or a repeat of an earlier cycle's (alpha-engine-config-I6919).
  Answers "are we making progress or looping", which red/green cannot.
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

from .completion_marker import (
    MARKER_PREFIX,
    augment_marker,
    decode_marker,
    marker_key,
    marker_verdict,
    merge_cycle_shape,
    read_marker,
)
from .coverage import (
    CoverageSweep,
    RowState,
    StageRow,
    publish_sweep,
    read_coverage_sweep,
    render_rows,
    sweep_coverage,
)
from .cycle_shape import (
    CONTRIBUTING_ROLES,
    CycleExecution,
    CycleShape,
    CycleVerdict,
    build_cycle_shape,
    read_cycle_shape,
)
from .cycles import (
    AttemptOutcome,
    CycleReliability,
    ReliabilityWindow,
    build_reliability_window,
    fingerprint,
)
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
    read_reliability_window,
)
from .registry import (
    PIPELINE_LABELS,
    PIPELINE_STAGE_ORDER,
    SKIP_TERMINALS,
    STATE_TO_ARCHIVE_PAGE,
    SUBSTANTIVE_RESOURCES,
    WAIT_GROUPING,
    ArchivePageRef,
    ArtifactReason,
    skip_terminals_for,
    stage_order_for,
)
from .roles import (
    ADHOC_ROLES,
    ALL_ROLES,
    CADENCE_ROLES,
    EXERCISE_ROLES,
    RECOVERY_ROLES,
    cadence_filter,
    classify,
)
from .templates import format_failure_message, format_success_message
from .work import (
    UndeclaredPipeline,
    WorkOutcome,
    WorkVerdict,
    classify_work,
    entered_states_from_history,
    read_work_outcome,
)

__all__ = [
    "ADHOC_ROLES",
    "CONTRIBUTING_ROLES",
    "MARKER_PREFIX",
    "augment_marker",
    "decode_marker",
    "marker_key",
    "marker_verdict",
    "merge_cycle_shape",
    "read_marker",
    "CoverageSweep",
    "CycleExecution",
    "CycleShape",
    "CycleVerdict",
    "RowState",
    "StageRow",
    "build_cycle_shape",
    "publish_sweep",
    "read_coverage_sweep",
    "read_cycle_shape",
    "render_rows",
    "sweep_coverage",
    "AttemptOutcome",
    "ALL_ROLES",
    "ArchivePageRef",
    "ArtifactReason",
    "CADENCE_ROLES",
    "CycleReliability",
    "EXERCISE_ROLES",
    "PIPELINE_LABELS",
    "PIPELINE_STAGE_ORDER",
    "SKIP_TERMINALS",
    "UndeclaredPipeline",
    "WorkOutcome",
    "WorkVerdict",
    "PipelineExecutionSummary",
    "PipelineRun",
    "RECOVERY_ROLES",
    "ReliabilityWindow",
    "RunStatus",
    "SFNAccessDenied",
    "SFNNoExecutions",
    "SFNThrottled",
    "STATE_TO_ARCHIVE_PAGE",
    "SUBSTANTIVE_RESOURCES",
    "TaskRow",
    "TaskStatus",
    "WAIT_GROUPING",
    "build_reliability_window",
    "cadence_filter",
    "classify_work",
    "classify",
    "entered_states_from_history",
    "fingerprint",
    "format_failure_message",
    "format_success_message",
    "list_recent_pipeline_runs",
    "read_pipeline_state",
    "read_reliability_window",
    "read_work_outcome",
    "skip_terminals_for",
    "stage_order_for",
]
