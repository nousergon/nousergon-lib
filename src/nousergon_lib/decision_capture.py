"""
Decision-artifact persistence schema for Alpha Engine agent capture.

Every LLM agent invocation across the Alpha Engine stack persists a
``DecisionArtifact`` to S3 at ``s3://alpha-engine-research/decision_artifacts/
{YYYY}/{MM}/{DD}/{agent_id}/{run_id}.json``. The artifact captures the full
prompt + input data snapshot + agent output + cost so each decision can be:

- replayed against a different model version (capability-delta measurement
  vs Claude 5 / Sonnet 5 / etc. as frontier models ship);
- analyzed for rationale clustering (does this agent do varied work or
  collapse to deterministic patterns?);
- judged for output quality (LLM-as-judge eval against held-out decisions);
- audited for cost regressions (token spend per agent over time).

**Public surface:**

- :class:`DecisionArtifact` — the schema (top-level capture record).
- :class:`ModelMetadata` / :class:`FullPromptContext` — nested metadata.
- :func:`truncate_snapshot` — 1MB cap enforcer for input snapshots.
- :func:`capture_decision` — construct a ``DecisionArtifact`` and write
  it to S3. Hard-fails on S3 errors per ``feedback_no_silent_fails``.
- :exc:`DecisionCaptureWriteError` — raised by ``capture_decision`` on
  S3 failure (do not swallow).

**Schema versioning rule:** ``schema_version`` accepts ``1`` (legacy
LLM-only artifacts, pre-2026-05-11) or ``2`` (current — adds support for
deterministic decisions with ``model_metadata = None`` and
``full_prompt_context = None``). New writes go out as v2. Reads accept
either. Fields are additive-only within a version; any rename or removal
would trigger ``schema_version=3``. Use :func:`is_llm_decision` to
discriminate LLM vs deterministic artifacts at read time.

**Compatibility posture:** the top-level ``DecisionArtifact`` model is
``extra="forbid"`` (the contract is locked); the per-agent ``input_data_snapshot``
and ``agent_output`` dicts use ``extra="allow"`` since their shapes vary by
agent. ``ModelMetadata`` and ``FullPromptContext`` lock down the cross-cutting
metadata fields.

Workstream design: ``alpha-engine-docs/private/alpha-engine-research-typed-
state-capture-260429.md``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Literal

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from krepis.model_metadata import ModelMetadata
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


# ── Cross-cutting metadata ────────────────────────────────────────────────



class FullPromptContext(BaseModel):
    """Captures the full prompt the agent saw — system prompt, user prompt,
    and any tool definitions registered for the call.

    ``prompt_version_hash`` ties the capture to a specific prompt revision
    via the prompt-versioning pipeline (P2.3 in the workstream design).
    None until prompt versioning ships, at which point the wrapper begins
    populating it.
    """

    model_config = ConfigDict(extra="forbid")

    system_prompt: str
    user_prompt: str
    tool_definitions: list[dict] = Field(default_factory=list)
    prompt_version_hash: str | None = None


# ── Top-level artifact ────────────────────────────────────────────────────


_INPUT_SNAPSHOT_DEFAULT_CAP_BYTES = 1_000_000  # 1 MB — see truncate_snapshot()


class DecisionArtifact(BaseModel):
    """One captured agent decision — the substrate for replay, rationale
    clustering, agent-justification, and LLM-as-judge eval.

    Fields:

    - ``schema_version``: ``1`` (legacy LLM-only) or ``2`` (current —
      supports deterministic decisions with ``model_metadata=None`` +
      ``full_prompt_context=None``). New writes go out as v2.
    - ``run_id``: unique per pipeline invocation; ties multiple agents'
      artifacts together for one Weekly Freshness SF run or weekday morning run.
    - ``timestamp``: ISO-8601 capture time (wall clock at the moment the
      wrapper writes to S3).
    - ``agent_id``: identifies which agent produced this — e.g.
      ``"sector_quant"``, ``"sector_qual"``, ``"macro_economist"``,
      ``"ic_cio"``, ``"executor:entry_triggers"``, ``"executor:risk_guard"``.
    - ``model_metadata``: model + version + cost. ``None`` for
      deterministic decisions (e.g. ``executor:*`` algorithmic agents).
      Required-paired with ``full_prompt_context``: both present or both
      ``None``, never half-populated.
    - ``full_prompt_context``: prompt + tool definitions. ``None`` for
      deterministic decisions. Required-paired with ``model_metadata``.
    - ``input_data_snapshot``: full input payload the agent saw at
      decision time (market state, portfolio state, retrieved RAG chunks,
      etc.). Truncated to fit ``_INPUT_SNAPSHOT_DEFAULT_CAP_BYTES`` if
      pathologically large; ``input_data_truncated_at`` records the
      pre-truncation byte size when truncation fires.
    - ``input_data_summary``: deterministic human-readable view of the
      snapshot (no LLM call). Populated by the wrapper from a typed
      input model's ``__str__`` or equivalent. Not load-bearing for
      replay — the full snapshot is.
    - ``input_data_truncated_at``: byte size of the original snapshot
      if truncation fired; None otherwise.
    - ``agent_output``: serialized agent output dict (typed Pydantic
      model dumped via ``model_dump()``). Includes reasoning, tool
      calls, final structured decision. For deterministic decisions,
      the decision verdict (chosen trigger / sizing / veto verdict).

    Use :func:`is_llm_decision` to discriminate LLM vs deterministic at
    read time — consumers (cost telemetry, replay harness, LLM-as-judge,
    rationale clustering) gate on it to skip deterministic rows.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1, 2] = 2
    run_id: str
    timestamp: str  # ISO-8601, e.g. "2026-04-29T22:30:00.000Z"
    agent_id: str
    model_metadata: ModelMetadata | None = None
    full_prompt_context: FullPromptContext | None = None
    input_data_snapshot: dict[str, Any]
    input_data_summary: str | None = None
    input_data_truncated_at: int | None = Field(default=None, ge=0)
    agent_output: dict[str, Any]
    # Reproducibility provenance (optional, additive). Stamp the producing
    # code revision + the data snapshot/version the agent's inputs were
    # built from, so a later faithful replay can re-run against the SAME
    # code + data by construction (the SOTA reproducibility contract —
    # MLflow/DVC-style: run = code SHA + data version). ``None`` on artifacts
    # written before the producer populates them; consumers must tolerate
    # absence. See ROADMAP L4567 (reproducible-replay arc).
    code_sha: str | None = None
    data_snapshot_id: str | None = None

    @model_validator(mode="after")
    def _llm_fields_paired(self) -> DecisionArtifact:
        """``model_metadata`` and ``full_prompt_context`` must both be
        present (LLM agent) or both be ``None`` (deterministic decision).

        Catches the silent-bug case where an LLM-agent producer accidentally
        omits one of the two fields. Deterministic-producer callers pass
        ``None`` for both, explicitly, by convention.
        """
        if (self.model_metadata is None) != (self.full_prompt_context is None):
            raise ValueError(
                "model_metadata and full_prompt_context must both be "
                "present (LLM agent) or both be None (deterministic "
                "decision); got "
                f"model_metadata={'set' if self.model_metadata is not None else 'None'}, "
                f"full_prompt_context={'set' if self.full_prompt_context is not None else 'None'}."
            )
        return self


def is_llm_decision(artifact: DecisionArtifact) -> bool:
    """True iff ``artifact`` was produced by an LLM agent (vs a deterministic
    decision).

    Consumers gate on this to skip rows they can't or shouldn't process:

    - **Cost telemetry** — skips deterministic rows ($0 cost, 0 tokens).
    - **Replay harness** — deterministic decisions don't replay through an
      LLM model; dispatch to a different replay path (or skip).
    - **LLM-as-judge** — judge has nothing to evaluate without a prompt.
    - **Rationale clustering** — clustering needs the reasoning text inside
      ``agent_output``, which deterministic rows don't carry.

    Discrimination rule: ``model_metadata is not None``. The
    :meth:`DecisionArtifact._llm_fields_paired` validator guarantees that
    ``model_metadata`` and ``full_prompt_context`` are both present or both
    ``None``, so either field is sufficient to decide.
    """
    return artifact.model_metadata is not None


# ── Truncation helper for the 1MB cap ─────────────────────────────────────


def _serialized_size(payload: dict) -> int:
    """JSON-serialize and return the byte length. Used as the size metric
    for the cap check (matches what S3 will actually store)."""
    return len(json.dumps(payload, default=str).encode("utf-8"))


def truncate_snapshot(
    payload: dict,
    cap_bytes: int = _INPUT_SNAPSHOT_DEFAULT_CAP_BYTES,
) -> tuple[dict, int | None]:
    """Truncate ``payload`` to fit under ``cap_bytes`` when serialized to JSON.

    Strategy:

    1. If serialized size ≤ ``cap_bytes`` → return ``(payload, None)``.
       No truncation marker.
    2. Otherwise, repeatedly drop the largest top-level field by size and
       replace it with a ``{"_truncated": True, "original_field": <name>,
       "original_size_bytes": <N>}`` marker until under cap.
    3. Return ``(truncated_payload, original_size)`` where ``original_size``
       is the pre-truncation byte size — the wrapper stores this on
       ``DecisionArtifact.input_data_truncated_at`` so consumers can detect
       truncation.

    Replay correctness note: a truncated artifact loses the dropped fields,
    so replay against a different model would feed it different inputs and
    produce non-comparable output. The truncation marker is a flag for
    consumers to skip replay on truncated artifacts (or fall back to the
    summary view). Steady-state agent inputs (sector_quant ~10-50KB,
    macro_economist ~20-50KB, sector_qual ~300-400KB) are well under the
    1 MB default cap; truncation is a safety net for pathological cases,
    not the steady-state path.
    """
    original_size = _serialized_size(payload)
    if original_size <= cap_bytes:
        return payload, None

    # Truncation: drop largest top-level field, replace with marker, repeat.
    truncated = dict(payload)
    while _serialized_size(truncated) > cap_bytes:
        # Find largest top-level field by serialized size.
        sized_fields = [
            (k, _serialized_size({k: v}))
            for k, v in truncated.items()
            if not (isinstance(v, dict) and v.get("_truncated") is True)
        ]
        if not sized_fields:
            # Every field is already a truncation marker; pathological case.
            # Replace whole payload with a single marker.
            truncated = {
                "_truncated": True,
                "reason": "exceeded_cap_after_full_field_drop",
                "original_size_bytes": original_size,
                "cap_bytes": cap_bytes,
            }
            break
        sized_fields.sort(key=lambda x: -x[1])  # largest first
        largest_field, largest_size = sized_fields[0]
        truncated[largest_field] = {
            "_truncated": True,
            "original_field": largest_field,
            "original_size_bytes": largest_size,
        }

    return truncated, original_size


# ── Capture wrapper (constructs DecisionArtifact + writes to S3) ──────────


_DEFAULT_S3_BUCKET = "alpha-engine-research"
_DEFAULT_S3_PREFIX = "decision_artifacts"


class DecisionCaptureWriteError(RuntimeError):
    """Raised when the S3 write of a captured ``DecisionArtifact`` fails.

    Per ``feedback_no_silent_fails``, the capture path does not swallow
    S3 errors — every captured artifact must land or the run hard-fails.
    Callers that wish to make capture best-effort should catch this
    exception explicitly at the call site (and add a CloudWatch metric
    so silent capture loss is observable).
    """


def _build_s3_key(
    *,
    s3_prefix: str,
    capture_dt: datetime,
    agent_id: str,
    run_id: str,
) -> str:
    """Compute the canonical S3 key for a captured artifact.

    Format: ``{s3_prefix}/{YYYY}/{MM}/{DD}/{agent_id}/{run_id}.json``

    Date-partitioned by capture date (UTC) so consumers (replay harness,
    rationale clustering, LLM-as-judge) can window by day cheaply.
    Per-agent partitioning lets a single agent's corpus be queried
    independently. ``run_id`` is the leaf so all artifacts from one
    pipeline run can be discovered by listing the directory.
    """
    yyyy = capture_dt.strftime("%Y")
    mm = capture_dt.strftime("%m")
    dd = capture_dt.strftime("%d")
    return f"{s3_prefix}/{yyyy}/{mm}/{dd}/{agent_id}/{run_id}.json"


def capture_decision(
    *,
    run_id: str,
    agent_id: str,
    model_metadata: ModelMetadata | None,
    full_prompt_context: FullPromptContext | None,
    input_data_snapshot: dict[str, Any],
    agent_output: dict[str, Any],
    input_data_summary: str | None = None,
    code_sha: str | None = None,
    data_snapshot_id: str | None = None,
    s3_bucket: str = _DEFAULT_S3_BUCKET,
    s3_prefix: str = _DEFAULT_S3_PREFIX,
    s3_client: Any | None = None,
    timestamp: datetime | None = None,
    snapshot_cap_bytes: int = _INPUT_SNAPSHOT_DEFAULT_CAP_BYTES,
) -> str:
    """Construct a :class:`DecisionArtifact` and write it to S3.

    Returns the S3 key the artifact was written to. Raises
    :exc:`DecisionCaptureWriteError` on any S3 failure — the caller MUST
    NOT swallow this silently per ``feedback_no_silent_fails``.

    Parameters
    ----------
    run_id
        Unique-per-pipeline-invocation identifier. Ties artifacts from
        multiple agents in one run together.
    agent_id
        Identifies which agent produced this — e.g. ``"sector_quant"``,
        ``"sector_qual"``, ``"macro_economist"``, ``"ic_cio"``,
        ``"executor:entry_triggers"``, ``"executor:risk_guard"``.
    model_metadata
        Model identifier + token counts + cost. Pass ``None`` for
        deterministic decisions (e.g. ``executor:*`` algorithmic agents).
        Must be paired with ``full_prompt_context`` — both ``None`` or
        both populated; half-populated raises.
    full_prompt_context
        System prompt + user prompt + tool definitions seen by the agent.
        Pass ``None`` for deterministic decisions; paired with
        ``model_metadata``.
    input_data_snapshot
        Full input payload — load-bearing for replay correctness. Will
        be truncated to ``snapshot_cap_bytes`` if oversized; truncation
        is recorded in ``DecisionArtifact.input_data_truncated_at``.
    agent_output
        Serialized agent output dict (typed Pydantic model dumped via
        ``model_dump()``).
    input_data_summary
        Optional human-readable summary derived deterministically from
        the typed input model. Not load-bearing for replay; useful for
        dashboards + click-through views.
    code_sha, data_snapshot_id
        Optional reproducibility provenance — the producing code revision
        and the data snapshot/version the agent's inputs were built from.
        Stamp both so a later faithful replay re-runs against the SAME
        code + data by construction. ``None`` when the producer hasn't
        wired them yet; consumers must tolerate absence.
    s3_bucket, s3_prefix
        S3 location overrides. Defaults to
        ``s3://alpha-engine-research/decision_artifacts/`` per the
        workstream design.
    s3_client
        For testing — pass a moto-mocked or otherwise stubbed boto3 S3
        client. Defaults to ``boto3.client("s3")`` if not provided.
    timestamp
        For testing — pass a fixed UTC datetime. Defaults to
        ``datetime.now(timezone.utc)`` if not provided.
    snapshot_cap_bytes
        Override the default 1MB truncation cap.

    Returns
    -------
    str
        The S3 key the artifact was written to (e.g.
        ``"decision_artifacts/2026/04/29/sector_quant/run-abc123.json"``).

    Raises
    ------
    DecisionCaptureWriteError
        If the S3 ``PutObject`` call fails for any reason
        (BotoCoreError, ClientError including AccessDenied, etc.).
    """
    capture_dt = timestamp if timestamp is not None else datetime.now(timezone.utc)

    # 1. Apply truncation cap on the snapshot.
    snapshot_for_artifact, truncated_at = truncate_snapshot(
        input_data_snapshot, cap_bytes=snapshot_cap_bytes,
    )

    # 2. Construct the artifact (Pydantic validates).
    artifact = DecisionArtifact(
        run_id=run_id,
        timestamp=capture_dt.isoformat(),
        agent_id=agent_id,
        model_metadata=model_metadata,
        full_prompt_context=full_prompt_context,
        input_data_snapshot=snapshot_for_artifact,
        input_data_summary=input_data_summary,
        input_data_truncated_at=truncated_at,
        agent_output=agent_output,
        code_sha=code_sha,
        data_snapshot_id=data_snapshot_id,
    )

    # 3. Compute S3 key + serialize.
    s3_key = _build_s3_key(
        s3_prefix=s3_prefix,
        capture_dt=capture_dt,
        agent_id=agent_id,
        run_id=run_id,
    )
    body = artifact.model_dump_json().encode("utf-8")

    # 4. Write to S3 — hard-fail on any error.
    client = s3_client if s3_client is not None else boto3.client("s3")
    try:
        client.put_object(
            Bucket=s3_bucket,
            Key=s3_key,
            Body=body,
            ContentType="application/json",
        )
    except (BotoCoreError, ClientError) as exc:
        raise DecisionCaptureWriteError(
            f"Failed to write decision artifact to "
            f"s3://{s3_bucket}/{s3_key}: {exc}"
        ) from exc

    if truncated_at is not None:
        logger.warning(
            "[decision_capture:%s] artifact for run_id=%s truncated "
            "(original size %d bytes > cap %d bytes); replay correctness "
            "may be reduced for this artifact",
            agent_id, run_id, truncated_at, snapshot_cap_bytes,
        )

    return s3_key
