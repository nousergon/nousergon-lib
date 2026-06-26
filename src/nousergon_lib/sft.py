"""Canonical SFT (supervised-fine-tuning) trace records for the Crucible/Metron
distillation corpus — the single chokepoint both producers emit through so the
record schema cannot drift between them.

Two producers feed one downstream curate/fine-tune/eval pipeline:

- ``crucible_research`` — per-LLM-call capture in alpha-engine-research's
  ``graph/llm_cost_tracker.py`` (LangChain off-the-wire, one JSONL file per run at
  ``decision_artifacts/_sft_raw/{date}/{run_id}/{agent_id}.jsonl``).
- ``metron_advisor`` — per-advisory-call capture in metron-ops'
  ``metron_ext/advisor/sft_capture.py`` (Anthropic SDK, one object per call at
  ``metron/_sft_raw/advisor/{date}/{portfolio}/…json``).

Before this module the two emitted *different* field names while both claiming
``schema_version = 1`` (research: ``timestamp``/``model_name`` + top-level frame
dims; metron: ``captured_at``/``model`` + ``meta``). This module defines the
canonical **v2** envelope they converge on.

**Scope boundary:** each producer serializes its own message objects (LangChain
``message_to_dict`` vs the Anthropic SDK) into the ``input_messages`` /
``output_message`` dicts — this module owns the ENVELOPE schema + the S3 write,
NOT the message serialization. Producer-specific dimensions (research frame dims:
``run_id``/``agent_id``/``node_name``/prompt provenance; metron:
``tenant_id``/``portfolio_id``/``source``/``posture``) live in the free-form
``meta`` map, so adding a producer dimension never touches the shared schema.

**Public surface:** :data:`SFT_SCHEMA_VERSION`, :class:`SftRecord`,
:func:`build_record`, :func:`to_json_bytes`, :func:`to_jsonl_bytes`,
:func:`write_object`, :func:`write_jsonl`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# v2: the canonical schema both producers converge on. v1 was the pre-convergence
# era where research and metron emitted divergent field names — readers of historical
# _sft_raw must branch on schema_version.
SFT_SCHEMA_VERSION = 2


class SftRecord(BaseModel):
    """One captured ``(rendered prompt → completion)`` training example.

    Envelope metadata is strict; the payload fields (``input_messages``,
    ``output_message``, …) are producer-serialized opaque JSON, so they are typed
    permissively. ``meta`` carries producer-specific dimensions.
    """

    # `model` is a plain field name here, not a Pydantic-protected one.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    schema_version: int = SFT_SCHEMA_VERSION
    producer: str
    captured_at: str
    model: str | None = None
    call_seq: int | None = None
    input_messages: Any | None = None
    invocation_params: Any | None = None
    output_message: Any | None = None
    output_text: str | None = None
    structured_output: Any | None = None
    usage: dict[str, Any] | None = None
    cost_usd: float | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("producer")
    @classmethod
    def _producer_non_empty(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("producer must be a non-empty identifier (e.g. 'crucible_research')")
        return v


def build_record(
    producer: str,
    *,
    captured_at: str,
    model: str | None = None,
    call_seq: int | None = None,
    input_messages: Any | None = None,
    invocation_params: Any | None = None,
    output_message: Any | None = None,
    output_text: str | None = None,
    structured_output: Any | None = None,
    usage: dict[str, Any] | None = None,
    cost_usd: float | None = None,
    meta: dict[str, Any] | None = None,
    schema_version: int = SFT_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Build a validated canonical SFT record as a plain dict (ready for ``json.dumps``).

    ``captured_at`` is an ISO-8601 string supplied by the caller (the producers own
    their clocks). ``meta`` defaults to ``{}``. Raises ``pydantic.ValidationError`` on
    a malformed envelope (e.g. empty ``producer``).
    """
    return SftRecord(
        schema_version=schema_version,
        producer=producer,
        captured_at=captured_at,
        model=model,
        call_seq=call_seq,
        input_messages=input_messages,
        invocation_params=invocation_params,
        output_message=output_message,
        output_text=output_text,
        structured_output=structured_output,
        usage=usage,
        cost_usd=cost_usd,
        meta=dict(meta or {}),
    ).model_dump()


def _as_dict(record: SftRecord | dict[str, Any]) -> dict[str, Any]:
    return record.model_dump() if isinstance(record, SftRecord) else record


def to_json_bytes(record: SftRecord | dict[str, Any]) -> bytes:
    """Serialize one record to a single JSON object (object-per-call layout)."""
    return json.dumps(_as_dict(record), default=str).encode("utf-8")


def to_jsonl_bytes(records: Sequence[SftRecord | dict[str, Any]]) -> bytes:
    """Serialize many records to newline-delimited JSON (JSONL-per-run layout)."""
    return "\n".join(json.dumps(_as_dict(r), default=str) for r in records).encode("utf-8")


def write_object(
    record: SftRecord | dict[str, Any],
    *,
    bucket: str,
    key: str,
    s3_client: Any | None = None,
) -> None:
    """PUT one record as a JSON object at ``s3://bucket/key`` (metron layout).

    ``s3_client`` is injectable for tests; constructed lazily otherwise.
    """
    if s3_client is None:
        import boto3

        s3_client = boto3.client("s3")
    s3_client.put_object(Bucket=bucket, Key=key, Body=to_json_bytes(record), ContentType="application/json")


def write_jsonl(
    records: Sequence[SftRecord | dict[str, Any]],
    *,
    bucket: str,
    key: str,
    s3_client: Any | None = None,
) -> None:
    """PUT many records as one JSONL object at ``s3://bucket/key`` (research layout)."""
    if s3_client is None:
        import boto3

        s3_client = boto3.client("s3")
    s3_client.put_object(Bucket=bucket, Key=key, Body=to_jsonl_bytes(records), ContentType="application/json")
