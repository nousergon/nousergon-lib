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

**Provenance (v3, config#1539):** every record carries a typed
:class:`SftProvenance` — ``source`` (``live`` | ``replay`` | ``synthetic``) so
teacher-REPLAYED and self-instruct SYNTHETIC traces are separable from
naturally-accrued LIVE ones, and a stable ``content_hash`` over the model INPUT
so live/replay duplicates of the same teacher call collapse cleanly. Teacher
segregation keys off the envelope ``model`` field (the dated teacher version,
e.g. ``claude-haiku-4-5-20251001``) — a distillation corpus must never silently
BLEND teacher versions (:func:`assert_single_teacher`).

**Public surface:** :data:`SFT_SCHEMA_VERSION`, :class:`SftRecord`,
:class:`SftProvenance`, :func:`build_record`, :func:`content_hash`,
:func:`record_source`, :func:`record_content_hash`, :func:`dedup`,
:func:`segregate_by_teacher`, :func:`assert_single_teacher`,
:func:`to_json_bytes`, :func:`to_jsonl_bytes`, :func:`write_object`,
:func:`write_jsonl`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# v3 (config#1539): adds the typed `provenance` envelope field (source +
# content_hash) so replay/synthetic corpora are separable from live and dupes
# collapse. v2 was the producer-convergence schema; v1 the pre-convergence era
# where research and metron emitted divergent field names. Readers of historical
# _sft_raw must branch on schema_version.
SFT_SCHEMA_VERSION = 3

Source = Literal["live", "replay", "synthetic"]

__all__ = [
    "SFT_SCHEMA_VERSION",
    "Source",
    "SftProvenance",
    "SftRecord",
    "build_record",
    "content_hash",
    "record_source",
    "record_content_hash",
    "dedup",
    "segregate_by_teacher",
    "assert_single_teacher",
    "to_json_bytes",
    "to_jsonl_bytes",
    "write_object",
    "write_jsonl",
]


def content_hash(input_messages: Any) -> str:
    """Stable SHA-256 over the model INPUT — the dedup key.

    Hashes the canonicalized ``input_messages`` (sorted keys, non-ASCII kept) so
    two captures of the SAME teacher call — e.g. a LIVE trace and a REPLAY of the
    same historical input — produce the same hash and collapse under :func:`dedup`.
    Hashes the input (not the completion) because that is what makes two examples
    "the same teacher call"; completions can vary run-to-run (sampling).
    """
    canon = json.dumps(input_messages, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


class SftProvenance(BaseModel):
    """Typed provenance for one SFT record (config#1539).

    ``source`` distinguishes naturally-accrued ``live`` traces from teacher-
    ``replay`` mints and self-instruct ``synthetic`` inputs — a distillation run
    must be able to filter to one source cleanly. ``content_hash`` is the stable
    dedup key over the model input (see :func:`content_hash`).
    """

    model_config = ConfigDict(extra="forbid")

    source: Source = "live"
    content_hash: str | None = None


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
    # Typed provenance (v3, config#1539). Optional on the model for read-back
    # tolerance of historical v1/v2 dicts; `build_record` always populates it.
    provenance: SftProvenance | None = None

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
    source: Source = "live",
    schema_version: int = SFT_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Build a validated canonical SFT record as a plain dict (ready for ``json.dumps``).

    ``captured_at`` is an ISO-8601 string supplied by the caller (the producers own
    their clocks). ``meta`` defaults to ``{}``. ``source`` stamps the record's
    :class:`SftProvenance` (``live`` for naturally-accrued traces — the default;
    ``replay`` for teacher-over-history mints; ``synthetic`` for self-instruct
    inputs); the ``content_hash`` is auto-computed from ``input_messages``. Raises
    ``pydantic.ValidationError`` on a malformed envelope (e.g. empty ``producer``).
    """
    provenance = SftProvenance(
        source=source,
        content_hash=content_hash(input_messages) if input_messages is not None else None,
    )
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
        provenance=provenance,
    ).model_dump()


def record_source(rec: dict[str, Any]) -> Source:
    """Canonical ``source`` extraction for a raw record dict (v1/v2/v3-tolerant).

    Prefers the typed ``provenance.source`` (v3); tolerates a legacy top-level or
    ``meta`` ``source`` ONLY if it is a valid :data:`Source` literal; otherwise
    defaults to ``"live"`` (naturally-accrued). A stray non-Source ``meta.source``
    (e.g. a producer-specific data-source string) is ignored, never misclassified.
    """
    prov = rec.get("provenance")
    if isinstance(prov, dict) and prov.get("source") in ("live", "replay", "synthetic"):
        return prov["source"]  # type: ignore[return-value]
    for candidate in (rec.get("source"), (rec.get("meta") or {}).get("source")):
        if candidate in ("live", "replay", "synthetic"):
            return candidate  # type: ignore[return-value]
    return "live"


def record_content_hash(rec: dict[str, Any]) -> str:
    """The record's dedup key — the stored ``provenance.content_hash`` (v3) if
    present, else computed from ``input_messages`` (v1/v2 back-compat)."""
    prov = rec.get("provenance")
    if isinstance(prov, dict) and prov.get("content_hash"):
        return prov["content_hash"]
    return content_hash(rec.get("input_messages"))


def dedup(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse records sharing a ``content_hash`` to one, keeping the EARLIEST
    (min ``captured_at``) — so a REPLAY mint never displaces the original LIVE
    trace of the same teacher call. Output is ordered by ``captured_at``.
    """
    by_hash: dict[str, dict[str, Any]] = {}
    for rec in records:
        h = record_content_hash(rec)
        prior = by_hash.get(h)
        if prior is None or _captured(rec) < _captured(prior):
            by_hash[h] = rec
    return sorted(by_hash.values(), key=_captured)


def segregate_by_teacher(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group records by teacher (the envelope ``model`` field). A distillation set
    must be built per-teacher-version — never blended (see :func:`assert_single_teacher`).
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault(rec.get("model") or "unknown", []).append(rec)
    return groups


def assert_single_teacher(records: Iterable[dict[str, Any]]) -> str:
    """Guard: raise ``ValueError`` if ``records`` mix >1 teacher (``model``) version.

    A silent blend of two teachers' behavior is a corpus-integrity defect — a
    distillation run must opt into a single teacher. Returns the sole teacher on
    success (``"unknown"`` if every record lacks ``model``).
    """
    teachers = sorted(segregate_by_teacher(records).keys())
    if len(teachers) > 1:
        raise ValueError(
            "SFT corpus blends multiple teacher versions "
            f"({', '.join(teachers)}) — segregate by teacher before training/eval; "
            "old traces are a DIFFERENT teacher's behavior."
        )
    return teachers[0] if teachers else "unknown"


def _captured(rec: dict[str, Any]) -> str:
    return str(rec.get("captured_at") or rec.get("timestamp") or "")


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
