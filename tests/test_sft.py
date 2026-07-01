"""Tests for nousergon_lib.sft — the canonical SFT record both producers converge on."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from nousergon_lib.sft import (
    SFT_SCHEMA_VERSION,
    SftProvenance,
    SftRecord,
    assert_single_teacher,
    build_record,
    content_hash,
    dedup,
    record_content_hash,
    record_source,
    segregate_by_teacher,
    to_json_bytes,
    to_jsonl_bytes,
    write_jsonl,
    write_object,
)


class _FakeS3:
    """Captures put_object calls instead of hitting S3."""

    def __init__(self):
        self.puts: list[dict] = []

    def put_object(self, **kw):
        self.puts.append(kw)


# --- envelope -----------------------------------------------------------------------


def test_build_record_canonical_fields_and_defaults():
    rec = build_record("crucible_research", captured_at="2026-06-26T00:00:00+00:00", model="claude-haiku-4-5")
    assert rec["schema_version"] == SFT_SCHEMA_VERSION == 3
    assert rec["producer"] == "crucible_research"
    assert rec["model"] == "claude-haiku-4-5"
    assert rec["meta"] == {}  # defaults to empty map, never None
    # All canonical keys present even when unset (stable shape for the reader).
    for k in ("call_seq", "input_messages", "output_message", "output_text", "structured_output", "usage", "cost_usd"):
        assert k in rec


def test_producer_must_be_non_empty():
    with pytest.raises(ValidationError):
        build_record("  ", captured_at="2026-06-26T00:00:00+00:00")


def test_extra_fields_forbidden():
    # A typo / stray field is rejected rather than silently carried (drift guard).
    with pytest.raises(ValidationError):
        SftRecord(producer="x", captured_at="t", model_name="oops")  # 'model_name' is the OLD v1 name


def test_both_producer_shapes_map_onto_one_schema():
    """The two real producers differ only in which optional fields + meta they fill."""
    research = build_record(
        "crucible_research",
        captured_at="2026-06-26T09:00:00+00:00",
        model="claude-sonnet-4-6",
        call_seq=3,
        input_messages=[{"type": "human", "data": {"content": "..."}}],
        invocation_params={"tools": [], "api_key": "<redacted>"},
        output_message={"type": "ai", "data": {"content": "..."}},
        output_text="...",
        meta={"run_id": "2026-06-26", "agent_id": "sector_team:tech", "node_name": "quant"},
    )
    metron = build_record(
        "metron_advisor",
        captured_at="2026-06-26T15:00:00+00:00",
        model="claude-haiku-4-5",
        invocation_params={"system": [{"text": "..."}], "messages": []},
        output_message={"content": [{"type": "tool_use", "input": {}}]},
        structured_output={"narrative": "N", "considerations": ["a"]},
        usage={"input_tokens": 1200, "output_tokens": 80},
        cost_usd=0.0012,
        meta={"tenant_id": "t1", "portfolio_id": "p9", "source": "interactive"},
    )
    # Same key set; producer-specific data lives in meta, not in divergent top-level fields.
    assert set(research) == set(metron)
    assert research["meta"]["agent_id"] == "sector_team:tech"
    assert metron["structured_output"]["narrative"] == "N"
    assert research["structured_output"] is None  # research has no forced-tool target
    assert metron["call_seq"] is None


# --- serialization ------------------------------------------------------------------


def test_to_json_and_jsonl_bytes():
    r1 = build_record("metron_advisor", captured_at="t1")
    r2 = build_record("metron_advisor", captured_at="t2")
    assert json.loads(to_json_bytes(r1))["captured_at"] == "t1"
    lines = to_jsonl_bytes([r1, r2]).decode().splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["captured_at"] == "t2"


def test_serializers_accept_model_instances_too():
    rec = SftRecord(producer="metron_advisor", captured_at="t")
    assert json.loads(to_json_bytes(rec))["producer"] == "metron_advisor"


# --- writers ------------------------------------------------------------------------


def test_write_object_puts_single_json(monkeypatch):
    s3 = _FakeS3()
    rec = build_record("metron_advisor", captured_at="t", meta={"portfolio_id": "p9"})
    write_object(rec, bucket="b", key="metron/_sft_raw/advisor/p9.json", s3_client=s3)
    assert len(s3.puts) == 1
    put = s3.puts[0]
    assert put["Bucket"] == "b" and put["Key"].endswith("p9.json")
    assert json.loads(put["Body"])["meta"]["portfolio_id"] == "p9"


def test_write_jsonl_puts_newline_delimited(monkeypatch):
    s3 = _FakeS3()
    recs = [build_record("crucible_research", captured_at=f"t{i}", call_seq=i) for i in range(3)]
    write_jsonl(recs, bucket="b", key="decision_artifacts/_sft_raw/d/run/agent.jsonl", s3_client=s3)
    body = s3.puts[0]["Body"].decode()
    assert len(body.splitlines()) == 3


# --- provenance (v3, config#1539) ---------------------------------------------------


def test_build_record_stamps_live_provenance_and_content_hash():
    rec = build_record("crucible_research", captured_at="t", input_messages=[{"c": "AMD"}])
    assert rec["schema_version"] == 3
    assert rec["provenance"]["source"] == "live"  # default
    assert rec["provenance"]["content_hash"] == content_hash([{"c": "AMD"}])


def test_build_record_source_replay():
    rec = build_record("crucible_research", captured_at="t", input_messages=[{"c": "x"}], source="replay")
    assert rec["provenance"]["source"] == "replay"


def test_provenance_no_input_leaves_hash_none():
    rec = build_record("metron_advisor", captured_at="t")
    assert rec["provenance"] == {"source": "live", "content_hash": None}


def test_provenance_extra_forbidden():
    with pytest.raises(ValidationError):
        SftProvenance(source="live", teacher="oops")  # unknown field rejected


def test_invalid_source_rejected():
    with pytest.raises(ValidationError):
        build_record("crucible_research", captured_at="t", source="bogus")  # not a Source literal


def test_content_hash_stable_and_input_sensitive():
    a = content_hash([{"role": "user", "content": "hi"}])
    assert a == content_hash([{"content": "hi", "role": "user"}])  # key order irrelevant
    assert a != content_hash([{"role": "user", "content": "bye"}])


# --- corpus utilities ---------------------------------------------------------------


def test_record_source_prefers_typed_then_ignores_stray_meta():
    v3 = build_record("crucible_research", captured_at="t", input_messages=[{"c": "1"}], source="replay")
    assert record_source(v3) == "replay"
    # legacy v2 with a non-Source meta.source (metron's real-world data-source key)
    legacy = {"producer": "metron_advisor", "captured_at": "t", "meta": {"source": "interactive"}}
    assert record_source(legacy) == "live"  # stray value ignored, defaults live
    # legacy carrying a genuine Source literal is honored
    assert record_source({"captured_at": "t", "source": "synthetic"}) == "synthetic"


def test_record_content_hash_v3_stored_vs_v2_computed():
    v3 = build_record("crucible_research", captured_at="t", input_messages=[{"c": "z"}])
    assert record_content_hash(v3) == content_hash([{"c": "z"}])
    v2 = {"captured_at": "t", "input_messages": [{"c": "z"}]}  # no provenance
    assert record_content_hash(v2) == record_content_hash(v3)  # same input → same key


def test_dedup_keeps_earliest_of_live_and_replay():
    live = build_record("crucible_research", captured_at="2026-06-27T00:00:00", input_messages=[{"c": "AMD"}])
    replay = build_record("crucible_research", captured_at="2026-07-01T00:00:00", input_messages=[{"c": "AMD"}], source="replay")
    other = build_record("crucible_research", captured_at="2026-06-28T00:00:00", input_messages=[{"c": "NVDA"}])
    out = dedup([replay, live, other])
    assert len(out) == 2  # AMD live+replay collapse to one
    amd = next(r for r in out if r["input_messages"] == [{"c": "AMD"}])
    assert amd["provenance"]["source"] == "live"  # earliest (2026-06-27) wins, not the replay


def test_segregate_and_assert_single_teacher():
    haiku = build_record("crucible_research", captured_at="t", model="claude-haiku-4-5-20251001", input_messages=[{"c": "1"}])
    sonnet = build_record("crucible_research", captured_at="t", model="claude-sonnet-4-6", input_messages=[{"c": "2"}])
    groups = segregate_by_teacher([haiku, sonnet, haiku])
    assert set(groups) == {"claude-haiku-4-5-20251001", "claude-sonnet-4-6"}
    assert len(groups["claude-haiku-4-5-20251001"]) == 2
    assert assert_single_teacher([haiku, haiku]) == "claude-haiku-4-5-20251001"
    with pytest.raises(ValueError, match="multiple teacher versions"):
        assert_single_teacher([haiku, sonnet])


def test_v2_record_reads_back_without_provenance():
    """Historical v2 dict (no provenance) still validates + defaults cleanly."""
    v2 = {"schema_version": 2, "producer": "crucible_research", "captured_at": "t",
          "input_messages": [{"c": "1"}]}
    rec = SftRecord(**v2)
    assert rec.provenance is None
    assert record_source(v2) == "live"
