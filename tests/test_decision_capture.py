"""
Round-trip + truncation tests for ``nousergon_lib.decision_capture``.

Schema is the cross-cutting contract every captured agent decision must
satisfy. These tests lock down: extra-field rejection at the artifact level,
range constraints on token counts + cost, schema_version pinning, and the
1 MB truncation helper's behavior on small / large / pathological payloads.

Workstream design: ``alpha-engine-docs/private/alpha-engine-research-typed-
state-capture-260429.md``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import boto3
import pytest
from moto import mock_aws

from nousergon_lib.decision_capture import (
    _INPUT_SNAPSHOT_DEFAULT_CAP_BYTES,
    DecisionArtifact,
    DecisionCaptureWriteError,
    FullPromptContext,
    ModelMetadata,
    _build_s3_key,
    _serialized_size,
    capture_decision,
    is_llm_decision,
    truncate_snapshot,
)

# ── ModelMetadata ─────────────────────────────────────────────────────────


class TestModelMetadata:
    def test_minimal_model_only(self):
        m = ModelMetadata(model_name="claude-haiku-4-5")
        assert m.input_tokens == 0
        assert m.output_tokens == 0
        assert m.cost_usd == 0.0
        assert m.model_version is None
        # Cost-telemetry context fields default to None.
        assert m.run_type is None
        assert m.node_name is None
        assert m.sector_team_id is None
        assert m.prompt_id is None
        assert m.prompt_version is None

    def test_full(self):
        m = ModelMetadata(
            model_name="claude-haiku-4-5",
            model_version="20250101",
            input_tokens=4000,
            output_tokens=1200,
            cache_read_tokens=2000,
            cache_create_tokens=500,
            cost_usd=0.04321,
        )
        assert m.cost_usd == 0.04321

    def test_full_with_telemetry_context(self):
        m = ModelMetadata(
            model_name="claude-haiku-4-5",
            input_tokens=1000,
            output_tokens=500,
            run_type="weekly_research",
            node_name="sector_quant_node",
            sector_team_id="technology",
            prompt_id="sector_quant_analyst",
            prompt_version="1.2.0",
        )
        assert m.run_type == "weekly_research"
        assert m.sector_team_id == "technology"
        assert m.prompt_version == "1.2.0"

    def test_run_type_literal_enforced(self):
        # run_type is a Literal — invalid value rejected at validation.
        with pytest.raises(ValueError):
            ModelMetadata(model_name="x", run_type="adhoc")

    def test_negative_token_counts_rejected(self):
        with pytest.raises(ValueError):
            ModelMetadata(model_name="x", input_tokens=-1)
        with pytest.raises(ValueError):
            ModelMetadata(model_name="x", output_tokens=-5)

    def test_negative_cost_rejected(self):
        with pytest.raises(ValueError):
            ModelMetadata(model_name="x", cost_usd=-0.01)

    def test_extra_fields_rejected(self):
        # Cross-cutting metadata is locked — adding a field requires a
        # schema_version bump on DecisionArtifact.
        with pytest.raises(ValueError):
            ModelMetadata(model_name="x", undocumented="value")


# ── FullPromptContext ─────────────────────────────────────────────────────


class TestFullPromptContext:
    def test_minimal(self):
        ctx = FullPromptContext(system_prompt="sys", user_prompt="user")
        assert ctx.tool_definitions == []
        assert ctx.prompt_version_hash is None

    def test_with_tools(self):
        ctx = FullPromptContext(
            system_prompt="sys",
            user_prompt="user",
            tool_definitions=[
                {"name": "quant_indicators", "args_schema": {"type": "object"}},
                {"name": "qual_news_search", "args_schema": {"type": "object"}},
            ],
        )
        assert len(ctx.tool_definitions) == 2

    def test_with_version_hash(self):
        ctx = FullPromptContext(
            system_prompt="sys", user_prompt="user",
            prompt_version_hash="abc123",
        )
        assert ctx.prompt_version_hash == "abc123"

    def test_extra_fields_rejected(self):
        with pytest.raises(ValueError):
            FullPromptContext(system_prompt="sys", user_prompt="user", extra="x")


# ── DecisionArtifact ──────────────────────────────────────────────────────


def _minimal_artifact_kwargs() -> dict:
    """Helper for tests: minimal valid kwargs for DecisionArtifact."""
    return {
        "run_id": "run-2026-04-29",
        "timestamp": "2026-04-29T22:30:00Z",
        "agent_id": "sector_quant",
        "model_metadata": ModelMetadata(model_name="claude-haiku-4-5"),
        "full_prompt_context": FullPromptContext(
            system_prompt="sys", user_prompt="user",
        ),
        "input_data_snapshot": {"market_regime": "neutral", "tickers": ["AAPL"]},
        "agent_output": {"recommendations": [{"ticker": "AAPL", "score": 75}]},
    }


class TestDecisionArtifactBasics:
    def test_minimal(self):
        art = DecisionArtifact(**_minimal_artifact_kwargs())
        # v0.10.0 (schema_version=2) default — new writes go out as v2.
        assert art.schema_version == 2
        assert art.input_data_summary is None
        assert art.input_data_truncated_at is None

    def test_schema_version_v1_accepted_for_legacy_reads(self):
        # Legacy v1 artifacts (pre-2026-05-11) must still validate so the
        # historical corpus remains readable after the v2 bump.
        kwargs = _minimal_artifact_kwargs()
        kwargs["schema_version"] = 1
        art = DecisionArtifact(**kwargs)
        assert art.schema_version == 1

    def test_schema_version_v3_rejected(self):
        # Future-proofing: only v1 and v2 are valid. A future v3 bump would
        # update this test alongside the schema change.
        kwargs = _minimal_artifact_kwargs()
        kwargs["schema_version"] = 3
        with pytest.raises(ValueError):
            DecisionArtifact(**kwargs)

    def test_extra_fields_rejected_at_top_level(self):
        # Top-level contract is locked — new fields require a schema_version
        # bump or an additive landing on v1 with a Pydantic field default.
        kwargs = _minimal_artifact_kwargs()
        kwargs["undocumented_field"] = "value"
        with pytest.raises(ValueError):
            DecisionArtifact(**kwargs)

    def test_input_data_snapshot_allows_arbitrary_dict_shapes(self):
        # Per-agent snapshot shapes vary; the capture layer treats them as
        # opaque dicts.
        kwargs = _minimal_artifact_kwargs()
        kwargs["input_data_snapshot"] = {
            "any": "shape",
            "nested": {"deep": {"value": [1, 2, 3]}},
            "list": [{"a": 1}, {"b": 2}],
        }
        art = DecisionArtifact(**kwargs)
        assert art.input_data_snapshot["nested"]["deep"]["value"] == [1, 2, 3]

    def test_agent_output_allows_arbitrary_dict_shapes(self):
        kwargs = _minimal_artifact_kwargs()
        kwargs["agent_output"] = {
            "reasoning": "long chain of thought...",
            "tool_calls": [{"tool": "x"}, {"tool": "y"}],
            "final_decision": {"recs": [], "score": 0},
        }
        art = DecisionArtifact(**kwargs)
        assert "tool_calls" in art.agent_output


def _minimal_deterministic_kwargs() -> dict:
    """Helper: minimal valid kwargs for a deterministic decision (v2 only).

    ``model_metadata`` and ``full_prompt_context`` both ``None`` — produced
    by an algorithmic agent (e.g. ``executor:entry_triggers``).
    """
    return {
        "run_id": "run-2026-05-11",
        "timestamp": "2026-05-11T20:30:00Z",
        "agent_id": "executor:entry_triggers",
        "model_metadata": None,
        "full_prompt_context": None,
        "input_data_snapshot": {
            "ticker": "AAPL",
            "current_price": 175.25,
            "day_high": 178.50,
            "thresholds": {"pullback_pct": 0.02},
        },
        "agent_output": {
            "fired_trigger": "pullback 1.8% from high $178.50",
            "trigger_kind": "pullback",
        },
    }


class TestDecisionArtifactDeterministic:
    """v0.10.0 — deterministic decisions (no LLM): both LLM fields are None."""

    def test_minimal_deterministic(self):
        art = DecisionArtifact(**_minimal_deterministic_kwargs())
        assert art.schema_version == 2
        assert art.model_metadata is None
        assert art.full_prompt_context is None
        assert art.agent_id == "executor:entry_triggers"

    def test_half_populated_model_metadata_raises(self):
        # model_metadata set + full_prompt_context None → validator fires
        kwargs = _minimal_deterministic_kwargs()
        kwargs["model_metadata"] = ModelMetadata(model_name="claude-haiku-4-5")
        # full_prompt_context stays None
        with pytest.raises(ValueError, match="must both be"):
            DecisionArtifact(**kwargs)

    def test_half_populated_prompt_context_raises(self):
        # full_prompt_context set + model_metadata None → validator fires
        kwargs = _minimal_deterministic_kwargs()
        kwargs["full_prompt_context"] = FullPromptContext(
            system_prompt="sys", user_prompt="user",
        )
        # model_metadata stays None
        with pytest.raises(ValueError, match="must both be"):
            DecisionArtifact(**kwargs)

    def test_both_populated_is_valid(self):
        # LLM agent path — both LLM fields populated, decision validates.
        kwargs = _minimal_deterministic_kwargs()
        kwargs["model_metadata"] = ModelMetadata(model_name="claude-haiku-4-5")
        kwargs["full_prompt_context"] = FullPromptContext(
            system_prompt="sys", user_prompt="user",
        )
        art = DecisionArtifact(**kwargs)
        assert art.model_metadata is not None
        assert art.full_prompt_context is not None

    def test_deterministic_round_trip_json(self):
        # Deterministic artifact survives model_dump_json() → model_validate_json().
        original = DecisionArtifact(**_minimal_deterministic_kwargs())
        roundtripped = DecisionArtifact.model_validate_json(original.model_dump_json())
        assert roundtripped == original
        assert roundtripped.model_metadata is None
        assert roundtripped.full_prompt_context is None


class TestIsLLMDecision:
    """``is_llm_decision`` helper — consumers gate on it to skip deterministic rows."""

    def test_llm_agent_returns_true(self):
        art = DecisionArtifact(**_minimal_artifact_kwargs())
        assert is_llm_decision(art) is True

    def test_deterministic_returns_false(self):
        art = DecisionArtifact(**_minimal_deterministic_kwargs())
        assert is_llm_decision(art) is False

    def test_discriminator_is_model_metadata(self):
        # Documented behavior: the discriminator key is ``model_metadata is not None``.
        # Pin this so a future refactor can't silently switch to checking the
        # other paired field (full_prompt_context) — they're equivalent today
        # thanks to the paired-presence validator, but the convention is to
        # check model_metadata.
        llm_art = DecisionArtifact(**_minimal_artifact_kwargs())
        det_art = DecisionArtifact(**_minimal_deterministic_kwargs())
        assert is_llm_decision(llm_art) == (llm_art.model_metadata is not None)
        assert is_llm_decision(det_art) == (det_art.model_metadata is not None)


class TestCaptureDecisionDeterministic:
    """``capture_decision`` accepts None for the LLM fields and writes
    a v2 artifact to S3 cleanly."""

    @mock_aws
    def test_capture_with_none_llm_fields(self):
        # Set up moto S3 bucket.
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="alpha-engine-research")

        # Capture a deterministic decision.
        s3_key = capture_decision(
            run_id="run-2026-05-11",
            agent_id="executor:entry_triggers",
            model_metadata=None,
            full_prompt_context=None,
            input_data_snapshot={"ticker": "AAPL", "current_price": 175.25},
            agent_output={"fired_trigger": "pullback", "trigger_kind": "pullback"},
            s3_client=s3,
            timestamp=datetime(2026, 5, 11, 20, 30, 0, tzinfo=timezone.utc),
        )

        # Verify object lands at the canonical path.
        assert s3_key == "decision_artifacts/2026/05/11/executor:entry_triggers/run-2026-05-11.json"
        body = s3.get_object(Bucket="alpha-engine-research", Key=s3_key)["Body"].read()
        artifact = DecisionArtifact.model_validate_json(body)

        # The persisted artifact carries None for both LLM fields.
        assert artifact.schema_version == 2
        assert artifact.model_metadata is None
        assert artifact.full_prompt_context is None
        assert artifact.agent_id == "executor:entry_triggers"
        assert is_llm_decision(artifact) is False


class TestDecisionArtifactRoundTrip:
    def test_dump_then_validate_yields_equal_model(self):
        original = DecisionArtifact(**_minimal_artifact_kwargs())
        dumped = original.model_dump()
        roundtripped = DecisionArtifact.model_validate(dumped)
        assert roundtripped == original

    def test_json_dump_then_validate(self):
        original = DecisionArtifact(**_minimal_artifact_kwargs())
        dumped_json = original.model_dump_json()
        roundtripped = DecisionArtifact.model_validate_json(dumped_json)
        assert roundtripped == original

    def test_with_optional_fields_populated(self):
        kwargs = _minimal_artifact_kwargs()
        kwargs["input_data_summary"] = "Sector=Technology, 28 candidates, ..."
        kwargs["input_data_truncated_at"] = 1_500_000
        art = DecisionArtifact(**kwargs)
        roundtripped = DecisionArtifact.model_validate(art.model_dump())
        assert roundtripped.input_data_summary == kwargs["input_data_summary"]
        assert roundtripped.input_data_truncated_at == 1_500_000


class TestDecisionArtifactValidation:
    def test_truncated_at_must_be_non_negative(self):
        kwargs = _minimal_artifact_kwargs()
        kwargs["input_data_truncated_at"] = -1
        with pytest.raises(ValueError):
            DecisionArtifact(**kwargs)


# ── truncate_snapshot ─────────────────────────────────────────────────────


class TestTruncateSnapshotNoTruncation:
    def test_small_payload_passes_through(self):
        payload = {"market_regime": "neutral", "tickers": ["AAPL", "MSFT"]}
        result, original_size = truncate_snapshot(payload)
        assert result == payload
        assert original_size is None

    def test_at_cap_passes_through(self):
        # Construct a payload exactly at cap (or just under).
        # Using small cap for test speed.
        payload = {"x": "a" * 100}
        result, original_size = truncate_snapshot(payload, cap_bytes=200)
        assert result == payload
        assert original_size is None


class TestTruncateSnapshotWithTruncation:
    def test_oversized_payload_drops_largest_field(self):
        # One huge field, one small field.
        payload = {
            "small_field": "small",
            "huge_field": "x" * 5000,
        }
        result, original_size = truncate_snapshot(payload, cap_bytes=500)

        # Original size reflects pre-truncation
        assert original_size is not None
        assert original_size > 5000

        # The huge field has been replaced with a marker
        assert isinstance(result["huge_field"], dict)
        assert result["huge_field"]["_truncated"] is True
        assert result["huge_field"]["original_field"] == "huge_field"
        assert result["huge_field"]["original_size_bytes"] > 5000

        # The small field is preserved
        assert result["small_field"] == "small"

    def test_repeated_truncation_until_under_cap(self):
        # Multiple oversized fields — truncator must drop them progressively
        # until the result fits.
        payload = {
            f"field_{i}": "x" * 1000 for i in range(10)
        }
        result, original_size = truncate_snapshot(payload, cap_bytes=2000)
        assert original_size > 10000
        # Final serialized size must fit under cap
        assert _serialized_size(result) <= 2000
        # At least some fields are truncated markers
        truncated_count = sum(
            1 for v in result.values()
            if isinstance(v, dict) and v.get("_truncated") is True
        )
        assert truncated_count > 0

    def test_pathological_payload_replaced_with_single_marker(self):
        # A payload where even the dropped-field markers exceed the cap
        # gets replaced with a single top-level marker.
        # (Cap so small that even one marker triggers the fallback.)
        payload = {"x": "a" * 50}
        result, original_size = truncate_snapshot(payload, cap_bytes=20)
        # Either the field is replaced with a marker AND serialized fits...
        # or the whole payload is the fallback marker.
        if "_truncated" in result and result["_truncated"] is True:
            # Fallback path
            assert result["reason"] == "exceeded_cap_after_full_field_drop"
            assert result["original_size_bytes"] == original_size
            assert result["cap_bytes"] == 20

    def test_truncated_payload_remains_json_serializable(self):
        # A truncation result must always serialize cleanly so the wrapper
        # can write it to S3 without surprises.
        payload = {f"f{i}": "x" * 500 for i in range(20)}
        result, _ = truncate_snapshot(payload, cap_bytes=1000)
        # No exception raised — payload is JSON-clean
        json.dumps(result)


class TestTruncateSnapshotDefaultCap:
    def test_default_cap_is_1mb(self):
        # Constant kept here so consumers can rely on the documented default.
        assert _INPUT_SNAPSHOT_DEFAULT_CAP_BYTES == 1_000_000

    def test_typical_agent_payloads_fit_under_default_cap(self):
        # Sector_qual is the largest steady-state input (~300-400 KB worst
        # case from the design doc). Synthesize a comparable payload and
        # verify it doesn't trigger truncation under the default cap.
        chunks = ["A" * 1500] * 200  # ~300 KB worth of content
        payload = {
            "sector": "Healthcare",
            "candidate_tickers": ["AAPL"] * 50,
            "rag_retrieved_chunks": chunks,
        }
        result, original_size = truncate_snapshot(payload)
        # 300 KB < 1 MB cap — should pass through untruncated
        assert original_size is None
        assert result == payload


# ── _build_s3_key ─────────────────────────────────────────────────────────


class TestBuildS3Key:
    def test_canonical_format(self):
        dt = datetime(2026, 4, 29, 22, 30, 0, tzinfo=timezone.utc)
        key = _build_s3_key(
            s3_prefix="decision_artifacts",
            capture_dt=dt,
            agent_id="sector_quant",
            run_id="run-abc123",
        )
        assert key == "decision_artifacts/2026/04/29/sector_quant/run-abc123.json"

    def test_zero_padded_month_and_day(self):
        dt = datetime(2026, 1, 5, 0, 0, 0, tzinfo=timezone.utc)
        key = _build_s3_key(
            s3_prefix="decision_artifacts",
            capture_dt=dt,
            agent_id="cio",
            run_id="r1",
        )
        # Single-digit month + day must be zero-padded for lexical sort
        assert key == "decision_artifacts/2026/01/05/cio/r1.json"

    def test_custom_prefix(self):
        dt = datetime(2026, 4, 29, tzinfo=timezone.utc)
        key = _build_s3_key(
            s3_prefix="staging/decision_artifacts",
            capture_dt=dt,
            agent_id="x",
            run_id="r",
        )
        assert key.startswith("staging/decision_artifacts/")


# ── capture_decision (happy path) ─────────────────────────────────────────


@pytest.fixture
def mocked_s3():
    """moto-mocked S3 client + pre-created bucket."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="alpha-engine-research")
        yield client


def _minimal_capture_kwargs(s3_client, **overrides):
    """Helper for tests: minimal valid kwargs for capture_decision."""
    base = {
        "run_id": "run-2026-04-29",
        "agent_id": "sector_quant",
        "model_metadata": ModelMetadata(model_name="claude-haiku-4-5"),
        "full_prompt_context": FullPromptContext(
            system_prompt="sys", user_prompt="user",
        ),
        "input_data_snapshot": {"market_regime": "neutral", "tickers": ["AAPL"]},
        "agent_output": {"recommendations": [{"ticker": "AAPL", "score": 75}]},
        "s3_client": s3_client,
        "timestamp": datetime(2026, 4, 29, 22, 30, 0, tzinfo=timezone.utc),
    }
    base.update(overrides)
    return base


class TestCaptureDecisionHappyPath:
    def test_writes_artifact_at_canonical_key(self, mocked_s3):
        s3_key = capture_decision(**_minimal_capture_kwargs(mocked_s3))
        assert s3_key == (
            "decision_artifacts/2026/04/29/sector_quant/run-2026-04-29.json"
        )

    def test_artifact_content_round_trips(self, mocked_s3):
        s3_key = capture_decision(**_minimal_capture_kwargs(mocked_s3))
        obj = mocked_s3.get_object(
            Bucket="alpha-engine-research", Key=s3_key,
        )
        body = json.loads(obj["Body"].read())
        # Round-trip via DecisionArtifact for full validation
        artifact = DecisionArtifact.model_validate(body)
        assert artifact.run_id == "run-2026-04-29"
        assert artifact.agent_id == "sector_quant"
        assert artifact.model_metadata.model_name == "claude-haiku-4-5"
        assert artifact.input_data_snapshot["market_regime"] == "neutral"
        assert artifact.agent_output["recommendations"][0]["ticker"] == "AAPL"
        assert artifact.input_data_truncated_at is None
        assert artifact.input_data_summary is None

    def test_content_type_is_json(self, mocked_s3):
        s3_key = capture_decision(**_minimal_capture_kwargs(mocked_s3))
        obj = mocked_s3.get_object(
            Bucket="alpha-engine-research", Key=s3_key,
        )
        assert obj["ContentType"] == "application/json"

    def test_summary_preserved_on_artifact(self, mocked_s3):
        kwargs = _minimal_capture_kwargs(
            mocked_s3,
            input_data_summary="Sector=Technology, 5 candidates",
        )
        s3_key = capture_decision(**kwargs)
        obj = mocked_s3.get_object(Bucket="alpha-engine-research", Key=s3_key)
        artifact = DecisionArtifact.model_validate(json.loads(obj["Body"].read()))
        assert artifact.input_data_summary == "Sector=Technology, 5 candidates"

    def test_custom_bucket_and_prefix(self, mocked_s3):
        # Pre-create the alternate bucket
        mocked_s3.create_bucket(Bucket="my-alt-bucket")
        s3_key = capture_decision(
            **_minimal_capture_kwargs(
                mocked_s3,
                s3_bucket="my-alt-bucket",
                s3_prefix="staging/dec_arts",
            )
        )
        assert s3_key.startswith("staging/dec_arts/2026/04/29/")
        # Content should be in the custom bucket
        obj = mocked_s3.get_object(Bucket="my-alt-bucket", Key=s3_key)
        assert obj["ContentType"] == "application/json"

    def test_idempotent_overwrite_on_same_key(self, mocked_s3):
        # Two captures with the same run_id + agent_id + date overwrite.
        # S3 PUT semantics — the second call wins.
        kwargs1 = _minimal_capture_kwargs(mocked_s3)
        kwargs1["agent_output"] = {"recommendations": [{"ticker": "AAPL"}]}
        capture_decision(**kwargs1)

        kwargs2 = _minimal_capture_kwargs(mocked_s3)
        kwargs2["agent_output"] = {"recommendations": [{"ticker": "MSFT"}]}
        s3_key = capture_decision(**kwargs2)

        obj = mocked_s3.get_object(Bucket="alpha-engine-research", Key=s3_key)
        body = json.loads(obj["Body"].read())
        # Second write wins
        assert body["agent_output"]["recommendations"][0]["ticker"] == "MSFT"


class TestCaptureDecisionTruncation:
    def test_oversized_snapshot_records_truncated_at(self, mocked_s3):
        # Snapshot just over the 1MB cap
        huge_field = "x" * 1_500_000
        kwargs = _minimal_capture_kwargs(
            mocked_s3,
            input_data_snapshot={"small": "ok", "huge_chunks": huge_field},
        )
        s3_key = capture_decision(**kwargs)
        obj = mocked_s3.get_object(Bucket="alpha-engine-research", Key=s3_key)
        artifact = DecisionArtifact.model_validate(json.loads(obj["Body"].read()))

        assert artifact.input_data_truncated_at is not None
        assert artifact.input_data_truncated_at > 1_500_000
        # Truncation marker preserved in the snapshot
        huge_value = artifact.input_data_snapshot["huge_chunks"]
        assert isinstance(huge_value, dict)
        assert huge_value["_truncated"] is True
        # Small field preserved
        assert artifact.input_data_snapshot["small"] == "ok"

    def test_under_cap_snapshot_no_truncation_marker(self, mocked_s3):
        # Steady-state-sized snapshot
        kwargs = _minimal_capture_kwargs(
            mocked_s3,
            input_data_snapshot={"a": "small", "b": [1, 2, 3]},
        )
        s3_key = capture_decision(**kwargs)
        obj = mocked_s3.get_object(Bucket="alpha-engine-research", Key=s3_key)
        artifact = DecisionArtifact.model_validate(json.loads(obj["Body"].read()))
        assert artifact.input_data_truncated_at is None

    def test_custom_cap_overrides_default(self, mocked_s3):
        # Force truncation by passing a tiny cap on a normal-sized payload.
        kwargs = _minimal_capture_kwargs(
            mocked_s3,
            input_data_snapshot={"x": "y" * 1000},
            snapshot_cap_bytes=200,
        )
        s3_key = capture_decision(**kwargs)
        obj = mocked_s3.get_object(Bucket="alpha-engine-research", Key=s3_key)
        artifact = DecisionArtifact.model_validate(json.loads(obj["Body"].read()))
        assert artifact.input_data_truncated_at is not None


class TestCaptureDecisionTimestamp:
    def test_caller_timestamp_used_for_iso_field(self, mocked_s3):
        ts = datetime(2026, 4, 29, 22, 30, 45, tzinfo=timezone.utc)
        kwargs = _minimal_capture_kwargs(mocked_s3, timestamp=ts)
        s3_key = capture_decision(**kwargs)
        obj = mocked_s3.get_object(Bucket="alpha-engine-research", Key=s3_key)
        artifact = DecisionArtifact.model_validate(json.loads(obj["Body"].read()))
        # ISO format starts with the supplied date+time
        assert artifact.timestamp.startswith("2026-04-29T22:30:45")

    def test_default_timestamp_is_now_utc(self, mocked_s3):
        # Don't pass timestamp — the wrapper should default to now()
        kwargs = _minimal_capture_kwargs(mocked_s3)
        del kwargs["timestamp"]
        before = datetime.now(timezone.utc)
        s3_key = capture_decision(**kwargs)
        after = datetime.now(timezone.utc)

        obj = mocked_s3.get_object(Bucket="alpha-engine-research", Key=s3_key)
        artifact = DecisionArtifact.model_validate(json.loads(obj["Body"].read()))
        captured = datetime.fromisoformat(artifact.timestamp)
        assert before <= captured <= after


# ── capture_decision hard-fail paths ──────────────────────────────────────


class TestCaptureDecisionHardFail:
    def test_missing_bucket_raises(self, mocked_s3):
        # mocked_s3 has only "alpha-engine-research"; route to a bucket
        # that doesn't exist
        kwargs = _minimal_capture_kwargs(
            mocked_s3,
            s3_bucket="this-bucket-does-not-exist",
        )
        with pytest.raises(DecisionCaptureWriteError, match=r"Failed to write"):
            capture_decision(**kwargs)

    def test_error_message_names_the_s3_path(self, mocked_s3):
        kwargs = _minimal_capture_kwargs(
            mocked_s3,
            s3_bucket="missing-bucket",
            run_id="run-x",
            agent_id="cio",
        )
        try:
            capture_decision(**kwargs)
            pytest.fail("expected DecisionCaptureWriteError")
        except DecisionCaptureWriteError as e:
            msg = str(e)
            # Bucket name + S3 key both named in the error so operator
            # can find the failed write quickly
            assert "missing-bucket" in msg
            assert "decision_artifacts/2026/04/29/cio/run-x.json" in msg

    def test_error_chains_underlying_boto_exception(self, mocked_s3):
        kwargs = _minimal_capture_kwargs(
            mocked_s3,
            s3_bucket="this-bucket-does-not-exist",
        )
        try:
            capture_decision(**kwargs)
            pytest.fail("expected DecisionCaptureWriteError")
        except DecisionCaptureWriteError as e:
            # __cause__ points at the underlying ClientError per the
            # ``raise ... from exc`` pattern
            assert e.__cause__ is not None

    def test_does_NOT_swallow_silently(self, mocked_s3):
        """Critical guard per feedback_no_silent_fails: capture_decision
        MUST raise on S3 failure, NOT log-and-continue."""
        kwargs = _minimal_capture_kwargs(
            mocked_s3, s3_bucket="missing-bucket",
        )
        # If this test ever fails, it means someone has loosened the
        # hard-fail contract. Capture the regression at the source.
        with pytest.raises(DecisionCaptureWriteError):
            capture_decision(**kwargs)


# ── DecisionCaptureWriteError exception class ─────────────────────────────


class TestDecisionCaptureWriteError:
    def test_is_runtime_error_subclass(self):
        # Subclassing RuntimeError lets callers catch it via the broader
        # type if they want best-effort capture (with their own metric
        # for the silent-loss case), without picking up unrelated
        # exceptions.
        assert issubclass(DecisionCaptureWriteError, RuntimeError)


class TestReproducibilityProvenance:
    """code_sha + data_snapshot_id — optional, additive, back-compat
    (ROADMAP L4567 reproducible-replay arc)."""

    def test_default_none(self):
        art = DecisionArtifact(**_minimal_artifact_kwargs())
        assert art.code_sha is None
        assert art.data_snapshot_id is None

    def test_fields_round_trip(self):
        kwargs = _minimal_artifact_kwargs()
        kwargs["code_sha"] = "abc1234"
        kwargs["data_snapshot_id"] = "arctic:universe@v812"
        art = DecisionArtifact(**kwargs)
        assert art.code_sha == "abc1234"
        assert art.data_snapshot_id == "arctic:universe@v812"

    def test_legacy_artifact_without_provenance_validates(self):
        # Pre-arc artifacts have neither key — must still validate (consumers
        # tolerate absence). Mirrors a stored JSON missing the fields.
        body = _minimal_artifact_kwargs()
        body["model_metadata"] = {"model_name": "claude-haiku-4-5"}
        body["full_prompt_context"] = {"system_prompt": "sys", "user_prompt": "user"}
        art = DecisionArtifact.model_validate(body)
        assert art.code_sha is None and art.data_snapshot_id is None

    def test_capture_stamps_provenance_to_s3(self, mocked_s3):
        kwargs = _minimal_capture_kwargs(
            mocked_s3,
            code_sha="deadbee",
            data_snapshot_id="arctic:universe@v812",
        )
        s3_key = capture_decision(**kwargs)
        obj = mocked_s3.get_object(Bucket="alpha-engine-research", Key=s3_key)
        art = DecisionArtifact.model_validate(json.loads(obj["Body"].read()))
        assert art.code_sha == "deadbee"
        assert art.data_snapshot_id == "arctic:universe@v812"
