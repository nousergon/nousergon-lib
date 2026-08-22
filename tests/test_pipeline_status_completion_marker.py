"""The completion marker's cycle block. ``alpha-engine-config-I8186``.

The ``LIVE_MARKER`` fixture is the verbatim body of
``_sf_completion/ne-weekly-freshness-pipeline/2026-08-22.json``, read from S3
on 2026-08-22 — including the ``States.Format`` double-encoding every object in
that prefix carries.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from nousergon_lib.pipeline_status.completion_marker import (
    CLAIM_SF_EXECUTION_TERMINAL,
    VERDICT_UNKNOWN,
    augment_marker,
    decode_marker,
    marker_key,
    marker_verdict,
    merge_cycle_shape,
    read_marker,
)
from nousergon_lib.pipeline_status.cycle_shape import build_cycle_shape
from nousergon_lib.pipeline_status.read import RunStatus
from nousergon_lib.pipeline_status.work import classify_work

PIPELINE = "ne-weekly-freshness-pipeline"
SPINE = ("A", "B", "C")

#: Verbatim, 2026-08-22.
LIVE_MARKER = {
    "sf": "ne-weekly-freshness-pipeline",
    "execution_arn": (
        "arn:aws:states:us-east-1:711398986525:execution:"
        "ne-weekly-freshness-pipeline:watch-rerun-2026-08-22-3"
    ),
    "status": "SUCCEEDED",
    "started_at": "2026-08-22T15:53:30.910Z",
    "completed_at": "2026-08-22T16:08:02.407Z",
    "cycle_key": "2026-08-22",
    "substrate_relaunches": 0,
}


def _outcome(status, entered, *, name, arn):
    return classify_work(
        state_machine_name=PIPELINE,
        status=status,
        entered_states=list(entered),
        execution_arn=arn,
        execution_name=name,
        stage_spine=SPINE,
        skip_terminals=frozenset(),
    )


def _recovered_shape():
    return build_cycle_shape(
        pipeline=PIPELINE,
        run_date="2026-08-22",
        outcomes=[
            (_outcome(RunStatus.FAILED, ["A", "B"], name="scheduled", arn="arn:a"), "weekly", ["A", "B"]),
            (_outcome(RunStatus.SUCCEEDED, ["C"], name="watch-rerun-2026-08-22-3", arn="arn:b"), "watch-rerun", ["C"]),
        ],
        stage_spine=SPINE,
    )


def _partial_shape():
    return build_cycle_shape(
        pipeline=PIPELINE,
        run_date="2026-08-22",
        outcomes=[
            (_outcome(RunStatus.SUCCEEDED, ["A"], name="watch-rerun-2026-08-22-3", arn="arn:b"), "watch-rerun", ["A"]),
        ],
        stage_spine=SPINE,
    )


# ── The double-encoding ──────────────────────────────────────────────────────


def test_the_states_format_double_encoding_is_unwrapped():
    """Every object in this prefix is a JSON STRING containing the JSON object.

    One ``json.loads`` yields a ``str``, so a consumer guarding with
    ``isinstance(marker, dict)`` falls through to its "no usable status" path
    on every read, silently, forever.
    """
    body = json.dumps(json.dumps(LIVE_MARKER)).encode()
    marker, levels = decode_marker(body)
    assert levels == 2
    assert marker["status"] == "SUCCEEDED"


def test_a_singly_encoded_marker_still_decodes():
    marker, levels = decode_marker(json.dumps(LIVE_MARKER).encode())
    assert levels == 1
    assert marker["cycle_key"] == "2026-08-22"


def test_a_marker_that_is_not_an_object_is_an_error_not_an_empty_dict():
    with pytest.raises(ValueError, match="not an object"):
        decode_marker(json.dumps(json.dumps([1, 2])).encode())


# ── The verdict reader ───────────────────────────────────────────────────────


def test_the_live_2026_08_22_marker_resolves_to_unknown_not_a_pass():
    """It says SUCCEEDED and names ONE execution. That is not a cycle verdict."""
    assert LIVE_MARKER["status"] == "SUCCEEDED"
    assert marker_verdict(LIVE_MARKER) == VERDICT_UNKNOWN


def test_an_absent_marker_is_unknown():
    assert marker_verdict(None) == VERDICT_UNKNOWN


def test_an_augmented_marker_carries_the_verdict():
    merged = merge_cycle_shape(LIVE_MARKER, _recovered_shape())
    assert marker_verdict(merged) == "completed"


# ── The merge ────────────────────────────────────────────────────────────────


def test_every_field_the_state_machine_wrote_survives_untouched():
    merged = merge_cycle_shape(LIVE_MARKER, _recovered_shape())
    for key, value in LIVE_MARKER.items():
        assert merged[key] == value, f"{key} was altered"
    assert LIVE_MARKER == LIVE_MARKER  # pure: the input is not mutated
    assert "cycle" not in LIVE_MARKER


def test_the_merged_marker_names_the_narrow_claim_the_sf_actually_made():
    merged = merge_cycle_shape(LIVE_MARKER, _recovered_shape())
    assert merged["claim"] == CLAIM_SF_EXECUTION_TERMINAL


def test_a_recovery_that_completed_the_cycle_is_completed_and_says_it_was_a_tail():
    """The overcorrection I8186 forbids: a legitimate recovery is not a failure."""
    merged = merge_cycle_shape(LIVE_MARKER, _recovered_shape())
    assert merged["cycle_verdict"] == "completed"
    assert merged["cycle"]["is_recovery_tail"] is True
    assert merged["cycle"]["execution_count"] == 2
    assert [e["execution_arn"] for e in merged["cycle"]["executions"]] == ["arn:a", "arn:b"]


def test_a_marker_written_by_a_tail_that_did_not_complete_the_cycle_says_so():
    merged = merge_cycle_shape(LIVE_MARKER, _partial_shape())
    assert merged["status"] == "SUCCEEDED", "the envelope claim is preserved"
    assert merged["cycle_verdict"] == "incomplete"
    assert merged["cycle"]["stages_missing"] == ["B", "C"]
    assert marker_verdict(merged) != "completed"


# ── The S3 round trip ────────────────────────────────────────────────────────


class _S3:
    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = objects or {}
        self.puts: list[dict[str, Any]] = []

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        key = kwargs["Key"]
        if key not in self.objects:
            exc = Exception("nope")
            exc.response = {"Error": {"Code": "NoSuchKey"}}  # type: ignore[attr-defined]
            raise exc

        body = self.objects[key]

        class _Body:
            @staticmethod
            def read() -> bytes:
                return body

        return {"Body": _Body()}

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        self.puts.append(kwargs)
        self.objects[kwargs["Key"]] = kwargs["Body"]
        return {}


KEY = marker_key(PIPELINE, "2026-08-22")


def test_augment_reads_merges_and_writes_back_to_the_same_key():
    s3 = _S3({KEY: json.dumps(json.dumps(LIVE_MARKER)).encode()})
    merged = augment_marker(_recovered_shape(), s3_client=s3, bucket="b")
    assert merged is not None
    assert s3.puts[0]["Key"] == KEY
    written = json.loads(s3.puts[0]["Body"])
    assert written["cycle_verdict"] == "completed"
    assert written["execution_arn"] == LIVE_MARKER["execution_arn"]


def test_a_missing_marker_is_reported_not_invented():
    """No marker for a cycle means the state machine's own write did not happen."""
    s3 = _S3()
    assert augment_marker(_recovered_shape(), s3_client=s3, bucket="b") is None
    assert not s3.puts


def test_a_denial_on_read_never_becomes_a_written_marker():
    class _Denied(_S3):
        def get_object(self, **_: Any) -> dict[str, Any]:
            exc = Exception("denied")
            exc.response = {"Error": {"Code": "AccessDenied"}}  # type: ignore[attr-defined]
            raise exc

    s3 = _Denied()
    assert augment_marker(_recovered_shape(), s3_client=s3, bucket="b") is None
    assert not s3.puts, "a denied read must never produce a fabricated marker"


def test_read_marker_raises_on_a_denial_rather_than_returning_none():
    class _Denied(_S3):
        def get_object(self, **_: Any) -> dict[str, Any]:
            exc = Exception("denied")
            exc.response = {"Error": {"Code": "AccessDenied"}}  # type: ignore[attr-defined]
            raise exc

    with pytest.raises(Exception, match="denied"):
        read_marker(_Denied(), bucket="b", key=KEY)


def test_a_failed_write_still_returns_the_merged_marker_and_never_raises():
    class _WriteFails(_S3):
        def put_object(self, **_: Any) -> dict[str, Any]:
            raise RuntimeError("boom")

    s3 = _WriteFails({KEY: json.dumps(LIVE_MARKER).encode()})
    merged = augment_marker(_recovered_shape(), s3_client=s3, bucket="b")
    assert merged is not None and merged["cycle_verdict"] == "completed"


def test_the_key_shape_matches_the_live_prefix():
    assert KEY == "_sf_completion/ne-weekly-freshness-pipeline/2026-08-22.json"
