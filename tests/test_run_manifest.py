"""`nousergon_lib.run_manifest` — the data-collection run record (I10773).

The tests that matter here are the ones about the FAILURE path: a record that
only appears when the run succeeded is worse than no record at all, because it
makes a fleet of dying units look like a fleet that is not running.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from nousergon_lib import contracts
from nousergon_lib.run_manifest import (
    DEFAULT_MANIFEST_PREFIX,
    NOT_APPLICABLE_REASONS,
    REASON_MAX_LEN,
    SCHEMA_VERSION,
    TRIGGERS,
    VERSION_CAPTURES,
    CodeShaError,
    LocalDirManifestSink,
    NotApplicable,
    S3ManifestSink,
    UnitRun,
    manifest_key,
    new_run_id,
    resolve_code_sha,
    run_unit,
)

SHA = "a" * 40
DAY = "2026-09-14"


class RecordingSink:
    def __init__(self):
        self.writes: list[tuple[str, dict]] = []

    def write(self, key: str, payload: bytes) -> str | None:
        self.writes.append((key, json.loads(payload.decode("utf-8"))))
        return "etag-1"


def _run(fn, sink, **kw):
    return run_unit(
        "D19",
        fn,
        sink=sink,
        trigger=kw.pop("trigger", "scheduled"),
        trading_day=kw.pop("trading_day", DAY),
        log_location=kw.pop("log_location", "/alpha-engine/data-spot:stream-1"),
        code_sha=kw.pop("code_sha", SHA),
        **kw,
    )


# ── the record exists on every path ───────────────────────────────────────


def test_success_writes_one_conforming_manifest():
    sink = RecordingSink()

    def body(ctx: UnitRun):
        ctx.record_input("metron/holdings_universe.json", etag="e0")
        ctx.rows_in = 903
        ctx.record_output("staging/daily_closes/2026-09-14.parquet", rows_out=896, etag="e1")
        ctx.reject("unpriced_symbol", 7)
        return "done"

    result = _run(body, sink)

    assert result.status == "ok"
    assert result.reason == ""
    assert result.value == "done"
    assert len(sink.writes) == 1
    key, manifest = sink.writes[0]
    assert key == f"{DEFAULT_MANIFEST_PREFIX}/D19/{DAY}/{manifest['run_id']}.json"
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["rows_out"] == 896
    assert manifest["rows_rejected"] == [{"reason": "unpriced_symbol", "count": 7}]
    assert contracts.conformance_errors("data_run_manifest", manifest) == []


def test_a_dying_run_writes_failed_with_its_cause_and_reraises():
    """`observability-policy` §3.1: the failure path writes the same telemetry
    as the success path, minus the completion claim."""
    sink = RecordingSink()

    def body(ctx: UnitRun):
        ctx.record_input("metron/holdings_universe.json", etag="e0")
        ctx.rows_in = 903
        raise RuntimeError("vendor returned 503")

    with pytest.raises(RuntimeError, match="vendor returned 503"):
        _run(body, sink)

    key, manifest = sink.writes[0]
    assert manifest["status"] == "failed"
    assert "vendor returned 503" in manifest["reason"]
    # The partial lineage the body HAD established survives — that is what
    # makes a failure manifest diagnostic rather than a tombstone.
    assert manifest["rows_in"] == 903
    assert manifest["inputs"][0]["key"] == "metron/holdings_universe.json"
    # …and no completion claim: nothing was published.
    assert manifest["outputs"] == []
    assert manifest["rows_out"] == 0
    assert contracts.conformance_errors("data_run_manifest", manifest) == []


def test_a_keyboard_interrupt_still_writes_a_manifest():
    """A BaseException is still a dying run, and the record is what says so."""
    sink = RecordingSink()

    def body(ctx: UnitRun):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run(body, sink)
    assert sink.writes[0][1]["status"] == "failed"


def test_not_applicable_is_recorded_not_skipped():
    sink = RecordingSink()

    def body(ctx: UnitRun):
        raise NotApplicable("not_a_trading_day", "2026-09-14 is a Saturday")

    result = _run(body, sink)
    assert result.status == "not_applicable"
    manifest = sink.writes[0][1]
    assert manifest["status"] == "not_applicable"
    assert manifest["reason"] == "not_a_trading_day"
    assert contracts.conformance_errors("data_run_manifest", manifest) == []


def test_not_applicable_reason_must_come_from_the_closed_list():
    with pytest.raises(ValueError, match="closed list"):
        NotApplicable("felt_like_it")


@pytest.mark.parametrize(
    "reason, detail",
    [
        ("disabled_by_declaration", "D15 disabled: enabled=false in config.yaml"),
        ("outside_session_window", "20:05 UTC is outside 13:30-20:00 UTC NYSE session"),
    ],
)
def test_each_new_not_applicable_reason_round_trips_against_the_contract(reason, detail):
    """I10831 deliverable 1: a manifest built with EITHER new reason validates
    against `data_run_manifest.v1` — the reason list and the schema cannot
    drift into two ideas of `closed` for these either."""
    sink = RecordingSink()

    def body(ctx: UnitRun):
        raise NotApplicable(reason, detail)

    result = _run(body, sink)

    assert result.status == "not_applicable"
    manifest = sink.writes[0][1]
    assert manifest["status"] == "not_applicable"
    assert manifest["reason"] == reason
    assert contracts.conformance_errors("data_run_manifest", manifest) == []


def test_no_third_success_state_is_representable():
    """The schema refuses `degraded` outright — there is no call site to guard."""
    sink = RecordingSink()
    _run(lambda ctx: None, sink)
    manifest = dict(sink.writes[0][1])
    manifest["status"] = "degraded"
    assert contracts.conformance_errors("data_run_manifest", manifest) != []


# ── a long reason keeps its tail, not just its head (I11358) ─────────────


def test_an_undersized_reason_is_returned_byte_identical():
    """A reason well under the cap is untouched — no marker, no truncation
    field on the manifest at all."""
    sink = RecordingSink()

    def body(ctx: UnitRun):
        raise RuntimeError("short and unremarkable failure")

    with pytest.raises(RuntimeError):
        _run(body, sink)

    manifest = sink.writes[0][1]
    assert manifest["reason"] == "RuntimeError: short and unremarkable failure"
    assert "reason_truncated_bytes" not in manifest
    assert contracts.conformance_errors("data_run_manifest", manifest) == []


def test_a_6000_char_reason_keeps_its_tail_where_the_failing_date_lives():
    """The 2026-09-21 defect this closes: a window scan's failing entry is the
    LAST thing the exception message says, and a head-only cut at 2000 chars
    dropped it silently. The tail must survive, and the manifest must say it
    was cut."""
    sink = RecordingSink()
    tail = "FAILING_DATE=2026-09-21 status=short_fetch_guard_refused".rjust(200, "-")
    body_text = ("2026-09-08 ok, " * 400)[: 6000 - len(tail)]
    long_message = f"{body_text}{tail}"
    assert len(long_message) == 6000
    assert long_message[-200:] == tail

    def body(ctx: UnitRun):
        raise RuntimeError(long_message)

    with pytest.raises(RuntimeError):
        _run(body, sink)

    manifest = sink.writes[0][1]
    assert len(manifest["reason"]) <= REASON_MAX_LEN
    assert "2026-09-21" in manifest["reason"]
    assert tail in manifest["reason"]
    assert "reason_truncated" in manifest["reason"]
    assert manifest["reason_truncated_bytes"] > 0
    assert contracts.conformance_errors("data_run_manifest", manifest) == []


# ── refusals that happen BEFORE the body runs ─────────────────────────────


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"trigger": "cron"}, "trigger"),
        ({"log_location": ""}, "log_location"),
    ],
)
def test_malformed_invocation_is_refused_before_the_body_runs(kwargs, match):
    sink = RecordingSink()
    ran = []

    with pytest.raises(ValueError, match=match):
        _run(lambda ctx: ran.append(1), sink, **kwargs)
    assert ran == []
    assert sink.writes == []


def test_a_non_unit_id_is_refused():
    with pytest.raises(ValueError, match="audit unit id"):
        run_unit(
            "daily_closes",
            lambda ctx: None,
            sink=RecordingSink(),
            trigger="scheduled",
            trading_day=DAY,
            log_location="x",
            code_sha=SHA,
        )


def test_a_placeholder_code_sha_is_refused(monkeypatch):
    monkeypatch.setenv("NE_DATA_CODE_SHA", "0" * 40)
    with pytest.raises(CodeShaError):
        resolve_code_sha()


def test_env_declared_code_sha_wins(monkeypatch):
    monkeypatch.setenv("NE_DATA_CODE_SHA", "b" * 40)
    assert resolve_code_sha() == "b" * 40


# ── dry run writes nothing, and says so ───────────────────────────────────


def test_dry_run_runs_the_body_and_writes_no_manifest(caplog):
    ran = []
    with caplog.at_level("INFO"):
        result = _run(lambda ctx: ran.append(1), None)
    assert ran == [1]
    assert result.key is None
    assert "NO manifest written" in caplog.text


# ── an unwritable manifest is LOUD ────────────────────────────────────────


def test_a_sink_that_cannot_write_raises_rather_than_swallowing():
    class BrokenSink:
        def write(self, key, payload):
            raise OSError("access denied")

    with pytest.raises(OSError, match="access denied"):
        _run(lambda ctx: None, BrokenSink())


# ── shapes and helpers ────────────────────────────────────────────────────


def test_local_dir_sink_uses_the_same_key_shape(tmp_path):
    sink = LocalDirManifestSink(root=str(tmp_path))
    result = _run(lambda ctx: None, sink)
    written = tmp_path / result.key
    assert written.is_file()
    assert json.loads(written.read_text())["unit_id"] == "D19"


def test_run_ids_are_ulids_and_sort_by_creation_time():
    base = dt.datetime(2026, 9, 14, tzinfo=dt.timezone.utc)
    early = new_run_id(base)
    late = new_run_id(base + dt.timedelta(seconds=5))
    assert len(early) == 26
    assert early < late


def test_manifest_key_shape():
    assert manifest_key("D19", DAY, "R" * 26) == f"data_collection/runs/D19/{DAY}/{'R' * 26}.json"


def test_rows_out_without_a_count_is_not_expressible():
    ctx = UnitRun(
        run_id="R" * 26,
        unit_id="D19",
        trading_day=DAY,
        calendar_date=DAY,
        trigger="scheduled",
        started=dt.datetime.now(dt.timezone.utc),
        code_sha=SHA,
        log_location="x",
    )
    with pytest.raises(TypeError):
        ctx.record_output("k")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="rows_out"):
        ctx.record_output("k", rows_out=-1)


def test_guard_verdicts_ride_on_the_manifest_including_passes():
    sink = RecordingSink()

    def body(ctx: UnitRun):
        ctx.record_output("k", rows_out=10)
        ctx.record_guard("empty_fresh", mode="observe", verdict="ok", detail="10 rows", key="k", value=10.0)

    _run(body, sink)
    manifest = sink.writes[0][1]
    assert manifest["guards"][0]["verdict"] == "ok"
    assert manifest["guards"][0]["mode"] == "observe"
    assert contracts.conformance_errors("data_run_manifest", manifest) == []


def test_not_applicable_reasons_carries_the_i10831_members():
    """A closed list that lacks the word forces the wrong one (I10831 finding
    1): `disabled_by_declaration` (an operator/config decision) and
    `outside_session_window` (a clock fact) are distinct from the
    `no_new_data_declared` catch-all and from each other."""
    assert {"disabled_by_declaration", "outside_session_window"} <= NOT_APPLICABLE_REASONS
    assert NOT_APPLICABLE_REASONS == {
        "not_a_trading_day",
        "no_new_data_declared",
        "disabled_by_declaration",
        "outside_session_window",
    }


# ── per-output version capture (alpha-engine-config-I10892) ───────────────


class _ClientError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3:
    """put_object for the manifest, head_object for the outputs."""

    def __init__(self, heads=None, fail_code=None):
        self.heads = heads or {}
        self.fail_code = fail_code
        self.head_calls: list[tuple[str, str]] = []
        self.puts: list[tuple[str, dict]] = []

    def put_object(self, *, Bucket, Key, Body, ContentType):
        self.puts.append((Key, json.loads(Body.decode("utf-8"))))
        return {"ETag": '"manifest-etag"'}

    def head_object(self, *, Bucket, Key):
        self.head_calls.append((Bucket, Key))
        if self.fail_code:
            raise _ClientError(self.fail_code)
        if (Bucket, Key) not in self.heads:
            raise _ClientError("404")
        return self.heads[(Bucket, Key)]


def _s3_run(body, client, **sink_kw):
    sink = S3ManifestSink(bucket="alpha-engine-research", s3_client=client, **sink_kw)
    _run(body, sink)
    return client.puts[0][1]


def test_record_output_measures_etag_version_and_bytes_without_the_caller_passing_them():
    client = FakeS3(
        {
            ("alpha-engine-research", "market_data/close_history/A.json"): {
                "ETag": '"abc123"',
                "VersionId": "v-3HL4kqtJlcpXroDTDmJ",
                "ContentLength": 48213,
            }
        }
    )
    manifest = _s3_run(lambda ctx: ctx.record_output("market_data/close_history/A.json", rows_out=2511), client)
    out = manifest["outputs"][0]
    assert out["etag"] == "abc123"
    assert out["version_id"] == "v-3HL4kqtJlcpXroDTDmJ"
    assert out["bytes"] == 48213
    assert out["version_capture"] == "head_object"
    assert contracts.conformance_errors("data_run_manifest", manifest) == []


def test_an_unversioned_objects_null_version_id_is_recorded_as_none():
    client = FakeS3({("alpha-engine-research", "k.json"): {"ETag": '"e"', "VersionId": "null", "ContentLength": 3}})
    out = _s3_run(lambda ctx: ctx.record_output("k.json", rows_out=1), client)["outputs"][0]
    assert out["etag"] == "e"
    assert out["version_id"] is None
    assert out["version_capture"] == "head_object"


def test_a_404_head_is_object_absent_not_a_failure():
    """`arcticdb/universe` is recorded as an output but is not one S3 object."""
    client = FakeS3()
    manifest = _s3_run(lambda ctx: ctx.record_output("arcticdb/universe", rows_out=900), client)
    out = manifest["outputs"][0]
    assert manifest["status"] == "ok"
    assert (out["etag"], out["version_id"], out["bytes"]) == (None, None, None)
    assert out["version_capture"] == "object_absent"
    assert contracts.conformance_errors("data_run_manifest", manifest) == []


def test_a_denied_head_is_recorded_as_unavailable_and_the_run_still_succeeds(caplog):
    client = FakeS3(fail_code="AccessDenied")
    with caplog.at_level("WARNING"):
        manifest = _s3_run(lambda ctx: ctx.record_output("k.json", rows_out=1), client)
    out = manifest["outputs"][0]
    assert manifest["status"] == "ok"
    assert out["version_capture"] == "unavailable"
    assert "AccessDenied" in out["version_capture_error"]
    assert out["version_id"] is None
    assert "HeadObject for output k.json failed" in caplog.text
    assert contracts.conformance_errors("data_run_manifest", manifest) == []


def test_caller_supplied_fields_win_and_skip_the_lookup():
    client = FakeS3()
    out = _s3_run(
        lambda ctx: ctx.record_output("k.json", rows_out=1, etag="e1", bytes_=10, version_id="v1"), client
    )["outputs"][0]
    assert (out["etag"], out["bytes"], out["version_id"], out["version_capture"]) == ("e1", 10, "v1", "caller")
    assert client.head_calls == []


def test_a_partially_supplied_record_is_completed_from_the_head_without_overwriting():
    client = FakeS3({("alpha-engine-research", "k.json"): {"ETag": '"store"', "VersionId": "v9", "ContentLength": 7}})
    out = _s3_run(lambda ctx: ctx.record_output("k.json", rows_out=1, etag="caller-etag"), client)["outputs"][0]
    assert (out["etag"], out["version_id"], out["bytes"]) == ("caller-etag", "v9", 7)
    assert out["version_capture"] == "head_object"


def test_head_key_redirects_the_lookup_and_s3_uris_address_their_own_bucket():
    client = FakeS3(
        {
            ("alpha-engine-research", "shadow/2026-09-14/k.json"): {"ETag": '"s"', "VersionId": "vs", "ContentLength": 1},
            ("other-bucket", "x/y.parquet"): {"ETag": '"o"', "VersionId": "vo", "ContentLength": 2},
        }
    )

    def body(ctx):
        ctx.record_output("k.json", rows_out=1)
        ctx.record_output("s3://other-bucket/x/y.parquet", rows_out=1)

    outs = _s3_run(body, client, head_key=lambda k: k if k.startswith("s3://") else f"shadow/2026-09-14/{k}")["outputs"]
    assert [o["version_id"] for o in outs] == ["vs", "vo"]
    assert outs[0]["key"] == "k.json"  # the RECORDED key is unchanged; only the lookup moved


def test_a_sink_without_head_records_not_captured():
    sink = RecordingSink()
    _run(lambda ctx: ctx.record_output("k.json", rows_out=1), sink)
    manifest = sink.writes[0][1]
    assert manifest["outputs"][0]["version_capture"] == "not_captured"
    assert contracts.conformance_errors("data_run_manifest", manifest) == []


def test_a_pre_i10892_manifest_with_null_etag_and_no_version_fields_still_validates():
    sink = RecordingSink()
    _run(lambda ctx: None, sink)
    manifest = dict(sink.writes[0][1])
    manifest["outputs"] = [{"key": "k.json", "etag": None, "schema_version": None, "rows_out": 5, "bytes": None}]
    manifest["rows_out"] = 5
    assert contracts.conformance_errors("data_run_manifest", manifest) == []


def test_version_capture_vocabulary_matches_the_schema():
    schema = contracts.load_schema("data_run_manifest")
    assert set(schema["$defs"]["OutputRef"]["properties"]["version_capture"]["enum"]) == VERSION_CAPTURES


def test_the_closed_vocabularies_match_the_schema():
    """The module and the contract cannot drift into two ideas of `closed`."""
    schema = contracts.load_schema("data_run_manifest")
    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION
    assert set(schema["properties"]["trigger"]["enum"]) == TRIGGERS
    assert set(schema["not_applicable_reasons"]["enum"]) == NOT_APPLICABLE_REASONS
    na_branch = schema["allOf"][2]["then"]["properties"]["reason"]["enum"]
    assert set(na_branch) == NOT_APPLICABLE_REASONS
