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
    SCHEMA_VERSION,
    TRIGGERS,
    CodeShaError,
    LocalDirManifestSink,
    NotApplicable,
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


def test_no_third_success_state_is_representable():
    """The schema refuses `degraded` outright — there is no call site to guard."""
    sink = RecordingSink()
    _run(lambda ctx: None, sink)
    manifest = dict(sink.writes[0][1])
    manifest["status"] = "degraded"
    assert contracts.conformance_errors("data_run_manifest", manifest) != []


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


def test_the_closed_vocabularies_match_the_schema():
    """The module and the contract cannot drift into two ideas of `closed`."""
    schema = contracts.load_schema("data_run_manifest")
    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION
    assert set(schema["properties"]["trigger"]["enum"]) == TRIGGERS
    assert set(schema["not_applicable_reasons"]["enum"]) == NOT_APPLICABLE_REASONS
    na_branch = schema["allOf"][2]["then"]["properties"]["reason"]["enum"]
    assert set(na_branch) == NOT_APPLICABLE_REASONS
