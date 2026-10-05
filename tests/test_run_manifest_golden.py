"""Golden-fixture contract tests for ``data_run_manifest.v1`` (alpha-engine-config-I10773).

``tests/test_run_manifest.py`` asserts the record's *semantics* field by field.
This module asserts its *bytes*: that the production writer path —
:func:`run_unit` through :class:`S3ManifestSink` — reproduces a manifest the
collector really wrote, byte-for-byte, at the key layout readers list.

Why bytes and not fields. ``nousergon-data``'s readers (the data gate's
evidence and exit-criteria passes, the completion check) parse these objects
out of S3, and a consumer pinned to an older lib reads manifests a newer one
wrote. A refactor that renames a key, drops an always-present field, changes
the status spelling, the timestamp format, the float rendering of
``cost_usd``, the serializer's key order or indent, or the
``{prefix}/{unit_id}/{trading_day}/{run_id}.json`` layout is a contract change
— and every one of those passes a field-by-field test written after it.

**Provenance of the fixtures** (``tests/fixtures/run_manifest/``). Each is a
real manifest the collector wrote to
``s3://alpha-engine-research/data_collection/runs/`` (fetched read-only on
2026-10-05), one per status:

* ``ok_D24.json``             — ``D24/2026-10-01/01M3WS6S6SAFC863RKA7AV78QW.json``
* ``not_applicable_D01.json`` — ``D01/2026-10-02/01M40XJVKT1V525TEZQVN3XVAK.json``
* ``failed_D02.json``         — ``D02/2026-10-02/01M40GE6V29Y40VMRZNVDG1QNE.json``

Manifests are run metadata only; nothing in them is a credential. They were
still scrubbed of deployment-identifying values before being committed to a
public repository — ``code_sha``, the EC2 instance id inside ``log_location``,
and each output's ``etag`` / ``version_id`` are replaced with synthetic values
of the same shape, and ``failed_D02.json``'s ``reason`` is shortened to its
head. Every key, status, nesting and type is exactly as written in production.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import types

import pytest

from nousergon_lib import contracts, run_manifest
from nousergon_lib.run_manifest import (
    DEFAULT_MANIFEST_PREFIX,
    SCHEMA_VERSION,
    NotApplicable,
    S3ManifestSink,
    UnitRun,
    run_unit,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "run_manifest"
BUCKET = "alpha-engine-research"

GOLDEN = {
    "ok": "ok_D24.json",
    "not_applicable": "not_applicable_D01.json",
    "failed": "failed_D02.json",
}

#: The keys every manifest carries, whatever its status and whatever the body
#: recorded. A literal on purpose: adding, removing or renaming one is a
#: contract change and must be made here, deliberately, in the same PR.
ALWAYS_PRESENT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "unit_id",
        "trigger",
        "trading_day",
        "calendar_date",
        "status",
        "reason",
        "started",
        "finished",
        "code_sha",
        "log_location",
        "inputs",
        "outputs",
        "rows_in",
        "rows_out",
        "rows_rejected",
        "cost_usd",
        "compute",
    }
)

#: Keys that appear only when the body recorded something under them.
OPTIONAL_KEYS = frozenset({"denominator", "excluded", "guards", "metrics", "reason_truncated_bytes"})

COMPUTE_KEYS = frozenset({"instance_type", "spot", "escalated_on_demand", "interruptions", "region"})

OUTPUT_KEYS = frozenset({"key", "etag", "schema_version", "rows_out", "bytes", "version_id", "version_capture"})

GUARD_KEYS = frozenset({"guard", "mode", "verdict", "detail", "key", "value", "baseline"})


def _golden_bytes(status: str) -> bytes:
    return (FIXTURES / GOLDEN[status]).read_bytes()


def _golden(status: str) -> dict:
    return json.loads(_golden_bytes(status).decode("utf-8"))


def _parse_utc(stamp: str) -> dt.datetime:
    return dt.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


class _NotFound(Exception):
    def __init__(self):
        super().__init__("404")
        self.response = {"Error": {"Code": "404"}}


class _RecordingS3:
    """A boto3-shaped S3 client: keeps the manifest PUT's raw bytes, and
    answers HeadObject for the outputs the golden manifest says were published."""

    def __init__(self, heads: dict[str, dict]):
        self.heads = heads
        self.puts: list[dict] = []

    def put_object(self, *, Bucket, Key, Body, ContentType):
        self.puts.append({"Bucket": Bucket, "Key": Key, "Body": Body, "ContentType": ContentType})
        return {"ETag": '"manifest-etag"'}

    def head_object(self, *, Bucket, Key):
        if Bucket != BUCKET or Key not in self.heads:
            raise _NotFound()
        return self.heads[Key]


def _replay(golden: dict, monkeypatch) -> tuple[_RecordingS3, object]:
    """Drive the PUBLIC API with the facts a real run recorded, and capture what
    the production sink writes.

    The body only records what the collector's own body recorded (inputs,
    guards, metrics, outputs, rejects, spend) and ends the way that run ended.
    Everything else on the manifest — its key set, ``schema_version``,
    ``rows_out``, the ``rows_rejected`` shape, ``compute``, both timestamps'
    format, the serialization, and the object key — is the library's to
    produce, which is what is under test.
    """
    compute = golden["compute"]
    monkeypatch.setenv("NE_DATA_INSTANCE_TYPE", compute["instance_type"])
    monkeypatch.setenv("NE_DATA_LIFECYCLE", "spot" if compute["spot"] else "on-demand")
    monkeypatch.setenv("NE_DATA_ESCALATED_ON_DEMAND", "true" if compute["escalated_on_demand"] else "false")
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setenv("AWS_REGION", compute["region"])

    # The two values a real run cannot reproduce on demand: its ULID (80 random
    # bits) and the wall-clock instant its `finally` ran.
    monkeypatch.setattr(run_manifest, "new_run_id", lambda _started: golden["run_id"])
    finished = _parse_utc(golden["finished"])

    class _FrozenDatetime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return finished

    monkeypatch.setattr(
        run_manifest,
        "dt",
        types.SimpleNamespace(datetime=_FrozenDatetime, timezone=dt.timezone, date=dt.date),
    )

    heads = {
        o["key"]: {"ETag": f'"{o["etag"]}"', "VersionId": o["version_id"], "ContentLength": o["bytes"]}
        for o in golden["outputs"]
        if o.get("version_capture") == "head_object"
    }
    client = _RecordingS3(heads)
    sink = S3ManifestSink(bucket=BUCKET, s3_client=client)

    def body(ctx: UnitRun):
        for ref in golden["inputs"]:
            ctx.record_input(ref["key"], etag=ref["etag"], version=ref["version"], schema_version=ref["schema_version"])
        ctx.rows_in = golden["rows_in"]
        for row in golden["rows_rejected"]:
            ctx.reject(row["reason"], row["count"])
        if golden["cost_usd"]:
            ctx.spend(golden["cost_usd"])
        for g in golden.get("guards", []):
            ctx.record_guard(
                g["guard"],
                mode=g["mode"],
                verdict=g["verdict"],
                detail=g["detail"],
                key=g["key"],
                value=g["value"],
                baseline=g["baseline"],
            )
        for m in golden.get("metrics", []):
            ctx.record_metric(m)
        for o in golden["outputs"]:
            ctx.record_output(o["key"], rows_out=o["rows_out"], schema_version=o["schema_version"])
        if golden["status"] == "not_applicable":
            raise NotApplicable(golden["reason"], "replayed from the golden fixture")
        if golden["status"] == "failed":
            type_name, _, message = golden["reason"].partition(": ")
            raise type(type_name, (Exception,), {})(message)
        return "done"

    def run():
        return run_unit(
            golden["unit_id"],
            body,
            sink=sink,
            trigger=golden["trigger"],
            trading_day=golden["trading_day"],
            calendar_date=golden["calendar_date"],
            log_location=golden["log_location"],
            code_sha=golden["code_sha"],
            now=_parse_utc(golden["started"]),
        )

    if golden["status"] == "failed":
        with pytest.raises(Exception) as excinfo:
            run()
        # The body's own exception is re-raised AFTER the record is durable.
        assert type(excinfo.value).__name__ == golden["reason"].partition(": ")[0]
        return client, None
    return client, run()


# ── the bytes ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("status", sorted(GOLDEN))
def test_the_production_writer_reproduces_a_real_manifest_byte_for_byte(status, monkeypatch):
    client, _ = _replay(_golden(status), monkeypatch)

    assert len(client.puts) == 1, "exactly one manifest per execution, on every path"
    put = client.puts[0]
    assert put["Body"] == _golden_bytes(status)
    assert put["ContentType"] == "application/json"


@pytest.mark.parametrize("status", sorted(GOLDEN))
def test_the_object_lands_at_the_layout_readers_list(status, monkeypatch):
    golden = _golden(status)
    client, result = _replay(golden, monkeypatch)

    expected = f"data_collection/runs/{golden['unit_id']}/{golden['trading_day']}/{golden['run_id']}.json"
    assert DEFAULT_MANIFEST_PREFIX == "data_collection/runs"
    assert client.puts[0]["Bucket"] == BUCKET
    assert client.puts[0]["Key"] == expected
    if result is not None:
        assert result.key == expected
        assert result.status == status


# ── the fixtures themselves ───────────────────────────────────────────────


@pytest.mark.parametrize("status", sorted(GOLDEN))
def test_each_golden_fixture_conforms_to_the_published_schema(status):
    pytest.importorskip("jsonschema")
    golden = _golden(status)
    assert golden["schema_version"] == SCHEMA_VERSION
    assert golden["status"] == status
    assert contracts.conformance_errors("data_run_manifest", golden) == []


@pytest.mark.parametrize("status", sorted(GOLDEN))
def test_each_golden_fixture_carries_exactly_the_contract_keys(status):
    golden = _golden(status)
    keys = set(golden)
    assert ALWAYS_PRESENT_KEYS <= keys
    assert keys - ALWAYS_PRESENT_KEYS <= OPTIONAL_KEYS
    assert set(golden["compute"]) == COMPUTE_KEYS
    for output in golden["outputs"]:
        assert set(output) == OUTPUT_KEYS
    for guard in golden.get("guards", []):
        assert set(guard) == GUARD_KEYS


def test_a_bare_run_writes_exactly_the_always_present_keys(monkeypatch):
    """A body that records nothing gets every always-present key and no
    optional one — so a reader may index the former unconditionally."""
    client = _RecordingS3({})
    run_unit(
        "D24",
        lambda ctx: None,
        sink=S3ManifestSink(bucket=BUCKET, s3_client=client),
        trigger="scheduled",
        trading_day="2026-10-01",
        log_location="cloudwatch:/alpha-engine/data-spot",
        code_sha=_golden("ok")["code_sha"],
    )
    manifest = json.loads(client.puts[0]["Body"].decode("utf-8"))
    assert set(manifest) == ALWAYS_PRESENT_KEYS
    assert set(manifest["compute"]) == COMPUTE_KEYS
    assert manifest["rows_rejected"] == []
    assert manifest["inputs"] == [] and manifest["outputs"] == []
