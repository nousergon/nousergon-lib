"""
Unit tests for ``nousergon_lib.health`` — the consolidated module-health
enrichment writer/reader (config#1727, Phase C of config#1724).

Two contracts are pinned:

1. :func:`derive_status` — a full TRUTH TABLE over
   ``(error, required-missing, non-required-missing, warnings)``, including
   the structural invariant that a run with an absent *required* deliverable
   can NEVER be ``"ok"`` (it is ``"failed"``), regardless of warnings.

2. :func:`write_health` → :func:`read_health` — a round-trip through a fake
   in-memory S3 client (no live S3, no moto needed), asserting the canonical
   eight legacy keys + the new ``deliverables`` field survive the write/read,
   and that the derived status is what lands on the wire.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from nousergon_lib.health import (
    DEFAULT_HEALTH_BUCKET,
    HEALTH_KEYS,
    STATUSES,
    Deliverable,
    build_health_payload,
    check_upstream_health,
    derive_status,
    health_key,
    missing_required,
    read_health,
    write_health,
)


# ── Fake S3 ──────────────────────────────────────────────────────────────────


class _FakeS3:
    """Minimal in-memory S3 double supporting ``put_object`` / ``get_object``.

    Bodies are stored as bytes; ``get_object`` returns a dict whose
    ``"Body"`` has a ``.read()`` returning those bytes — matching the shape
    :func:`read_health` consumes. ``get_object`` on a missing key raises, so
    the reader's "collapse to None" path is exercised too.
    """

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], bytes] = {}
        self.put_calls: list[dict] = []

    def put_object(self, *, Bucket, Key, Body, ContentType=None):
        self.put_calls.append(
            {"Bucket": Bucket, "Key": Key, "ContentType": ContentType}
        )
        self.store[(Bucket, Key)] = Body

    def get_object(self, *, Bucket, Key):
        try:
            body = self.store[(Bucket, Key)]
        except KeyError as exc:
            raise KeyError(f"no such key s3://{Bucket}/{Key}") from exc
        return {"Body": _Body(body)}


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


# ── Deliverable helpers ──────────────────────────────────────────────────────


def _req(produced: bool) -> Deliverable:
    return Deliverable(name="req", required=True, produced=produced)


def _opt(produced: bool) -> Deliverable:
    return Deliverable(name="opt", required=False, produced=produced)


# ── derive_status TRUTH TABLE ────────────────────────────────────────────────
#
# Columns: deliverables, error, warnings → expected status.
# Precedence under test: error > required-missing > (warnings | opt-missing) > ok.

_TRUTH_TABLE = [
    # id,                          deliverables,               error,   warnings,      expected
    ("all-good-ok",                [_req(True), _opt(True)],   None,    [],            "ok"),
    ("empty-deliverables-ok",      [],                         None,    [],            "ok"),
    ("warnings-only-degraded",     [_req(True)],               None,    ["w"],         "degraded"),
    ("opt-missing-degraded",       [_req(True), _opt(False)],  None,    [],            "degraded"),
    ("opt-missing+warn-degraded",  [_req(True), _opt(False)],  None,    ["w"],         "degraded"),
    ("required-missing-failed",    [_req(False)],              None,    [],            "failed"),
    # INVARIANT: required-missing can NEVER be "ok" even with zero warnings.
    ("required-missing-no-warn",   [_req(False), _opt(True)],  None,    [],            "failed"),
    # INVARIANT: required-missing outranks warnings → still failed, never degraded.
    ("required-missing+warn",      [_req(False)],              None,    ["w"],         "failed"),
    ("error-failed",               [_req(True)],               "boom",  [],            "failed"),
    # error outranks a clean deliverable set.
    ("error-outranks-ok",          [_req(True), _opt(True)],   "boom",  [],            "failed"),
    # error AND required-missing → still failed (both point the same way).
    ("error+required-missing",     [_req(False)],              "boom",  ["w"],         "failed"),
]


@pytest.mark.parametrize(
    "deliverables,error,warnings,expected",
    [row[1:] for row in _TRUTH_TABLE],
    ids=[row[0] for row in _TRUTH_TABLE],
)
def test_derive_status_truth_table(deliverables, error, warnings, expected):
    assert derive_status(deliverables, error=error, warnings=warnings) == expected


def test_ok_is_structurally_impossible_with_missing_required():
    """The load-bearing invariant, asserted directly: no combination of
    warnings can lift a required-missing run to "ok"."""
    deliverables = [_req(False), _opt(True)]
    for warnings in ([], ["w"], ["a", "b"]):
        for error in (None, "e"):
            assert (
                derive_status(deliverables, error=error, warnings=warnings) != "ok"
            )


def test_missing_required_identifies_blockers():
    dels = [_req(False), _opt(False), Deliverable(name="ok", produced=True)]
    blockers = missing_required(dels)
    assert [b.name for b in blockers] == ["req"]


def test_statuses_ordering_worst_first():
    # Consumers rely on STATUSES being worst → best for worst-case reduction.
    assert STATUSES == ("failed", "degraded", "ok")


# ── Payload shape ────────────────────────────────────────────────────────────

_LEGACY_KEYS = {
    "module",
    "status",
    "last_success",
    "run_date",
    "duration_seconds",
    "summary",
    "warnings",
    "error",
}


def test_payload_has_legacy_keys_plus_deliverables():
    payload = build_health_payload(
        module_name="predictor",
        deliverables=[_req(True)],
        run_date="2026-07-05",
        duration_seconds=12.34,
    )
    assert _LEGACY_KEYS.issubset(payload.keys())
    assert "deliverables" in payload
    # exactly the legacy keys + deliverables, nothing extra
    assert set(payload.keys()) == _LEGACY_KEYS | {"deliverables"}
    assert payload["duration_seconds"] == 12.3  # rounded to 1dp like the copies
    assert payload["deliverables"] == [
        {"name": "req", "required": True, "produced": True, "detail": ""}
    ]


def test_last_success_nulled_on_failed():
    ok = build_health_payload("m", [_req(True)], "2026-07-05", 1.0)
    assert ok["status"] == "ok"
    assert ok["last_success"] is not None

    failed = build_health_payload("m", [_req(False)], "2026-07-05", 1.0)
    assert failed["status"] == "failed"
    assert failed["last_success"] is None


# ── HEALTH_KEYS registry ─────────────────────────────────────────────────────


def test_health_keys_cover_the_five_modules():
    assert set(HEALTH_KEYS) >= {
        "data",
        "research",
        "predictor",
        "backtester",
        "executor",
    }
    # Registry-aligned primary keys (config#1728 — live producers, not stale shorthands)
    assert HEALTH_KEYS["data"] == "health/daily_data.json"
    assert HEALTH_KEYS["predictor"] == "health/predictor_inference.json"
    assert HEALTH_KEYS["research"] == "health/research.json"


def test_registry_health_artifacts_match_health_keys_values():
    from nousergon_lib.health import REGISTRY_HEALTH_ARTIFACTS

    registry_values = set(REGISTRY_HEALTH_ARTIFACTS.values())
    health_key_values = {
        HEALTH_KEYS["data"],
        HEALTH_KEYS["research"],
        HEALTH_KEYS["predictor"],
        HEALTH_KEYS["backtester"],
    }
    assert registry_values == health_key_values


def test_health_key_falls_back_for_unknown_module():
    assert health_key("executor") == "health/executor.json"
    assert health_key("some_new_module") == "health/some_new_module.json"


# ── write → read round-trip ──────────────────────────────────────────────────


def test_write_read_round_trip():
    s3 = _FakeS3()
    deliverables = [
        Deliverable(name="signals.json", required=True, produced=True, detail="512 rows"),
        Deliverable(name="debug_dump", required=False, produced=False, detail="skipped"),
    ]
    written = write_health(
        module_name="predictor_inference",
        deliverables=deliverables,
        run_date="2026-07-05",
        duration_seconds=8.0,
        summary={"rows": 512},
        warnings=["slow fetch"],
        s3_client=s3,
    )

    # non-required missing + warnings, no required missing → degraded
    assert written["status"] == "degraded"

    # landed at the canonical key in the default bucket
    assert s3.put_calls == [
        {
            "Bucket": DEFAULT_HEALTH_BUCKET,
            "Key": "health/predictor_inference.json",
            "ContentType": "application/json",
        }
    ]

    got = read_health("predictor_inference", s3_client=s3)
    assert got == written
    raw = json.loads(s3.store[(DEFAULT_HEALTH_BUCKET, "health/predictor_inference.json")])
    assert raw["module"] == "predictor_inference"
    assert raw["summary"] == {"rows": 512}
    assert len(raw["deliverables"]) == 2


def test_read_missing_returns_none():
    s3 = _FakeS3()
    assert read_health("nope", s3_client=s3) is None


def test_write_derives_failed_and_nulls_last_success_on_missing_required():
    s3 = _FakeS3()
    written = write_health(
        module_name="research",
        deliverables=[Deliverable(name="factors", required=True, produced=False)],
        run_date="2026-07-05",
        duration_seconds=3.0,
        s3_client=s3,
    )
    assert written["status"] == "failed"
    assert written["last_success"] is None
    assert read_health("research", s3_client=s3)["status"] == "failed"


def test_write_never_raises_on_s3_failure():
    class _Boom:
        def put_object(self, **_):
            raise RuntimeError("s3 down")

    # enrichment must not take down the run: returns payload, swallows error
    payload = write_health(
        module_name="executor",
        deliverables=[_req(True)],
        run_date="2026-07-05",
        duration_seconds=1.0,
        s3_client=_Boom(),
    )
    assert payload["status"] == "ok"


# ── check_upstream_health ────────────────────────────────────────────────────


def test_check_upstream_health_unknown_stale_and_fresh():
    s3 = _FakeS3()
    # fresh: written just now with last_success populated
    write_health(
        module_name="data",
        deliverables=[_req(True)],
        run_date="2026-07-05",
        duration_seconds=1.0,
        s3_client=s3,
    )
    # stale: a failed stamp (last_success = None) → treated as stale
    write_health(
        module_name="research",
        deliverables=[_req(False)],
        run_date="2026-07-05",
        duration_seconds=1.0,
        s3_client=s3,
    )

    result = check_upstream_health(
        ["data", "research", "predictor"], s3_client=s3, max_age_hours=48
    )

    assert result["data"]["status"] == "ok"
    assert result["data"]["stale"] is False
    assert result["data"]["age_hours"] >= 0

    assert result["research"]["status"] == "failed"
    assert result["research"]["stale"] is True  # last_success None → age -1

    assert result["predictor"] == {"status": "unknown", "age_hours": -1, "stale": True}


def test_check_upstream_health_flags_old_stamp_as_stale():
    s3 = _FakeS3()
    stale_payload = {
        "module": "data",
        "status": "ok",
        "last_success": datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat(),
        "run_date": "2020-01-01",
        "duration_seconds": 1.0,
        "summary": {},
        "warnings": [],
        "error": None,
        "deliverables": [],
    }
    s3.store[(DEFAULT_HEALTH_BUCKET, "health/daily_data.json")] = json.dumps(
        stale_payload
    ).encode("utf-8")

    result = check_upstream_health(["data"], s3_client=s3, max_age_hours=48)
    assert result["data"]["status"] == "ok"
    assert result["data"]["stale"] is True  # years old
    assert result["data"]["age_hours"] > 48
