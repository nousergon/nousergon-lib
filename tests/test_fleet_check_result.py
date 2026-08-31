"""Tests for the shared fleet check-result envelope.

Lifted from `alpha-engine-config/scripts/test_fleet_check_result.py` alongside
the module (alpha-engine-config-I5863). The repo-local adoption test did not
come with it — "every producer in alpha-engine-config imports this" is a fact
about that repo's tree, not about the lib, and asserting it from here would
either pass vacuously or couple the lib's suite to a private repo's layout.
That assertion stays where it can actually see the producers.

This module is the PRODUCER half of a cross-repo contract:
`crucible-dashboard/loaders/fleet_checks_loader.py` is the consumer. The fields
asserted here are the ones that side reads — changing any of them without a
paired dashboard PR silently blanks a console row, which is the failure this
surface exists to prevent.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from nousergon_lib import fleet_check_result as fcr

NOW = datetime(2026, 7, 29, 20, 0, tzinfo=timezone.utc)


def _ok(**over):
    kw = {"check_id": "c", "label": "C", "status": fcr.STATUS_OK,
          "summary": "fine", "cadence_minutes": 1440, "now": NOW}
    kw.update(over)
    return fcr.build(**kw)


# --- the consumer contract -------------------------------------------------

def test_envelope_carries_every_field_the_console_reads():
    """crucible-dashboard loaders/fleet_checks_loader.interpret() reads exactly
    these. A missing one blanks a console row rather than erroring."""
    e = _ok()
    for field in ("schema_version", "check_id", "label", "ran_at", "status",
                  "summary", "cadence_minutes", "deep_link", "findings"):
        assert field in e, field


def test_status_values_match_the_consumer_vocabulary():
    """The console maps ok/attention/error and derives stale/unreadable itself.
    A producer inventing a fifth value renders as its literal string."""
    assert (fcr.STATUS_OK, fcr.STATUS_ATTENTION, fcr.STATUS_ERROR) == (
        "ok", "attention", "error")


def test_schema_version_is_pinned():
    """The consumer branches on this. A silent bump is a silent contract break."""
    assert _ok()["schema_version"] == 1


def test_envelope_is_json_serialisable():
    json.dumps(_ok(findings=[{"key": "k", "detail": "d"}]))


def test_ran_at_is_tz_aware_iso():
    assert _ok()["ran_at"] == NOW.isoformat()
    assert "+00:00" in _ok()["ran_at"]


# --- validation ------------------------------------------------------------

def test_invalid_status_is_rejected_at_build_time():
    """Better to fail the producer's own tests than to publish a string the
    console renders verbatim as a status."""
    with pytest.raises(ValueError, match="status must be one of"):
        _ok(status="green")


def test_check_id_must_be_one_path_segment():
    """check_id becomes an S3 path segment; a slash silently writes to a
    different prefix the console never scans."""
    with pytest.raises(ValueError, match="single path segment"):
        _ok(check_id="a/b")
    with pytest.raises(ValueError, match="single path segment"):
        _ok(check_id="")


@pytest.mark.parametrize("bad", [0, -1, -1440])
def test_nonpositive_cadence_is_rejected(bad):
    """The console derives its staleness threshold from cadence_minutes; zero
    or negative makes every publish instantly stale."""
    with pytest.raises(ValueError, match="cadence_minutes must be positive"):
        _ok(cadence_minutes=bad)


# --- event-driven cadence (alpha-engine-config-I9033) -----------------------


def test_cadence_minutes_none_is_accepted_and_written_as_null():
    """An event-driven producer — merge-triggered, no clock — declares no
    cadence rather than a manufactured one. The key still appears (`null`),
    matching the always-present consumer contract, and the checks-envelope
    adapter treats a missing cadence as no freshness input, never MISSED."""
    e = _ok(cadence_minutes=None)
    assert "cadence_minutes" in e
    assert e["cadence_minutes"] is None


def test_cadence_minutes_none_is_json_serialisable():
    json.dumps(_ok(cadence_minutes=None))


# --- emission --------------------------------------------------------------

def test_key_is_derived_from_check_id():
    assert fcr.key_for("lib_pin_drift") == "ops/checks/lib_pin_drift/latest.json"


def test_dry_run_writes_nothing_and_returns_none():
    assert fcr.emit(_ok(), dry_run=True) is None


def test_emit_never_raises_when_s3_is_unavailable(monkeypatch):
    """A check must not go red because its telemetry did. The console renders a
    missing artifact as `unreadable`, never `ok`, so the gap stays visible.

    The allow-writes escape hatch is set deliberately: without it the test-write
    guard below returns None BEFORE boto3 is ever imported, and this test would
    pass while exercising nothing — the assertion is `is None` either way.
    """
    monkeypatch.setenv(fcr.ALLOW_TEST_WRITES_ENV, "1")
    import builtins
    real_import = builtins.__import__

    def boom(name, *a, **k):
        if name == "boto3":
            raise RuntimeError("no credentials")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    assert fcr.emit(_ok()) is None


# ── The test-write guard (alpha-engine-config-I9052) ──────────────────────


def test_a_test_process_never_publishes_to_the_live_surface(monkeypatch):
    """`ops/checks/<id>/latest.json` is ONE mutable object the console reads as
    a component's live state. A producer test that reaches `emit` unstubbed
    republishes its own fixture as the fleet's truth, and because `emit`
    swallows every exception it does so silently. Measured instance: on
    2026-08-31 a nous-ergon-ops drill test published `status: error` /
    `deep_link: "s3://test"` over the real run, and the console rendered the
    component FAILED.
    """
    published = []
    monkeypatch.setattr(fcr, "key_for", lambda cid: published.append(cid) or f"k/{cid}")
    monkeypatch.delenv(fcr.ALLOW_TEST_WRITES_ENV, raising=False)

    def must_not_run(*_a, **_k):
        raise AssertionError("emit reached the S3 write from inside a test")

    import builtins
    real_import = builtins.__import__
    monkeypatch.setattr(
        builtins, "__import__",
        lambda name, *a, **k: must_not_run() if name == "boto3"
        else real_import(name, *a, **k))
    assert fcr.emit(_ok()) is None


def test_the_guard_reads_the_current_test_not_the_interpreter(monkeypatch):
    """`PYTEST_CURRENT_TEST` is set only for the duration of a test, so the
    guard answers "am I inside a test right now" — not the weaker "was this
    interpreter started by pytest", which would also gag a production script
    that happened to be launched from a pytest-installed venv.
    """
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("UNITTEST_CURRENT_TEST", raising=False)
    assert fcr._running_under_test() is False
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/x.py::test_y (call)")
    assert fcr._running_under_test() is True


def test_the_escape_hatch_lets_a_deliberate_write_through(monkeypatch):
    """A test whose SUBJECT is the write must still be able to run it."""
    monkeypatch.setenv(fcr.ALLOW_TEST_WRITES_ENV, "1")
    wrote = []

    class _S3:
        def put_object(self, **kw):
            wrote.append(kw["Key"])

    class _Boto:
        @staticmethod
        def client(_name, **_k):
            return _S3()

    monkeypatch.setitem(__import__("sys").modules, "boto3", _Boto)
    assert fcr.emit(_ok()) is not None
    assert wrote == [fcr.key_for(_ok()["check_id"])]


def test_findings_default_to_an_empty_list_not_none():
    """The console does `len(findings)`; None would raise inside the renderer."""
    assert _ok()["findings"] == []


def test_emit_result_is_build_plus_emit():
    assert fcr.emit_result(check_id="c", label="C", status=fcr.STATUS_OK,
                           summary="s", cadence_minutes=60, dry_run=True) is None
