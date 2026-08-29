"""Tests for the shared alert-transport liveness probe (alpha-engine-config-I9335/I9334).

Lifted from `claude-code-config/laptop-checks/alert_transport_liveness.py`
(claude-code-config-PR218, alpha-engine-config-I9209) on its second adoption —
`shared-code-policy.md` §2. The laptop script becomes a thin wrapper around
these functions; the assertions here are the ones every future substrate
wrapper (GHA, Lambda, EC2) depends on.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from nousergon_lib import alert_transport_probe as atp


def test_synthetic_detail_declares_no_condition():
    """`state=None` is the v1 contract's "this producer tracks no condition" —
    a synthetic probe must never look like an open incident to the drain."""
    detail = atp.synthetic_detail(origin="alert-transport-liveness", source_suffix="x", nonce="abc123")
    assert detail["state"] is None
    assert detail["severity"] == "info"
    assert "SYNTHETIC LIVENESS PROBE" in detail["body"]
    assert "abc123" in detail["body"]


def test_new_nonce_is_short_and_unique():
    a, b = atp.new_nonce(), atp.new_nonce()
    assert a != b
    assert len(a) == 12


def test_probe_primary_ok():
    session = MagicMock()
    session.client.return_value.put_events.return_value = {
        "FailedEntryCount": 0,
        "Entries": [{"EventId": "abc"}],
    }
    finding = atp.probe_primary(session, bus_name="bus", profile_label="p", detail={})
    assert finding["ok"] is True
    assert finding["key"] == "primary/p"


def test_probe_primary_reports_rejected_entry_not_just_call_success():
    """`FailedEntryCount` > 0 with a 200 response is a silent-discard mode
    `PutEvents` succeeding at the transport layer does not catch."""
    session = MagicMock()
    session.client.return_value.put_events.return_value = {
        "FailedEntryCount": 1,
        "Entries": [{"ErrorCode": "AccessDenied", "ErrorMessage": "nope"}],
    }
    finding = atp.probe_primary(session, bus_name="bus", profile_label="p", detail={})
    assert finding["ok"] is False
    assert "AccessDenied" in finding["detail"]


def test_probe_primary_raises_are_findings_not_exceptions():
    session = MagicMock()
    session.client.return_value.put_events.side_effect = RuntimeError("boom")
    finding = atp.probe_primary(session, bus_name="bus", profile_label="p", detail={})
    assert finding["ok"] is False
    assert "boom" in finding["detail"]


def test_probe_fallback_reads_back_by_exact_key():
    session = MagicMock()
    detail = {"body": "nonce=abc123"}
    written = {}

    def put_object(Bucket, Key, Body, ContentType):
        written["key"] = Key
        written["body"] = Body

    def get_object(Bucket, Key):
        assert Key == written["key"]
        import json
        payload = json.dumps({"detail": detail}).encode()
        return {"Body": MagicMock(read=lambda: payload)}

    session.client.return_value.put_object.side_effect = put_object
    session.client.return_value.get_object.side_effect = get_object
    finding = atp.probe_fallback(
        session, bucket="b", prefix="pre", profile_label="p", nonce="abc123", detail=detail
    )
    assert finding["ok"] is True


def test_probe_fallback_mismatched_readback_fails():
    session = MagicMock()

    def get_object(Bucket, Key):
        import json
        payload = json.dumps({"detail": {"body": "WRONG"}}).encode()
        return {"Body": MagicMock(read=lambda: payload)}

    session.client.return_value.get_object.side_effect = get_object
    finding = atp.probe_fallback(
        session, bucket="b", prefix="pre", profile_label="p", nonce="n",
        detail={"body": "nonce=n"},
    )
    assert finding["ok"] is False


def test_probe_arrival_by_counter_zero_expected_is_not_a_second_failure():
    session = MagicMock()
    finding = atp.probe_arrival_by_counter(
        session, bus_name="b", rule_name="r", expected=0,
        window_start=datetime.now(timezone.utc),
    )
    assert finding["ok"] is False
    assert "not measured" in finding["detail"]


def test_probe_arrival_by_counter_moves_are_ok(monkeypatch):
    session = MagicMock()
    session.client.return_value.get_metric_data.return_value = {
        "MetricDataResults": [{"Values": [3.0]}]
    }
    monkeypatch.setattr(atp.time, "sleep", lambda s: None)
    finding = atp.probe_arrival_by_counter(
        session, bus_name="b", rule_name="r", expected=1,
        window_start=datetime.now(timezone.utc), timeout_s=1, poll_s=0,
    )
    assert finding["ok"] is True
    assert "COUNTER proof only" in finding["detail"]


def test_probe_arrival_by_nonce_finds_exact_match():
    session = MagicMock()
    session.client.return_value.filter_log_events.return_value = {
        "events": [{"message": "nonce=abc123"}]
    }
    finding = atp.probe_arrival_by_nonce(
        session, log_group="/alpha-engine/x", nonce="abc123",
        window_start=datetime.now(timezone.utc), timeout_s=1, poll_s=0,
    )
    assert finding["ok"] is True
    assert finding["key"] == "arrival/nonce-exact"


def test_probe_arrival_by_nonce_missing_log_group_names_the_gap():
    session = MagicMock()

    class _ResourceNotFound(Exception):
        pass

    session.client.return_value.exceptions.ResourceNotFoundException = _ResourceNotFound
    session.client.return_value.filter_log_events.side_effect = _ResourceNotFound("no group")
    finding = atp.probe_arrival_by_nonce(
        session, log_group="/alpha-engine/missing", nonce="n",
        window_start=datetime.now(timezone.utc), timeout_s=1, poll_s=0,
    )
    assert finding["ok"] is False
    assert "have not been provisioned" in finding["detail"]


def test_probe_arrival_by_nonce_no_match_within_timeout(monkeypatch):
    session = MagicMock()
    session.client.return_value.exceptions.ResourceNotFoundException = Exception
    session.client.return_value.filter_log_events.return_value = {"events": []}
    monkeypatch.setattr(atp.time, "sleep", lambda s: None)
    finding = atp.probe_arrival_by_nonce(
        session, log_group="/alpha-engine/x", nonce="n",
        window_start=datetime.now(timezone.utc), timeout_s=0.01, poll_s=0,
    )
    assert finding["ok"] is False
    assert "not found" in finding["detail"]
