"""Tests for :mod:`nousergon_lib.collector_results`.

Validates the helper that surfaces collector-style error dicts to Flow
Doctor's ERROR-level logging handler. The shape under test is the
return-dict pattern used by alpha-engine-data weekly_collector and
similar orchestrators where exceptions are caught into per-collector
status entries rather than propagated.
"""

from __future__ import annotations

import logging

import pytest

from nousergon_lib.collector_results import report_collector_errors


def test_logs_one_error_per_error_status_entry(caplog: pytest.LogCaptureFixture):
    """Each error-status collector emits a distinct ERROR log record.

    Distinct signatures are load-bearing: Flow Doctor's dedup keys off
    the rendered message. Two different collector failures in the same
    run must produce two different alerts, not one deduplicated alert.
    """
    collectors = {
        "constituents": {"status": "ok"},
        "prices": {"status": "ok", "refreshed": 1},
        "arcticdb": {
            "status": "error",
            "error": "Backfill regression preflight failed: 38 symbols would regress",
        },
        "fundamentals": {"status": "error", "error": "Polygon 429 rate limit"},
    }
    with caplog.at_level(logging.ERROR):
        n = report_collector_errors(collectors)

    assert n == 2
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 2
    messages = [r.getMessage() for r in error_records]
    assert any(
        "collector arcticdb failed: Backfill regression preflight failed" in m
        for m in messages
    )
    assert any("collector fundamentals failed: Polygon 429" in m for m in messages)


def test_no_op_on_all_ok(caplog: pytest.LogCaptureFixture):
    """All-ok results emit nothing, return zero."""
    collectors = {
        "constituents": {"status": "ok"},
        "prices": {"status": "ok"},
        "macro": {"status": "ok_dry_run"},
    }
    with caplog.at_level(logging.ERROR):
        n = report_collector_errors(collectors)

    assert n == 0
    assert [r for r in caplog.records if r.levelno == logging.ERROR] == []


def test_missing_error_field_uses_placeholder(caplog: pytest.LogCaptureFixture):
    """Status=error without an ``error`` field still logs a record.

    Defensive: callers shouldn't ship status=error with no message, but
    if they do we still want the alert to fire (with a placeholder
    string) rather than silently swallow the failure.
    """
    collectors = {"buggy": {"status": "error"}}
    with caplog.at_level(logging.ERROR):
        n = report_collector_errors(collectors)

    assert n == 1
    msg = caplog.records[0].getMessage()
    assert "collector buggy failed" in msg
    assert "<no error message>" in msg


def test_ignores_non_mapping_entries(caplog: pytest.LogCaptureFixture):
    """Malformed entries (non-dict values) are skipped, never raise."""
    collectors = {
        "good": {"status": "error", "error": "boom"},
        "weird_string": "this should never happen",  # type: ignore[dict-item]
        "weird_none": None,  # type: ignore[dict-item]
    }
    with caplog.at_level(logging.ERROR):
        n = report_collector_errors(collectors)  # type: ignore[arg-type]

    assert n == 1


def test_accepts_custom_logger(caplog: pytest.LogCaptureFixture):
    """When given an explicit logger, records emit through it.

    Lets call sites attach their own filters/handlers (e.g. a per-flow
    logger that adds context tags) without going through the root.
    """
    custom = logging.getLogger("test_collector_results.custom")
    collectors = {"x": {"status": "error", "error": "oops"}}
    with caplog.at_level(logging.ERROR, logger=custom.name):
        n = report_collector_errors(collectors, logger=custom)

    assert n == 1
    assert any(r.name == custom.name for r in caplog.records)


def test_empty_dict():
    """No collectors → no work, returns zero, no exception."""
    assert report_collector_errors({}) == 0
