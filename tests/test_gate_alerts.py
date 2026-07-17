"""Tests for nousergon_lib.gate_alerts.alert_gate_failure (config#2459
scope item 4 — L1/L2/L3 alerting backbone).

These simulate a synthetic gate failure and assert the alert path fires by
attaching a fake logging.Handler that mimics flow-doctor's own
``FlowDoctorHandler`` — a level=ERROR handler on the root logger that
receives every dispatched record (see krepis.logging._attach_flow_doctor,
which does exactly this: ``logging.getLogger().addHandler(FlowDoctorHandler
(fd, level=logging.ERROR))``). We don't require the real ``flow_doctor``
package to be installed to prove ``alert_gate_failure`` reaches that
boundary correctly — mocking the boundary (a handler capturing at the root
logger) is the right unit-of-test isolation per the module's own contract:
"relies entirely on the calling process having already run setup_logging".
"""

from __future__ import annotations

import logging

import pytest

from nousergon_lib.gate_alerts import alert_gate_failure


class _CapturingFlowDoctorHandler(logging.Handler):
    """Stand-in for ``flow_doctor.FlowDoctorHandler`` — records every
    LogRecord that reaches it at/above its level, exactly like the real
    handler krepis.logging._attach_flow_doctor installs at level=ERROR."""

    def __init__(self, level=logging.ERROR):
        super().__init__(level=level)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def fake_flow_doctor_handler():
    """Attach a capturing handler to the root logger, matching where the
    real FlowDoctorHandler lives, and detach it after the test."""
    handler = _CapturingFlowDoctorHandler(level=logging.ERROR)
    root = logging.getLogger()
    root.addHandler(handler)
    prior_level = root.level
    root.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        root.setLevel(prior_level)


def test_alert_gate_failure_reaches_flow_doctor_boundary(fake_flow_doctor_handler):
    """A synthetic L2 gate failure must produce exactly one ERROR-level
    record at the flow-doctor dispatch boundary, carrying layer/series/
    detail in a greppable message plus structured `extra`."""
    alert_gate_failure(
        layer="L2",
        series="AAPL",
        detail="continuity gap: 2026-07-11 missing, expected trading day",
    )

    assert len(fake_flow_doctor_handler.records) == 1
    record = fake_flow_doctor_handler.records[0]
    assert record.levelno == logging.ERROR
    assert "[gate-failure]" in record.getMessage()
    assert "layer=L2" in record.getMessage()
    assert "series=AAPL" in record.getMessage()
    assert "continuity gap" in record.getMessage()
    assert record.layer == "L2"
    assert record.series == "AAPL"
    assert record.severity == "error"


def test_alert_gate_failure_default_severity_is_error(fake_flow_doctor_handler):
    """Default severity ('error') must map to logging.ERROR so it always
    crosses the FlowDoctorHandler's level=ERROR threshold — a gate FAILURE
    silently logged below that threshold would never reach flow-doctor at
    all."""
    alert_gate_failure(layer="L3", series="NAV", detail="T+1 mismatch")

    assert len(fake_flow_doctor_handler.records) == 1
    assert fake_flow_doctor_handler.records[0].levelno == logging.ERROR


def test_alert_gate_failure_critical_severity_crosses_error_threshold(
    fake_flow_doctor_handler,
):
    """'critical' must also cross the ERROR-level handler threshold —
    it must never be dispatched at a level BELOW error (that would silently
    drop it from flow-doctor's dispatch pipeline)."""
    alert_gate_failure(
        layer="L1", series="SPY", detail="cross-source disagreement",
        severity="critical",
    )

    assert len(fake_flow_doctor_handler.records) == 1
    assert fake_flow_doctor_handler.records[0].levelno == logging.ERROR
    assert fake_flow_doctor_handler.records[0].severity == "critical"


def test_alert_gate_failure_warning_severity_does_not_cross_error_threshold(
    fake_flow_doctor_handler,
):
    """A 'warning' severity gate check is intentionally sub-alert-threshold
    for a level=ERROR-only FlowDoctorHandler — this pins that behavior
    explicitly rather than leaving it as an accident of log-level plumbing."""
    alert_gate_failure(
        layer="L2", series="MSFT", detail="minor stale quote", severity="warning",
    )

    assert len(fake_flow_doctor_handler.records) == 0


def test_alert_gate_failure_invalid_severity_raises():
    with pytest.raises(ValueError, match="severity"):
        alert_gate_failure(layer="L2", series="X", detail="y", severity="bogus")


def test_log_level_for_critical_is_error_not_stdlib_critical():
    """Pins the intentional critical->ERROR (not logging.CRITICAL) mapping
    documented in _log_level_for's docstring — logging.CRITICAL would
    still cross the FlowDoctorHandler's level=ERROR threshold, but using
    it would misleadingly imply a level distinction flow-doctor doesn't
    make at the handler-attach layer."""
    from nousergon_lib.gate_alerts import _log_level_for

    assert _log_level_for("critical") == logging.ERROR
    assert _log_level_for("error") == logging.ERROR
    assert _log_level_for("warning") == logging.WARNING
    assert _log_level_for("info") == logging.INFO


def test_alert_gate_failure_no_handler_does_not_raise():
    """Local dev / no flow-doctor attached: must degrade to a plain log
    line, never crash the caller's gate-checking pipeline."""
    alert_gate_failure(layer="L2", series="AAPL", detail="no handler attached")
