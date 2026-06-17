"""
Unit tests for :mod:`nousergon_lib.phase_registry` — the phase-execution
framework lifted from alpha-engine-backtester (L4528).

Covers the outcome taxonomy, the marker read/write contract, the should_run
decision matrix (incl. L4524 artifact-validated checkpoints), the injectable
``marker_prefix`` (the key cross-repo generalization), and ``load_phase_hard_caps``.
S3 is stubbed with a minimal in-memory fake so the tests run offline.
"""

from __future__ import annotations

import json

import pytest
from botocore.exceptions import ClientError

from nousergon_lib.phase_registry import (
    PhaseContext,
    PhaseOutcome,
    PhaseRegistry,
    PhaseStatus,
    _marker_key,
    load_phase_hard_caps,
    phase,
)


class _FakeS3:
    def __init__(self):
        self.store: dict[tuple[str, str], bytes] = {}
        self.put_calls: list[dict] = []
        self.get_calls: list[dict] = []
        self.head_calls: list[dict] = []

    def get_object(self, *, Bucket, Key):
        self.get_calls.append({"Bucket": Bucket, "Key": Key})
        if (Bucket, Key) not in self.store:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "not found"}}, "GetObject"
            )
        body = self.store[(Bucket, Key)]

        class _Body:
            def __init__(self, b): self._b = b
            def read(self): return self._b

        return {"Body": _Body(body)}

    def head_object(self, *, Bucket, Key):
        self.head_calls.append({"Bucket": Bucket, "Key": Key})
        if (Bucket, Key) not in self.store:
            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
            )
        return {"ContentLength": len(self.store[(Bucket, Key)])}

    def put_object(self, *, Bucket, Key, Body, ContentType=None):
        self.put_calls.append({"Bucket": Bucket, "Key": Key, "Body": Body})
        if isinstance(Body, str):
            Body = Body.encode()
        self.store[(Bucket, Key)] = Body

    def seed(self, bucket, prefix, date, phase_name, marker):
        self.store[(bucket, _marker_key(prefix, date, phase_name))] = json.dumps(marker).encode()


@pytest.fixture
def s3():
    return _FakeS3()


def _reg(s3, **kw):
    defaults = dict(date="2026-04-23", bucket="test-bucket", marker_prefix="backtest", s3_client=s3)
    defaults.update(kw)
    return PhaseRegistry(**defaults)


# ── Outcome taxonomy ────────────────────────────────────────────────────────


def test_phase_outcome_status_helpers_and_serialization():
    o = PhaseOutcome(status=PhaseStatus.EMPTY, phase="param_sweep", reason="no combo")
    assert o.is_empty and not o.is_success and not o.is_failure
    d = o.to_dict()
    assert d["status"] == "empty" and d["phase"] == "param_sweep"
    json.dumps(d)  # must be serializable


# ── Marker key prefix is injectable (the cross-repo generalization) ─────────


def test_marker_key_uses_prefix():
    assert _marker_key("backtest", "2026-04-23", "simulate") == "backtest/2026-04-23/.phases/simulate.json"
    assert _marker_key("predictor", "2026-04-23", "train") == "predictor/2026-04-23/.phases/train.json"


def test_registry_writes_marker_under_its_prefix(s3):
    r = _reg(s3, marker_prefix="predictor")
    with r.phase("train", supports_auto_skip=True) as ctx:
        ctx.record_artifact("predictor/2026-04-23/weights.json")
    assert s3.put_calls[0]["Key"] == "predictor/2026-04-23/.phases/train.json"


# ── should_run decision matrix ──────────────────────────────────────────────


def test_default_runs(s3):
    assert _reg(s3).should_run("simulate", supports_auto_skip=True) == (True, "default_run")


def test_explicit_skip(s3):
    assert _reg(s3, skip_phases=["simulate"]).should_run("simulate", True) == (False, "explicit_skip")


def test_only_phases_filter(s3):
    r = _reg(s3, only_phases=["param_sweep"])
    assert r.should_run("simulate", True) == (False, "only_phases_filter")


def test_not_auto_skippable_runs_despite_marker(s3):
    s3.seed("test-bucket", "backtest", "2026-04-23", "simulate", {"status": "ok"})
    assert _reg(s3).should_run("simulate", supports_auto_skip=False) == (True, "not_auto_skippable")


def test_auto_skip_when_marker_ok_and_no_declared_artifacts(s3):
    s3.seed("test-bucket", "backtest", "2026-04-23", "simulate", {"status": "ok"})
    assert _reg(s3).should_run("simulate", True) == (False, "auto_skip_marker_ok")


def test_force_overrides_marker(s3):
    s3.seed("test-bucket", "backtest", "2026-04-23", "simulate", {"status": "ok"})
    assert _reg(s3, force=True).should_run("simulate", True) == (True, "force_rerun")


def test_error_marker_does_not_auto_skip(s3):
    s3.seed("test-bucket", "backtest", "2026-04-23", "simulate", {"status": "error"})
    assert _reg(s3).should_run("simulate", True) == (True, "default_run")


# ── L4524 artifact-validated checkpoints ────────────────────────────────────


def test_marker_honored_when_declared_artifact_present(s3):
    art = "backtest/2026-04-23/portfolio_stats.json"
    s3.store[("test-bucket", art)] = b"x"
    s3.seed("test-bucket", "backtest", "2026-04-23", "simulate",
            {"status": "ok", "artifact_keys": [art]})
    run, reason = _reg(s3).should_run("simulate", True)
    assert (run, reason) == (False, "auto_skip_marker_ok")
    assert any(c["Key"] == art for c in s3.head_calls)


def test_marker_invalid_when_declared_artifact_absent(s3):
    s3.seed("test-bucket", "backtest", "2026-04-23", "param_sweep",
            {"status": "ok", "artifact_keys": ["backtest/2026-04-23/sweep_df.parquet"]})
    assert _reg(s3).should_run("param_sweep", True) == (True, "marker_artifact_missing")


def test_artifact_validation_non_404_raises(s3):
    s3.seed("test-bucket", "backtest", "2026-04-23", "simulate",
            {"status": "ok", "artifact_keys": ["backtest/2026-04-23/x.json"]})

    def _boom(*, Bucket, Key):
        raise ClientError({"Error": {"Code": "AccessDenied"}}, "HeadObject")

    s3.head_object = _boom
    with pytest.raises(ClientError):
        _reg(s3).should_run("simulate", True)


# ── Marker read/write contract ──────────────────────────────────────────────


def test_phase_context_writes_ok_marker_with_schema(s3):
    r = _reg(s3)
    with r.phase("simulate", supports_auto_skip=True) as ctx:
        assert ctx.skipped is False
        ctx.record_artifact("backtest/2026-04-23/sim.parquet")
    marker = json.loads(s3.put_calls[0]["Body"])
    assert marker["status"] == "ok"
    assert marker["schema_version"] == 1
    assert marker["artifact_keys"] == ["backtest/2026-04-23/sim.parquet"]
    assert marker["error"] is None


def test_phase_context_writes_error_marker_and_tracks(s3):
    r = _reg(s3)
    with pytest.raises(RuntimeError, match="boom"):
        with r.phase("simulate", supports_auto_skip=True):
            raise RuntimeError("boom")
    marker = json.loads(s3.put_calls[0]["Body"])
    assert marker["status"] == "error" and "RuntimeError" in marker["error"]
    assert r.phase_errors == ["simulate"]


def test_skipped_phase_does_not_write_marker(s3):
    s3.seed("test-bucket", "backtest", "2026-04-23", "simulate", {"status": "ok"})
    r = _reg(s3)
    with r.phase("simulate", supports_auto_skip=True) as ctx:
        assert ctx.skipped is True
    assert s3.put_calls == []


def test_transient_s3_error_on_marker_read_raises(s3):
    def _boom(*, Bucket, Key):
        raise ClientError({"Error": {"Code": "InternalError"}}, "GetObject")

    s3.get_object = _boom
    with pytest.raises(ClientError):
        _reg(s3).should_run("simulate", True)


def test_record_artifact_rejects_empty():
    ctx = PhaseContext(name="x", skipped=False, skip_reason="default_run")
    with pytest.raises(ValueError):
        ctx.record_artifact("")


# ── Watchdog hard cap ───────────────────────────────────────────────────────


def test_hard_cap_trips_watchdog_and_raises_timeout(s3):
    from nousergon_lib.phase_registry import PhaseTimeoutError
    r = _reg(s3, hard_caps={"slow": 0.05})
    with pytest.raises(PhaseTimeoutError):
        with r.phase("slow", supports_auto_skip=True):
            import time
            time.sleep(2.0)
    marker = json.loads(s3.put_calls[0]["Body"])
    assert marker["status"] == "error"


# ── Simple phase() context manager ──────────────────────────────────────────


def test_simple_phase_logs_start_end(caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="nousergon_lib.phase"):
        with phase("ingest", rows=10):
            pass
    msgs = [r.getMessage() for r in caplog.records]
    assert any("PHASE_START name=ingest" in m for m in msgs)
    assert any("PHASE_END name=ingest" in m and "status=ok" in m for m in msgs)


# ── load_phase_hard_caps (path-agnostic) ────────────────────────────────────


def test_load_hard_caps_reads_block(tmp_path):
    p = tmp_path / "caps.yaml"
    p.write_text("full_run_hard_caps_seconds:\n  simulate: 120\n  param_sweep: 300.5\n")
    caps = load_phase_hard_caps(p)
    assert caps == {"simulate": 120.0, "param_sweep": 300.5}


def test_load_hard_caps_missing_file_returns_empty(tmp_path):
    assert load_phase_hard_caps(tmp_path / "nope.yaml") == {}


def test_load_hard_caps_drops_non_numeric(tmp_path):
    p = tmp_path / "caps.yaml"
    p.write_text("full_run_hard_caps_seconds:\n  good: 5\n  bad: not_a_number\n")
    assert load_phase_hard_caps(p) == {"good": 5.0}


def test_load_hard_caps_custom_key(tmp_path):
    p = tmp_path / "caps.yaml"
    p.write_text("smoke_caps:\n  smoke: 30\n")
    assert load_phase_hard_caps(p, caps_key="smoke_caps") == {"smoke": 30.0}
