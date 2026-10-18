"""A weekly-SF rehearsal's bookkeeping never lands at a live key (alpha-engine-config-I12351).

`rehearsal-2026-10-16-2` overwrote `health/backtester.json` and
`health/evaluator.json`, `ops/checks/ae-research-self-test/latest.json`, and
moved `backtest/cumulative_trial_count.json` by 137 trials. These tests pin the
three writers in this library that did it.
"""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

from nousergon_lib import fleet_check_result as fcr
from nousergon_lib import rehearsal
from nousergon_lib.health import Deliverable, health_key, write_health
from nousergon_lib.phase_registry import REHEARSAL_EXECUTION_PREFIX
from nousergon_lib.quant.stats.trial_accumulator import (
    DEFAULT_KEY,
    increment_trial_count,
    read_cumulative_trial_count,
)

TOKEN = "rehearsal-2026-10-16-2"
BUCKET = "alpha-engine-research"


@pytest.fixture(autouse=True)
def _no_ambient_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(rehearsal.RUN_TOKEN_ENV, raising=False)
    monkeypatch.delenv(rehearsal.PIPELINE_ROLE_ENV, raising=False)


# ── the signal ───────────────────────────────────────────────────────────────


def test_the_prefix_is_the_one_phase_markers_key_on() -> None:
    assert rehearsal.REHEARSAL_EXECUTION_PREFIX == REHEARSAL_EXECUTION_PREFIX


@pytest.mark.parametrize(
    ("env", "want"),
    [
        ({"RUN_TOKEN": TOKEN}, TOKEN),
        ({"RUN_TOKEN": "rehearsal-a/b c"}, "rehearsal-abc"),
        ({"AE_PIPELINE_ROLE": "rehearsal"}, rehearsal.UNNAMED_TOKEN),
        ({"RUN_TOKEN": "f4b49786-4d8b-793d-cfa7-419f26714770_32a27b1a"}, None),
        ({"RUN_TOKEN": "weekly-rerun-2026-10-10-1"}, None),
        ({}, None),
    ],
)
def test_rehearsal_token(env: dict, want: str | None) -> None:
    assert rehearsal.rehearsal_token(env) == want


def test_namespaced_key() -> None:
    assert (
        rehearsal.namespaced_key("health/backtester.json", TOKEN)
        == f"health/.rehearsal/{TOKEN}/backtester.json"
    )
    assert rehearsal.namespaced_key("health/backtester.json", None) == "health/backtester.json"
    assert rehearsal.namespaced_key("x.json", TOKEN) == f".rehearsal/{TOKEN}/x.json"


# ── health/<module>.json ─────────────────────────────────────────────────────


class _S3:
    def __init__(self) -> None:
        self.keys: list[str] = []

    def put_object(self, *, Bucket, Key, Body, ContentType=None):  # noqa: N803
        self.keys.append(Key)


def _write(s3: _S3) -> None:
    write_health(
        module_name="backtester",
        deliverables=[Deliverable(name="report", required=True, produced=True)],
        run_date="2026-10-15",
        duration_seconds=5.0,
        s3_client=s3,
    )


def test_a_production_heartbeat_keeps_its_live_key() -> None:
    s3 = _S3()
    _write(s3)
    assert s3.keys == [health_key("backtester")]


def test_a_rehearsal_heartbeat_goes_to_its_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUN_TOKEN", TOKEN)
    s3 = _S3()
    _write(s3)
    assert s3.keys == [f"health/.rehearsal/{TOKEN}/backtester.json"]


# ── ops/checks/<id>/latest.json ──────────────────────────────────────────────


def _publish(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    keys: list[str] = []

    class _Client:
        def put_object(self, *, Bucket, Key, Body, ContentType=None):  # noqa: N803
            keys.append(Key)

    monkeypatch.setenv(fcr.ALLOW_TEST_WRITES_ENV, "1")
    monkeypatch.setattr(boto3, "client", lambda *_a, **_k: _Client())
    envelope = fcr.build(
        check_id="ae-research-self-test",
        label="Research self-test",
        status=fcr.STATUS_OK,
        summary="ok",
        cadence_minutes=10080,
    )
    fcr.emit(envelope)
    return keys


def test_a_production_check_row_keeps_its_live_key(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _publish(monkeypatch) == ["ops/checks/ae-research-self-test/latest.json"]


def test_a_rehearsal_check_row_stays_out_of_the_console_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUN_TOKEN", TOKEN)
    keys = _publish(monkeypatch)
    assert keys == [f"ops/.rehearsal/{TOKEN}/checks/ae-research-self-test/latest.json"]
    assert not keys[0].startswith(fcr.CHECKS_PREFIX)


# ── backtest/cumulative_trial_count.json ─────────────────────────────────────


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _seed(s3) -> None:
    s3.put_object(
        Bucket=BUCKET,
        Key=DEFAULT_KEY,
        Body=json.dumps(
            {"total": 1859, "last_updated": "2026-10-09",
             "by_producer": {"predictor_param_sweep": 900}}
        ).encode(),
    )


def test_a_rehearsal_never_moves_the_production_trial_count(
    s3, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(s3)
    monkeypatch.setenv("RUN_TOKEN", TOKEN)
    out = increment_trial_count("predictor_param_sweep", 60, "2026-10-15", s3_client=s3)
    assert out["total"] == 1919
    assert out["by_producer"]["predictor_param_sweep"] == 960
    live = read_cumulative_trial_count(s3_client=s3)
    assert live["total"] == 1859
    assert live["by_producer"] == {"predictor_param_sweep": 900}
    assert [o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET).get("Contents", [])] == [
        DEFAULT_KEY
    ]


def test_production_still_counts(s3) -> None:
    _seed(s3)
    increment_trial_count("predictor_param_sweep", 60, "2026-10-16", s3_client=s3)
    assert read_cumulative_trial_count(s3_client=s3)["total"] == 1919
