"""Tests for nousergon_lib.artifact_resolution — the shared windowed
artifact-resolution keystone (alpha-engine-config#1190).

Mirrors the proofs that lived in crucible-evaluator's
``tests/test_artifacts.py::TestGetJsonWindowed`` (the helper's origin) so the
consolidation is behaviour-preserving, and adds coverage for the generic
:func:`resolve_windowed_artifact` resolver + the ``latest`` pointer-first path.
"""

import json

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from nousergon_lib.artifact_resolution import (
    DEFAULT_ARTIFACT_MAX_AGE_DAYS,
    ResolvedArtifact,
    get_json_windowed,
    resolve_windowed_artifact,
)

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-06-20"
TPL = "backtest/{date}/e2e_lift.json"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _put_on(s3, date, data, raw=False):
    body = data if raw else json.dumps(data).encode("utf-8")
    s3.put_object(Bucket=BUCKET, Key=f"backtest/{date}/e2e_lift.json", Body=body)


class TestGetJsonWindowed:
    """The resilience keystone: grade off the freshest artifact within the
    trailing window, so a partial / retried / off-cycle Saturday run still
    resolves instead of reading N/A. A corrupt mid-write is skipped."""

    def test_exact_date_age_zero(self, s3):
        _put_on(s3, "2026-06-20", {"status": "ok"})
        doc, src_date, age, key = get_json_windowed(s3, BUCKET, TPL, RUN_DATE)
        assert doc == {"status": "ok"} and src_date == "2026-06-20" and age == 0
        assert key == "backtest/2026-06-20/e2e_lift.json"

    def test_finds_earlier_artifact_within_window(self, s3):
        # SF never ran on run_date 2026-06-20, but a partial run on 06-18 produced it.
        _put_on(s3, "2026-06-18", {"status": "ok", "from": "partial"})
        doc, src_date, age, key = get_json_windowed(s3, BUCKET, TPL, RUN_DATE)
        assert doc["from"] == "partial" and src_date == "2026-06-18" and age == 2

    def test_outside_window_is_none(self, s3):
        # Older than the window → genuinely N/A (not silently graded stale).
        old = "2026-06-01"  # > DEFAULT_ARTIFACT_MAX_AGE_DAYS before run_date
        _put_on(s3, old, {"status": "ok"})
        doc, src_date, age, key = get_json_windowed(s3, BUCKET, TPL, RUN_DATE)
        assert doc is None and src_date is None and age is None and key is None

    def test_freshest_wins_over_older(self, s3):
        _put_on(s3, "2026-06-15", {"v": "older"})
        _put_on(s3, "2026-06-19", {"v": "fresher"})
        doc, src_date, _, _ = get_json_windowed(s3, BUCKET, TPL, RUN_DATE)
        assert doc["v"] == "fresher" and src_date == "2026-06-19"

    def test_corrupt_candidate_skipped_for_older_good(self, s3):
        # A crashed mid-write leaves an empty file on the freshest date; the
        # resolver skips it and returns the last GOOD artifact.
        _put_on(s3, "2026-06-19", b"", raw=True)  # corrupt/empty
        _put_on(s3, "2026-06-17", {"v": "good"})
        doc, src_date, _, _ = get_json_windowed(s3, BUCKET, TPL, RUN_DATE)
        assert doc["v"] == "good" and src_date == "2026-06-17"

    def test_default_window_is_ten_days(self, s3):
        assert DEFAULT_ARTIFACT_MAX_AGE_DAYS == 10

    def test_window_boundary_inclusive(self, s3):
        # Exactly max_age_days back is still in-window; one more day out is not.
        _put_on(s3, "2026-06-10", {"edge": True})  # 10 days before 06-20
        doc, src_date, age, _ = get_json_windowed(s3, BUCKET, TPL, RUN_DATE)
        assert doc == {"edge": True} and src_date == "2026-06-10" and age == 10

    def test_non_iso_run_date_falls_back_to_exact(self, s3):
        s3.put_object(
            Bucket=BUCKET,
            Key="backtest/latest/e2e_lift.json",
            Body=json.dumps({"v": "literal"}).encode("utf-8"),
        )
        doc, src_date, age, key = get_json_windowed(s3, BUCKET, TPL, "latest")
        assert doc == {"v": "literal"} and src_date == "latest" and age == 0
        assert key == "backtest/latest/e2e_lift.json"

    def test_real_s3_error_propagates(self, s3):
        # A non-404 error (wrong bucket) is an upstream contract violation, raised.
        with pytest.raises(ClientError):
            get_json_windowed(s3, "no-such-bucket-xyz", TPL, RUN_DATE)


class TestResolveWindowedArtifact:
    """The generic resolver — returns the freshest existing key (HEAD-only),
    for non-JSON / pointer-style consumers."""

    def test_finds_freshest_key_within_window(self, s3):
        _put_on(s3, "2026-06-14", {"a": 1})
        _put_on(s3, "2026-06-18", {"a": 2})
        res = resolve_windowed_artifact(s3, BUCKET, TPL, RUN_DATE)
        assert isinstance(res, ResolvedArtifact)
        assert res.found and res.key == "backtest/2026-06-18/e2e_lift.json"
        assert res.src_date == "2026-06-18" and res.age_days == 2
        assert res.used_pointer is False

    def test_no_instance_in_window(self, s3):
        res = resolve_windowed_artifact(s3, BUCKET, TPL, RUN_DATE)
        assert not res.found and res.key is None and res.src_date is None

    def test_latest_pointer_short_circuits_scan(self, s3):
        # The pointer is HEADed first; a hit returns it without any date scan.
        s3.put_object(Bucket=BUCKET, Key="signals/latest.json", Body=b"{}")
        res = resolve_windowed_artifact(
            s3, BUCKET, "signals/{date}/signals.json", RUN_DATE,
            latest_pointer_key="signals/latest.json",
        )
        assert res.found and res.used_pointer and res.key == "signals/latest.json"

    def test_missing_pointer_falls_through_to_scan(self, s3):
        _put_on(s3, "2026-06-16", {"a": 1})
        res = resolve_windowed_artifact(
            s3, BUCKET, TPL, RUN_DATE,
            latest_pointer_key="backtest/does-not-exist.json",
        )
        assert res.found and not res.used_pointer
        assert res.key == "backtest/2026-06-16/e2e_lift.json"

    def test_custom_window(self, s3):
        # A 14-day window (the executor signals reader's default) reaches further.
        _put_on(s3, "2026-06-08", {"a": 1})  # 12 days back
        assert not resolve_windowed_artifact(s3, BUCKET, TPL, RUN_DATE).found
        res = resolve_windowed_artifact(s3, BUCKET, TPL, RUN_DATE, max_age_days=14)
        assert res.found and res.src_date == "2026-06-08"
