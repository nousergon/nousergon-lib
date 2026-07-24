"""Tests for ``nousergon_lib.signals``.

Validates:
- ``fallback_research_date_keys`` — weekday skipping, sentinel
  fallback, malformed date handling, configurable max_weekdays.
- ``try_read_s3_json`` — happy path, NoSuchKey, AccessDenied,
  malformed JSON, auth errors that must propagate.
- ``load_json_with_fallback`` — first-hit semantics, full-chain miss,
  empty key list.
"""
from __future__ import annotations

import json

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from nousergon_lib.signals import (
    fallback_research_date_keys,
    load_json_with_fallback,
    try_read_s3_json,
)


# ── fallback_research_date_keys ──────────────────────────────────────


class TestFallbackResearchDateKeys:

    def test_monday_sees_previous_friday(self):
        """Monday 2026-07-20: last weekday is Friday 2026-07-17.
        The chain starts with 2026-07-20, walks back through prior
        weekdays skipping Sat(18) and Sun(19), then ends with latest.json.
        """
        keys = fallback_research_date_keys("2026-07-20")
        assert keys[0] == "signals/2026-07-20/signals.json"
        assert "signals/2026-07-18/signals.json" not in keys  # Saturday
        assert "signals/2026-07-19/signals.json" not in keys  # Sunday
        assert keys[-1] == "signals/latest.json"
        # Expect: Mon, Fri, Thu, Wed + latest.json sentinel
        # (Sat+Sun are skipped; range(6) covers today + 5 prior calendar days)
        assert len(keys) == 5

    def test_friday_gets_full_week(self):
        """Friday 2026-07-24: all 5 prior weekdays (Mon-Thu) should
        appear, plus Friday itself, plus latest.json.
        """
        keys = fallback_research_date_keys("2026-07-24")
        assert keys[0] == "signals/2026-07-24/signals.json"
        # range(6) = today + 5 prior calendar days → Fri, Thu, Wed, Tue, Mon
        # (Sat/Sun wouldn't appear in the 5 prior days starting from Friday)
        dated = [k for k in keys if k != "signals/latest.json"]
        assert len(dated) == 5  # Fri, Thu, Wed, Tue, Mon

    def test_bad_date_returns_only_sentinel(self):
        keys = fallback_research_date_keys("not-a-date")
        assert keys == ["signals/latest.json"]

    def test_custom_max_weekdays(self):
        keys = fallback_research_date_keys("2026-07-24", max_weekdays=1)
        # Only Fri + the sentinel, since max_weekdays=1 only goes back 1
        # weekday (Thursday — Saturday/Sunday skipped)
        assert keys[0] == "signals/2026-07-24/signals.json"
        assert keys[-1] == "signals/latest.json"
        dated = [k for k in keys if k != "signals/latest.json"]
        assert len(dated) == 2  # Fri + Thu (max_weekdays=1 after self)

    def test_empty_date_str_falls_back_to_sentinel(self):
        keys = fallback_research_date_keys("")
        assert keys == ["signals/latest.json"]


# ── try_read_s3_json ────────────────────────────────────────────────


@pytest.fixture
def s3():
    with mock_aws():
        conn = boto3.client("s3", region_name="us-east-1")
        conn.create_bucket(Bucket="test-bucket")
        yield conn


class TestTryReadS3Json:

    def test_reads_existing_json(self, s3):
        s3.put_object(Bucket="test-bucket", Key="signals/2026-07-20/signals.json",
                      Body=json.dumps({"market_regime": "bullish"}))
        result = try_read_s3_json(s3, "test-bucket", "signals/2026-07-20/signals.json")
        assert result == {"market_regime": "bullish"}

    def test_returns_none_on_no_such_key(self, s3):
        result = try_read_s3_json(s3, "test-bucket", "signals/nonexistent/signals.json")
        assert result is None

    def test_returns_none_on_access_denied(self, s3):
        """Simulate AccessDenied via a key that exists but with a
        different credential — moto doesn't enforce real IAM, so we
        mock the ClientError directly in a unit context.
        """
        # We can't easily make moto return 403, so verify the code-path
        # by checking the Python exception handling logic.
        import botocore

        class MockClient:
            def get_object(self, Bucket, Key):
                raise ClientError(
                    {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
                    "GetObject",
                )

        result = try_read_s3_json(MockClient(), "b", "k")
        assert result is None

    def test_returns_none_on_malformed_body(self, s3):
        s3.put_object(Bucket="test-bucket", Key="signals/bad.json",
                      Body=b"not json")
        result = try_read_s3_json(s3, "test-bucket", "signals/bad.json")
        assert result is None

    def test_raises_on_credential_error(self):
        """A ClientError with a code other than the expected miss codes
        must propagate — we never silently eat an auth infrastructure
        failure.
        """
        from botocore.exceptions import ClientError

        class MockClient:
            def get_object(self, Bucket, Key):
                raise ClientError(
                    {"Error": {"Code": "ExpiredToken", "Message": "Token expired"}},
                    "GetObject",
                )

        with pytest.raises(ClientError):
            try_read_s3_json(MockClient(), "b", "k")

    def test_empty_body_returns_none(self, s3):
        s3.put_object(Bucket="test-bucket", Key="empty/signals.json",
                      Body=b"")
        result = try_read_s3_json(s3, "test-bucket", "empty/signals.json")
        assert result is None


# ── load_json_with_fallback ─────────────────────────────────────────


class TestLoadJsonWithFallback:

    def test_returns_first_hit(self, s3):
        s3.put_object(Bucket="test-bucket",
                      Key="signals/2026-07-20/signals.json",
                      Body=json.dumps({"date": "2026-07-20", "universe": ["SPY"]}))
        s3.put_object(Bucket="test-bucket",
                      Key="signals/latest.json",
                      Body=json.dumps({"date": "latest"}))

        keys = ["signals/2026-07-20/signals.json", "signals/latest.json"]
        result = load_json_with_fallback(s3, "test-bucket", keys)
        assert result == {"date": "2026-07-20", "universe": ["SPY"]}

    def test_skips_missing_then_uses_fallback(self, s3):
        s3.put_object(Bucket="test-bucket",
                      Key="signals/latest.json",
                      Body=json.dumps({"date": "latest"}))

        keys = ["signals/2026-07-20/signals.json", "signals/latest.json"]
        result = load_json_with_fallback(s3, "test-bucket", keys)
        assert result == {"date": "latest"}

    def test_returns_none_when_all_miss(self, s3):
        keys = ["signals/missing1/signals.json", "signals/missing2/signals.json"]
        result = load_json_with_fallback(s3, "test-bucket", keys)
        assert result is None

    def test_empty_keys_list_returns_none(self, s3):
        result = load_json_with_fallback(s3, "test-bucket", [])
        assert result is None
