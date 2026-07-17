"""Tests for nousergon_lib.quant.stats.trial_accumulator (config#2454).

Pins:
  1. First increment on an absent artifact creates it (IfNoneMatch path).
  2. Repeated increments from the same producer accumulate (not overwrite).
  3. Increments from different producers accumulate into separate
     ``by_producer`` keys and a combined ``total``.
  4. n_trials <= 0 is a no-op (returns current state unchanged, does not
     raise) — models a producer that legitimately swept zero cells.
  5. read_cumulative_trial_count on a never-written artifact returns the
     empty state (total=0), not an error.
  6. Concurrent-write race: a conditional-PUT precondition failure is
     retried against the fresh state rather than clobbering it.
  7. backfill_cumulative_trial_count seeds all 4 producers' historical
     sums in one shot.
  8. backfill refuses to clobber a non-empty counter unless overwrite=True.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import boto3
import pytest
from moto import mock_aws

from nousergon_lib.quant.stats.trial_accumulator import (
    DEFAULT_KEY,
    backfill_cumulative_trial_count,
    increment_trial_count,
    read_cumulative_trial_count,
)

BUCKET = "alpha-engine-research"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


class TestReadEmpty:
    def test_read_never_written_returns_empty_state(self, s3):
        state = read_cumulative_trial_count(BUCKET, s3_client=s3)
        assert state == {"total": 0, "last_updated": "", "by_producer": {}}


class TestIncrement:
    def test_first_increment_creates_artifact(self, s3):
        state = increment_trial_count(
            "gamma_sweep", 5, "2026-07-14", bucket=BUCKET, s3_client=s3,
        )
        assert state["total"] == 5
        assert state["by_producer"] == {"gamma_sweep": 5}
        assert state["last_updated"] == "2026-07-14"

        # Round-trips through S3, not just the in-memory return value.
        obj = s3.get_object(Bucket=BUCKET, Key=DEFAULT_KEY)
        on_disk = json.loads(obj["Body"].read())
        assert on_disk == state

    def test_same_producer_accumulates_across_cycles(self, s3):
        increment_trial_count("gamma_sweep", 5, "2026-07-01", bucket=BUCKET, s3_client=s3)
        state = increment_trial_count("gamma_sweep", 7, "2026-07-08", bucket=BUCKET, s3_client=s3)
        assert state["total"] == 12
        assert state["by_producer"] == {"gamma_sweep": 12}
        assert state["last_updated"] == "2026-07-08"

    def test_multiple_producers_accumulate_independently(self, s3):
        increment_trial_count("optimizer_param_sweep", 9, "2026-07-14", bucket=BUCKET, s3_client=s3)
        increment_trial_count("gamma_sweep", 5, "2026-07-14", bucket=BUCKET, s3_client=s3)
        increment_trial_count("cov_estimator_sweep", 8, "2026-07-14", bucket=BUCKET, s3_client=s3)
        state = increment_trial_count(
            "predictor_param_sweep", 137, "2026-07-14", bucket=BUCKET, s3_client=s3,
        )
        assert state["total"] == 9 + 5 + 8 + 137
        assert state["by_producer"] == {
            "optimizer_param_sweep": 9,
            "gamma_sweep": 5,
            "cov_estimator_sweep": 8,
            "predictor_param_sweep": 137,
        }

    def test_non_positive_n_trials_is_noop(self, s3):
        increment_trial_count("gamma_sweep", 5, "2026-07-01", bucket=BUCKET, s3_client=s3)
        state = increment_trial_count("gamma_sweep", 0, "2026-07-08", bucket=BUCKET, s3_client=s3)
        assert state["total"] == 5
        assert state["last_updated"] == "2026-07-01"  # unchanged — no write happened

        state = increment_trial_count("gamma_sweep", -3, "2026-07-08", bucket=BUCKET, s3_client=s3)
        assert state["total"] == 5


class TestConcurrentWriteRace:
    def test_precondition_failure_retries_against_fresh_state(self, s3):
        """Simulate a racing writer landing between our GET and PUT: the
        first put_object call raises PreconditionFailed once, then the
        retry (against the now-current ETag) succeeds and the racing
        writer's contribution is preserved (both increments land)."""

        real_put_object = s3.put_object
        call_count = {"n": 0}

        def flaky_put_object(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First attempt: simulate a racing writer having already
                # landed 100 trials under a different producer key just
                # before our conditional PUT — respond as S3 would.
                real_put_object(
                    Bucket=BUCKET, Key=DEFAULT_KEY,
                    Body=json.dumps(
                        {"total": 100, "last_updated": "2026-07-13",
                         "by_producer": {"cov_estimator_sweep": 100}},
                    ).encode(),
                    ContentType="application/json",
                )
                from botocore.exceptions import ClientError
                raise ClientError(
                    {"Error": {"Code": "PreconditionFailed"}}, "PutObject",
                )
            return real_put_object(**kwargs)

        with patch.object(s3, "put_object", side_effect=flaky_put_object):
            state = increment_trial_count(
                "gamma_sweep", 5, "2026-07-14", bucket=BUCKET, s3_client=s3,
            )

        # Our 5 landed on top of the racing writer's 100 — nothing lost.
        assert state["total"] == 105
        assert state["by_producer"]["cov_estimator_sweep"] == 100
        assert state["by_producer"]["gamma_sweep"] == 5
        assert call_count["n"] == 2

    def test_persistent_contention_raises_after_max_retries(self, s3):
        from botocore.exceptions import ClientError

        def always_conflict(**kwargs):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")

        with patch.object(s3, "put_object", side_effect=always_conflict):
            with patch("time.sleep"):  # don't actually sleep in tests
                with pytest.raises(RuntimeError, match="conditional-PUT retries"):
                    increment_trial_count(
                        "gamma_sweep", 5, "2026-07-14", bucket=BUCKET, s3_client=s3,
                    )


class TestBackfill:
    def test_backfill_seeds_all_producers(self, s3):
        state = backfill_cumulative_trial_count(
            {
                "optimizer_param_sweep": 4230,
                "gamma_sweep": 2350,
                "cov_estimator_sweep": 3760,
                "predictor_param_sweep": 61_884,
            },
            "2026-07-14",
            bucket=BUCKET,
            s3_client=s3,
        )
        assert state["total"] == 4230 + 2350 + 3760 + 61_884
        assert state["by_producer"]["predictor_param_sweep"] == 61_884

        # Persisted, not just returned.
        on_disk = read_cumulative_trial_count(BUCKET, s3_client=s3)
        assert on_disk == state

    def test_backfill_refuses_to_clobber_existing_nonzero_counter(self, s3):
        increment_trial_count("gamma_sweep", 5, "2026-07-01", bucket=BUCKET, s3_client=s3)
        with pytest.raises(RuntimeError, match="refusing backfill"):
            backfill_cumulative_trial_count(
                {"gamma_sweep": 9999}, "2026-07-14", bucket=BUCKET, s3_client=s3,
            )
        # Original state untouched.
        state = read_cumulative_trial_count(BUCKET, s3_client=s3)
        assert state["total"] == 5

    def test_backfill_overwrite_true_forces_reseed(self, s3):
        increment_trial_count("gamma_sweep", 5, "2026-07-01", bucket=BUCKET, s3_client=s3)
        state = backfill_cumulative_trial_count(
            {"gamma_sweep": 9999}, "2026-07-14",
            bucket=BUCKET, s3_client=s3, overwrite=True,
        )
        assert state["total"] == 9999
