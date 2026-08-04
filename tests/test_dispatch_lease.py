"""Unit tests for ``nousergon_lib.dispatch_lease``.

alpha-engine-config-I6460 (groom-sweep-policy.md §5.9): a dispatcher that
cannot take a lane's lease must yield immediately — never queue behind the
holder, never run beside it — and a lease left by a holder that died
uncleanly must self-release via TTL so the next dispatch is never wedged
forever. These tests pin both properties plus the ``force`` override a
caller uses when it has independent proof (not asserted here — that proof
is the caller's job) that the recorded holder is dead before its TTL.

No live AWS: ``s3_client`` is a tiny in-memory fake reproducing exactly the
two behaviors this module depends on — conditional PUT via ``IfNoneMatch``
raising ``ClientError(PreconditionFailed)`` on conflict, and GET raising
``ClientError(NoSuchKey)`` when absent — mirroring the real botocore
contract closely enough that swapping in a real S3 client changes nothing
about the module under test.
"""

from __future__ import annotations

import pytest
from botocore.exceptions import ClientError

from nousergon_lib import dispatch_lease


class FakeS3:
    """In-memory stand-in for the boto3 S3 client surface this module uses."""

    def __init__(self):
        self._objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket, Key, Body, IfNoneMatch=None, ContentType=None):
        key = (Bucket, Key)
        if IfNoneMatch == "*" and key in self._objects:
            raise ClientError(
                {"Error": {"Code": "PreconditionFailed", "Message": "conflict"}},
                "PutObject",
            )
        self._objects[key] = Body

    def get_object(self, *, Bucket, Key):
        key = (Bucket, Key)
        if key not in self._objects:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
            )
        body = self._objects[key]

        class _Body:
            def read(_self):
                return body

        return {"Body": _Body()}

    def delete_object(self, *, Bucket, Key):
        self._objects.pop((Bucket, Key), None)


BUCKET = "alpha-engine-research-test"
LOCK_KEY = "locks/groom-lane-mid-only.lock"


def _patch_clock(monkeypatch, epoch: int):
    monkeypatch.setattr(dispatch_lease, "_now_epoch", lambda: epoch)
    monkeypatch.setattr(dispatch_lease, "_now_iso", lambda: "2026-08-04T00:00:00Z")


class TestAcquireLease:
    def test_first_acquire_succeeds(self):
        s3 = FakeS3()
        result = dispatch_lease.acquire_lease(
            LOCK_KEY, owner_id="cycle-a", ttl_seconds=3600, bucket=BUCKET, s3_client=s3,
        )
        assert result.acquired is True
        assert result.holder.owner_id == "cycle-a"

    def test_second_acquire_while_held_yields_without_queuing(self):
        """The core §5.9 property: a second dispatch that cannot take the
        lease must be told NO immediately — it must never block, retry
        internally, or otherwise wait for the holder. It gets back the
        CONFLICTING holder's identity so it can log/record who it yielded
        to, and it does not launch anything."""
        s3 = FakeS3()
        first = dispatch_lease.acquire_lease(
            LOCK_KEY, owner_id="cycle-a", ttl_seconds=3600, bucket=BUCKET, s3_client=s3,
        )
        assert first.acquired is True

        second = dispatch_lease.acquire_lease(
            LOCK_KEY, owner_id="cycle-b", ttl_seconds=3600, bucket=BUCKET, s3_client=s3,
        )
        assert second.acquired is False
        assert second.holder.owner_id == "cycle-a"  # the holder we yielded to

    def test_lease_self_recovers_after_ttl_expiry(self, monkeypatch):
        """A holder that dies without releasing (kill -9, spot reclaim, OOM)
        must not wedge the lane forever — the next acquirer past the TTL
        self-recovers with no operator action."""
        s3 = FakeS3()
        _patch_clock(monkeypatch, epoch=1_000_000)
        first = dispatch_lease.acquire_lease(
            LOCK_KEY, owner_id="cycle-a", ttl_seconds=100, bucket=BUCKET, s3_client=s3,
        )
        assert first.acquired is True
        assert first.holder.ttl_epoch == 1_000_100

        # Still within TTL — must yield, not recover.
        _patch_clock(monkeypatch, epoch=1_000_050)
        still_held = dispatch_lease.acquire_lease(
            LOCK_KEY, owner_id="cycle-b", ttl_seconds=100, bucket=BUCKET, s3_client=s3,
        )
        assert still_held.acquired is False

        # Past TTL — self-recovery lets a fresh acquirer take it.
        _patch_clock(monkeypatch, epoch=1_000_200)
        recovered = dispatch_lease.acquire_lease(
            LOCK_KEY, owner_id="cycle-c", ttl_seconds=100, bucket=BUCKET, s3_client=s3,
        )
        assert recovered.acquired is True
        assert recovered.holder.owner_id == "cycle-c"

    def test_force_overrides_a_live_unexpired_lease(self):
        """A bounded SF relaunch that has independently confirmed (via live
        EC2 state, outside this module) that the prior holder's box is dead
        must be able to take the lease even though its TTL has not yet
        elapsed — this is the relaunch-ladder case, distinct from ordinary
        TTL self-recovery."""
        s3 = FakeS3()
        first = dispatch_lease.acquire_lease(
            LOCK_KEY, owner_id="cycle-a:attempt0", ttl_seconds=3600, bucket=BUCKET, s3_client=s3,
        )
        assert first.acquired is True

        relaunch = dispatch_lease.acquire_lease(
            LOCK_KEY, owner_id="cycle-a:attempt1", ttl_seconds=3600, bucket=BUCKET,
            s3_client=s3, force=True,
        )
        assert relaunch.acquired is True
        assert relaunch.holder.owner_id == "cycle-a:attempt1"

    def test_force_still_reports_conflict_on_a_genuine_race(self):
        """force is a caller-asserted override, not an unconditional write —
        if a third party wins the delete-and-recover race in the same
        instant, force reports the loss rather than silently succeeding
        against a lease it never actually wrote."""
        s3 = FakeS3()
        real_put = s3.put_object

        def racing_put(**kwargs):
            # Simulate someone else re-acquiring between our delete and our put.
            s3._objects[(kwargs["Bucket"], kwargs["Key"])] = b"placeholder"
            raise ClientError(
                {"Error": {"Code": "PreconditionFailed", "Message": "raced"}}, "PutObject"
            )

        first = dispatch_lease.acquire_lease(
            LOCK_KEY, owner_id="cycle-a", ttl_seconds=3600, bucket=BUCKET, s3_client=s3,
        )
        assert first.acquired is True

        s3.put_object = racing_put
        result = dispatch_lease.acquire_lease(
            LOCK_KEY, owner_id="cycle-b", ttl_seconds=3600, bucket=BUCKET,
            s3_client=s3, force=True,
        )
        assert result.acquired is False
        s3.put_object = real_put

    def test_malformed_lease_body_is_treated_as_absent_and_recovered(self):
        s3 = FakeS3()
        s3._objects[(BUCKET, LOCK_KEY)] = b"not json"
        result = dispatch_lease.acquire_lease(
            LOCK_KEY, owner_id="cycle-a", ttl_seconds=3600, bucket=BUCKET, s3_client=s3,
        )
        assert result.acquired is True


class TestReleaseLease:
    def test_release_deletes_the_object_and_allows_reacquisition(self):
        s3 = FakeS3()
        first = dispatch_lease.acquire_lease(
            LOCK_KEY, owner_id="cycle-a", ttl_seconds=3600, bucket=BUCKET, s3_client=s3,
        )
        assert first.acquired is True

        dispatch_lease.release_lease(LOCK_KEY, bucket=BUCKET, s3_client=s3)

        second = dispatch_lease.acquire_lease(
            LOCK_KEY, owner_id="cycle-b", ttl_seconds=3600, bucket=BUCKET, s3_client=s3,
        )
        assert second.acquired is True

    def test_release_of_a_never_acquired_key_never_raises(self):
        s3 = FakeS3()
        dispatch_lease.release_lease(LOCK_KEY, bucket=BUCKET, s3_client=s3)  # no raise
