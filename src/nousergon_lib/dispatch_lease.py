"""
Per-lane singleton dispatch lease for scheduled reconciliation loops.

groom-sweep-policy.md §5.9: "The loop holds a singleton lease per lane. A
dispatch that cannot take the lease exits, recording that it yielded; it
does not queue behind the holder and it does not run beside it." This module
is the primitive that closes that gap for dispatchers whose actual unit of
work outlives the process that launches it — e.g. a Lambda that fires an
async, detached EC2/SSM launch and returns in seconds while the box it
launched runs for hours. That shape cannot use a context-manager-held lock
(``krepis.locks.universe_writer_lock``): there is no live process to hold
the ``with`` block open for the lane's whole lifetime. This module exposes
the SAME mechanism — an S3 ``PutObject`` with ``IfNoneMatch="*"`` (atomic
compare-and-swap) plus a soft TTL for self-recovery when a holder dies
without releasing — as an explicit ``acquire_lease``/``release_lease`` pair
instead of a context manager, so the lease can be acquired once and left
held across process boundaries, then released by TTL expiry (the common
case: the holder dies without an actor to run the ``finally``) or by an
explicit call from whatever later observes the work as done.

**Relationship to krepis.locks.universe_writer_lock.** Same underlying
mechanism (conditional PUT, ``PreconditionFailed``/``412`` on conflict,
soft-TTL delete-and-retry recovery), independently implemented here rather
than reached into via krepis's private helpers, because the two lease
shapes are genuinely different at the API surface: ``universe_writer_lock``
is scoped to the lifetime of a ``with`` block; a dispatch lane's lease is
scoped to the lifetime of work a different process (the launched box) does
after the acquiring process has already returned. Forcing the async case
through a context manager would mean either holding the ``with`` block open
in a still-running Lambda (impossible — Lambdas do not babysit multi-hour
launches, see ``scheduled-groom-dispatcher``'s module docstring) or
releasing on ``__exit__`` immediately after acquiring (which defeats the
lease's purpose entirely).

**The ``force`` override.** A per-lane lease must not block a dispatcher's
OWN legitimate bounded-retry relaunch of a lane whose prior attempt died
before its TTL naturally expired (spot reclaim, box crash, watchdog kill —
all confirmed independently via live EC2 state by the caller, not by this
module). ``force=True`` tells :func:`acquire_lease` the caller already has
that independent proof and to take the lease unconditionally, regardless of
TTL. This module never infers "the holder is dead" on its own — the caller
must establish that (e.g. via ``spot_dispatch.termination_imminent``) before
passing ``force=True``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import socket
import time
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "alpha-engine-research"


@dataclasses.dataclass(frozen=True)
class LeaseHolder:
    """The body written into S3 at acquire time — who holds (or held) the lease."""

    owner_id: str
    started_at: str  # ISO-8601 UTC
    ttl_epoch: int  # unix-seconds; soft expiry the next acquirer honors
    hostname: str
    pid: int

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "LeaseHolder":
        d = json.loads(payload)
        return cls(
            owner_id=str(d["owner_id"]),
            started_at=str(d["started_at"]),
            ttl_epoch=int(d["ttl_epoch"]),
            hostname=str(d["hostname"]),
            pid=int(d["pid"]),
        )


@dataclasses.dataclass(frozen=True)
class LeaseAcquireResult:
    """Outcome of :func:`acquire_lease`.

    ``acquired=True``: ``holder`` is OUR holder body (already written to S3).
    ``acquired=False``: ``holder`` is the CONFLICTING holder currently on
    record — the caller should treat this as a yield, never a queue.
    """

    acquired: bool
    holder: LeaseHolder


def _now_epoch() -> int:
    """Indirection for monkeypatching in tests."""
    return int(time.time())


def _now_iso() -> str:
    """Indirection for monkeypatching in tests."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _build_holder(owner_id: str, ttl_seconds: int) -> LeaseHolder:
    return LeaseHolder(
        owner_id=owner_id,
        started_at=_now_iso(),
        ttl_epoch=_now_epoch() + ttl_seconds,
        hostname=socket.gethostname(),
        pid=os.getpid(),
    )


def _read_existing_holder(s3_client, bucket: str, key: str) -> Optional[LeaseHolder]:
    """Read + parse the current lease body, or None if absent / malformed.

    A malformed body can't be interpreted as "held by someone we know
    about", so it is treated as not-held — the caller's own conditional PUT
    still governs correctness (a malformed reader never grants a lease by
    itself, it only decides whether to attempt the recovery PUT).
    """
    try:
        from botocore.exceptions import ClientError
    except ImportError:  # pragma: no cover - botocore is a hard dep
        raise

    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise
    body = obj["Body"].read().decode("utf-8")
    try:
        return LeaseHolder.from_json(body)
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning(
            "Malformed lease body at s3://%s/%s — treating as absent: %s",
            bucket, key, exc,
        )
        return None


def _try_conditional_put(s3_client, bucket: str, key: str, holder: LeaseHolder) -> bool:
    """Attempt ``put_object(IfNoneMatch='*')``. Return True on acquired,
    False on PreconditionFailed (lease now held by someone else)."""
    from botocore.exceptions import ClientError

    try:
        s3_client.put_object(
            Bucket=bucket, Key=key,
            Body=holder.to_json().encode("utf-8"),
            IfNoneMatch="*", ContentType="application/json",
        )
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        # S3 returns "PreconditionFailed" on If-None-Match conflict; legacy
        # mocks / older boto3 surface "412" as the bare HTTP status.
        if code in ("PreconditionFailed", "412"):
            return False
        raise


def _delete_best_effort(s3_client, bucket: str, key: str) -> None:
    try:
        s3_client.delete_object(Bucket=bucket, Key=key)
    except Exception as exc:  # noqa: BLE001 - best-effort cleanup
        logger.warning("lease delete failed at s3://%s/%s (non-fatal): %s", bucket, key, exc)


def acquire_lease(
    lock_key: str,
    *,
    owner_id: str,
    ttl_seconds: int,
    bucket: str = DEFAULT_BUCKET,
    s3_client=None,
    force: bool = False,
) -> LeaseAcquireResult:
    """Attempt to acquire a singleton lease at ``s3://{bucket}/{lock_key}``.

    Returns immediately either way — this function NEVER retries against a
    live, unexpired conflicting holder and never blocks waiting for one to
    release (groom-sweep-policy §5.9: "does not queue behind the holder").

    :param lock_key: S3 key identifying the lane (or shared resource) being
        leased. Callers sharing a resource (§5.9.3) pass the SAME key.
    :param owner_id: Logical identity of this acquirer, carried into the
        lease body for diagnostics. NOT used for acquisition uniqueness.
    :param ttl_seconds: Soft TTL. A stale lease (elapsed ``ttl_epoch``) is
        deleted and re-acquired automatically — this is how a killed holder
        (the normal case on interruptible compute) releases the lease
        without an actor.
    :param force: Skip the TTL/liveness check entirely and take the lease
        unconditionally (delete-then-PUT). Use ONLY when the caller has
        independent, live confirmation the current holder is gone (e.g. a
        bounded SF relaunch after confirming the prior box is
        termination-imminent) — see module docstring.
    :param s3_client: Optional boto3 S3 client override (for tests).
    """
    if s3_client is None:
        import boto3
        s3_client = boto3.client("s3")

    holder = _build_holder(owner_id=owner_id, ttl_seconds=ttl_seconds)

    if force:
        _delete_best_effort(s3_client, bucket, lock_key)
        if _try_conditional_put(s3_client, bucket, lock_key, holder):
            logger.info(
                "Acquired lease (forced) at s3://%s/%s (owner_id=%s, ttl_epoch=%d)",
                bucket, lock_key, owner_id, holder.ttl_epoch,
            )
            return LeaseAcquireResult(acquired=True, holder=holder)
        # A genuine race against a fresh acquirer mid-force — do not loop;
        # report the conflict like any other failed acquire.
        existing = _read_existing_holder(s3_client, bucket, lock_key) or holder
        return LeaseAcquireResult(acquired=False, holder=existing)

    if _try_conditional_put(s3_client, bucket, lock_key, holder):
        logger.info(
            "Acquired lease at s3://%s/%s (owner_id=%s, ttl_epoch=%d)",
            bucket, lock_key, owner_id, holder.ttl_epoch,
        )
        return LeaseAcquireResult(acquired=True, holder=holder)

    # Lease is held. Inspect it; if the soft TTL has elapsed, self-recover.
    existing = _read_existing_holder(s3_client, bucket, lock_key)
    if existing is not None and existing.ttl_epoch > _now_epoch():
        return LeaseAcquireResult(acquired=False, holder=existing)

    logger.warning(
        "Stale or malformed lease at s3://%s/%s — self-recovering "
        "(existing_ttl_epoch=%s, now=%s)",
        bucket, lock_key, getattr(existing, "ttl_epoch", None), _now_epoch(),
    )
    _delete_best_effort(s3_client, bucket, lock_key)
    if _try_conditional_put(s3_client, bucket, lock_key, holder):
        return LeaseAcquireResult(acquired=True, holder=holder)

    # Someone else raced us to the delete-and-recover. Do NOT loop — report
    # the new conflicting holder; the caller decides whether/when to retry.
    existing = _read_existing_holder(s3_client, bucket, lock_key)
    if existing is None:
        existing = holder  # race window with no holder visible; shape-compatible placeholder
    return LeaseAcquireResult(acquired=False, holder=existing)


def release_lease(lock_key: str, *, bucket: str = DEFAULT_BUCKET, s3_client=None) -> None:
    """Best-effort release. Never raises — the hard S3-lifecycle TTL (if
    configured on the ``locks/`` prefix) is the authoritative purger for a
    release that fails to land, mirroring ``krepis.locks``' own posture."""
    if s3_client is None:
        import boto3
        s3_client = boto3.client("s3")
    _delete_best_effort(s3_client, bucket, lock_key)
    logger.info("Released lease at s3://%s/%s", bucket, lock_key)
