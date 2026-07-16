"""Shared EC2-spot dispatch primitives for the fleet's Lambda dispatchers.

Extracted (config#2106) from two independent implementations of the same
three primitives — tag-based concurrency lock, spot-launch-with-on-demand-
fallback, terminate-on-post-launch-failure — that had accumulated in
`scheduled-groom-dispatcher/index.py` (config#1432) and
`ci-watch-dispatcher/index.py` (config#2001). Both files are in
nousergon-data. A third independent copy (`sf-watch-spot-dispatcher`) was
about to be written when this module was extracted instead, per the fleet's
own "second adoption is a strong consolidation signal" convention.

No raise/return-clean posture opinion here — callers choose. These
functions RAISE on failure (matches groom's native fail-loud posture,
appropriate for an EventBridge-triggered caller where retry-on-error is
correct). A caller wanting a synchronous fail-soft contract (ci-watch's
`lambda invoke` RequestResponse caller needs a clean JSON verdict, not an
invocation error to unwrap) wraps calls in its own try/except and converts
to `{"launched": False, "reason": ...}` — exactly as
`ci-watch-dispatcher/index.py` already does.
"""

from __future__ import annotations

import logging
import time

import boto3
from krepis import alerts
from nousergon_lib import ec2_spot

# ec2_spot.py is a sys.modules rebind shim to krepis.ec2_spot; pyright can't
# see through the dynamic rebind (verified correct at runtime), so each
# symbol needs its own reportAttributeAccessIssue ignore on the line pyright
# actually flags (a single ignore on the `from ... import (` line, as this
# used to be written as a one-liner, only silences the first diagnostic).
from nousergon_lib.ec2_spot import (
    SpotCapacityExhausted,  # noqa: F401 - re-exported for callers  # pyright: ignore[reportAttributeAccessIssue]
    SpotLaunchError,  # noqa: F401 - re-exported for callers  # pyright: ignore[reportAttributeAccessIssue]
    SpotQuotaExceededError,  # noqa: F401 - re-exported for callers  # pyright: ignore[reportAttributeAccessIssue]
)

logger = logging.getLogger(__name__)


class SpotProbeError(Exception):
    """The pre-launch concurrency probe (``running_instance_ids``) could not
    determine whether a matching box is live — DescribeInstances itself
    failed. Distinct from the probe's clean-empty ``[]`` result ("no
    instances"): a degraded EC2 API must never masquerade as "no duplicate
    running" (config#2267 site 1 — the old fail-open ``return []`` silently
    dropped the duplicate-box guard). Callers choose their posture explicitly:
    fail the dispatch loudly, or proceed to launch with a recorded
    ``dedupe_degraded`` marker (coverage beats dedupe)."""


def launch_with_fallback(
    instance_types: list[str],
    subnets: list[str],
    *,
    image_id: str,
    key_name: str,
    security_group_ids: list[str],
    iam_instance_profile: str,
    volume_size_gb: int,
    tag_name: str,
    region: str,
    force_on_demand: bool = False,
    extra_tags: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Launch a box; spot first, on-demand fallback on SpotCapacityExhausted
    OR SpotQuotaExceededError (config#2698 — an account-wide spot quota
    ceiling, e.g. ``MaxSpotInstanceCountExceeded``, gets the identical
    on-demand fallback as ordinary per-AZ capacity exhaustion, plus an
    operator page since a quota ceiling only clears via a human-requested
    increase), or immediately on-demand if ``force_on_demand`` (e.g.
    config#1645's bounded relaunch escalation after repeated mid-run spot
    interruption). Returns ``(instance_id, market)`` where market is
    ``"spot"`` or ``"on-demand"``. Raises ``SpotLaunchError`` (or its
    ``SpotCapacityExhausted``/``SpotQuotaExceededError`` subclasses) if
    every attempt is exhausted/fails.

    ``extra_tags`` (config#2292, root fix for config#2267 site 2): additional
    ``{key: value}`` instance tags threaded straight through to
    :func:`krepis.ec2_spot.launch`, which merges them into the SAME
    RunInstances ``TagSpecifications`` entry as ``tag_name`` — the box is
    never observably untagged. Callers that previously wrote load-bearing
    discriminator tags via a separate post-launch ``create_tags`` call (with
    its own bounded retry) should pass them here instead and delete that
    retry path entirely.
    """
    common = {
        "image_id": image_id,
        "key_name": key_name,
        "security_group_ids": list(security_group_ids),
        "iam_instance_profile": iam_instance_profile,
        "volume_size_gb": volume_size_gb,
        "shutdown_behavior": "terminate",
        "tag_name": tag_name,
        "extra_tags": extra_tags,
        "region": region,
    }
    # ec2_spot.launch(...) below: ec2_spot.py is a sys.modules rebind shim
    # to krepis.ec2_spot; pyright can't see through the dynamic rebind,
    # verified correct at runtime — see the ignore-reason on the import
    # above.
    if force_on_demand:
        logger.warning("force_on_demand set — launching ON-DEMAND directly")
        iid = ec2_spot.launch(list(instance_types), list(subnets), spot=False, **common)  # pyright: ignore[reportAttributeAccessIssue]
        return iid, "on-demand"
    try:
        iid = ec2_spot.launch(list(instance_types), list(subnets), spot=True, **common)  # pyright: ignore[reportAttributeAccessIssue]
        return iid, "spot"
    except SpotCapacityExhausted:
        logger.warning(
            "spot capacity exhausted across all type×subnet pools — relaunching ON-DEMAND"
        )
        iid = ec2_spot.launch(list(instance_types), list(subnets), spot=False, **common)  # pyright: ignore[reportAttributeAccessIssue]
        return iid, "on-demand"
    except SpotQuotaExceededError as exc:
        # Account-wide (config#2698) — distinct from ordinary capacity
        # rotation exhaustion, so this gets its own operator page: capacity
        # exhaustion self-heals as AWS capacity shifts, but a quota ceiling
        # only clears via a service-quota increase, which needs a human to
        # notice and request.
        logger.warning("spot quota exceeded (%s) — relaunching ON-DEMAND", exc)
        alerts.publish(
            f"EC2 spot quota exceeded for {tag_name!r} in {region} — "
            f"falling back to on-demand: {exc}",
            severity="warning",
            source="nousergon_lib.spot_dispatch.launch_with_fallback",
            dedup_key=f"spot-quota-exceeded-{region}",
        )
        iid = ec2_spot.launch(list(instance_types), list(subnets), spot=False, **common)  # pyright: ignore[reportAttributeAccessIssue]
        return iid, "on-demand"


def wait_ssm_online(instance_id: str, *, region: str, ssm_online_budget_sec: int = 180) -> None:
    """Block until the instance is running AND its SSM agent registers Online."""
    ec2 = boto3.client("ec2", region_name=region)
    ssm = boto3.client("ssm", region_name=region)
    ec2.get_waiter("instance_running").wait(
        InstanceIds=[instance_id], WaiterConfig={"Delay": 5, "MaxAttempts": 40}
    )
    deadline = time.time() + ssm_online_budget_sec
    while time.time() < deadline:
        info = ssm.describe_instance_information(
            Filters=[{"Key": "InstanceIds", "Values": [instance_id]}]
        ).get("InstanceInformationList", [])
        if info and info[0].get("PingStatus") == "Online":
            logger.info("SSM agent Online for %s", instance_id)
            return
        time.sleep(5)
    raise RuntimeError(f"SSM agent not Online after {ssm_online_budget_sec}s for {instance_id}")


def send_async_command(
    instance_id: str,
    command: str,
    *,
    comment: str,
    region: str,
    cw_log_group: str,
    execution_timeout_seconds: int,
    start_timeout_seconds: int = 600,
) -> str:
    """Fire an async, detached ``AWS-RunShellScript`` SSM command. Returns the
    command id. ``execution_timeout_seconds`` bounds the command itself (NOT
    the start timeout) — without it SSM kills the command at its own 3600s
    default, guillotining a longer-running workload."""
    ssm = boto3.client("ssm", region_name=region)
    resp = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Comment=comment,
        Parameters={
            "commands": [command],
            "executionTimeout": [str(execution_timeout_seconds)],
        },
        TimeoutSeconds=start_timeout_seconds,
        CloudWatchOutputConfig={
            "CloudWatchLogGroupName": cw_log_group,
            "CloudWatchOutputEnabled": True,
        },
    )
    return resp["Command"]["CommandId"]


def running_instance_ids(
    tag_name: str, discriminator_tags: dict[str, str], *, region: str
) -> list[str]:
    """Instance ids for a LIVE (``pending``/``running``) box matching
    ``tag_name`` AND every key/value in ``discriminator_tags``.

    ``[]`` means exactly one thing: the probe RAN and found no matching box.
    A DescribeInstances failure raises :class:`SpotProbeError` (chained) —
    it never returns ``[]`` (config#2267 site 1: the old fail-open ``[]``
    made a degraded EC2 API indistinguishable from "no duplicate", silently
    vanishing the duplicate-box guard). The caller decides whether a failed
    probe blocks the launch or degrades to launch-with-logged-flag."""
    try:
        ec2 = boto3.client("ec2", region_name=region)
        filters = [
            {"Name": "tag:Name", "Values": [tag_name]},
            {"Name": "instance-state-name", "Values": ["pending", "running"]},
        ]
        for key, value in discriminator_tags.items():
            filters.append({"Name": f"tag:{key}", "Values": [value]})
        resp = ec2.describe_instances(Filters=filters)
        return [
            i["InstanceId"]
            for r in resp.get("Reservations", [])
            for i in r.get("Instances", [])
        ]
    except Exception as exc:  # noqa: BLE001 - re-raised as SpotProbeError; never swallowed
        logger.error(
            "concurrency probe DescribeInstances failed for tag_name=%s: %s: %s",
            tag_name, type(exc).__name__, exc,
        )
        raise SpotProbeError(
            f"concurrency probe failed for tag_name={tag_name!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def terminate_on_failure(instance_id: str, *, region: str, label: str = "spot") -> None:
    """Best-effort terminate of a just-launched box whose post-launch steps
    failed, to avoid orphaning it (no watchdog/trap is armed yet — that only
    happens inside the box's own bootstrap script). Never masks the original
    error (logged, not raised) — the caller still surfaces/raises/returns
    the original failure."""
    try:
        boto3.client("ec2", region_name=region).terminate_instances(InstanceIds=[instance_id])
        logger.warning(
            "terminated %s box %s after post-launch failure (no orphan)", label, instance_id
        )
    except Exception as exc:  # noqa: BLE001 - cleanup; caller still surfaces the original error
        logger.error(
            "FAILED to terminate %s box %s after a post-launch error (%s) — MANUAL cleanup needed",
            label, instance_id, exc,
        )
