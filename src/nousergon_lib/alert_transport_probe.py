#!/usr/bin/env python3
"""Reusable end-to-end alert-transport liveness probe (alpha-engine-config-I9335).

WHY THIS IS LIFTED HERE, AND WHY NOW
-------------------------------------
`claude-code-config/laptop-checks/alert_transport_liveness.py`
(claude-code-config-PR218, alpha-engine-config-I9209) proved that a laptop
alert emitted through `krepis.fleet_events.emit_alert_event` actually ARRIVES,
not merely that the call returned. alpha-engine-config-I9335 measured that the
laptop is the ONLY origin with that proof — every other substrate (a Lambda, an
EC2 spot box, a scheduled GHA workflow, a Step Functions execution) is
IAM-configured but never probed end-to-end. Generalising the check to a second
substrate is this module's second adoption, which is exactly
`shared-code-policy.md` §2's lift trigger: copy once, lift on the second
consumer. `laptop-checks/alert_transport_liveness.py` keeps running unchanged;
it now imports its three probe functions from here instead of defining them.

WHAT THIS MODULE DOES AND DOES NOT KNOW
----------------------------------------
It knows nothing about launchd, GitHub Actions, or systemd. It takes a boto3
Session (or a profile name, laptop-side) and three ARNs/paths, and returns
finding dicts in the exact shape every caller already expects:
`{"key": ..., "ok": bool, "detail": ...}`. Cadence, plist/workflow wiring, and
the `fleet_check_result` envelope stay in each substrate's own thin wrapper —
those differ per substrate by construction and do not belong in a shared lib.

THE NONCE-EXACT ARRIVAL GAP (alpha-engine-config-I9334)
--------------------------------------------------------
`probe_arrival_by_counter` is the ORIGINAL, weaker arrival proof: it watches the
`AWS/Events` `SuccessfulInvocationAttempts` counter on the intake rule move by
at least the number of accepted `PutEvents`. A counter cannot distinguish "our
synthetic event arrived" from "somebody else's real alert arrived in the same
window" — measured 2026-08-29 at 8-72 events/day on the production rule. It is
kept ONLY as an explicit degraded fallback for a caller that has not yet
provisioned the nonce-exact log-group route (`probe_arrival_by_nonce`), and it
must never be silently preferred when the exact route is configured.
`probe_arrival_by_nonce` reads the event back by its exact nonce from a
CloudWatch Logs group an EventBridge rule delivers into
(`/alpha-engine/alert-transport-liveness` for the production bus), which is an
exact, unambiguous, per-run proof with no collision window at all.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any


def synthetic_detail(
    *, origin: str, source_suffix: str, nonce: str, extra_body: str = ""
) -> dict[str, Any]:
    """A schema-valid `nousergon.alert.v1` detail, marked synthetic.

    Deliberately hand-built rather than imported from `krepis.fleet_events`:
    this probe must keep working when the krepis install on the probed
    substrate is the thing that is broken, and a liveness probe sharing a
    failure mode with the path it probes is not a probe.
    """
    return {
        "schema_version": 1,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "origin": origin,
        "source": f"{origin}/{source_suffix}",
        "severity": "info",
        "severity_raw": "info",
        "body": (
            f"[SYNTHETIC LIVENESS PROBE - no action] alert transport liveness "
            f"nonce={nonce} identity={source_suffix} {extra_body}".rstrip()
        ),
        "dedup_key": None,
        # None is the v1 contract's "this producer tracks no condition" — NOT
        # "the condition is open". A synthetic probe must never look like an
        # open incident to the drain.
        "state": None,
        "identity_key": f"{origin}/{source_suffix}",
        "channels": None,
        "disable_notification": True,
        "runtime": {"lambda_function_name": None, "hostname": None},
    }


def new_nonce() -> str:
    return uuid.uuid4().hex[:12]


def probe_primary(
    session: Any, *, bus_name: str, profile_label: str, detail: dict[str, Any]
) -> dict[str, Any]:
    """PutEvents onto the real bus. Returns a finding dict."""
    try:
        resp = session.client("events").put_events(
            Entries=[
                {
                    "Source": "nousergon.krepis",
                    "DetailType": "nousergon.alert.v1",
                    "Detail": json.dumps(detail),
                    "EventBusName": bus_name,
                }
            ]
        )
    except Exception as exc:  # noqa: BLE001 - the exception IS the finding
        return {
            "key": f"primary/{profile_label}",
            "ok": False,
            "detail": f"events:PutEvents raised: {exc!r}",
        }
    failed = resp.get("FailedEntryCount") or 0
    if failed:
        entry = resp["Entries"][0]
        return {
            "key": f"primary/{profile_label}",
            "ok": False,
            "detail": (
                f"PutEvents rejected the entry: "
                f"{entry.get('ErrorCode')} {entry.get('ErrorMessage')}"
            ),
        }
    return {
        "key": f"primary/{profile_label}",
        "ok": True,
        "detail": f"PutEvents accepted (event id {resp['Entries'][0].get('EventId')})",
    }


def probe_fallback(
    session: Any,
    *,
    bucket: str,
    prefix: str,
    profile_label: str,
    nonce: str,
    detail: dict[str, Any],
) -> dict[str, Any]:
    """Write the real fallback shape, then read it back by exact key.

    No attribution ambiguity: the object is addressed by a key only this run
    knows, so a successful read-back is proof that THIS identity's write
    arrived and nothing else's did the proving.
    """
    now = datetime.now(timezone.utc)
    key = f"{prefix}/{now:%Y-%m-%d}/{now:%H%M%S}-liveness-{profile_label}-{nonce}.json"
    body = json.dumps(
        {"source": "nousergon.krepis", "detail_type": "nousergon.alert.v1", "detail": detail}
    ).encode()
    try:
        s3 = session.client("s3")
        s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    except Exception as exc:  # noqa: BLE001
        return {
            "key": f"fallback/{profile_label}",
            "ok": False,
            "detail": f"s3:PutObject to {key} raised: {exc!r}",
        }
    try:
        back = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as exc:  # noqa: BLE001
        return {
            "key": f"fallback/{profile_label}",
            "ok": False,
            "detail": (
                f"wrote {key} but could not read it back: {exc!r} — a write "
                f"this identity cannot verify is not a proven transport"
            ),
        }
    if json.loads(back)["detail"]["body"] != detail["body"]:
        return {
            "key": f"fallback/{profile_label}",
            "ok": False,
            "detail": f"read back {key} but the body did not match the nonce",
        }
    return {"key": f"fallback/{profile_label}", "ok": True, "detail": f"wrote and read back {key}"}


def probe_arrival_by_counter(
    session: Any,
    *,
    bus_name: str,
    rule_name: str,
    expected: int,
    window_start: datetime,
    timeout_s: int = 300,
    poll_s: int = 20,
) -> dict[str, Any]:
    """DEGRADED arrival proof — a counter, not a nonce (alpha-engine-config-I9334).

    Keep only as an explicit fallback when the nonce-exact route
    (`probe_arrival_by_nonce`) is not yet provisioned for this bus. A lost
    synthetic event can be masked by a real one arriving in the same window;
    callers that have the log-group route MUST prefer it and must not run
    this as a "second opinion" alongside it — the weaker signal silently
    papering over the stronger one's absence is the failure mode this exists
    to avoid.
    """
    if expected == 0:
        return {
            "key": "arrival/intake-rule",
            "ok": False,
            "detail": (
                "no identity's PutEvents was accepted, so nothing could arrive "
                "— arrival not measured (this is a consequence, not a "
                "second independent failure)"
            ),
        }
    deadline = time.monotonic() + timeout_s
    seen = 0.0
    while time.monotonic() < deadline:
        time.sleep(poll_s)
        try:
            resp = session.client("cloudwatch").get_metric_data(
                MetricDataQueries=[
                    {
                        "Id": "arrivals",
                        "MetricStat": {
                            "Metric": {
                                "Namespace": "AWS/Events",
                                "MetricName": "SuccessfulInvocationAttempts",
                                "Dimensions": [
                                    {"Name": "EventBusName", "Value": bus_name},
                                    {"Name": "RuleName", "Value": rule_name},
                                ],
                            },
                            "Period": 60,
                            "Stat": "Sum",
                        },
                        "ReturnData": True,
                    }
                ],
                StartTime=window_start,
                EndTime=datetime.now(timezone.utc) + timedelta(minutes=1),
            )
            seen = float(sum(resp["MetricDataResults"][0]["Values"]))
        except Exception as exc:  # noqa: BLE001
            return {
                "key": "arrival/intake-rule",
                "ok": False,
                "detail": (
                    f"could not read AWS/Events SuccessfulInvocationAttempts on "
                    f"{rule_name}: {exc!r} — arrival is UNMEASURED, which is "
                    f"not the same as not-arrived"
                ),
            }
        if seen >= expected:
            return {
                "key": "arrival/intake-rule",
                "ok": True,
                "detail": (
                    f"{seen:.0f} successful invocation(s) of {rule_name} in the "
                    f"probe window (expected at least {expected}); COUNTER proof "
                    f"only — see alpha-engine-config-I9334 for the exact-nonce gap"
                ),
            }
    return {
        "key": "arrival/intake-rule",
        "ok": False,
        "detail": (
            f"PutEvents was accepted but only {seen:.0f} of {expected} expected "
            f"invocation(s) of {rule_name} appeared within {timeout_s}s. An "
            f"event accepted onto a bus that no rule matches is discarded "
            f"silently and PutEvents still returns success — check the rule's "
            f"event pattern and its target."
        ),
    }


def probe_arrival_by_nonce(
    session: Any,
    *,
    log_group: str,
    nonce: str,
    window_start: datetime,
    timeout_s: int = 180,
    poll_s: int = 10,
) -> dict[str, Any]:
    """EXACT arrival proof (alpha-engine-config-I9334): find OUR nonce.

    Requires an EventBridge rule on the bus routing this probe's synthetic
    events (pattern `detail.origin == <origin>`) into `log_group`, and a
    resource policy on that log group permitting `events.amazonaws.com` to
    `PutLogEvents` — EventBridge-to-Logs delivery fails silently without it.
    """
    logs = session.client("logs")
    deadline = time.monotonic() + timeout_s
    start_ms = int(window_start.timestamp() * 1000)
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            resp = logs.filter_log_events(
                logGroupName=log_group, startTime=start_ms, filterPattern=f'"{nonce}"'
            )
            if resp.get("events"):
                return {
                    "key": "arrival/nonce-exact",
                    "ok": True,
                    "detail": (
                        f"nonce {nonce} found in {log_group} — exact, "
                        f"unambiguous end-to-end arrival, no collision window"
                    ),
                }
        except logs.exceptions.ResourceNotFoundException as exc:
            last_exc = exc
            return {
                "key": "arrival/nonce-exact",
                "ok": False,
                "detail": (
                    f"log group {log_group} does not exist — the EventBridge "
                    f"rule + log group + resource policy for I9334 have not "
                    f"been provisioned yet: {exc!r}"
                ),
            }
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            break
        time.sleep(poll_s)
    return {
        "key": "arrival/nonce-exact",
        "ok": False,
        "detail": (
            f"nonce {nonce} not found in {log_group} within {timeout_s}s"
            + (f" (last error: {last_exc!r})" if last_exc else "")
        ),
    }
