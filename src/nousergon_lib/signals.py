"""Shared signals.json S3 fallback-chain utilities for the Alpha Engine fleet.

Research (Scanner / SignalsEnvelope) writes signals.json only on Saturdays
during the weekly freshness pipeline.  Weekday consumers that need a recent
snapshot must walk a fallback chain: today's dated key → prior weekdays →
``signals/latest.json``.

Before this module existed, the fallback algorithm was independently
duplicated in three repos (predictor, executor, dashboard), each with
slight variations.  This module consolidates the core patterns into one
place — callers compose the generic building blocks to match their own
API shape.

Usage::

    from nousergon_lib.signals import fallback_research_date_keys, try_read_s3_json

    keys = fallback_research_date_keys("2026-07-23")
    for key in keys:
        payload = try_read_s3_json(s3, bucket, key)
        if payload:
            return payload
    return {}
"""

from __future__ import annotations

import json
from datetime import date as _date, timedelta as _td
from typing import Any, Optional


def fallback_research_date_keys(
    date_str: str,
    max_weekdays: int = 5,
) -> list[str]:
    """Return S3 keys for signals.json in priority order.

    The order matches the Alpha Engine research pipeline's Saturday-only
    write cadence: today's dated snapshot → the most recent *N* prior
    weekdays' dated snapshots → the rolling ``signals/latest.json``
    pointer.

    Weekends (weekday() >= 5) are skipped because research never writes
    on Saturday or Sunday, so there is never anything to find there.

    Parameters
    ----------
    date_str:
        ISO-format date string (e.g. ``"2026-07-23"``) to start the
        chain from.  Typically the trading day the caller is processing.
    max_weekdays:
        How many prior weekdays to include.  5 (the default) covers
        Monday–Friday of a standard week, so a Monday run sees the prior
        Saturday's snapshot after skipping Saturday+Sunday.

    Returns
    -------
    list[str]
        At least one entry (the ``signals/latest.json`` sentinel is
        always appended).  On a malformed *date_str* an empty prefix is
        returned with just the sentinel, so callers always have
        something to try.
    """
    keys: list[str] = []
    try:
        start = _date.fromisoformat(date_str)
        for days_back in range(max_weekdays + 1):
            candidate = start - _td(days=days_back)
            if candidate.weekday() >= 5:
                continue
            keys.append(f"signals/{candidate.isoformat()}/signals.json")
    except (ValueError, TypeError):
        pass
    keys.append("signals/latest.json")
    return keys


def try_read_s3_json(
    s3_client: Any,
    bucket: str,
    key: str,
) -> Optional[dict[str, Any]]:
    """Read a single S3 JSON object, returning ``None`` on a resolvable miss.

    This is deliberately forgiving on transient errors (permission,
    parse, network) — the caller is walking a fallback chain and should
    silently continue to the next key rather than crashing because one
    candidate in the list is unreachable.  Only re-raises when the error
    code is definitively NOT a "this key doesn't exist" signal, so an
    auth/credential failure on the *first* key propagates up.

    Parameters
    ----------
    s3_client:
        A ``boto3.client("s3")`` instance.
    bucket:
        S3 bucket name.
    key:
        S3 object key to read.

    Returns
    -------
    dict or None
        Parsed JSON dict, or ``None`` if the key does not exist, access
        is denied, the body is not valid JSON, or any non-terminal
        ``ClientError`` is hit.
    """
    from botocore.exceptions import ClientError

    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read()
        if not body:
            return None
        return json.loads(body.decode("utf-8"))
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        # "NoSuchKey", "AccessDenied", 403, 404 — these are expected for
        # a key that simply doesn't exist or isn't readable.
        if code in ("NoSuchKey", "AccessDenied", "403", "404"):
            return None
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
        return None


def load_json_with_fallback(
    s3_client: Any,
    bucket: str,
    keys: list[str],
) -> Optional[dict[str, Any]]:
    """Walk a priority-ordered list of S3 keys, returning the first payload found.

    Parameters
    ----------
    s3_client:
        A ``boto3.client("s3")`` instance.
    bucket:
        S3 bucket name.
    keys:
        Priority-ordered S3 key list — the first one that resolves to a
        non-empty JSON dict is returned.  Typically produced by
        :func:`fallback_research_date_keys`.

    Returns
    -------
    dict or None
        The first non-empty payload in *keys*, or ``None`` if none
        resolved.
    """
    for key in keys:
        payload = try_read_s3_json(s3_client, bucket, key)
        if payload:
            return payload
    return None
