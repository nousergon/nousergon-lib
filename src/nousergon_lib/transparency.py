"""
Transparency inventory substrate health checker.

Reads ``transparency_inventory.yaml``, validates that each row's
expected artifact exists with the expected cadence and content, and
returns per-row results. The Saturday and weekday Step Functions both
invoke this checker; the cadence flag determines which subset of rows
runs.

Phase 2 → 3 gate: ≥ 99% of inventory rows pass for 8 consecutive
weeks. The check fires per-row CloudWatch metrics so individual rows
have their own alarms — a failed row pages immediately, the gate
denominator is decremented for that row, and the 8-week clock resets.

Source kinds supported in v1:

  s3_json           HEAD + GET an S3 JSON object; assert_keys_present,
                    assert (path / op / value).
  s3_csv            HEAD + GET an S3 CSV; assert_columns_present,
                    assert_columns_non_null_for_rows_after,
                    assert_value_on_latest_row.
  s3_parquet        HEAD + GET an S3 parquet; assert_columns_present,
                    assert_column_non_null.
  sqlite_via_s3     Download SQLite DB from S3, run PRAGMA table_info
                    against ``table``, assert_columns_present.
  cloudwatch        GetMetricData over ``window_days``, assert
                    success_rate_pct_gte | datapoints_gte.

Source kinds not in v1 (deferred): cloudwatch_search,
custom_python_callable.

The checker is read-only — it does not write artifacts of its own.
The caller (CLI ``main()``) emits CloudWatch metrics from the result
list and optionally publishes SNS.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

log = logging.getLogger(__name__)

INVENTORY_PATH = Path(__file__).parent / "transparency_inventory.yaml"

DEFAULT_BUCKET = "alpha-engine-research"
DEFAULT_NAMESPACE_OUT = "AlphaEngine/Substrate"
DEFAULT_SNS_TOPIC = "arn:aws:sns:us-east-1:711398986525:alpha-engine-alerts"


@dataclass
class CheckResult:
    """Outcome of validating one inventory row."""

    row_id: str
    cadence: str
    # "ok" | "fail" | "degraded" | "not_yet_effective" | "error"
    # "degraded" = non-fatal: either a diagnostic row (non_fatal: true, e.g.
    # pipeline_execution success_rate — observability, not a gate) or a present
    # artifact carrying a benign producer status (non_fatal_statuses, e.g.
    # no_recent_sf_run = no upstream data this cycle, not a missing diagnostic).
    # Degraded does NOT count as a failure: no SNS alert, exit 0, CW value 1.0.
    status: str
    detail: str
    effective_date: str
    artifact: str | None = None
    sub_failures: list[str] = field(default_factory=list)


def load_inventory(path: Path | None = None) -> dict:
    """Load and parse the inventory YAML.

    Imports yaml lazily so the rest of the lib stays import-light for
    consumers that don't use this module.
    """
    import yaml

    p = path or INVENTORY_PATH
    with p.open() as fh:
        return yaml.safe_load(fh)


def check_inventory(
    cadence: str,
    *,
    today: date | None = None,
    inventory: dict | None = None,
    s3_client: Any = None,
    cloudwatch_client: Any = None,
) -> list[CheckResult]:
    """Validate every inventory row whose ``cadence`` matches the input.

    The Saturday SF passes ``cadence="weekly"`` to validate weekly +
    daily rows (since daily artifacts from Friday should be readable
    on Saturday). The weekday SF passes ``cadence="daily"`` to
    validate only daily rows.

    Rows with ``effective_date`` > today are returned with
    ``status="not_yet_effective"`` and contribute to neither
    pass-rate calculation.
    """
    today = today or _today_utc()
    inv = inventory or load_inventory()

    rows = list(_filter_rows(inv["inventory"], cadence))
    results: list[CheckResult] = []

    for row in rows:
        results.append(_check_row(row, today, s3_client, cloudwatch_client))

    return results


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _filter_rows(rows: Iterable[dict], cadence: str) -> Iterable[dict]:
    """Pick rows that the given run should validate.

    Saturday (cadence='weekly') validates everything; weekday
    (cadence='daily') validates only daily rows; per-event cadence
    is validated only when explicitly requested.
    """
    if cadence == "weekly":
        wanted = {"weekly", "daily"}
    elif cadence == "daily":
        wanted = {"daily"}
    elif cadence == "per_event":
        wanted = {"per_event"}
    else:
        raise ValueError(f"Unknown cadence: {cadence}")
    for row in rows:
        if row["cadence"] in wanted:
            yield row


def _check_row(
    row: dict,
    today: date,
    s3_client: Any,
    cloudwatch_client: Any,
) -> CheckResult:
    eff = date.fromisoformat(str(row["effective_date"]))
    if today < eff:
        return CheckResult(
            row_id=row["id"],
            cadence=row["cadence"],
            status="not_yet_effective",
            detail=f"effective_date={eff} > today={today}",
            effective_date=str(eff),
        )

    sub: list[str] = []
    artifact_hint: str | None = None
    degraded_detail: str | None = None
    for src in row["sources"]:
        try:
            ok, detail, artifact, status_hint = _check_source(
                src, today, s3_client, cloudwatch_client
            )
        except Exception as exc:  # pragma: no cover — defensive
            ok, detail, artifact, status_hint = (
                False, f"checker error: {exc!r}", None, None
            )
        if artifact and artifact_hint is None:
            artifact_hint = artifact
        if ok:
            return CheckResult(
                row_id=row["id"],
                cadence=row["cadence"],
                status="ok",
                detail=detail,
                effective_date=str(eff),
                artifact=artifact_hint,
            )
        if status_hint == "degraded" and degraded_detail is None:
            degraded_detail = detail
        sub.append(detail)

    # All sources failed. Classify non-fatal degradation vs hard fail:
    #  - row-level ``non_fatal: true`` → diagnostic/observability row demoted
    #    from a gate (Phase 1c: pipeline_execution success_rate).
    #  - any source signalled "degraded" → present artifact carrying a benign
    #    producer status (Phase 1a: e.g. no_recent_sf_run = no upstream data
    #    this cycle, not a missing diagnostic).
    # Either way the cycle isn't "broken" — surface it without failing the gate.
    if row.get("non_fatal") or degraded_detail is not None:
        return CheckResult(
            row_id=row["id"],
            cadence=row["cadence"],
            status="degraded",
            detail=degraded_detail or "; ".join(sub),
            effective_date=str(eff),
            artifact=artifact_hint,
            sub_failures=sub,
        )

    return CheckResult(
        row_id=row["id"],
        cadence=row["cadence"],
        status="fail",
        detail="; ".join(sub),
        effective_date=str(eff),
        artifact=artifact_hint,
        sub_failures=sub,
    )


# ---------------------------------------------------------------------------
# Source-kind dispatchers
# ---------------------------------------------------------------------------


def _check_source(
    src: dict,
    today: date,
    s3_client: Any,
    cloudwatch_client: Any,
) -> tuple[bool, str, str | None, str | None]:
    """Run a source handler, normalized to ``(ok, detail, artifact, status_hint)``.

    Handlers may return a 3-tuple (the common case) or a 4-tuple whose 4th
    element is a ``status_hint`` ("degraded") used to mark a non-fatal
    non-pass. Normalizing here keeps handlers that don't care unchanged.
    """
    kind = src["kind"]
    handler = _SOURCE_HANDLERS.get(kind)
    if handler is None:
        return False, f"unsupported source kind: {kind}", None, None
    result = handler(src, today, s3_client, cloudwatch_client)
    if len(result) == 4:
        return result
    ok, detail, artifact = result
    return ok, detail, artifact, None


def _resolve_key(src: dict, today: date) -> tuple[str, str]:
    """Return (key, age_window_label).

    Two patterns:
      key            — fixed S3 key, no date templating
      key_pattern    — contains {date}; checker walks back N days to
                       find the most recent matching object
    """
    if "key" in src:
        return src["key"], "fixed"
    if "key_pattern" not in src:
        raise ValueError(f"source missing key/key_pattern: {src}")
    return src["key_pattern"], "templated"


def _walk_back(
    pattern: str,
    today: date,
    max_age_days: int,
    exists: Callable[[str], bool],
) -> tuple[str | None, int]:
    """Walk back day-by-day, return first key whose object exists.

    Returns (key, age_in_days) or (None, age_at_limit).
    """
    for i in range(max_age_days + 1):
        d = today - timedelta(days=i)
        key = pattern.format(date=d.isoformat())
        if exists(key):
            return key, i
    return None, max_age_days + 1


def _s3_head(s3_client: Any, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def _s3_get_bytes(s3_client: Any, bucket: str, key: str) -> bytes:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def _s3_age_days(s3_client: Any, bucket: str, key: str) -> int | None:
    try:
        resp = s3_client.head_object(Bucket=bucket, Key=key)
        modified = resp["LastModified"]
        return (datetime.now(timezone.utc) - modified).days
    except Exception:
        return None


def _resolve_and_age(
    src: dict, today: date, s3_client: Any
) -> tuple[str | None, int | None, str]:
    """Locate the artifact key + report its age. Common to all S3 kinds."""
    bucket = src.get("bucket", DEFAULT_BUCKET)
    key, mode = _resolve_key(src, today)
    max_age = src.get("max_age_days", 8)
    if mode == "fixed":
        age = _s3_age_days(s3_client, bucket, key)
        if age is None:
            return None, None, f"missing s3://{bucket}/{key}"
        if age > max_age:
            return key, age, (
                f"stale s3://{bucket}/{key} (age={age}d > {max_age}d)"
            )
        return key, age, "ok"
    # templated
    resolved_key, age = _walk_back(
        key,
        today,
        max_age,
        lambda k: _s3_head(s3_client, bucket, k),
    )
    if resolved_key is None:
        return None, None, (
            f"no object matching s3://{bucket}/{key} within {max_age}d"
        )
    return resolved_key, age, "ok"


def _check_s3_json(
    src: dict, today: date, s3_client: Any, _cw: Any
) -> tuple[bool, str, str | None] | tuple[bool, str, str | None, str | None]:
    bucket = src.get("bucket", DEFAULT_BUCKET)
    key, age, status = _resolve_and_age(src, today, s3_client)
    if key is None:
        # Companion fallback for "ok_if_companion_present"
        if src.get("treat_absent_as") == "ok_if_companion_present":
            comp_pattern = src.get("companion_key_pattern")
            if comp_pattern:
                comp_key, _ = _walk_back(
                    comp_pattern,
                    today,
                    src.get("max_age_days", 8),
                    lambda k: _s3_head(s3_client, bucket, k),
                )
                if comp_key:
                    return True, (
                        f"primary absent, companion present: "
                        f"s3://{bucket}/{comp_key}"
                    ), comp_key
        return False, status, None
    if status != "ok":
        return False, status, key

    body = _s3_get_bytes(s3_client, bucket, key)
    try:
        payload = json.loads(body)
    except Exception as exc:
        return False, f"json parse error on s3://{bucket}/{key}: {exc!r}", key

    # Phase 1a: a present artifact carrying a benign producer status is a
    # legitimate cycle state (no upstream data), NOT a missing diagnostic and
    # NOT a hard failure. Short-circuit BEFORE evaluating asserts so we don't
    # report a misleading "coverage 0% < 99". Always-emit (producer side) is
    # what makes this distinguishable from absence.
    non_fatal_statuses = src.get("non_fatal_statuses", [])
    prod_status = payload.get("status") if isinstance(payload, dict) else None
    if non_fatal_statuses and prod_status in non_fatal_statuses:
        return (
            False,
            f"degraded: producer status='{prod_status}' — no upstream data "
            f"this cycle (s3://{bucket}/{key})",
            key,
            "degraded",
        )

    failures: list[str] = []
    for required in src.get("assert_keys_present", []):
        if required not in payload:
            failures.append(f"missing key '{required}'")
    for assertion in src.get("assert", []):
        ok, detail = _eval_path_assertion(payload, assertion)
        if not ok:
            failures.append(detail)
    if failures:
        return False, "; ".join(failures), key
    return True, f"ok (age={age}d)", key


def _check_s3_csv(
    src: dict, today: date, s3_client: Any, _cw: Any
) -> tuple[bool, str, str | None]:
    bucket = src.get("bucket", DEFAULT_BUCKET)
    key, age, status = _resolve_and_age(src, today, s3_client)
    if key is None or status != "ok":
        return False, status, key

    body = _s3_get_bytes(s3_client, bucket, key)
    try:
        import pandas as pd

        df = pd.read_csv(io.BytesIO(body))
    except Exception as exc:
        return False, f"csv parse error on s3://{bucket}/{key}: {exc!r}", key

    failures: list[str] = []
    for col in src.get("assert_columns_present", []):
        if col not in df.columns:
            failures.append(f"missing column '{col}'")

    rule = src.get("assert_columns_non_null_for_rows_after")
    if rule and not failures:
        date_col = rule["date_column"]
        threshold = date.fromisoformat(str(rule["rows_after"]))
        cols = rule["columns"]
        action_filter = rule.get("action_filter")
        if date_col not in df.columns:
            failures.append(f"missing date_column '{date_col}'")
        else:
            try:
                # Coerce date_column to date for comparison; tolerate
                # both 'YYYY-MM-DD' and ISO timestamps.
                d_col = pd.to_datetime(df[date_col], errors="coerce").dt.date
                mask = d_col > threshold
                sub = df[mask]
                if action_filter:
                    a_col = action_filter["column"]
                    a_val = action_filter["equals"]
                    if a_col in sub.columns:
                        sub = sub[sub[a_col] == a_val]
                if not sub.empty:
                    for col in cols:
                        if col not in sub.columns:
                            failures.append(f"missing column '{col}' for non-null assertion")
                            continue
                        nulls = sub[col].isna().sum()
                        if nulls > 0:
                            failures.append(
                                f"column '{col}' has {int(nulls)} null rows after {threshold}"
                            )
            except Exception as exc:
                failures.append(f"non-null check error: {exc!r}")

    latest = src.get("assert_value_on_latest_row")
    if latest and not failures:
        col = latest["column"]
        if col not in df.columns:
            failures.append(f"missing column '{col}' for latest-row assertion")
        elif df.empty:
            failures.append(f"csv empty — cannot evaluate '{col}' on latest row")
        else:
            val = df[col].iloc[-1]
            ok, detail = _eval_op(val, latest["op"], latest["value"])
            if not ok:
                failures.append(detail)

    if failures:
        return False, "; ".join(failures), key
    return True, f"ok (age={age}d, rows={len(df)})", key


def _check_s3_parquet(
    src: dict, today: date, s3_client: Any, _cw: Any
) -> tuple[bool, str, str | None]:
    bucket = src.get("bucket", DEFAULT_BUCKET)
    key, age, status = _resolve_and_age(src, today, s3_client)
    if key is None or status != "ok":
        return False, status, key

    body = _s3_get_bytes(s3_client, bucket, key)
    try:
        import pandas as pd

        df = pd.read_parquet(io.BytesIO(body))
    except Exception as exc:
        return False, f"parquet parse error on s3://{bucket}/{key}: {exc!r}", key

    failures: list[str] = []
    for col in src.get("assert_columns_present", []):
        if col not in df.columns:
            failures.append(f"missing column '{col}'")
    for col in src.get("assert_column_non_null", []):
        if col not in df.columns:
            failures.append(f"missing column '{col}' for non-null check")
            continue
        nulls = df[col].isna().sum()
        if nulls > 0:
            failures.append(f"column '{col}' has {int(nulls)} null rows")

    if failures:
        return False, "; ".join(failures), key
    return True, f"ok (age={age}d, rows={len(df)})", key


def _check_sqlite_via_s3(
    src: dict, today: date, s3_client: Any, _cw: Any
) -> tuple[bool, str, str | None]:
    bucket = src.get("bucket", DEFAULT_BUCKET)
    key, age, status = _resolve_and_age(src, today, s3_client)
    if key is None or status != "ok":
        return False, status, key

    table = src["table"]
    body = _s3_get_bytes(s3_client, bucket, key)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as fh:
        fh.write(body)
        db_path = fh.name
    try:
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.execute(f"PRAGMA table_info({table})")
            cols = {row[1] for row in cur.fetchall()}
        finally:
            conn.close()
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass

    if not cols:
        return False, f"table '{table}' missing in s3://{bucket}/{key}", key

    failures = [
        f"missing column '{c}' in table '{table}'"
        for c in src.get("assert_columns_present", [])
        if c not in cols
    ]
    if failures:
        return False, "; ".join(failures), key
    return True, f"ok (age={age}d, table='{table}')", key


def _check_cloudwatch(
    src: dict, today: date, _s3: Any, cloudwatch_client: Any
) -> tuple[bool, str, str | None]:
    if cloudwatch_client is None:
        import boto3

        cloudwatch_client = boto3.client("cloudwatch", region_name="us-east-1")

    namespace = src["namespace"]
    metric = src["metric"]
    window_days = src.get("window_days", 7)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=window_days)
    # AWS GetMetricStatistics requires Period to be a multiple of 60. Aim
    # for ~100 datapoints across the window, then round down to a multiple
    # of 60 (with a 60s floor for the smallest windows).
    raw_period = max(60, window_days * 86400 // 100)
    period = max(60, (raw_period // 60) * 60)

    artifact = f"cw://{namespace}/{metric}"
    assertion = src.get("assert", {})
    op = assertion.get("op")

    if op == "success_rate_pct_gte":
        return _check_cw_success_rate(
            cloudwatch_client, src, start, end, period, assertion["value"]
        )
    if op == "datapoints_gte":
        return _check_cw_datapoints(
            cloudwatch_client, namespace, metric, start, end, period,
            assertion["value"], artifact,
        )
    return False, f"unsupported cloudwatch assert op: {op}", artifact


def _check_cw_success_rate(
    cw: Any, src: dict, start: datetime, end: datetime, period: int, threshold: float
) -> tuple[bool, str, str | None]:
    namespace = src["namespace"]
    dim_field = src.get("dimensions", {})
    arns = list(dim_field.get("StateMachineArn", [])) or [None]

    failures: list[str] = []
    for arn in arns:
        kw = {
            "Namespace": namespace,
            "Period": period,
            "Statistics": ["Sum"],
            "StartTime": start,
            "EndTime": end,
        }
        if arn:
            kw["Dimensions"] = [{"Name": "StateMachineArn", "Value": arn}]
        succ = cw.get_metric_statistics(MetricName="ExecutionsSucceeded", **kw)
        fail = cw.get_metric_statistics(MetricName="ExecutionsFailed", **kw)
        s = sum(p["Sum"] for p in succ.get("Datapoints", []))
        f = sum(p["Sum"] for p in fail.get("Datapoints", []))
        denom = s + f
        if denom == 0:
            failures.append(f"{arn or 'aggregate'}: no datapoints in window")
            continue
        pct = 100.0 * s / denom
        if pct < threshold:
            failures.append(
                f"{arn or 'aggregate'}: success_rate={pct:.2f}% < {threshold}%"
            )
    if failures:
        return False, "; ".join(failures), f"cw://{namespace}/ExecutionsSucceeded"
    return True, "ok", f"cw://{namespace}/ExecutionsSucceeded"


def _check_cw_datapoints(
    cw: Any,
    namespace: str,
    metric: str,
    start: datetime,
    end: datetime,
    period: int,
    threshold: int,
    artifact: str,
) -> tuple[bool, str, str | None]:
    resp = cw.get_metric_statistics(
        Namespace=namespace,
        MetricName=metric,
        Period=period,
        Statistics=["SampleCount"],
        StartTime=start,
        EndTime=end,
    )
    n = sum(p["SampleCount"] for p in resp.get("Datapoints", []))
    if n < threshold:
        return False, (
            f"only {int(n)} datapoints in {namespace}/{metric} (need ≥ {threshold})"
        ), artifact
    return True, f"ok (n={int(n)})", artifact


_SOURCE_HANDLERS: dict[str, Callable] = {
    "s3_json": _check_s3_json,
    "s3_csv": _check_s3_csv,
    "s3_parquet": _check_s3_parquet,
    "sqlite_via_s3": _check_sqlite_via_s3,
    "cloudwatch": _check_cloudwatch,
}


# ---------------------------------------------------------------------------
# Assertion primitives
# ---------------------------------------------------------------------------


def _eval_path_assertion(payload: Any, assertion: dict) -> tuple[bool, str]:
    path = assertion["path"]
    cur: Any = payload
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return False, f"path '{path}' not found"
    return _eval_op(cur, assertion["op"], assertion["value"], path=path)


def _eval_op(value: Any, op: str, target: Any, path: str | None = None) -> tuple[bool, str]:
    label = path or "value"
    try:
        v = float(value) if not isinstance(value, bool) else value
        t = float(target) if not isinstance(target, bool) else target
    except (TypeError, ValueError):
        return False, f"{label}={value!r} not comparable to {target!r}"
    if op == "gte":
        return (v >= t, f"{label}={v} {'>=' if v >= t else '<'} {t}")
    if op == "gt":
        return (v > t, f"{label}={v} {'>' if v > t else '<='} {t}")
    if op == "lte":
        return (v <= t, f"{label}={v} {'<=' if v <= t else '>'} {t}")
    if op == "lt":
        return (v < t, f"{label}={v} {'<' if v < t else '>='} {t}")
    if op == "eq":
        return (v == t, f"{label}={v} {'==' if v == t else '!='} {t}")
    return False, f"unsupported op '{op}'"


# ---------------------------------------------------------------------------
# CLI + side effects
# ---------------------------------------------------------------------------


def emit_cloudwatch_metrics(results: list[CheckResult], cloudwatch_client: Any = None) -> None:
    """Publish per-row + aggregate metrics to ``AlphaEngine/Substrate``."""
    if cloudwatch_client is None:
        import boto3

        cloudwatch_client = boto3.client("cloudwatch", region_name="us-east-1")

    metric_data = []
    for r in results:
        # 1 = ok / not_yet_effective / degraded (all non-failing), 0 = fail.
        # Degraded is non-fatal so it must not trip the SubstrateRowOK alarm.
        value = 1.0 if r.status in ("ok", "not_yet_effective", "degraded") else 0.0
        metric_data.append({
            "MetricName": "SubstrateRowOK",
            "Dimensions": [{"Name": "RowID", "Value": r.row_id}],
            "Value": value,
            "Unit": "Count",
        })
    n_ok = sum(1 for r in results if r.status == "ok")
    n_fail = sum(1 for r in results if r.status == "fail")
    n_degraded = sum(1 for r in results if r.status == "degraded")
    n_pending = sum(1 for r in results if r.status == "not_yet_effective")
    metric_data.extend([
        {"MetricName": "SubstrateChecksOK", "Value": float(n_ok), "Unit": "Count"},
        {"MetricName": "SubstrateChecksFailed", "Value": float(n_fail), "Unit": "Count"},
        {"MetricName": "SubstrateChecksDegraded", "Value": float(n_degraded), "Unit": "Count"},
        {"MetricName": "SubstrateChecksPending", "Value": float(n_pending), "Unit": "Count"},
    ])

    for i in range(0, len(metric_data), 20):
        cloudwatch_client.put_metric_data(
            Namespace=DEFAULT_NAMESPACE_OUT,
            MetricData=metric_data[i : i + 20],
        )


def format_report(results: list[CheckResult]) -> str:
    lines = ["Substrate Health Report", "=" * 50]
    n_ok = sum(1 for r in results if r.status == "ok")
    n_fail = sum(1 for r in results if r.status == "fail")
    n_degraded = sum(1 for r in results if r.status == "degraded")
    n_pending = sum(1 for r in results if r.status == "not_yet_effective")
    n_total = len(results)
    # Gating denominator excludes pending (not yet effective) AND degraded
    # (non-fatal, can't be scored pass/fail this cycle).
    n_gating = n_total - n_pending - n_degraded
    pct = (100.0 * n_ok / n_gating) if n_gating > 0 else 100.0
    lines.append(
        f"OK: {n_ok}  Failed: {n_fail}  Degraded: {n_degraded}  "
        f"Pending: {n_pending}  ({pct:.1f}% of gating rows passing)"
    )
    lines.append("")
    icon = {
        "ok": "OK ", "fail": "FAIL", "degraded": "DEGR",
        "not_yet_effective": "PEND", "error": "ERR ",
    }
    for r in results:
        lines.append(f"  [{icon.get(r.status, '?')}] {r.row_id:30s} {r.detail}")
    failures = [r for r in results if r.status == "fail"]
    if failures:
        lines.append("")
        lines.append("ACTIONS NEEDED:")
        for r in failures:
            lines.append(f"  - {r.row_id}: {r.detail}")
    degraded = [r for r in results if r.status == "degraded"]
    if degraded:
        lines.append("")
        lines.append("DEGRADED (non-fatal — observability, no action gate):")
        for r in degraded:
            lines.append(f"  - {r.row_id}: {r.detail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cadence",
        choices=["daily", "weekly", "per_event"],
        required=True,
        help="Run weekly (Saturday SF) or daily (weekday SF) check.",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--alert", action="store_true", help="Publish SNS on failure.")
    parser.add_argument("--no-emit", action="store_true", help="Skip CloudWatch emission.")
    parser.add_argument(
        "--inventory", type=Path, default=None, help="Override inventory path."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    inv = load_inventory(args.inventory) if args.inventory else None

    import boto3

    s3 = boto3.client("s3")
    cw = boto3.client("cloudwatch", region_name="us-east-1")

    results = check_inventory(
        args.cadence, inventory=inv, s3_client=s3, cloudwatch_client=cw
    )

    if args.json:
        print(json.dumps([r.__dict__ for r in results], indent=2, default=str))
    else:
        print(format_report(results))

    if not args.no_emit:
        try:
            emit_cloudwatch_metrics(results, cw)
        except Exception as exc:  # pragma: no cover — non-fatal
            log.warning("CloudWatch emission failed: %s", exc)

    failures = [r for r in results if r.status == "fail"]
    if failures and args.alert:
        try:
            sns = boto3.client("sns", region_name="us-east-1")
            topic = os.environ.get("SNS_TOPIC_ARN", DEFAULT_SNS_TOPIC)
            sns.publish(
                TopicArn=topic,
                Subject=(
                    f"Alpha Engine — Substrate Health "
                    f"({args.cadence}): {len(failures)} row(s) failed"
                ),
                Message=format_report(results),
            )
        except Exception as exc:  # pragma: no cover — non-fatal
            log.warning("SNS publish failed: %s", exc)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
