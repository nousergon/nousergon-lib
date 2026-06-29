"""Tests for the transparency substrate health checker."""

from __future__ import annotations

import io
import json
import sqlite3
import textwrap
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from nousergon_lib import transparency
from nousergon_lib.transparency import (
    CheckResult,
    INVENTORY_PATH,
    check_inventory,
    emit_cloudwatch_metrics,
    format_report,
    load_inventory,
)


# ---------------------------------------------------------------------------
# Stub clients
# ---------------------------------------------------------------------------


class StubS3:
    """Minimal in-memory stand-in for boto3 s3 client used by the checker."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, datetime]] = {}

    def put(self, bucket: str, key: str, body: bytes, age_days: int = 0) -> None:
        ts = datetime.now(timezone.utc).replace(microsecond=0)
        if age_days:
            ts = ts.replace(day=max(1, ts.day))  # placeholder; we replace below
            from datetime import timedelta

            ts = ts - timedelta(days=age_days)
        self.objects[(bucket, key)] = (body, ts)

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        if (Bucket, Key) not in self.objects:
            raise KeyError(f"NoSuchKey: {Bucket}/{Key}")
        _, ts = self.objects[(Bucket, Key)]
        return {"LastModified": ts}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        body, _ = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(body)}


class StubCloudWatch:
    def __init__(self) -> None:
        self.put_calls: list[dict] = []
        self.stats: dict[tuple[str, str, tuple[tuple[str, str], ...]], list[dict]] = {}
        self.get_calls: list[dict] = []

    def set_stats(
        self,
        *,
        namespace: str,
        metric: str,
        dimensions: list[tuple[str, str]] | None = None,
        datapoints: list[dict],
    ) -> None:
        key = (namespace, metric, tuple(dimensions or ()))
        self.stats[key] = datapoints

    def get_metric_statistics(self, **kw):
        self.get_calls.append(kw)
        ns = kw["Namespace"]
        m = kw["MetricName"]
        dims = tuple(
            (d["Name"], d["Value"]) for d in kw.get("Dimensions", [])
        )
        return {"Datapoints": self.stats.get((ns, m, dims), [])}

    def put_metric_data(self, *, Namespace: str, MetricData: list[dict]) -> None:
        self.put_calls.append({"Namespace": Namespace, "MetricData": MetricData})


# ---------------------------------------------------------------------------
# Inventory shape
# ---------------------------------------------------------------------------


def test_inventory_yaml_loads_and_is_well_formed():
    inv = load_inventory()
    assert inv["version"] == 1
    assert isinstance(inv["inventory"], list)
    assert len(inv["inventory"]) == 10
    ids = {row["id"] for row in inv["inventory"]}
    expected = {
        "pipeline_execution",
        "agent_decisions",
        "predictor_decisions",
        "trade_execution_lineage",
        "risk_events",
        "pnl_attribution",
        "config_changes",
        "data_quality",
        "agent_quality",
        "cost_telemetry",
    }
    assert ids == expected


def test_every_row_has_required_fields():
    inv = load_inventory()
    for row in inv["inventory"]:
        assert "id" in row
        assert "cadence" in row
        assert row["cadence"] in {"daily", "weekly", "per_event"}
        assert "effective_date" in row
        # parseable
        date.fromisoformat(str(row["effective_date"]))
        assert "description" in row
        assert "sources" in row and len(row["sources"]) >= 1
        for src in row["sources"]:
            assert "kind" in src


def test_inventory_yaml_is_packaged():
    """The YAML must live next to the module so the wheel ships it."""
    assert INVENTORY_PATH.is_file()
    assert INVENTORY_PATH.parent.name == "nousergon_lib"


def test_data_quality_row_matches_actual_parquet_columns():
    """Regression for the 2026-05-06 first-substrate-exercise drift —
    inventory expected ``ticker`` (which is the parquet *index*, not a
    column) and lowercase ``close`` (actual column is ``Close``). Pin the
    fix so the row's column-presence assertions match the
    daily_closes producer contract: parquet has columns ``date``,
    ``Open/High/Low/Close``, ``Adj_Close``, ``Volume``, ``VWAP``,
    ``source``; ``ticker`` is the index name. The producer contract has
    been stable since the staging-prefix migration (data #112,
    2026-04-29) and shouldn't move; future drift on this side
    indicates an inventory bug, not a producer bug."""
    inv = load_inventory()
    row = next(r for r in inv["inventory"] if r["id"] == "data_quality")
    src = row["sources"][0]
    assert src["kind"] == "s3_parquet"
    cols = src["assert_columns_present"]
    assert "Close" in cols, (
        "data_quality must check the canonical capital-C ``Close`` column "
        "the daily_closes parquet actually emits"
    )
    assert "close" not in cols, (
        "lowercase ``close`` is not a column on daily_closes parquet — "
        "regression of the 2026-05-06 inventory drift"
    )
    assert "ticker" not in cols, (
        "``ticker`` is the parquet *index*, not a column — "
        "regression of the 2026-05-06 inventory drift"
    )
    assert "source" in cols  # core attribution column from data #159


def test_cost_telemetry_row_pins_producer_contract():
    """Cost telemetry was added 2026-05-10 as a substrate row so the
    backtester's cost_report.detect_anomaly can read the registry's
    effective_date as the anomaly-baseline floor (rather than infer it
    from S3 listing). Pin the row's producer contract so a producer
    schema drift surfaces at the substrate check, not silently in
    downstream baseline classification.
    """
    inv = load_inventory()
    row = next(
        (r for r in inv["inventory"] if r["id"] == "cost_telemetry"), None,
    )
    assert row is not None, (
        "cost_telemetry row is required; consumers (backtester "
        "detect_anomaly) read effective_date from this registry"
    )
    assert row["cadence"] == "weekly"
    # Weekly Freshness SF cron produces the parquet — first capture was 5/2.
    assert str(row["effective_date"]) == "2026-05-02"
    src = row["sources"][0]
    assert src["kind"] == "s3_parquet"
    assert src["bucket"] == "alpha-engine-research"
    assert src["key_pattern"] == (
        "decision_artifacts/_cost/{date}/cost.parquet"
    )
    cols = src["assert_columns_present"]
    # Producer contract surface from llm_cost_tracker._PER_CALL_SCHEMA
    # — a missing column on this list means the row was edited but the
    # producer contract drifted.
    for required in ("schema_version", "run_id", "agent_id",
                     "model_name", "cost_usd"):
        assert required in cols, f"cost_telemetry must check {required!r}"
    assert "cost_usd" in src["assert_column_non_null"]


def test_pipeline_execution_row_uses_full_state_machine_arns():
    """Regression for the 2026-05-08 drift surfaced by the lib v0.5.3
    Period fix: AWS/States dimension Value is the full ARN, not the
    short SF name. GetMetricStatistics filters by exact dimension-value
    match, so short names returned zero datapoints and the row reported
    ``no datapoints in window`` for all three pipelines despite the SFs
    succeeding."""
    inv = load_inventory()
    row = next(r for r in inv["inventory"] if r["id"] == "pipeline_execution")
    src = row["sources"][0]
    assert src["kind"] == "cloudwatch"
    arns = src["dimensions"]["StateMachineArn"]
    assert len(arns) == 3
    expected_suffixes = {
        "ne-weekly-freshness-pipeline",
        "ne-preopen-trading-pipeline",
        "ne-postclose-trading-pipeline",
    }
    for arn in arns:
        assert arn.startswith("arn:aws:states:"), (
            f"StateMachineArn must be a full ARN, got {arn!r} — short SF "
            "names return zero datapoints from GetMetricStatistics"
        )
        assert ":stateMachine:" in arn
    suffixes = {arn.rsplit(":", 1)[-1] for arn in arns}
    assert suffixes == expected_suffixes


# ---------------------------------------------------------------------------
# check_inventory: cadence filtering + effective_date gating
# ---------------------------------------------------------------------------


def _mini_inventory() -> dict:
    return {
        "version": 1,
        "inventory": [
            {
                "id": "weekly_row",
                "cadence": "weekly",
                "effective_date": "2026-01-01",
                "description": "test weekly row",
                "sources": [
                    {
                        "kind": "s3_json",
                        "bucket": "b",
                        "key": "weekly.json",
                        "max_age_days": 8,
                        "assert_keys_present": ["foo"],
                    }
                ],
            },
            {
                "id": "daily_row",
                "cadence": "daily",
                "effective_date": "2026-01-01",
                "description": "test daily row",
                "sources": [
                    {
                        "kind": "s3_json",
                        "bucket": "b",
                        "key": "daily.json",
                        "max_age_days": 4,
                        "assert_keys_present": ["bar"],
                    }
                ],
            },
            {
                "id": "future_row",
                "cadence": "daily",
                "effective_date": "2099-01-01",
                "description": "row not yet effective",
                "sources": [
                    {
                        "kind": "s3_json",
                        "bucket": "b",
                        "key": "future.json",
                    }
                ],
            },
        ],
    }


def test_weekly_cadence_includes_daily_rows():
    """Sat SF check sweeps everything — weekly + daily rows."""
    inv = _mini_inventory()
    s3 = StubS3()
    s3.put("b", "weekly.json", json.dumps({"foo": 1}).encode())
    s3.put("b", "daily.json", json.dumps({"bar": 2}).encode())

    res = check_inventory(
        "weekly", today=date(2026, 6, 1), inventory=inv, s3_client=s3
    )

    by_id = {r.row_id: r for r in res}
    assert by_id["weekly_row"].status == "ok"
    assert by_id["daily_row"].status == "ok"
    assert by_id["future_row"].status == "not_yet_effective"


def test_daily_cadence_excludes_weekly_rows():
    """Pre-open Trading SF check skips weekly rows."""
    inv = _mini_inventory()
    s3 = StubS3()
    s3.put("b", "daily.json", json.dumps({"bar": 2}).encode())

    res = check_inventory(
        "daily", today=date(2026, 6, 1), inventory=inv, s3_client=s3
    )
    ids = {r.row_id for r in res}
    assert "weekly_row" not in ids
    assert "daily_row" in ids


def test_future_effective_date_returns_not_yet_effective():
    inv = _mini_inventory()
    s3 = StubS3()
    res = check_inventory(
        "daily", today=date(2026, 6, 1), inventory=inv, s3_client=s3
    )
    by_id = {r.row_id: r for r in res}
    assert by_id["future_row"].status == "not_yet_effective"


# ---------------------------------------------------------------------------
# s3_json source kind
# ---------------------------------------------------------------------------


def test_s3_json_missing_artifact_fails():
    inv = _mini_inventory()
    s3 = StubS3()  # nothing uploaded
    res = check_inventory(
        "daily", today=date(2026, 6, 1), inventory=inv, s3_client=s3
    )
    daily = next(r for r in res if r.row_id == "daily_row")
    assert daily.status == "fail"
    assert "missing" in daily.detail


def test_s3_json_assert_keys_present_failure():
    inv = _mini_inventory()
    s3 = StubS3()
    s3.put("b", "daily.json", json.dumps({"baz": 1}).encode())  # wrong key
    res = check_inventory(
        "daily", today=date(2026, 6, 1), inventory=inv, s3_client=s3
    )
    daily = next(r for r in res if r.row_id == "daily_row")
    assert daily.status == "fail"
    assert "missing key 'bar'" in daily.detail


def test_s3_json_path_assertion_gte_passes():
    inv = {
        "version": 1,
        "inventory": [
            {
                "id": "row",
                "cadence": "daily",
                "effective_date": "2026-01-01",
                "description": "x",
                "sources": [
                    {
                        "kind": "s3_json",
                        "bucket": "b",
                        "key": "k.json",
                        "max_age_days": 4,
                        "assert": [{"path": "coverage_pct", "op": "gte", "value": 99}],
                    }
                ],
            }
        ],
    }
    s3 = StubS3()
    s3.put("b", "k.json", json.dumps({"coverage_pct": 99.5}).encode())
    res = check_inventory(
        "daily", today=date(2026, 6, 1), inventory=inv, s3_client=s3
    )
    assert res[0].status == "ok"


def test_s3_json_path_assertion_gte_fails_when_below_threshold():
    inv = {
        "version": 1,
        "inventory": [
            {
                "id": "row",
                "cadence": "daily",
                "effective_date": "2026-01-01",
                "description": "x",
                "sources": [
                    {
                        "kind": "s3_json",
                        "bucket": "b",
                        "key": "k.json",
                        "max_age_days": 4,
                        "assert": [{"path": "coverage_pct", "op": "gte", "value": 99}],
                    }
                ],
            }
        ],
    }
    s3 = StubS3()
    s3.put("b", "k.json", json.dumps({"coverage_pct": 95.0}).encode())
    res = check_inventory(
        "daily", today=date(2026, 6, 1), inventory=inv, s3_client=s3
    )
    assert res[0].status == "fail"
    assert "95" in res[0].detail


def test_s3_json_walks_back_for_templated_key():
    inv = {
        "version": 1,
        "inventory": [
            {
                "id": "row",
                "cadence": "weekly",
                "effective_date": "2026-01-01",
                "description": "x",
                "sources": [
                    {
                        "kind": "s3_json",
                        "bucket": "b",
                        "key_pattern": "backtest/{date}/decision_capture_coverage.json",
                        "max_age_days": 8,
                        "assert_keys_present": ["coverage_pct"],
                    }
                ],
            }
        ],
    }
    s3 = StubS3()
    # Only 3-day-old object exists; checker should walk back and find it.
    s3.put(
        "b",
        "backtest/2026-05-30/decision_capture_coverage.json",
        json.dumps({"coverage_pct": 99.5}).encode(),
    )
    res = check_inventory(
        "weekly", today=date(2026, 6, 2), inventory=inv, s3_client=s3
    )
    assert res[0].status == "ok"
    assert "2026-05-30" in (res[0].artifact or "")


# ---------------------------------------------------------------------------
# s3_csv source kind
# ---------------------------------------------------------------------------


def test_s3_csv_missing_columns_fails():
    inv = {
        "version": 1,
        "inventory": [
            {
                "id": "row",
                "cadence": "daily",
                "effective_date": "2026-01-01",
                "description": "x",
                "sources": [
                    {
                        "kind": "s3_csv",
                        "bucket": "b",
                        "key": "trades.csv",
                        "max_age_days": 4,
                        "assert_columns_present": ["signal_date", "prediction_date"],
                    }
                ],
            }
        ],
    }
    s3 = StubS3()
    csv = b"date,ticker\n2026-06-01,AAPL\n"
    s3.put("b", "trades.csv", csv)
    res = check_inventory(
        "daily", today=date(2026, 6, 1), inventory=inv, s3_client=s3
    )
    assert res[0].status == "fail"
    assert "signal_date" in res[0].detail


def test_s3_csv_non_null_for_rows_after_passes():
    inv = {
        "version": 1,
        "inventory": [
            {
                "id": "row",
                "cadence": "daily",
                "effective_date": "2026-01-01",
                "description": "x",
                "sources": [
                    {
                        "kind": "s3_csv",
                        "bucket": "b",
                        "key": "trades.csv",
                        "max_age_days": 4,
                        "assert_columns_present": ["signal_date", "prediction_date"],
                        "assert_columns_non_null_for_rows_after": {
                            "date_column": "date",
                            "rows_after": "2026-05-07",
                            "action_filter": {"column": "action", "equals": "BUY"},
                            "columns": ["signal_date", "prediction_date"],
                        },
                    }
                ],
            }
        ],
    }
    s3 = StubS3()
    csv = textwrap.dedent(
        """\
        date,action,ticker,signal_date,prediction_date
        2026-04-01,BUY,AAPL,,
        2026-05-08,BUY,NVDA,2026-05-08,2026-05-08
        2026-05-09,SELL,AAPL,,
        """
    ).encode()
    s3.put("b", "trades.csv", csv)
    res = check_inventory(
        "daily", today=date(2026, 6, 1), inventory=inv, s3_client=s3
    )
    # Pre-2026-05-07 nulls are excluded; SELL rows are filtered out;
    # only the BUY row on 2026-05-08 is checked.
    assert res[0].status == "ok"


def test_s3_csv_non_null_fails_when_buy_after_threshold_has_null():
    inv = {
        "version": 1,
        "inventory": [
            {
                "id": "row",
                "cadence": "daily",
                "effective_date": "2026-01-01",
                "description": "x",
                "sources": [
                    {
                        "kind": "s3_csv",
                        "bucket": "b",
                        "key": "trades.csv",
                        "max_age_days": 4,
                        "assert_columns_present": ["signal_date"],
                        "assert_columns_non_null_for_rows_after": {
                            "date_column": "date",
                            "rows_after": "2026-05-07",
                            "action_filter": {"column": "action", "equals": "BUY"},
                            "columns": ["signal_date"],
                        },
                    }
                ],
            }
        ],
    }
    s3 = StubS3()
    csv = textwrap.dedent(
        """\
        date,action,ticker,signal_date
        2026-05-08,BUY,NVDA,
        """
    ).encode()
    s3.put("b", "trades.csv", csv)
    res = check_inventory(
        "daily", today=date(2026, 6, 1), inventory=inv, s3_client=s3
    )
    assert res[0].status == "fail"
    assert "null" in res[0].detail


def test_s3_csv_assert_value_on_latest_row_lte():
    inv = {
        "version": 1,
        "inventory": [
            {
                "id": "row",
                "cadence": "daily",
                "effective_date": "2026-01-01",
                "description": "x",
                "sources": [
                    {
                        "kind": "s3_csv",
                        "bucket": "b",
                        "key": "eod.csv",
                        "max_age_days": 4,
                        "assert_columns_present": ["unattributed_residual_pct"],
                        "assert_value_on_latest_row": {
                            "column": "unattributed_residual_pct",
                            "op": "lte",
                            "value": 1.0,
                        },
                    }
                ],
            }
        ],
    }
    s3 = StubS3()
    csv_pass = b"date,unattributed_residual_pct\n2026-06-01,0.4\n"
    s3.put("b", "eod.csv", csv_pass)
    assert check_inventory(
        "daily", today=date(2026, 6, 1), inventory=inv, s3_client=s3
    )[0].status == "ok"

    s3 = StubS3()
    csv_fail = b"date,unattributed_residual_pct\n2026-06-01,2.5\n"
    s3.put("b", "eod.csv", csv_fail)
    assert check_inventory(
        "daily", today=date(2026, 6, 1), inventory=inv, s3_client=s3
    )[0].status == "fail"


# ---------------------------------------------------------------------------
# sqlite_via_s3 source kind
# ---------------------------------------------------------------------------


def test_sqlite_via_s3_validates_table_schema(tmp_path: Path):
    db_path = tmp_path / "trades.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE risk_events (rule TEXT, reason TEXT, value REAL, threshold REAL)"
    )
    conn.commit()
    conn.close()

    inv = {
        "version": 1,
        "inventory": [
            {
                "id": "row",
                "cadence": "daily",
                "effective_date": "2026-01-01",
                "description": "x",
                "sources": [
                    {
                        "kind": "sqlite_via_s3",
                        "bucket": "b",
                        "key": "trades.db",
                        "max_age_days": 4,
                        "table": "risk_events",
                        "assert_columns_present": ["rule", "reason", "value", "threshold"],
                    }
                ],
            }
        ],
    }
    s3 = StubS3()
    s3.put("b", "trades.db", db_path.read_bytes())
    res = check_inventory(
        "daily", today=date(2026, 6, 1), inventory=inv, s3_client=s3
    )
    assert res[0].status == "ok"


def test_sqlite_via_s3_missing_column_fails(tmp_path: Path):
    db_path = tmp_path / "trades.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE risk_events (rule TEXT, reason TEXT)")
    conn.commit()
    conn.close()

    inv = {
        "version": 1,
        "inventory": [
            {
                "id": "row",
                "cadence": "daily",
                "effective_date": "2026-01-01",
                "description": "x",
                "sources": [
                    {
                        "kind": "sqlite_via_s3",
                        "bucket": "b",
                        "key": "trades.db",
                        "max_age_days": 4,
                        "table": "risk_events",
                        "assert_columns_present": ["rule", "reason", "value", "threshold"],
                    }
                ],
            }
        ],
    }
    s3 = StubS3()
    s3.put("b", "trades.db", db_path.read_bytes())
    res = check_inventory(
        "daily", today=date(2026, 6, 1), inventory=inv, s3_client=s3
    )
    assert res[0].status == "fail"
    assert "value" in res[0].detail


# ---------------------------------------------------------------------------
# CloudWatch source kind
# ---------------------------------------------------------------------------


def test_cloudwatch_success_rate_passes():
    inv = {
        "version": 1,
        "inventory": [
            {
                "id": "row",
                "cadence": "weekly",
                "effective_date": "2026-01-01",
                "description": "x",
                "sources": [
                    {
                        "kind": "cloudwatch",
                        "namespace": "AWS/States",
                        "metric": "ExecutionsSucceeded",
                        "window_days": 7,
                        "dimensions": {
                            "StateMachineArn": ["ne-weekly-freshness-pipeline"]
                        },
                        "assert": {"op": "success_rate_pct_gte", "value": 99},
                    }
                ],
            }
        ],
    }
    cw = StubCloudWatch()
    cw.set_stats(
        namespace="AWS/States",
        metric="ExecutionsSucceeded",
        dimensions=[("StateMachineArn", "ne-weekly-freshness-pipeline")],
        datapoints=[{"Sum": 100.0}],
    )
    cw.set_stats(
        namespace="AWS/States",
        metric="ExecutionsFailed",
        dimensions=[("StateMachineArn", "ne-weekly-freshness-pipeline")],
        datapoints=[{"Sum": 0.0}],
    )
    res = check_inventory(
        "weekly", today=date(2026, 6, 1), inventory=inv,
        s3_client=StubS3(), cloudwatch_client=cw,
    )
    assert res[0].status == "ok"


@pytest.mark.parametrize("window_days", [1, 7, 14, 28, 90])
def test_cloudwatch_period_is_multiple_of_60(window_days):
    """AWS GetMetricStatistics rejects ``Period`` values that are not multiples
    of 60. Regression for the 2026-05-06 first-substrate-exercise bug where
    ``window_days=7`` produced ``Period=6048`` (=7*86400//100), getting
    rejected with ``InvalidParameterValue: The parameter Period must be a
    multiple of 60``."""
    inv = {
        "version": 1,
        "inventory": [
            {
                "id": "row",
                "cadence": "weekly",
                "effective_date": "2026-01-01",
                "description": "x",
                "sources": [
                    {
                        "kind": "cloudwatch",
                        "namespace": "AWS/States",
                        "metric": "ExecutionsSucceeded",
                        "window_days": window_days,
                        "dimensions": {
                            "StateMachineArn": ["ne-weekly-freshness-pipeline"]
                        },
                        "assert": {"op": "success_rate_pct_gte", "value": 99},
                    }
                ],
            }
        ],
    }
    cw = StubCloudWatch()
    cw.set_stats(
        namespace="AWS/States",
        metric="ExecutionsSucceeded",
        dimensions=[("StateMachineArn", "ne-weekly-freshness-pipeline")],
        datapoints=[{"Sum": 100.0}],
    )
    cw.set_stats(
        namespace="AWS/States",
        metric="ExecutionsFailed",
        dimensions=[("StateMachineArn", "ne-weekly-freshness-pipeline")],
        datapoints=[{"Sum": 0.0}],
    )
    check_inventory(
        "weekly", today=date(2026, 6, 1), inventory=inv,
        s3_client=StubS3(), cloudwatch_client=cw,
    )
    assert cw.get_calls, "expected at least one get_metric_statistics call"
    for call in cw.get_calls:
        period = call["Period"]
        assert period >= 60, f"Period={period} below 60s minimum"
        assert period % 60 == 0, (
            f"Period={period} for window_days={window_days} is not a multiple "
            f"of 60 — AWS GetMetricStatistics will reject this"
        )


@pytest.mark.parametrize("window_days", [1, 7, 14, 28])
def test_cloudwatch_datapoints_period_is_multiple_of_60(window_days):
    """Same multiple-of-60 invariant for the datapoints_gte assertion path
    (used by agent_quality row in the inventory)."""
    inv = {
        "version": 1,
        "inventory": [
            {
                "id": "row",
                "cadence": "weekly",
                "effective_date": "2026-01-01",
                "description": "x",
                "sources": [
                    {
                        "kind": "cloudwatch",
                        "namespace": "AlphaEngine/Eval",
                        "metric": "agent_quality_score_4w_mean_min",
                        "window_days": window_days,
                        "assert": {"op": "datapoints_gte", "value": 1},
                    }
                ],
            }
        ],
    }
    cw = StubCloudWatch()
    cw.set_stats(
        namespace="AlphaEngine/Eval",
        metric="agent_quality_score_4w_mean_min",
        datapoints=[{"SampleCount": 1.0}],
    )
    check_inventory(
        "weekly", today=date(2026, 6, 1), inventory=inv,
        s3_client=StubS3(), cloudwatch_client=cw,
    )
    assert cw.get_calls
    for call in cw.get_calls:
        period = call["Period"]
        assert period >= 60
        assert period % 60 == 0, (
            f"Period={period} for window_days={window_days} is not a multiple "
            f"of 60 (datapoints_gte path)"
        )


def test_cloudwatch_success_rate_fails_below_threshold():
    inv = {
        "version": 1,
        "inventory": [
            {
                "id": "row",
                "cadence": "weekly",
                "effective_date": "2026-01-01",
                "description": "x",
                "sources": [
                    {
                        "kind": "cloudwatch",
                        "namespace": "AWS/States",
                        "metric": "ExecutionsSucceeded",
                        "window_days": 7,
                        "dimensions": {
                            "StateMachineArn": ["ne-weekly-freshness-pipeline"]
                        },
                        "assert": {"op": "success_rate_pct_gte", "value": 99},
                    }
                ],
            }
        ],
    }
    cw = StubCloudWatch()
    cw.set_stats(
        namespace="AWS/States",
        metric="ExecutionsSucceeded",
        dimensions=[("StateMachineArn", "ne-weekly-freshness-pipeline")],
        datapoints=[{"Sum": 90.0}],
    )
    cw.set_stats(
        namespace="AWS/States",
        metric="ExecutionsFailed",
        dimensions=[("StateMachineArn", "ne-weekly-freshness-pipeline")],
        datapoints=[{"Sum": 10.0}],
    )
    res = check_inventory(
        "weekly", today=date(2026, 6, 1), inventory=inv,
        s3_client=StubS3(), cloudwatch_client=cw,
    )
    assert res[0].status == "fail"
    assert "90" in res[0].detail


# ---------------------------------------------------------------------------
# Companion-key fallback (config_changes pattern)
# ---------------------------------------------------------------------------


def test_companion_key_fallback_when_primary_absent():
    inv = {
        "version": 1,
        "inventory": [
            {
                "id": "row",
                "cadence": "weekly",
                "effective_date": "2026-01-01",
                "description": "x",
                "sources": [
                    {
                        "kind": "s3_json",
                        "bucket": "b",
                        "key_pattern": "backtest/{date}/optimizer_changes.json",
                        "max_age_days": 8,
                        "treat_absent_as": "ok_if_companion_present",
                        "companion_key_pattern": "backtest/{date}/metrics.json",
                    }
                ],
            }
        ],
    }
    s3 = StubS3()
    s3.put("b", "backtest/2026-05-30/metrics.json", b"{}")
    res = check_inventory(
        "weekly", today=date(2026, 6, 2), inventory=inv, s3_client=s3
    )
    assert res[0].status == "ok"
    assert "companion present" in res[0].detail


# ---------------------------------------------------------------------------
# CloudWatch metric emission + report formatting
# ---------------------------------------------------------------------------


def test_emit_cloudwatch_metrics_sends_per_row_plus_aggregate():
    cw = StubCloudWatch()
    results = [
        CheckResult(row_id="a", cadence="daily", status="ok", detail="", effective_date="2026-01-01"),
        CheckResult(row_id="b", cadence="daily", status="fail", detail="x", effective_date="2026-01-01"),
        CheckResult(row_id="c", cadence="daily", status="not_yet_effective", detail="", effective_date="2099-01-01"),
    ]
    emit_cloudwatch_metrics(results, cloudwatch_client=cw)

    flat = [m for call in cw.put_calls for m in call["MetricData"]]
    per_row = [m for m in flat if m["MetricName"] == "SubstrateRowOK"]
    assert len(per_row) == 3
    by_id = {m["Dimensions"][0]["Value"]: m["Value"] for m in per_row}
    assert by_id["a"] == 1.0
    assert by_id["b"] == 0.0
    assert by_id["c"] == 1.0  # not_yet_effective counts as healthy

    aggregates = {m["MetricName"]: m["Value"] for m in flat if "Dimensions" not in m}
    assert aggregates["SubstrateChecksOK"] == 1.0
    assert aggregates["SubstrateChecksFailed"] == 1.0
    assert aggregates["SubstrateChecksPending"] == 1.0


def test_format_report_lists_actions_for_failed_rows():
    results = [
        CheckResult(row_id="a", cadence="daily", status="ok", detail="ok", effective_date="2026-01-01"),
        CheckResult(row_id="b", cadence="daily", status="fail", detail="missing column", effective_date="2026-01-01"),
    ]
    out = format_report(results)
    assert "ACTIONS NEEDED" in out
    assert "b: missing column" in out


# ---------------------------------------------------------------------------
# Phase 1 — degraded (non-fatal) status: non_fatal_statuses + non_fatal row
# ---------------------------------------------------------------------------


def _s3_json_row(extra_src=None):
    src = {
        "kind": "s3_json",
        "bucket": "b",
        "key": "k.json",
        "max_age_days": 4,
        "assert": [{"path": "coverage_pct", "op": "gte", "value": 99}],
    }
    if extra_src:
        src.update(extra_src)
    return {
        "version": 1,
        "inventory": [{
            "id": "agent_decisions", "cadence": "daily",
            "effective_date": "2026-01-01", "description": "x",
            "sources": [src],
        }],
    }


def test_s3_json_non_fatal_status_degrades_not_fails():
    """A present artifact carrying a benign producer status (no_recent_sf_run)
    degrades — it is NOT a coverage failure and NOT a missing diagnostic."""
    inv = _s3_json_row({"non_fatal_statuses": ["no_recent_sf_run"]})
    s3 = StubS3()
    s3.put("b", "k.json", json.dumps(
        {"status": "no_recent_sf_run", "coverage_pct": 0.0}).encode())
    res = check_inventory("daily", today=date(2026, 6, 1), inventory=inv, s3_client=s3)
    assert res[0].status == "degraded"
    assert "no_recent_sf_run" in res[0].detail


def test_s3_json_non_fatal_status_ok_passes_normally():
    """status=ok + coverage passing → ok (non_fatal_statuses doesn't interfere)."""
    inv = _s3_json_row({"non_fatal_statuses": ["no_recent_sf_run"]})
    s3 = StubS3()
    s3.put("b", "k.json", json.dumps(
        {"status": "ok", "coverage_pct": 100.0}).encode())
    res = check_inventory("daily", today=date(2026, 6, 1), inventory=inv, s3_client=s3)
    assert res[0].status == "ok"


def test_s3_json_status_not_in_non_fatal_set_still_fails():
    """A failing status NOT in non_fatal_statuses (here: low coverage) still
    fails — only the listed benign statuses degrade."""
    inv = _s3_json_row({"non_fatal_statuses": ["no_recent_sf_run"]})
    s3 = StubS3()
    s3.put("b", "k.json", json.dumps(
        {"status": "ok", "coverage_pct": 50.0}).encode())
    res = check_inventory("daily", today=date(2026, 6, 1), inventory=inv, s3_client=s3)
    assert res[0].status == "fail"


def test_missing_artifact_still_fails_even_with_non_fatal_statuses():
    """Absence is a real failure: always-emit (producer) means a missing
    object = diagnostic never ran, distinct from a benign present status."""
    inv = _s3_json_row({"non_fatal_statuses": ["no_recent_sf_run"]})
    s3 = StubS3()  # nothing put
    res = check_inventory("daily", today=date(2026, 6, 1), inventory=inv, s3_client=s3)
    assert res[0].status == "fail"


def test_row_non_fatal_degrades_on_source_failure():
    """A row marked non_fatal: true degrades instead of failing when its
    source fails (Phase 1c: pipeline_execution success_rate is a diagnostic)."""
    inv = _s3_json_row()
    inv["inventory"][0]["non_fatal"] = True
    inv["inventory"][0]["id"] = "pipeline_execution"
    s3 = StubS3()
    s3.put("b", "k.json", json.dumps({"coverage_pct": 10.0}).encode())
    res = check_inventory("daily", today=date(2026, 6, 1), inventory=inv, s3_client=s3)
    assert res[0].status == "degraded"


def test_format_report_separates_degraded_from_actions():
    from nousergon_lib.transparency import CheckResult, format_report
    results = [
        CheckResult("ok_row", "daily", "ok", "fine", "2026-01-01"),
        CheckResult("bad_row", "daily", "fail", "broke", "2026-01-01"),
        CheckResult("degr_row", "daily", "degraded", "no upstream data", "2026-01-01"),
    ]
    report = format_report(results)
    assert "Degraded: 1" in report
    assert "[DEGR] degr_row" in report
    # degraded must NOT appear under ACTIONS NEEDED
    actions = report.split("ACTIONS NEEDED:")[1].split("DEGRADED")[0]
    assert "degr_row" not in actions
    assert "bad_row" in actions
    assert "DEGRADED (non-fatal" in report


def test_real_inventory_pipeline_execution_is_non_fatal():
    inv = load_inventory()
    row = next(r for r in inv["inventory"] if r["id"] == "pipeline_execution")
    assert row.get("non_fatal") is True


def test_real_inventory_agent_decisions_degrades_on_no_recent_sf_run():
    inv = load_inventory()
    row = next(r for r in inv["inventory"] if r["id"] == "agent_decisions")
    src = next(s for s in row["sources"] if s["kind"] == "s3_json")
    assert "no_recent_sf_run" in src.get("non_fatal_statuses", [])
