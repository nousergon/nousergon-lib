"""Tests for ``nousergon_lib.eval_artifacts``.

Validates the canonical eval-style artifact partition convention:
- ``new_eval_run_id`` returns a YYMMDDHHMM string from a UTC moment
- ``eval_artifact_key`` formats {prefix}/{run_id}.json (default) or
  {prefix}/{run_id}_{basename} (named). Flat layout — the YYMMDDHHMM
  run_id encodes the date itself, so no date sub-partition is needed.
- ``eval_latest_key`` returns {prefix}/latest.json
- Trailing/leading slashes are normalized away so callers don't have
  to think about prefix shape
- Both helpers compose with each other and with ``now_dual`` to produce
  fully-canonical paths in one call site
"""
from __future__ import annotations

import io as _io
import json as _json
from datetime import datetime, timezone
from unittest.mock import MagicMock as _MagicMock

from nousergon_lib.eval_artifacts import (
    EVAL_LATEST_FILENAME,
    eval_artifact_key,
    eval_latest_key,
    list_eval_artifacts,
    load_latest_eval_artifact,
    new_eval_run_id,
)


class TestNewEvalRunId:

    def test_format_is_yymmddhhmm(self):
        # Inject a known UTC moment → exact YYMMDDHHMM expected
        moment = datetime(2026, 5, 10, 14, 37, 0, tzinfo=timezone.utc)
        assert new_eval_run_id(now=moment) == "2605101437"

    def test_length_is_ten_chars(self):
        moment = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        rid = new_eval_run_id(now=moment)
        assert len(rid) == 10
        assert rid == "2601010000"

    def test_minute_resolution_distinct_minutes_yield_distinct_ids(self):
        m1 = datetime(2026, 5, 10, 14, 37, 0, tzinfo=timezone.utc)
        m2 = datetime(2026, 5, 10, 14, 38, 0, tzinfo=timezone.utc)
        assert new_eval_run_id(now=m1) != new_eval_run_id(now=m2)

    def test_seconds_within_minute_collide_by_design(self):
        # Two runs within the same UTC minute MUST produce the same
        # run_id — the convention is minute-granularity.
        m1 = datetime(2026, 5, 10, 14, 37, 1, tzinfo=timezone.utc)
        m2 = datetime(2026, 5, 10, 14, 37, 59, tzinfo=timezone.utc)
        assert new_eval_run_id(now=m1) == new_eval_run_id(now=m2)

    def test_naive_datetime_treated_as_utc(self):
        # Mirrors dates.now_dual semantics: naive inputs assumed UTC,
        # callers responsible for their own TZ awareness if not.
        naive = datetime(2026, 5, 10, 14, 37, 0)
        assert new_eval_run_id(now=naive) == "2605101437"

    def test_lexicographic_sort_yields_chronological(self):
        ids = [
            new_eval_run_id(now=datetime(2026, 5, 9, 23, 59, 0, tzinfo=timezone.utc)),
            new_eval_run_id(now=datetime(2026, 5, 10, 0, 0, 0, tzinfo=timezone.utc)),
            new_eval_run_id(now=datetime(2026, 5, 10, 14, 37, 0, tzinfo=timezone.utc)),
            new_eval_run_id(now=datetime(2026, 5, 10, 14, 38, 0, tzinfo=timezone.utc)),
            new_eval_run_id(now=datetime(2026, 12, 31, 23, 59, 0, tzinfo=timezone.utc)),
        ]
        assert sorted(ids) == ids, (
            f"YYMMDDHHMM should sort lexicographically into chronological "
            f"order; got {sorted(ids)} != expected {ids}"
        )

    def test_default_uses_now_utc(self):
        # No injected datetime → uses datetime.now(timezone.utc). Smoke
        # check that the result is parseable as YYMMDDHHMM and falls
        # within a reasonable window.
        rid = new_eval_run_id()
        assert len(rid) == 10
        assert rid.isdigit()
        # Year-prefix should be in [25, 99] for any realistic now() call
        # within the project's lifetime
        year_prefix = int(rid[:2])
        assert 25 <= year_prefix <= 99


class TestEvalArtifactKey:

    def test_default_basename_simplifies_to_run_id_dot_json(self):
        key = eval_artifact_key(
            prefix="predictor/variant_gates/triple_barrier",
            run_id="2605101437",
        )
        assert key == "predictor/variant_gates/triple_barrier/2605101437.json"

    def test_custom_basename_keeps_run_id_prefix(self):
        # Multi-file-per-run pipelines (eval-judge): per-file basename
        # gets the run_id prefix so files for one run group together
        # in path listings.
        key = eval_artifact_key(
            prefix="decision_artifacts/_eval",
            run_id="2605101437",
            basename="haiku_eval.json",
        )
        assert key == "decision_artifacts/_eval/2605101437_haiku_eval.json"

    def test_prefix_trailing_slash_normalized(self):
        key = eval_artifact_key(
            prefix="predictor/variant_gates/triple_barrier/",
            run_id="2605101437",
        )
        assert key == "predictor/variant_gates/triple_barrier/2605101437.json"

    def test_prefix_leading_slash_normalized(self):
        key = eval_artifact_key(
            prefix="/predictor/variant_gates/triple_barrier",
            run_id="2605101437",
        )
        assert key == "predictor/variant_gates/triple_barrier/2605101437.json"

    def test_composes_with_now_eval_run_id(self):
        moment = datetime(2026, 5, 10, 14, 37, 0, tzinfo=timezone.utc)
        rid = new_eval_run_id(now=moment)
        key = eval_artifact_key(
            prefix="predictor/variant_gates/triple_barrier",
            run_id=rid,
        )
        assert key == "predictor/variant_gates/triple_barrier/2605101437.json"

    def test_no_date_partition_in_path(self):
        # The flat layout is the institutional canonical form per
        # 2026-05-10 design discussion: YYMMDDHHMM run_id encodes the
        # date, so a {calendar_date}/ sub-partition would be pure
        # redundancy. This test pins the flat shape against future
        # well-meaning re-introduction of date partitioning.
        key = eval_artifact_key(prefix="x/y", run_id="2605101437")
        # Path is exactly two components after the prefix: x/y/run_id.json
        # (no x/y/2026-05-10/run_id.json shape)
        assert key.count("/") == 2
        assert "/2026-" not in key  # no ISO date sub-partition


class TestEvalLatestKey:

    def test_basic(self):
        assert (
            eval_latest_key("predictor/variant_gates/triple_barrier")
            == "predictor/variant_gates/triple_barrier/latest.json"
        )

    def test_trailing_slash_normalized(self):
        assert (
            eval_latest_key("predictor/variant_gates/triple_barrier/")
            == "predictor/variant_gates/triple_barrier/latest.json"
        )

    def test_filename_constant_exposed(self):
        # Constant is part of the public API so dashboards / scripts
        # can hard-code the filename without re-inventing it.
        assert EVAL_LATEST_FILENAME == "latest.json"


# ─ Reader tests ──────────────────────────────────────────────────────


class _FakeS3:
    """Minimal in-memory S3 stub for reader tests — supports get_object
    and a paginator for list_objects_v2."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_json(self, bucket: str, key: str, payload: dict) -> None:
        self.objects[(bucket, key)] = _json.dumps(payload).encode("utf-8")

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        if (Bucket, Key) not in self.objects:
            raise KeyError(f"no object at {Bucket}/{Key}")
        return {"Body": _io.BytesIO(self.objects[(Bucket, Key)])}

    def get_paginator(self, op: str):
        assert op == "list_objects_v2"
        paginator = _MagicMock()
        contents = [{"Key": k} for (_, k) in self.objects.keys()]
        paginator.paginate.return_value = [{"Contents": contents}]
        return paginator


class TestLoadLatestEvalArtifact:

    def test_resolves_sidecar_to_artifact(self):
        s3 = _FakeS3()
        s3.put_json("buck", "regime/latest.json", {
            "run_id": "2605170230",
            "artifact_key": "regime/2605170230.json",
        })
        s3.put_json("buck", "regime/2605170230.json", {
            "run_id": "2605170230",
            "hmm": {"argmax": "neutral"},
        })
        result = load_latest_eval_artifact(s3, bucket="buck", prefix="regime")
        assert result["run_id"] == "2605170230"
        assert result["hmm"]["argmax"] == "neutral"

    def test_returns_none_when_sidecar_missing(self):
        s3 = _FakeS3()
        assert load_latest_eval_artifact(s3, bucket="buck", prefix="regime") is None

    def test_returns_none_when_sidecar_lacks_artifact_key(self):
        s3 = _FakeS3()
        s3.put_json("buck", "regime/latest.json", {"run_id": "2605170230"})
        assert load_latest_eval_artifact(s3, bucket="buck", prefix="regime") is None

    def test_returns_none_when_artifact_body_missing(self):
        """Sidecar points at a key that doesn't exist (transient hiccup
        or partial publish). Loader returns None instead of crashing."""
        s3 = _FakeS3()
        s3.put_json("buck", "regime/latest.json", {
            "artifact_key": "regime/2605170230.json",
        })
        assert load_latest_eval_artifact(s3, bucket="buck", prefix="regime") is None

    def test_prefix_with_trailing_slash_normalized(self):
        """Trailing/leading slashes on prefix shouldn't matter."""
        s3 = _FakeS3()
        s3.put_json("buck", "regime/latest.json", {"artifact_key": "regime/X.json"})
        s3.put_json("buck", "regime/X.json", {"k": "v"})
        assert load_latest_eval_artifact(s3, bucket="buck", prefix="regime/")["k"] == "v"


class TestListEvalArtifacts:

    def test_lists_chronologically(self):
        s3 = _FakeS3()
        # Three artifacts out of order
        s3.put_json("buck", "regime/2605170230.json", {"run_id": "2605170230"})
        s3.put_json("buck", "regime/2604120230.json", {"run_id": "2604120230"})
        s3.put_json("buck", "regime/2604260230.json", {"run_id": "2604260230"})
        s3.put_json("buck", "regime/latest.json", {"artifact_key": "regime/2605170230.json"})
        results = list_eval_artifacts(s3, bucket="buck", prefix="regime")
        assert [r["run_id"] for r in results] == [
            "2604120230", "2604260230", "2605170230",
        ]

    def test_takes_only_n_recent(self):
        s3 = _FakeS3()
        for m in range(1, 11):
            run_id = f"26{m:02d}010230"
            s3.put_json("buck", f"regime/{run_id}.json", {"run_id": run_id})
        results = list_eval_artifacts(s3, bucket="buck", prefix="regime", n_recent=3)
        assert len(results) == 3
        assert [r["run_id"] for r in results] == [
            "2608010230", "2609010230", "2610010230",
        ]

    def test_skips_sidecar_and_nonconforming_keys(self):
        s3 = _FakeS3()
        s3.put_json("buck", "regime/2605170230.json", {"run_id": "2605170230"})
        s3.put_json("buck", "regime/latest.json", {"artifact_key": "x"})
        # Non-conforming keys — must all be skipped
        s3.objects[("buck", "regime/notnumeric.json")] = b"{}"
        s3.objects[("buck", "regime/12345.json")] = b"{}"        # wrong length
        s3.objects[("buck", "regime/2605170230.parquet")] = b""  # wrong ext
        s3.objects[("buck", "regime/sub/2605170230.json")] = b"{}"  # nested
        results = list_eval_artifacts(s3, bucket="buck", prefix="regime")
        assert len(results) == 1
        assert results[0]["run_id"] == "2605170230"

    def test_empty_when_no_artifacts(self):
        s3 = _FakeS3()
        assert list_eval_artifacts(s3, bucket="buck", prefix="regime") == []

    def test_partial_progress_on_body_fetch_failures(self):
        """One bad artifact body shouldn't drop the rest of the window."""
        s3 = _FakeS3()
        s3.put_json("buck", "regime/2604120230.json", {"run_id": "2604120230"})
        s3.put_json("buck", "regime/2605170230.json", {"run_id": "2605170230"})
        # Sidecar that says "middle" artifact exists but body is missing
        # — list operation discovers all 3 keys, body fetch fails on middle.
        original_get_object = s3.get_object

        def _flaky_get(*, Bucket: str, Key: str):
            if Key == "regime/2604260230.json":
                raise KeyError("transient")
            return original_get_object(Bucket=Bucket, Key=Key)

        # Insert the middle key into objects so list sees it, but the
        # body fetch will fail via our patched get_object.
        s3.objects[("buck", "regime/2604260230.json")] = b"{}"
        s3.get_object = _flaky_get  # type: ignore[assignment]
        results = list_eval_artifacts(s3, bucket="buck", prefix="regime")
        run_ids = [r["run_id"] for r in results]
        assert run_ids == ["2604120230", "2605170230"]
