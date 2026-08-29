"""The weekly partition families and the cutover deadline.

``alpha-engine-config-I8809`` — one weekly cycle was written across two S3
date partitions (28 ``_stage_coverage`` verdicts under ``2026-08-21`` and 11
under ``2026-08-22``, same cycle, measured 2026-08-27). The reader unions both
families during a migration window; these tests pin the window's shape and,
critically, make its END unforgettable.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from nousergon_lib.pipeline_status import coverage as cov
from nousergon_lib.pipeline_status.partition import (
    CUTOVER_DATE,
    PARTITION_FAMILIES,
    dual_partition_active,
    partition_dates,
)


def test_cutover_date_is_the_ratified_one():
    """Brian ruling 2026-08-27: dual-write one cycle, cut over 2026-09-05."""
    assert CUTOVER_DATE == date(2026, 9, 5)


def test_partition_families_are_a_closed_vocabulary():
    assert PARTITION_FAMILIES == {"trading_day", "calendar_date"}


def test_window_is_open_before_the_cutover_and_shut_on_it():
    assert dual_partition_active(CUTOVER_DATE - timedelta(days=1)) is True
    # The cutover date itself is OUTSIDE the window — a window including its
    # own end date is one more day of ambiguity for no stated reason.
    assert dual_partition_active(CUTOVER_DATE) is False
    assert dual_partition_active(CUTOVER_DATE + timedelta(days=30)) is False


def test_canonical_partition_is_first_and_legacy_only_while_open():
    before = CUTOVER_DATE - timedelta(days=1)
    assert partition_dates("2026-08-28", "2026-08-29", today=before) == (
        "2026-08-28",
        "2026-08-29",
    )
    assert partition_dates("2026-08-28", "2026-08-29", today=CUTOVER_DATE) == ("2026-08-28",)


def test_identical_or_missing_calendar_date_never_duplicates():
    before = CUTOVER_DATE - timedelta(days=1)
    assert partition_dates("2026-08-28", "2026-08-28", today=before) == ("2026-08-28",)
    assert partition_dates("2026-08-28", None, today=before) == ("2026-08-28",)
    assert partition_dates("2026-08-28", "  ", today=before) == ("2026-08-28",)


# ── The deadline that cannot be forgotten ────────────────────────────────────


def test_the_calendar_fallback_is_GONE_once_the_cutover_has_passed():
    """FAILS from 2026-09-05 onward while the legacy fallback still exists.

    This is the whole point of the module. A compatibility fallback with no
    expiry is indistinguishable from the defect it hides: once both partitions
    are read forever, a writer silently reverting to the calendar family
    produces no signal at all.

    To clear this test you DELETE the fallback — remove ``calendar_date`` from
    ``read_coverage_sweep``, drop ``partition_dates``'s second family, and drop
    the SF's dual-write of ``_sf_completion``. Moving ``CUTOVER_DATE`` forward
    is not a fix and the assertion message says so.
    """
    today = datetime.now(timezone.utc).date()
    if today < CUTOVER_DATE:
        pytest.skip(f"migration window open until {CUTOVER_DATE}; this test arms itself on that date")
    assert not dual_partition_active(today), (
        f"The alpha-engine-config-I8809 migration window closed on {CUTOVER_DATE} and the "
        "calendar-partition fallback is STILL live. Delete the fallback — do not move "
        "CUTOVER_DATE. See nousergon_lib/pipeline_status/partition.py."
    )


# ── The reader actually unions, and says that it did ─────────────────────────


class _FakeS3:
    def __init__(self, objects: dict[str, bytes]):
        self._objects = objects

    def list_objects_v2(self, **kwargs):
        prefix = kwargs["Prefix"]
        return {
            "Contents": [{"Key": k} for k in self._objects if k.startswith(prefix)],
            "IsTruncated": False,
        }

    def get_object(self, Bucket, Key):  # noqa: N803 — boto3 kwarg casing
        class _B:
            def __init__(self, b):
                self._b = b

            def read(self):
                return self._b

        return {"Body": _B(self._objects[Key])}


_REGISTRY = {
    "pipeline_stages": [
        {"stage": "Backtester", "stage_class": "product", "output": "registered"},
        {"stage": "Scanner", "stage_class": "product", "output": "registered"},
        {"stage": "Director", "stage_class": "product", "output": "registered"},
    ]
}


def _verdict(status="COVERED", is_finding=False):
    import json

    return json.dumps({"status": status, "is_finding": is_finding}).encode()


def test_a_split_cycle_reads_as_ONE_cycle_with_no_false_absences():
    """The 2026-08-22 shape: some stages in the trading-day partition, some in
    the calendar one. Before I8809 the second group read as ABSENT and paged."""
    s3 = _FakeS3(
        {
            "_stage_coverage/2026-08-28/Backtester.json": _verdict(),
            "_stage_coverage/2026-08-29/Scanner.json": _verdict(),
            "_stage_coverage/2026-08-29/Director.json": _verdict(),
        }
    )
    verdicts = cov._load_verdicts(
        s3,
        bucket="b",
        run_date="2026-08-28",
        partitions=partition_dates("2026-08-28", "2026-08-29", today=date(2026, 8, 29)),
    )
    sweep = cov.sweep_coverage(
        pipeline="ne-weekly-freshness-pipeline",
        run_date="2026-08-28",
        registry=_REGISTRY,
        verdicts=verdicts,
        entered_states=["Backtester", "Scanner", "Director"],
        partitions_read=("2026-08-28", "2026-08-29"),
    )
    assert sweep.absent == 0
    assert sweep.covered == 3
    assert sweep.should_alert is False
    assert sweep.legacy_partition_rows == 2
    assert "partitions unioned" in sweep.explain()


def test_the_canonical_partition_wins_a_duplicate():
    s3 = _FakeS3(
        {
            "_stage_coverage/2026-08-28/Backtester.json": _verdict("COVERED"),
            "_stage_coverage/2026-08-29/Backtester.json": _verdict("MISSING", True),
        }
    )
    verdicts = cov._load_verdicts(s3, bucket="b", run_date="2026-08-28", partitions=("2026-08-28", "2026-08-29"))
    assert verdicts["Backtester"]["status"] == "COVERED"
    assert verdicts["Backtester"]["_partition_date"] == "2026-08-28"


def test_an_absent_stage_names_every_partition_it_was_looked_for_in():
    sweep = cov.sweep_coverage(
        pipeline="p",
        run_date="2026-08-28",
        registry={"pipeline_stages": [{"stage": "Backtester"}]},
        verdicts={},
        entered_states=["Backtester"],
        partitions_read=("2026-08-28", "2026-08-29"),
    )
    (row,) = sweep.rows
    assert row.state is cov.RowState.ABSENT
    assert "_stage_coverage/2026-08-28/" in row.reason
    assert "_stage_coverage/2026-08-29/" in row.reason


def test_a_single_partition_sweep_reports_no_union():
    sweep = cov.sweep_coverage(
        pipeline="p",
        run_date="2026-08-28",
        registry={"pipeline_stages": [{"stage": "Backtester"}]},
        verdicts={},
        entered_states=[],
    )
    assert sweep.partitions_read == ("2026-08-28",)
    assert "partitions unioned" not in sweep.explain()
    assert sweep.legacy_partition_rows == 0


def test_dual_partition_active_defaults_to_today_utc():
    """No argument = the real clock. Pinning the branch, not the answer."""
    assert dual_partition_active() is (datetime.now(timezone.utc).date() < CUTOVER_DATE)


def test_partition_dates_defaults_to_the_real_clock_too():
    got = partition_dates("2026-08-28", "2026-08-29")
    assert got[0] == "2026-08-28"
    assert (len(got) == 2) is dual_partition_active()


def test_non_json_keys_and_nested_keys_are_ignored():
    s3 = _FakeS3(
        {
            "_stage_coverage/2026-08-28/Backtester.json": _verdict(),
            "_stage_coverage/2026-08-28/README.txt": b"not json",
            "_stage_coverage/2026-08-28/_sweep/nested.json": b"{}",
        }
    )
    verdicts = cov._load_verdicts(s3, bucket="b", run_date="2026-08-28")
    assert set(verdicts) == {"Backtester"}


def test_a_paginated_partition_is_read_to_the_end():
    pages = [
        {
            "Contents": [{"Key": "_stage_coverage/2026-08-28/Backtester.json"}],
            "IsTruncated": True,
            "NextContinuationToken": "t1",
        },
        {
            "Contents": [{"Key": "_stage_coverage/2026-08-28/Scanner.json"}],
            "IsTruncated": False,
        },
    ]
    bodies = {
        "_stage_coverage/2026-08-28/Backtester.json": _verdict(),
        "_stage_coverage/2026-08-28/Scanner.json": _verdict(),
    }

    class _Paged(_FakeS3):
        def __init__(self):
            super().__init__(bodies)
            self._i = 0

        def list_objects_v2(self, **kwargs):
            page = pages[self._i]
            self._i += 1
            return page

    verdicts = cov._load_verdicts(_Paged(), bucket="b", run_date="2026-08-28")
    assert set(verdicts) == {"Backtester", "Scanner"}


def test_the_marker_is_augmented_in_every_partition_the_sweep_unioned():
    """The SF dual-writes the envelope marker; both copies must carry the
    cycle verdict, or a consumer on the legacy family reads UNKNOWN beside a
    known verdict (alpha-engine-config-I8809)."""
    import json

    from nousergon_lib.pipeline_status import completion_marker as cm

    written: dict[str, bytes] = {}
    base = json.dumps(
        {
            "sf": "ne-weekly-freshness-pipeline",
            "status": "SUCCEEDED",
            "claim": "sf_execution_terminal",
            "cycle_verdict": "unknown",
        }
    ).encode()

    class _S3:
        def get_object(self, Bucket, Key):  # noqa: N803
            class _B:
                def read(self_inner):
                    return written.get(Key, base)

            return {"Body": _B()}

        def put_object(self, Bucket, Key, Body, ContentType):  # noqa: N803
            written[Key] = Body

    class _Shape:
        pipeline = "ne-weekly-freshness-pipeline"
        run_date = "2026-08-28"
        executions = ()

        def to_dict(self):
            return {"run_date": self.run_date, "verdict": "completed"}

        class verdict:  # noqa: N801
            value = "completed"

    cm.augment_marker(
        _Shape(),
        s3_client=_S3(),
        bucket="b",
        also_dates=("2026-08-28", "2026-08-29"),
    )
    assert set(written) == {
        "_sf_completion/ne-weekly-freshness-pipeline/2026-08-28.json",
        "_sf_completion/ne-weekly-freshness-pipeline/2026-08-29.json",
    }


# ── The CYCLE key is not the artifact partition, and does not expire ─────────


def test_cycle_keys_admits_the_calendar_identity_permanently():
    """``cycle_keys`` must NOT be gated on the migration window.

    ``partition_dates`` answers *which S3 prefixes hold this cycle's
    artifacts* and closes at the cutover. ``cycle_keys`` answers *which
    executions ARE this cycle*, and a scheduled weekly execution's identity
    falls through to ``startDate`` — its wall-clock day — forever, because
    Step Functions stamps the start time and no NYSE calendar reaches it.
    Expiring this would silently re-break the cycle lookup on 2026-09-05.
    """
    from nousergon_lib.pipeline_status.partition import cycle_keys

    assert cycle_keys("2026-08-28", "2026-08-29") == ("2026-08-28", "2026-08-29")
    # partition_dates DOES expire; cycle_keys must not. Pinned side by side so
    # a future edit cannot collapse the two into one helper.
    assert partition_dates("2026-08-28", "2026-08-29", today=CUTOVER_DATE) == ("2026-08-28",)


def test_cycle_keys_is_canonical_first_and_never_duplicates():
    from nousergon_lib.pipeline_status.partition import cycle_keys

    assert cycle_keys("2026-08-28", "2026-08-28") == ("2026-08-28",)
    assert cycle_keys("2026-08-28", None) == ("2026-08-28",)
    assert cycle_keys("2026-08-28", "   ") == ("2026-08-28",)


class _CycleKeySFN:
    """Two executions of ONE weekly cycle with two different identities.

    ``sat-uuid`` is the Saturday cadence run: no ``run_date`` in its input and
    an opaque name, so its cycle key derives from ``startDate`` — the CALENDAR
    day, ``2026-08-29``. ``fri-uuid`` is the Friday run-day-gate tick, which
    declares itself and skips. The cycle is keyed on the TRADING day,
    ``2026-08-28``.
    """

    _EXECUTIONS = [
        {
            "executionArn": "arn:aws:states:us-east-1:1:execution:sf:sat-uuid",
            "name": "sat-uuid",
            "status": "SUCCEEDED",
            "startDate": datetime(2026, 8, 29, 9, 0, tzinfo=timezone.utc),
        },
        {
            "executionArn": "arn:aws:states:us-east-1:1:execution:sf:fri-uuid",
            "name": "fri-uuid",
            "status": "SUCCEEDED",
            "startDate": datetime(2026, 8, 28, 9, 0, tzinfo=timezone.utc),
        },
    ]
    _HISTORY = {
        "sat-uuid": [
            {"type": "TaskStateEntered", "stateEnteredEventDetails": {"name": n}}
            for n in ("MorningEnrich", "DataPhase1", "Director")
        ],
        "fri-uuid": [
            {"type": "TaskStateEntered", "stateEnteredEventDetails": {"name": "WeeklyRunDayGate"}}
        ],
    }

    def list_executions(self, **_):
        return {"executions": list(self._EXECUTIONS)}

    def describe_execution(self, *, executionArn):
        name = executionArn.rsplit(":", 1)[-1]
        row = next(e for e in self._EXECUTIONS if e["name"] == name)
        return {
            "executionArn": executionArn,
            "name": name,
            "status": row["status"],
            "startDate": row["startDate"],
            "input": '{"pipeline_role":"weekly"}',
        }

    def get_execution_history(self, *, executionArn, **_):
        return {"events": self._HISTORY[executionArn.rsplit(":", 1)[-1]]}


def test_a_trading_day_cycle_admits_its_own_saturday_execution():
    """The regression this exists for, measured live 2026-08-29.

    ``read_cycle_shape(arn, "2026-08-21")`` against the real
    ``ne-weekly-freshness-pipeline`` returned ``skipped (declared_skip),
    0/16 stages in 1 execution`` — the Friday gate tick — while the four
    executions that actually ran the 2026-08-22 cycle were not contributors.
    Passing the calendar identity returned ``incomplete (partial_cycle),
    14/16 across 5 executions``.

    Two consequences, and the second is the serious one: the coverage
    denominator becomes the skip run's entered set (so ``ABSENT`` is
    unreachable), and ``augment_marker`` stamps ``cycle.verdict: skipped``
    onto the completion marker of a cycle that completed — which is
    ``alpha-engine-config-I8186`` with the sign flipped, on the object every
    ``gate:*`` ``Verified-when:`` predicate reads.
    """
    from nousergon_lib.pipeline_status.cycle_shape import read_cycle_shape

    arn = "arn:aws:states:us-east-1:1:stateMachine:ne-weekly-freshness-pipeline"
    spine = ["MorningEnrich", "DataPhase1", "Director"]

    without = read_cycle_shape(arn, "2026-08-28", client=_CycleKeySFN(), stage_spine=spine)
    assert [e.execution_name for e in without.executions] == ["fri-uuid"]
    # The Saturday run that did the week's work is not a contributor to its
    # own cycle: the verdict is derived entirely from the Friday tick.
    assert without.verdict.value == "incomplete"
    assert without.stage_coverage == "0/3"

    with_calendar = read_cycle_shape(
        arn, "2026-08-28", calendar_date="2026-08-29", client=_CycleKeySFN(), stage_spine=spine
    )
    assert {e.execution_name for e in with_calendar.executions} == {"fri-uuid", "sat-uuid"}
    assert with_calendar.verdict.value == "completed"
    # The shape keeps the CANONICAL key, so the marker it augments keeps its
    # canonical S3 key rather than moving to the legacy partition.
    assert with_calendar.run_date == "2026-08-28"


def test_the_sweep_hands_the_calendar_identity_to_the_cycle_lookup():
    """The wiring, not just the primitive: a coverage sweep given both dates
    must pass BOTH down, or the fix above is unreachable in production."""
    seen: dict[str, object] = {}

    def _fake_read_cycle_shape(arn, run_date, *, calendar_date=None, client=None):
        seen["run_date"] = run_date
        seen["calendar_date"] = calendar_date
        raise RuntimeError("stop here — the call signature is what is under test")

    original = cov.read_cycle_shape
    cov.read_cycle_shape = _fake_read_cycle_shape
    try:
        cov.read_coverage_sweep(
            pipeline="ne-weekly-freshness-pipeline",
            run_date="2026-08-28",
            calendar_date="2026-08-29",
            state_machine_arn="arn:aws:states:us-east-1:1:stateMachine:x",
            registry={"pipeline_stages": []},
            s3_client=_EmptyS3(),
        )
    finally:
        cov.read_cycle_shape = original

    assert seen == {"run_date": "2026-08-28", "calendar_date": "2026-08-29"}


class _EmptyS3:
    def list_objects_v2(self, **_):
        return {"Contents": [], "IsTruncated": False}
