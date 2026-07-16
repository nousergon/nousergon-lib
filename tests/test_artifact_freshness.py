"""
Unit tests for ``nousergon_lib.artifact_freshness``.

Pins the substrate contract for the artifact-freshness monitor arc
(plan doc: ``~/Development/alpha-engine-docs/private/artifact-freshness-monitor-260527.md``).

The substrate is the lib-side piece of the cascade closing the silent
absence-of-artifact bug class — the 2026-05-17→27 ``pit_parity.json``
incident is the proximate trigger; the 2026-05-18 factor-profiles
orphan and the 2026-05-23 missing-signals.json incident are siblings.

Tests cover the five branches of :func:`check_freshness`:

  1. Grace-period gate (spec younger than ``grace_period_cycles``).
  2. Calendar-holiday short-circuit (NYSE holiday weekday SFs).
  3. HEAD canonical (fresh / missing / probe_failed paths).
  4. Stale check (object present but last_modified < cycle_tick).
  5. Recovery substitution (canonical missing/stale + recovery fresh).

Plus the pure-helper layer:

  - :func:`resolve_current_cycle` per cadence symbol.
  - :func:`resolve_dedup_key` stability + uniqueness.
  - :class:`ArtifactSpec` validation surface.

See ``[[feedback_no_silent_fails]]`` + the alpha-engine SOTA
sub-sub-rule (this module IS the second-adoption signal — schema-
contract chokepoint is the first; registry-coverage CI guards are
the third).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest import mock

import pytest

from nousergon_lib.artifact_freshness import (
    CADENCE_SYMBOLS,
    ArtifactSpec,
    CheckResult,
    CycleCompletion,
    DependencyGraph,
    build_dependency_graph,
    check_freshness,
    cycle_completion,
    leaf_alert_decisions,
    localize_root_causes,
    resolve_current_cycle,
    resolve_dedup_key,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


def _spec(**overrides) -> ArtifactSpec:
    """Build a baseline saturday_sf spec. Override fields per-test."""
    defaults = {
        "artifact_id": "test_artifact",
        "s3_bucket": "bkt",
        "s3_key_template": "path/{date}/file.json",
        "cadence": "saturday_sf",
        "sla_minutes_after_cron": 180,  # 3hr after Sat 09:00 UTC = 12:00 UTC
        "severity": "warning",
        "owner_repo": "alpha-engine-test",
        "created_at": date(2025, 1, 1),  # ancient — past any grace
    }
    defaults.update(overrides)
    return ArtifactSpec(**defaults)


def _fake_s3(
    head_returns: dict[str, dict] | None = None,
    head_raises: dict[str, Exception] | None = None,
    objects: dict[str, datetime] | None = None,
    list_raises: dict[str, Exception] | None = None,
):
    """Build a mock S3 client supporting ``head_object`` (fixed-key probes)
    and ``get_paginator("list_objects_v2")`` (date-templated recency probes).

    ``head_returns[key]`` ⇒ that HEAD response dict.
    ``head_raises[key]`` ⇒ HEAD raises that exception.
    ``objects[key] = last_modified`` ⇒ object visible to LIST. When omitted,
        the listing is derived from ``head_returns`` (key → its LastModified),
        so a single ``head_returns=`` keeps working for both probe shapes.
    ``list_raises[prefix]`` ⇒ paginate raises for that Prefix.
    """
    head_returns = head_returns or {}
    head_raises = head_raises or {}
    list_raises = list_raises or {}
    if objects is None:
        objects = {
            k: v["LastModified"]
            for k, v in head_returns.items()
            if isinstance(v, dict) and "LastModified" in v
        }

    def _head(*, Bucket, Key):
        if Key in head_raises:
            raise head_raises[Key]
        if Key in head_returns:
            return head_returns[Key]
        raise _ClientError404()

    def _paginate(*, Bucket, Prefix):
        for pfx, exc in list_raises.items():
            if Prefix.startswith(pfx) or pfx.startswith(Prefix):
                raise exc
        contents = [
            {"Key": k, "LastModified": lm}
            for k, lm in objects.items()
            if k.startswith(Prefix)
        ]
        return iter([{"Contents": contents}])

    paginator = mock.Mock()
    paginator.paginate.side_effect = _paginate

    client = mock.Mock()
    client.head_object.side_effect = _head
    client.get_paginator.return_value = paginator
    return client


class _ClientError404(Exception):
    """Duck-typed boto3 ClientError for 404. Avoids the botocore dep
    in the lib-side substrate tests — _classify_client_error is
    intentionally duck-typed against ``err.response``."""

    def __init__(self):
        super().__init__("Not Found")
        self.response = {
            "Error": {"Code": "404", "Message": "Not Found"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        }


class _ClientError403(Exception):
    def __init__(self):
        super().__init__("Access Denied")
        self.response = {
            "Error": {"Code": "AccessDenied", "Message": "Access Denied"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        }


# ── ArtifactSpec validation ─────────────────────────────────────────────────


class TestArtifactSpecValidation:
    """The spec's __post_init__ is the producer-side chokepoint —
    bad rows fail at registry-load time, not at probe time."""

    def test_baseline_spec_validates(self):
        s = _spec()
        assert s.cadence == "saturday_sf"

    def test_rejects_unknown_cadence(self):
        with pytest.raises(ValueError, match="not in"):
            _spec(cadence="hourly")

    def test_rejects_unknown_severity(self):
        with pytest.raises(ValueError, match="severity"):
            _spec(severity="info")

    def test_rejects_negative_sla(self):
        with pytest.raises(ValueError, match="sla_minutes_after_cron"):
            _spec(sla_minutes_after_cron=-1)

    def test_rejects_negative_grace(self):
        with pytest.raises(ValueError, match="grace_period_cycles"):
            _spec(grace_period_cycles=-1)

    def test_continuous_requires_interval(self):
        with pytest.raises(ValueError, match="interval_minutes"):
            _spec(cadence="continuous")

    def test_continuous_with_interval_validates(self):
        s = _spec(cadence="continuous", interval_minutes=15)
        assert s.interval_minutes == 15


# ── resolve_current_cycle ───────────────────────────────────────────────────


class TestResolveCurrentCycle:
    """The cycle resolver is the per-cadence semantic — gets wrong and
    every downstream piece (dedup keys, SLA arithmetic, expected-key
    formatting) gets wrong too."""

    def test_saturday_after_cron_returns_this_saturday(self):
        # Sat 2026-05-30 18:00 UTC — cron at 09:00 UTC has fired.
        now = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        tick, label = resolve_current_cycle(_spec(), now)
        assert tick == datetime(2026, 5, 30, 9, 0, tzinfo=timezone.utc)
        assert label == "2026-W22"

    def test_saturday_before_cron_returns_last_saturday(self):
        # Sat 2026-05-30 08:00 UTC — cron at 09:00 UTC has not fired.
        now = datetime(2026, 5, 30, 8, 0, tzinfo=timezone.utc)
        tick, label = resolve_current_cycle(_spec(), now)
        assert tick == datetime(2026, 5, 23, 9, 0, tzinfo=timezone.utc)
        assert label == "2026-W21"

    def test_midweek_returns_last_saturday(self):
        # Wed 2026-05-27 — last Saturday is 2026-05-23.
        now = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
        tick, label = resolve_current_cycle(_spec(), now)
        assert tick == datetime(2026, 5, 23, 9, 0, tzinfo=timezone.utc)
        assert label == "2026-W21"

    def test_weekday_sf_after_cron_returns_today_trading_day(self):
        # Wed 2026-05-27 15:00 UTC — cron at 13:00 UTC has fired.
        now = datetime(2026, 5, 27, 15, 0, tzinfo=timezone.utc)
        spec = _spec(cadence="weekday_sf", sla_minutes_after_cron=60)
        tick, label = resolve_current_cycle(spec, now)
        assert tick == datetime(2026, 5, 27, 13, 0, tzinfo=timezone.utc)
        assert label == "2026-05-27"

    def test_weekday_sf_before_cron_returns_yesterday(self):
        # Wed 2026-05-27 11:00 UTC — cron at 13:00 UTC has not fired.
        now = datetime(2026, 5, 27, 11, 0, tzinfo=timezone.utc)
        spec = _spec(cadence="weekday_sf", sla_minutes_after_cron=60)
        tick, label = resolve_current_cycle(spec, now)
        assert tick == datetime(2026, 5, 26, 13, 0, tzinfo=timezone.utc)
        assert label == "2026-05-26"

    def test_weekday_sf_monday_morning_returns_friday(self):
        # Mon 2026-05-25 11:00 UTC — cron not fired; weekend before is Fri 5/22.
        # But 2026-05-25 is Memorial Day! Calendar-aware should land on 5/22.
        now = datetime(2026, 5, 25, 11, 0, tzinfo=timezone.utc)
        spec = _spec(cadence="weekday_sf", sla_minutes_after_cron=60)
        tick, label = resolve_current_cycle(spec, now)
        # Most-recent weekday whose 13:00 UTC has passed is Friday 2026-05-22.
        assert tick == datetime(2026, 5, 22, 13, 0, tzinfo=timezone.utc)
        assert label == "2026-05-22"

    def test_weekday_sf_calendar_aware_returns_today_on_trading_day(self):
        # Tue 2026-05-26 14:00 UTC — cron fired. Monday 2026-05-25 was
        # Memorial Day but doesn't affect Tuesday's cycle. The cycle
        # for "today" is 5/26 (a trading day) — verify the resolver
        # doesn't corrupt non-holiday cycles by walking past them.
        now = datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc)
        spec = _spec(cadence="weekday_sf", sla_minutes_after_cron=60)
        tick, label = resolve_current_cycle(spec, now)
        assert label == "2026-05-26"
        assert tick.date() == date(2026, 5, 26)

    def test_weekday_sf_cycle_on_nyse_holiday_keeps_holiday_date(self):
        # Memorial Day Mon 2026-05-25 14:00 UTC — cron candidate is
        # today's 13:00 UTC. resolver does NOT snap holidays; it
        # returns the holiday weekday so :func:`check_freshness` can
        # explicitly route to the holiday short-circuit and produce
        # one distinct dedup cycle per calendar day.
        now = datetime(2026, 5, 25, 14, 0, tzinfo=timezone.utc)
        spec = _spec(cadence="weekday_sf", sla_minutes_after_cron=60)
        tick, label = resolve_current_cycle(spec, now)
        assert label == "2026-05-25"
        assert tick.date() == date(2026, 5, 25)

    def test_eod_sf_anchors_to_21_utc(self):
        # Wed 2026-05-27 22:00 UTC — EOD anchor at 21:00 UTC has fired.
        now = datetime(2026, 5, 27, 22, 0, tzinfo=timezone.utc)
        spec = _spec(cadence="eod_sf", sla_minutes_after_cron=60)
        tick, label = resolve_current_cycle(spec, now)
        assert tick == datetime(2026, 5, 27, 21, 0, tzinfo=timezone.utc)
        assert label == "2026-05-27"

    def test_continuous_buckets_by_interval(self):
        spec = _spec(
            cadence="continuous",
            interval_minutes=15,
            calendar_aware=False,
        )
        # Two calls within the same 15min bucket → same label.
        now1 = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
        now2 = datetime(2026, 5, 27, 12, 14, tzinfo=timezone.utc)
        _, label1 = resolve_current_cycle(spec, now1)
        _, label2 = resolve_current_cycle(spec, now2)
        assert label1 == label2
        # Across the bucket boundary → different labels.
        now3 = datetime(2026, 5, 27, 12, 15, tzinfo=timezone.utc)
        _, label3 = resolve_current_cycle(spec, now3)
        assert label1 != label3

    def test_naive_now_treated_as_utc(self):
        # Naive datetime ⇒ assumed UTC (matches the alerts module convention).
        now_aware = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        now_naive = datetime(2026, 5, 30, 18, 0)
        assert resolve_current_cycle(_spec(), now_aware) == \
            resolve_current_cycle(_spec(), now_naive)


# ── resolve_dedup_key ───────────────────────────────────────────────────────


class TestResolveDedupKey:

    def test_shape_is_freshness_prefix(self):
        now = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        key = resolve_dedup_key(_spec(), now)
        assert key.startswith("freshness_test_artifact_")
        assert key == "freshness_test_artifact_2026-W22"

    def test_same_cycle_same_key(self):
        # Both Sat 2026-05-30 18:00 and Sun 2026-05-31 04:00 are in W22.
        now1 = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        now2 = datetime(2026, 5, 31, 4, 0, tzinfo=timezone.utc)
        assert resolve_dedup_key(_spec(), now1) == \
            resolve_dedup_key(_spec(), now2)

    def test_different_cycles_different_keys(self):
        # 5/30 (W22) vs 6/6 (W23).
        now1 = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        now2 = datetime(2026, 6, 6, 18, 0, tzinfo=timezone.utc)
        assert resolve_dedup_key(_spec(), now1) != \
            resolve_dedup_key(_spec(), now2)


# ── check_freshness — grace period ──────────────────────────────────────────


class TestCheckFreshnessGracePeriod:

    def test_new_spec_within_grace_returns_grace_period(self):
        # Spec created 1 day ago; saturday_sf grace = 2 × 7d = 14d window.
        now = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        spec = _spec(created_at=date(2026, 5, 29), grace_period_cycles=2)
        result = check_freshness(_fake_s3(), spec, now)
        assert result.state == "grace_period"
        assert "grace period" in result.reason.lower()

    def test_grace_zero_disables_grace(self):
        # grace_period_cycles=0 ⇒ no grace at all even on day-zero spec.
        now = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        spec = _spec(created_at=date(2026, 5, 30), grace_period_cycles=0)
        result = check_freshness(_fake_s3(), spec, now)
        assert result.state == "missing"


# ── check_freshness — NYSE-holiday short-circuit ────────────────────────────


class TestCheckFreshnessHoliday:

    def test_weekday_sf_on_nyse_holiday_returns_fresh(self):
        # Memorial Day 2026-05-25 (Mon) is an NYSE holiday. Run the
        # probe at Tue 5/26 11:00 UTC so the current weekday_sf cycle
        # candidate is 5/25 (yesterday's 13:00 UTC cron) — calendar
        # snap should land on 5/22.
        # Actually let's run AT 5/25 14:00 UTC so cron-candidate is
        # today (Monday 13:00 UTC) and the holiday snap triggers.
        now = datetime(2026, 5, 25, 14, 0, tzinfo=timezone.utc)
        spec = _spec(
            cadence="weekday_sf",
            sla_minutes_after_cron=60,
            calendar_aware=True,
        )
        result = check_freshness(_fake_s3(), spec, now)
        # The cycle should snap to a trading day; absence is correct.
        # Note: the snap returns 5/22 (last trading day before MLK / Memorial).
        # check_freshness then HEADs path/2026-05-22/file.json which 404s.
        # But calendar_aware short-circuits BEFORE the HEAD when the
        # ORIGINAL candidate weekday is a holiday — that's the spec.
        # In this test, current cycle is 5/25 (Memorial Day) which is
        # NOT a trading day, so the gate fires.
        assert result.state == "fresh"
        assert "holiday" in result.reason.lower()

    def test_weekday_sf_calendar_aware_false_skips_holiday_gate(self):
        # Same Memorial Day setup but calendar_aware=False ⇒ no short-circuit.
        now = datetime(2026, 5, 25, 14, 0, tzinfo=timezone.utc)
        spec = _spec(
            cadence="weekday_sf",
            sla_minutes_after_cron=60,
            calendar_aware=False,
        )
        result = check_freshness(_fake_s3(), spec, now)
        # No holiday gate ⇒ falls through to HEAD which 404s ⇒ missing.
        assert result.state == "missing"

    def test_saturday_sf_no_holiday_short_circuit(self):
        # Saturday cron has no calendar-aware short-circuit (Saturday is
        # never a "trading day" but the SF runs anyway).
        now = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        spec = _spec(calendar_aware=True)
        result = check_freshness(_fake_s3(), spec, now)
        # Falls through to HEAD which 404s ⇒ missing.
        assert result.state == "missing"


# ── check_freshness — canonical HEAD paths ─────────────────────────────────


class TestCheckFreshnessCanonical:

    def test_fresh_when_head_returns_last_modified_after_cycle(self):
        now = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        cycle_tick = datetime(2026, 5, 30, 9, 0, tzinfo=timezone.utc)
        s3 = _fake_s3(head_returns={
            "path/2026-05-30/file.json": {
                "LastModified": cycle_tick + timedelta(hours=2),
            },
        })
        result = check_freshness(s3, _spec(), now)
        assert result.state == "fresh"
        assert result.last_modified == cycle_tick + timedelta(hours=2)
        assert result.canonical_key == "path/2026-05-30/file.json"

    def test_missing_when_head_404s(self):
        # Past SLA (Sat 18:00 UTC — 9hr after cron, SLA grace is 180min).
        now = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        result = check_freshness(_fake_s3(), _spec(), now)
        assert result.state == "missing"
        # SLA breach: now=18:00 - (cron 09:00 + 180min = 12:00) = 6hr = 360min.
        assert result.sla_violated_by_minutes == 360
        assert result.canonical_key == "path/2026-05-30/file.json"

    def test_missing_within_sla_grace_still_reported(self):
        # 10:00 UTC Saturday — past cron (09:00) but inside the 180min
        # SLA window. Probe still classifies as missing — the substrate
        # is pure; the *Lambda* is what decides whether to alert.
        now = datetime(2026, 5, 30, 10, 0, tzinfo=timezone.utc)
        result = check_freshness(_fake_s3(), _spec(), now)
        assert result.state == "missing"
        # 10:00 - (09:00 + 180min = 12:00) = -120min ⇒ clipped to 0.
        assert result.sla_violated_by_minutes == 0

    def test_stale_when_newest_instance_predates_floor(self):
        # Recency model, 10-day max-age floor anchored to NOW: now is Sat 5/30
        # 18:00 UTC, floor = now − 10d = 5/20 18:00 UTC. The freshest instance
        # is from 5/15 — older than 10 calendar days ⇒ STALE (a genuinely-
        # skipped week, not normal jitter).
        now = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        s3 = _fake_s3(objects={
            "path/2026-05-15/file.json":
                datetime(2026, 5, 15, 10, 0, tzinfo=timezone.utc),
        })
        result = check_freshness(s3, _spec(), now)
        assert result.state == "stale"
        # floor 5/20 18:00 - newest 5/15 10:00 = 5d8h = 7680min before floor
        # (== now − (newest + 10d), the issue's deadline arithmetic).
        assert result.sla_violated_by_minutes == 7680
        assert result.last_modified == datetime(2026, 5, 15, 10, 0, tzinfo=timezone.utc)

    def test_last_week_instance_fresh_at_saturday_tick(self):
        # Regression for the 2026-06-27 burst: at the moment THIS Saturday's
        # 09:00 cron ticks (the SF has only just started and hasn't produced
        # this week's artifacts), last week's instance must read FRESH — the
        # prior 5-CALENDAR-day floor flipped it stale here, paging at 2am.
        # now = Sat 5/30 09:30 (just after the tick); newest = last Saturday
        # 5/23's run (7 calendar days back, within the 10-day max-age window).
        now = datetime(2026, 5, 30, 9, 30, tzinfo=timezone.utc)
        s3 = _fake_s3(objects={
            "path/2026-05-23/file.json":
                datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc),
        })
        result = check_freshness(s3, _spec(), now)
        assert result.state == "fresh"
        assert result.last_modified == datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)

    def test_fresh_when_off_cycle_instance_within_slack(self):
        # The off-cycle regression: an instance written the day BEFORE the
        # Saturday cron (a `run_weekly_offcycle.sh full` Friday run) under a
        # DIFFERENT date key than the cron date must read FRESH — the monitor
        # finds the most recent instance, not an exact cron-date key.
        now = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        s3 = _fake_s3(objects={
            # Friday-anchored key, written Friday — not the 5/30 cron date.
            "path/2026-05-29/file.json":
                datetime(2026, 5, 29, 16, 0, tzinfo=timezone.utc),
        })
        result = check_freshness(s3, _spec(), now)
        assert result.state == "fresh"
        assert result.last_modified == datetime(2026, 5, 29, 16, 0, tzinfo=timezone.utc)

    def test_probe_failed_on_403(self):
        # 403 on the LIST probe ⇒ probe_failed (the monitor itself is broken).
        now = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        s3 = _fake_s3(list_raises={"path/": _ClientError403()})
        result = check_freshness(s3, _spec(), now)
        assert result.state == "probe_failed"
        assert "403" in result.reason or "AccessDenied" in result.reason

    def test_probe_failed_on_network_error(self):
        # Random exception (not a ClientError shape) ⇒ probe_failed.
        now = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        s3 = _fake_s3(list_raises={"path/": RuntimeError("network down")})
        result = check_freshness(s3, _spec(), now)
        assert result.state == "probe_failed"
        assert "network" in result.reason.lower() or "probe error" in result.reason.lower()


# ── check_freshness — saturday_sf 10-day max-age staleness (config#1297) ─────


class TestSaturdaySf10DayMaxAge:
    """The 2026-06-27 Saturday-SF-start false-positive burst + run-day-jitter
    regression set. The ``saturday_sf`` freshness floor is now a 10-CALENDAR-
    day max-age window anchored to ``now`` (``_SATURDAY_SF_STALE_DAYS = 10``),
    NOT a slack anchored to the Saturday cron tick. A normal weekly artifact
    (~7d old) must read FRESH even at the instant the Saturday cron ticks
    before this week's multi-hour SF has produced its replacement; Fri→Sun
    run-day jitter (~9d) must read FRESH; a genuinely-missed run (>10d) must
    read STALE; nothing at all must read MISSING."""

    def test_seven_day_old_keyed_instance_is_fresh(self):
        # THE regression. now = Sat 6/27 09:05 UTC — the instant this Saturday's
        # 09:00 cron ticks (the SF has only just started, this week's artifacts
        # don't exist yet). The newest instance is last Saturday's 6/20 run,
        # 7 calendar days old. Must read FRESH — the false-positive burst that
        # paged ~22 SNS alerts at 2am PT must be structurally impossible.
        now = datetime(2026, 6, 27, 9, 5, tzinfo=timezone.utc)
        s3 = _fake_s3(objects={
            "path/2026-06-20/file.json":
                datetime(2026, 6, 20, 12, 0, tzinfo=timezone.utc),
        })
        result = check_freshness(s3, _spec(), now)
        assert result.state == "fresh"
        assert result.sla_violated_by_minutes == 0

    def test_friday_run_nine_days_old_is_fresh(self):
        # Run-day jitter: a weekly run landed on a Friday and the next has not
        # yet started by the following Sunday — ≈9 calendar days apart, normal
        # jitter that must NOT alert. now = Sun 6/28 12:00; newest = Fri 6/19
        # 16:00 (9d ago, under the {trading_day}=Fri key). 9 ≤ 10 ⇒ FRESH.
        now = datetime(2026, 6, 28, 12, 0, tzinfo=timezone.utc)
        s3 = _fake_s3(objects={
            "path/2026-06-19/file.json":
                datetime(2026, 6, 19, 16, 0, tzinfo=timezone.utc),
        })
        result = check_freshness(s3, _spec(), now)
        assert result.state == "fresh"

    def test_eleven_day_old_instance_is_stale(self):
        # A genuinely-missed run: newest instance is 11 calendar days old —
        # past the 10-day max-age window ⇒ STALE (the absence backstop must
        # still fire on a real miss). now = Sat 6/27 18:00; newest = 6/16 12:00.
        now = datetime(2026, 6, 27, 18, 0, tzinfo=timezone.utc)
        s3 = _fake_s3(objects={
            "path/2026-06-16/file.json":
                datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc),
        })
        result = check_freshness(s3, _spec(), now)
        assert result.state == "stale"
        # deadline = newest + 10d = 6/26 12:00; violated = now − deadline
        # = 6/27 18:00 − 6/26 12:00 = 1d6h = 1800min.
        assert result.sla_violated_by_minutes == 1800
        assert result.last_modified == datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc)

    def test_none_in_window_is_missing(self):
        # No instance at all under the prefix ⇒ MISSING (not stale). The SLA
        # breach anchors to the cron tick + sla grace, not the max-age window.
        now = datetime(2026, 6, 27, 18, 0, tzinfo=timezone.utc)
        result = check_freshness(_fake_s3(), _spec(), now)
        assert result.state == "missing"

    def test_pointer_artifact_within_10d_is_fresh(self):
        # Pointer artifact (fixed key, no {...} placeholder, e.g.
        # config/scoring_weights.json): fresh iff now − last_modified ≤ 10d.
        # now = 6/27 18:00; last_modified = 6/18 18:00 (9d) ⇒ FRESH.
        now = datetime(2026, 6, 27, 18, 0, tzinfo=timezone.utc)
        spec = _spec(s3_key_template="config/scoring_weights.json")
        s3 = _fake_s3(head_returns={
            "config/scoring_weights.json": {
                "LastModified": datetime(2026, 6, 18, 18, 0, tzinfo=timezone.utc),
            },
        })
        result = check_freshness(s3, spec, now)
        assert result.state == "fresh"

    def test_pointer_artifact_age_boundary_just_over_10d_is_stale(self):
        # Pointer-age boundary: last_modified exactly 10d + 1min before now ⇒
        # just over the window ⇒ STALE. now = 6/27 18:00;
        # last_modified = 6/17 17:59 (10d 1min ago).
        now = datetime(2026, 6, 27, 18, 0, tzinfo=timezone.utc)
        spec = _spec(s3_key_template="regime/latest.json")
        s3 = _fake_s3(head_returns={
            "regime/latest.json": {
                "LastModified": datetime(2026, 6, 17, 17, 59, tzinfo=timezone.utc),
            },
        })
        result = check_freshness(s3, spec, now)
        assert result.state == "stale"

    def test_pointer_artifact_age_boundary_just_under_10d_is_fresh(self):
        # The other side of the boundary: last_modified exactly 10d − 1min
        # before now ⇒ inside the window ⇒ FRESH.
        now = datetime(2026, 6, 27, 18, 0, tzinfo=timezone.utc)
        spec = _spec(s3_key_template="regime/latest.json")
        s3 = _fake_s3(head_returns={
            "regime/latest.json": {
                "LastModified": datetime(2026, 6, 17, 18, 1, tzinfo=timezone.utc),
            },
        })
        result = check_freshness(s3, spec, now)
        assert result.state == "fresh"


# ── check_freshness — recovery substitution ─────────────────────────────────


class TestCheckFreshnessRecovery:

    def test_recovery_satisfies_when_canonical_missing(self):
        # Canonical 404; recovery present + within cycle ⇒ fresh.
        now = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        cycle_tick = datetime(2026, 5, 30, 9, 0, tzinfo=timezone.utc)
        spec = _spec(recovery_key_template="recovery/{date}/file.json")
        s3 = _fake_s3(head_returns={
            "recovery/2026-05-30/file.json": {
                "LastModified": cycle_tick + timedelta(hours=4),
            },
        })
        result = check_freshness(s3, spec, now)
        assert result.state == "fresh"
        assert result.recovery_substituted is True
        # canonical_key still reports the canonical (for operator
        # diagnostic) even though recovery satisfied.
        assert result.canonical_key == "path/2026-05-30/file.json"

    def test_recovery_satisfies_when_canonical_stale(self):
        now = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        cycle_tick = datetime(2026, 5, 30, 9, 0, tzinfo=timezone.utc)
        spec = _spec(recovery_key_template="recovery/{date}/file.json")
        s3 = _fake_s3(head_returns={
            "path/2026-05-30/file.json": {
                "LastModified": datetime(2026, 5, 23, 10, 0, tzinfo=timezone.utc),
            },
            "recovery/2026-05-30/file.json": {
                "LastModified": cycle_tick + timedelta(hours=4),
            },
        })
        result = check_freshness(s3, spec, now)
        assert result.state == "fresh"
        assert result.recovery_substituted is True

    def test_recovery_too_old_does_not_substitute(self):
        # Recovery's freshest instance predates the floor ⇒ does NOT satisfy:
        # an old instance exists, so it reads STALE (not missing), and the
        # fresh-only recovery_substituted flag stays False.
        now = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        spec = _spec(recovery_key_template="recovery/{date}/file.json")
        # 5/15 is past the 10-day max-age floor (now 5/30 − 10d = 5/20) —
        # a genuinely-old recovery instance, so it reads STALE and does not
        # substitute.
        s3 = _fake_s3(objects={
            "recovery/2026-05-15/file.json":
                datetime(2026, 5, 15, 10, 0, tzinfo=timezone.utc),
        })
        result = check_freshness(s3, spec, now)
        assert result.state == "stale"
        assert result.recovery_substituted is False

    def test_probe_failed_canonical_bypasses_recovery(self):
        # 403 on canonical ⇒ probe_failed even if recovery is fresh.
        # The monitor itself is broken; operator needs to know.
        now = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        cycle_tick = datetime(2026, 5, 30, 9, 0, tzinfo=timezone.utc)
        spec = _spec(recovery_key_template="recovery/{date}/file.json")
        s3 = _fake_s3(
            list_raises={"path/": _ClientError403()},
            objects={
                "recovery/2026-05-30/file.json": cycle_tick + timedelta(hours=4),
            },
        )
        result = check_freshness(s3, spec, now)
        assert result.state == "probe_failed"


# ── Cadence-symbol coverage sanity ──────────────────────────────────────────


def test_cadence_symbols_match_documented_set():
    """The set is closed by plan §4. Adding a symbol here without adding
    a cycle-resolution + dedup-key-label branch is the failure mode.

    ``event_driven`` (config#1718) is a first-class member: it carries a
    cycle-resolution branch (per-UTC-day label), a dedup-key label, a
    freshness-floor branch, and a check_freshness short-circuit — see
    TestEventDriven* below."""
    assert CADENCE_SYMBOLS == frozenset(
        {"saturday_sf", "weekday_sf", "eod_sf", "continuous", "event_driven"}
    )


# ── Per-cycle completion rollup (Phase 1b) ──────────────────────────────────


def _critical(artifact_id: str) -> ArtifactSpec:
    return _spec(artifact_id=artifact_id, severity="critical")


def _warning(artifact_id: str) -> ArtifactSpec:
    return _spec(artifact_id=artifact_id, severity="warning")


def _res(state: str) -> CheckResult:
    return CheckResult(state=state, reason=f"test {state}")


class TestCycleCompletion:
    def test_all_critical_fresh_is_complete(self):
        pairs = [
            (_critical("a"), _res("fresh")),
            (_critical("b"), _res("fresh")),
            (_critical("c"), _res("fresh")),
        ]
        v = cycle_completion(pairs, cycle_label="2026-W22")
        assert isinstance(v, CycleCompletion)
        assert v.state == "complete"
        assert v.complete is True
        assert v.n_required == 3
        assert v.n_satisfied == 3
        assert v.cycle_label == "2026-W22"

    def test_one_missing_is_incomplete(self):
        v = cycle_completion([
            (_critical("a"), _res("fresh")),
            (_critical("b"), _res("missing")),
        ])
        assert v.state == "incomplete"
        assert v.complete is False
        assert v.missing == ["b"]
        assert v.n_satisfied == 1

    def test_one_stale_is_incomplete(self):
        v = cycle_completion([
            (_critical("a"), _res("fresh")),
            (_critical("b"), _res("stale")),
        ])
        assert v.state == "incomplete"
        assert v.stale == ["b"]

    def test_probe_failed_only_is_indeterminate(self):
        v = cycle_completion([
            (_critical("a"), _res("fresh")),
            (_critical("b"), _res("probe_failed")),
        ])
        assert v.state == "indeterminate"
        assert v.complete is False
        assert v.probe_failed == ["b"]

    def test_real_gap_outranks_probe_failure(self):
        """A confirmed miss is more actionable than an unconfirmable probe."""
        v = cycle_completion([
            (_critical("a"), _res("missing")),
            (_critical("b"), _res("probe_failed")),
        ])
        assert v.state == "incomplete"
        assert v.missing == ["a"]
        assert v.probe_failed == ["b"]  # still localized, but doesn't set the verdict

    def test_grace_period_counts_as_satisfied(self):
        v = cycle_completion([
            (_critical("a"), _res("fresh")),
            (_critical("b"), _res("grace_period")),
        ])
        assert v.state == "complete"
        assert v.complete is True
        assert v.n_satisfied == 2
        assert v.grace_period == ["b"]

    def test_warning_severity_excluded_from_required_set(self):
        """A missing WARNING artifact must not fail the cycle — only
        critical rows gate the completion verdict."""
        v = cycle_completion([
            (_critical("a"), _res("fresh")),
            (_warning("b"), _res("missing")),
        ])
        assert v.state == "complete"
        assert v.n_required == 1
        assert v.missing == []

    def test_empty_required_set_is_vacuously_complete(self):
        v = cycle_completion([(_warning("a"), _res("missing"))])
        assert v.state == "complete"
        assert v.complete is True
        assert v.n_required == 0

    def test_mixed_states_incomplete_localizes_all_gaps(self):
        v = cycle_completion([
            (_critical("a"), _res("fresh")),
            (_critical("b"), _res("grace_period")),
            (_critical("c"), _res("missing")),
            (_critical("d"), _res("stale")),
            (_critical("e"), _res("probe_failed")),
        ])
        assert v.state == "incomplete"
        assert v.n_required == 5
        assert v.n_satisfied == 2  # fresh + grace_period
        assert v.missing == ["c"]
        assert v.stale == ["d"]
        assert v.probe_failed == ["e"]
        assert v.grace_period == ["b"]


# ── Continuous active-window (market-hours-only producers) ───────────────────


def _open_orders_spec(**overrides) -> ArtifactSpec:
    """A continuous, fixed-key spec modelling the executor daemon's
    ``trades/open_orders/latest.json`` snapshot — written every ~30min
    ONLY while paper-trading during the NYSE session."""
    defaults = {
        "artifact_id": "open_orders_latest",
        "s3_bucket": "bkt",
        "s3_key_template": "trades/open_orders/latest.json",
        "cadence": "continuous",
        "interval_minutes": 30,
        "sla_minutes_after_cron": 15,
        "severity": "warning",
        "owner_repo": "alpha-engine",
        "created_at": date(2025, 1, 1),
        "active_hours_utc": [14, 21],
    }
    defaults.update(overrides)
    return ArtifactSpec(**defaults)


class TestContinuousActiveWindow:
    """The 2026-06-26 open_orders overnight alert storm: a market-hours-only
    daemon modelled as 24/7 continuous false-alarmed every 30min outside its
    write window. The active-window short-circuit suppresses the off-window
    false positive while preserving the in-window mid-session-death signal."""

    def test_outside_hours_fresh_despite_absent_artifact(self):
        # Fri 03:00 UTC — outside [14,21). Even a 404 probe must NOT alarm:
        # the producer is idle by design. Gate short-circuits before HEAD.
        now = datetime(2026, 6, 26, 3, 0, tzinfo=timezone.utc)
        result = check_freshness(_fake_s3(), _open_orders_spec(), now)
        assert result.state == "fresh"
        assert "outside active production window" in result.reason

    def test_weekend_fresh_when_trading_days_only(self):
        # Sat 16:00 UTC — inside the hour window but a non-trading day.
        now = datetime(2026, 6, 27, 16, 0, tzinfo=timezone.utc)
        result = check_freshness(_fake_s3(), _open_orders_spec(), now)
        assert result.state == "fresh"
        assert "non-trading day" in result.reason

    def test_in_window_stale_artifact_flags_stale(self):
        # Fri 16:00 UTC (trading day, in window). Newest write 12:00 — older
        # than the floor (16:00 - 45min = 15:15). The daemon died mid-session:
        # this is the genuine signal the gate must STILL surface.
        now = datetime(2026, 6, 26, 16, 0, tzinfo=timezone.utc)
        s3 = _fake_s3(head_returns={
            "trades/open_orders/latest.json": {
                "LastModified": datetime(2026, 6, 26, 12, 0, tzinfo=timezone.utc),
            },
        })
        result = check_freshness(s3, _open_orders_spec(), now)
        assert result.state == "stale"

    def test_in_window_fresh_artifact_fresh(self):
        # Fri 16:00 UTC, newest write 15:50 — within the 45min floor.
        now = datetime(2026, 6, 26, 16, 0, tzinfo=timezone.utc)
        s3 = _fake_s3(head_returns={
            "trades/open_orders/latest.json": {
                "LastModified": datetime(2026, 6, 26, 15, 50, tzinfo=timezone.utc),
            },
        })
        result = check_freshness(s3, _open_orders_spec(), now)
        assert result.state == "fresh"

    def test_in_window_missing_flags_missing(self):
        # In-window 404 ⇒ missing (gate does not suppress a real in-window gap).
        now = datetime(2026, 6, 26, 16, 0, tzinfo=timezone.utc)
        result = check_freshness(_fake_s3(), _open_orders_spec(), now)
        assert result.state == "missing"

    def test_hours_implies_market_hours_trading_day_gate(self):
        # Semantic consolidation (run_calendar, 2026-06-28): a legacy spec
        # with active_hours_utc set (and no run_calendar) resolves to
        # run_calendar="market_hours", which is trading-day AND hours gated.
        # So a weekend hour inside the window short-circuits to fresh — there
        # is no "hours-only on all calendar days" producer in the fleet, and
        # the enum has no value for it. Previously these were independent
        # bounds and a bare active_hours_utc evaluated on Saturday.
        spec = _open_orders_spec()
        now = datetime(2026, 6, 27, 16, 0, tzinfo=timezone.utc)  # Sat, in hours
        result = check_freshness(_fake_s3(), spec, now)
        assert result.state == "fresh"
        assert "non-trading day" in result.reason


class TestActiveWindowValidation:

    def test_active_hours_list_coerced_to_tuple(self):
        s = _open_orders_spec(active_hours_utc=[14, 21])
        assert s.active_hours_utc == (14, 21)

    def test_active_hours_on_non_continuous_raises(self):
        with pytest.raises(ValueError, match="active_hours_utc"):
            _spec(active_hours_utc=[14, 21])  # default cadence saturday_sf

    def test_active_hours_bad_bounds_raises(self):
        with pytest.raises(ValueError, match="0 <= start < end <= 24"):
            _open_orders_spec(active_hours_utc=[21, 14])

    def test_active_hours_wrong_length_raises(self):
        with pytest.raises(ValueError, match="2-tuple"):
            _open_orders_spec(active_hours_utc=[14])

    def test_active_hours_end_24_allowed(self):
        s = _open_orders_spec(active_hours_utc=[14, 24])
        assert s.active_hours_utc == (14, 24)


# ── run_calendar (the trading-day-by-default continuous fix) ──────────────────


def _daily_health_spec(**overrides) -> ArtifactSpec:
    """A continuous, fixed-key DAILY (1440) spec modelling
    ``health/daily_data.json`` — written on each weekday + Weekly Freshness SF run,
    never on Sunday. The 2026-06-28 false-positive subject."""
    defaults = {
        "artifact_id": "health_alpha_engine_data",
        "s3_bucket": "bkt",
        "s3_key_template": "health/daily_data.json",
        "cadence": "continuous",
        "interval_minutes": 1440,
        "sla_minutes_after_cron": 60,
        "severity": "warning",
        "owner_repo": "alpha-engine-data",
        "created_at": date(2025, 1, 1),
        "run_calendar": "trading_days",
    }
    defaults.update(overrides)
    return ArtifactSpec(**defaults)


def _health_s3(last_modified: datetime):
    return _fake_s3(head_returns={
        "health/daily_data.json": {"LastModified": last_modified},
    })


class TestRunCalendarTradingDays:
    """Daily trading-day continuous producer: the weekend gap must not be
    counted against it (the bug the alert surfaced 2026-06-28)."""

    def test_sunday_short_circuits_fresh_even_when_absent(self):
        # Sun 2026-06-28 10:00 UTC — producer idle (no Sunday run). Even an
        # absent artifact must NOT alarm.
        now = datetime(2026, 6, 28, 10, 0, tzinfo=timezone.utc)
        result = check_freshness(_fake_s3(), _daily_health_spec(), now)
        assert result.state == "fresh"
        assert "non-trading day" in result.reason

    def test_monday_morning_fresh_with_saturday_write(self):
        # THE FIX. Mon 2026-06-29 10:00 UTC (trading day, pre-12:45 run).
        # Freshest write is Sat 2026-06-27 09:30 — ~48h old. Under the old
        # wall-clock floor (now - 1440 - 60 = ~25h) this flips STALE every
        # Monday morning. Under the trading-day floor it is FRESH (Saturday's
        # write >= previous_trading_day(last_closed) = Thursday).
        sat_write = datetime(2026, 6, 27, 9, 30, tzinfo=timezone.utc)
        now = datetime(2026, 6, 29, 10, 0, tzinfo=timezone.utc)
        result = check_freshness(_health_s3(sat_write), _daily_health_spec(), now)
        assert result.state == "fresh"

    def test_monday_morning_would_be_stale_under_all_days(self):
        # Control: the SAME data under run_calendar="all_days" is stale —
        # proves the trading-day floor (not some other change) is what fixes it.
        sat_write = datetime(2026, 6, 27, 9, 30, tzinfo=timezone.utc)
        now = datetime(2026, 6, 29, 10, 0, tzinfo=timezone.utc)
        result = check_freshness(
            _health_s3(sat_write),
            _daily_health_spec(run_calendar="all_days"),
            now,
        )
        assert result.state == "stale"

    def test_genuine_multiday_miss_caught_stale(self):
        # Wed 2026-07-01 10:00 UTC: newest write still Sat 06-27 (Mon+Tue runs
        # both failed). Trading-day floor = previous_trading_day(last_closed
        # =Tue) = Monday → Saturday < Monday → STALE. The absence backstop
        # still fires; trading-day awareness only tolerates the weekend gap.
        sat_write = datetime(2026, 6, 27, 9, 30, tzinfo=timezone.utc)
        now = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)
        result = check_freshness(_health_s3(sat_write), _daily_health_spec(), now)
        assert result.state == "stale"


class TestRunCalendarAllDays:
    """all_days = the documented wall-clock exception (7-day producers)."""

    def test_weekend_not_short_circuited(self):
        # A genuine 7-day producer (e.g. the changelog aggregator) is judged
        # every calendar day — a Sunday absence is a real gap, not idle.
        now = datetime(2026, 6, 28, 10, 0, tzinfo=timezone.utc)  # Sunday
        spec = _daily_health_spec(
            artifact_id="changelog_daily_view", run_calendar="all_days",
        )
        result = check_freshness(_fake_s3(), spec, now)
        assert result.state == "missing"


class TestRunCalendarResolutionAndValidation:

    def test_default_resolves_all_days(self):
        # Nothing declared ⇒ conservative all_days (no silent flip of an
        # un-migrated continuous spec).
        now = datetime(2026, 6, 28, 10, 0, tzinfo=timezone.utc)  # Sunday
        spec = _daily_health_spec(run_calendar=None)
        result = check_freshness(_fake_s3(), spec, now)
        assert result.state == "missing"  # all_days ⇒ evaluated, absent ⇒ missing

    def test_market_hours_requires_active_hours(self):
        with pytest.raises(ValueError, match="market_hours.*active_hours_utc"):
            _daily_health_spec(run_calendar="market_hours")

    def test_run_calendar_on_non_continuous_raises(self):
        with pytest.raises(ValueError, match="run_calendar.*continuous"):
            _spec(run_calendar="trading_days")  # default cadence saturday_sf

    def test_invalid_run_calendar_value_raises(self):
        with pytest.raises(ValueError, match="run_calendar"):
            _daily_health_spec(run_calendar="weekly")


# ── Phase 2: dependency DAG + root-cause localization ─────────────────────────


def _node(artifact_id: str, *, depends_on=(), severity="critical") -> ArtifactSpec:
    """A registry spec with lineage edges, for DAG tests."""
    return _spec(artifact_id=artifact_id, severity=severity, depends_on=depends_on)


def _pairs(states: dict) -> list:
    """Build ``(spec, CheckResult)`` pairs from ``{spec: state}``."""
    return [(spec, _res(state)) for spec, state in states.items()]


class TestArtifactSpecLineageValidation:
    def test_lists_coerced_to_tuples(self):
        s = _spec(produces=["x", "y"], depends_on=["a", "b"])
        assert s.produces == ("x", "y")
        assert s.depends_on == ("a", "b")

    def test_depends_on_deduped_order_preserving(self):
        s = _spec(depends_on=["a", "b", "a", "c", "b"])
        assert s.depends_on == ("a", "b", "c")

    def test_default_edges_empty(self):
        s = _spec()
        assert s.produces == () and s.depends_on == ()

    def test_self_edge_rejected(self):
        with pytest.raises(ValueError, match="references itself"):
            _spec(artifact_id="a", depends_on=["a"])

    def test_string_instead_of_list_rejected(self):
        with pytest.raises(ValueError, match="list/tuple"):
            _spec(depends_on="a")

    def test_empty_string_edge_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            _spec(depends_on=["a", ""])


class TestBuildDependencyGraph:
    def test_roots_and_leaves(self):
        specs = [_node("a"), _node("b", depends_on=["a"]), _node("c", depends_on=["b"])]
        g = build_dependency_graph(specs)
        assert g.roots == ("a",)
        assert g.leaves == ("c",)
        assert g.upstream("c") == ("b",)
        assert g.downstream("a") == ("b",)

    def test_reachable_leaves_walks_transitively(self):
        specs = [_node("a"), _node("b", depends_on=["a"]), _node("c", depends_on=["b"])]
        g = build_dependency_graph(specs)
        assert g.reachable_leaves("a") == ("c",)

    def test_reachable_leaves_of_a_leaf_is_itself(self):
        g = build_dependency_graph([_node("solo")])
        assert g.reachable_leaves("solo") == ("solo",)

    def test_diamond_reachable_leaves(self):
        specs = [
            _node("a"),
            _node("b", depends_on=["a"]),
            _node("c", depends_on=["a"]),
            _node("d", depends_on=["b", "c"]),
        ]
        g = build_dependency_graph(specs)
        assert g.reachable_leaves("a") == ("d",)
        assert set(g.downstream("a")) == {"b", "c"}

    def test_dangling_edge_raises(self):
        with pytest.raises(ValueError, match="unknown artifact 'ghost'"):
            build_dependency_graph([_node("a", depends_on=["ghost"])])

    def test_cycle_raises(self):
        specs = [
            _node("a", depends_on=["c"]),
            _node("b", depends_on=["a"]),
            _node("c", depends_on=["b"]),
        ]
        with pytest.raises(ValueError, match="dependency cycle"):
            build_dependency_graph(specs)

    def test_duplicate_id_raises(self):
        with pytest.raises(ValueError, match="duplicate artifact_id"):
            build_dependency_graph([_node("a"), _node("a")])


# ── event_driven cadence (config#1718) ────────────────────────────────────────


def _event_spec(**overrides) -> ArtifactSpec:
    """An event_driven config-row spec with a proxy liveness anchor."""
    defaults = {
        "cadence": "event_driven",
        "liveness_via": "optimizer_run_report",
        # event_driven rows never self-page; grace/interval are moot.
        "grace_period_cycles": 0,
    }
    defaults.update(overrides)
    return _spec(**defaults)


class TestEventDrivenSpecValidation:
    """The event_driven ↔ liveness_via coupling is the per-spec chokepoint
    that replaces the grace_period_cycles=999 blind-spot with a contract."""

    def test_event_driven_in_cadence_symbols(self):
        assert "event_driven" in CADENCE_SYMBOLS

    def test_event_driven_requires_liveness_via(self):
        with pytest.raises(ValueError, match="requires liveness_via"):
            _spec(cadence="event_driven", liveness_via=None)

    def test_event_driven_rejects_empty_liveness_via(self):
        with pytest.raises(ValueError, match="requires liveness_via"):
            _spec(cadence="event_driven", liveness_via="")

    def test_event_driven_rejects_self_anchor(self):
        with pytest.raises(ValueError, match="references itself"):
            _spec(
                artifact_id="cfg", cadence="event_driven", liveness_via="cfg"
            )

    def test_non_event_driven_rejects_liveness_via(self):
        with pytest.raises(ValueError, match="only valid for"):
            _spec(cadence="saturday_sf", liveness_via="proxy")

    def test_valid_event_driven_spec(self):
        s = _event_spec(artifact_id="config_research_params")
        assert s.cadence == "event_driven"
        assert s.liveness_via == "optimizer_run_report"

    def test_event_driven_does_not_require_interval(self):
        # Unlike continuous, event_driven carries no interval — construction
        # must not demand one.
        s = _event_spec()
        assert s.interval_minutes is None


class TestEventDrivenCheckFreshness:
    """The core acceptance: an event_driven row never self-pages on age
    ('producer ran and correctly declined to write' ⇒ no alert), while a
    genuinely-dead producer is caught via the liveness_via PROXY going
    stale (not via the event_driven row itself)."""

    def test_declined_to_write_no_artifact_is_fresh_not_missing(self):
        # The producer evaluated its gate and correctly wrote nothing — the
        # config key is 63 days old / absent. Pre-fix this false-alarmed
        # (or needed grace=999); now it short-circuits to fresh.
        spec = _event_spec(artifact_id="config_research_params")
        now = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        result = check_freshness(_fake_s3(), spec, now)  # empty S3 → no object
        assert result.state == "fresh"
        assert "optimizer_run_report" in result.reason
        assert "absence is correct" in result.reason

    def test_stale_artifact_still_fresh_never_pages_on_age(self):
        # Even a 63-day-old written instance must NOT read stale: the row's
        # own age is never the liveness signal.
        spec = _event_spec(artifact_id="config_research_params")
        now = datetime(2026, 7, 4, 18, 0, tzinfo=timezone.utc)
        old = datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc)  # ~63d stale
        s3 = _fake_s3(objects={"path/2026-05-02/file.json": old})
        result = check_freshness(s3, spec, now)
        assert result.state == "fresh"

    def test_liveness_rides_the_proxy_which_can_go_stale(self):
        # The proxy is an ordinary saturday_sf row: when the optimizer STAGE
        # itself stops running, the run-report ages out → the proxy pages.
        # This is the "producer never ran in N cycles → alert" case, now
        # carried by a signal that CAN go stale.
        proxy = _spec(
            artifact_id="optimizer_run_report",
            s3_key_template="optimizer_run/{date}/report.json",
        )
        now = datetime(2026, 7, 4, 18, 0, tzinfo=timezone.utc)
        stale = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)  # >10d old
        s3 = _fake_s3(objects={"optimizer_run/2026-06-01/report.json": stale})
        result = check_freshness(s3, proxy, now)
        assert result.state == "stale"

    def test_event_driven_dedup_key_is_stable_per_day(self):
        spec = _event_spec()
        now = datetime(2026, 7, 4, 3, 0, tzinfo=timezone.utc)
        later = datetime(2026, 7, 4, 21, 0, tzinfo=timezone.utc)
        assert resolve_dedup_key(spec, now) == resolve_dedup_key(spec, later)
        assert "2026-07-04" in resolve_dedup_key(spec, now)

    def test_event_driven_counts_as_satisfied_in_cycle_completion(self):
        # An event_driven required member must not drag a cycle to
        # 'incomplete' just because it correctly declined to write.
        proxy = _spec(
            artifact_id="optimizer_run_report",
            severity="critical",
            s3_key_template="optimizer_run/{date}/report.json",
        )
        cfg = _event_spec(
            artifact_id="config_research_params", severity="critical"
        )
        now = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        fresh_lm = datetime(2026, 5, 30, 11, 0, tzinfo=timezone.utc)
        s3 = _fake_s3(objects={
            "optimizer_run/2026-05-30/report.json": fresh_lm,  # proxy fresh
        })
        pairs = [
            (proxy, check_freshness(s3, proxy, now)),
            (cfg, check_freshness(s3, cfg, now)),  # event_driven → fresh
        ]
        cc = cycle_completion(pairs)
        assert cc.n_required == 2
        assert cc.n_satisfied == 2
        assert cc.state == "complete"
        assert cfg.artifact_id not in cc.stale
        assert cfg.artifact_id not in cc.missing


class TestEventDrivenLivenessAnchorReferentialIntegrity:
    """The registry-global half: build_dependency_graph enforces that every
    event_driven row's liveness_via names a real, non-event_driven anchor —
    so a dangling or chained anchor can never silently blind the fleet."""

    def test_valid_anchor_builds(self):
        specs = [
            _node("optimizer_run_report"),
            _event_spec(artifact_id="cfg", liveness_via="optimizer_run_report"),
        ]
        g = build_dependency_graph(specs)
        assert "cfg" in g.depends_on

    def test_dangling_anchor_raises(self):
        specs = [_event_spec(artifact_id="cfg", liveness_via="ghost")]
        with pytest.raises(ValueError, match="dangling liveness anchor"):
            build_dependency_graph(specs)

    def test_anchor_that_is_itself_event_driven_raises(self):
        specs = [
            _event_spec(artifact_id="proxy", liveness_via="cfg"),
            _event_spec(artifact_id="cfg", liveness_via="proxy"),
        ]
        with pytest.raises(ValueError, match="itself event_driven"):
            build_dependency_graph(specs)


class TestLeafAlertDecisions:
    def test_linear_chain_pages_only_root(self):
        a, b, c = _node("a"), _node("b", depends_on=["a"]), _node("c", depends_on=["b"])
        d = leaf_alert_decisions(_pairs({a: "missing", b: "missing", c: "missing"}))
        assert d["a"].should_page and d["a"].classification == "failed"
        assert not d["b"].should_page and d["b"].classification == "blocked"
        assert not d["c"].should_page and d["c"].classification == "blocked"
        assert d["c"].root_cause_ids == ("a",)

    def test_gap_below_a_fresh_root_pages_the_broken_stage(self):
        a, b, c = _node("a"), _node("b", depends_on=["a"]), _node("c", depends_on=["b"])
        d = leaf_alert_decisions(_pairs({a: "fresh", b: "missing", c: "missing"}))
        assert not d["a"].should_page
        assert d["b"].should_page and d["b"].classification == "failed"
        assert not d["c"].should_page and d["c"].root_cause_ids == ("b",)

    def test_healed_downstream_suppresses_upstream_miss(self):
        # Recovery produced the leaf directly ⇒ the upstream miss is moot.
        a, b, c = _node("a"), _node("b", depends_on=["a"]), _node("c", depends_on=["b"])
        d = leaf_alert_decisions(_pairs({a: "missing", b: "missing", c: "fresh"}))
        assert not d["a"].should_page and "healed" in d["a"].reason
        assert not d["b"].should_page
        assert not d["c"].should_page and d["c"].classification == "ok"

    def test_partial_diamond_root_still_pages_when_one_leaf_lost(self):
        a = _node("a")
        b = _node("b", depends_on=["a"])  # leaf, missing
        c = _node("c", depends_on=["a"])  # leaf, fresh (recovered)
        d = leaf_alert_decisions(_pairs({a: "missing", b: "missing", c: "fresh"}))
        assert d["a"].should_page  # b never landed ⇒ a's failure is real
        assert not d["b"].should_page and d["b"].root_cause_ids == ("a",)
        assert not d["c"].should_page

    def test_unedged_spec_behaviour_preserved(self):
        # No edges ⇒ root+leaf ⇒ a confirmed miss pages exactly as today.
        a = _node("a")
        d = leaf_alert_decisions(_pairs({a: "missing"}))
        assert d["a"].should_page and d["a"].classification == "failed"

    def test_stale_counts_as_gap(self):
        a, b = _node("a"), _node("b", depends_on=["a"])
        d = leaf_alert_decisions(_pairs({a: "stale", b: "missing"}))
        assert d["a"].should_page
        assert not d["b"].should_page and d["b"].root_cause_ids == ("a",)

    def test_probe_failed_upstream_does_not_suppress_downstream_miss(self):
        # An unconfirmable upstream must NOT block a real downstream miss.
        a, b = _node("a"), _node("b", depends_on=["a"])
        d = leaf_alert_decisions(_pairs({a: "probe_failed", b: "missing"}))
        assert d["b"].should_page and d["b"].classification == "failed"
        assert d["a"].should_page and d["a"].classification == "degraded"

    def test_probe_failed_root_pages_as_degraded(self):
        a = _node("a")
        d = leaf_alert_decisions(_pairs({a: "probe_failed"}))
        assert d["a"].should_page and d["a"].classification == "degraded"

    def test_probe_failed_blocked_by_confirmed_upstream_gap(self):
        a, b = _node("a"), _node("b", depends_on=["a"])
        d = leaf_alert_decisions(_pairs({a: "missing", b: "probe_failed"}))
        assert d["a"].should_page
        assert not d["b"].should_page and d["b"].classification == "blocked"

    def test_grace_period_counts_as_satisfied(self):
        a, b = _node("a"), _node("b", depends_on=["a"])
        d = leaf_alert_decisions(_pairs({a: "grace_period", b: "missing"}))
        assert not d["a"].should_page and d["a"].classification == "ok"
        # b's upstream is satisfied (grace) ⇒ b is its own root failure.
        assert d["b"].should_page and d["b"].classification == "failed"

    def test_diamond_two_roots_localized(self):
        # d depends on two independently-broken roots.
        a = _node("a")
        b = _node("b")
        d_ = _node("d", depends_on=["a", "b"])
        out = leaf_alert_decisions(_pairs({a: "missing", b: "missing", d_: "missing"}))
        assert out["a"].should_page and out["b"].should_page
        assert not out["d"].should_page
        assert out["d"].root_cause_ids == ("a", "b")

    def test_accepts_prebuilt_graph(self):
        a, b = _node("a"), _node("b", depends_on=["a"])
        g = build_dependency_graph([a, b])
        assert isinstance(g, DependencyGraph)
        d = leaf_alert_decisions(_pairs({a: "missing", b: "missing"}), graph=g)
        assert d["a"].should_page and not d["b"].should_page


class TestLocalizeRootCauses:
    def test_projection_omits_satisfied_and_maps_gaps(self):
        a, b, c = _node("a"), _node("b", depends_on=["a"]), _node("c", depends_on=["b"])
        rc = localize_root_causes(_pairs({a: "missing", b: "missing", c: "missing"}))
        assert rc == {"a": ("a",), "b": ("a",), "c": ("a",)}

    def test_healed_root_still_localizes_but_does_not_page(self):
        a, b = _node("a"), _node("b", depends_on=["a"])
        # a healed (b fresh) ⇒ a omitted only if satisfied; a is 'failed' class
        # (suppressed), so it still appears in the localization map.
        rc = localize_root_causes(_pairs({a: "missing", b: "fresh"}))
        assert "b" not in rc  # satisfied leaf omitted
        assert rc["a"] == ("a",)
