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

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from unittest import mock

import pytest

from nousergon_lib.artifact_freshness import (
    ArtifactSpec,
    CADENCE_SYMBOLS,
    CheckResult,
    CycleCompletion,
    check_freshness,
    cycle_completion,
    resolve_current_cycle,
    resolve_dedup_key,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _spec(**overrides) -> ArtifactSpec:
    """Build a baseline saturday_sf spec. Override fields per-test."""
    defaults = dict(
        artifact_id="test_artifact",
        s3_bucket="bkt",
        s3_key_template="path/{date}/file.json",
        cadence="saturday_sf",
        sla_minutes_after_cron=180,  # 3hr after Sat 09:00 UTC = 12:00 UTC
        severity="warning",
        owner_repo="alpha-engine-test",
        created_at=date(2025, 1, 1),  # ancient — past any grace
    )
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
        # Recency model: the freshest instance is from 5/23 (last week's
        # cycle), but now is Sat 5/30 — this week's cron tick is 5/30 09:00,
        # floor = 5/30 09:00 - 5d = 5/25 09:00. 5/23 < 5/25 ⇒ STALE (this
        # week's artifact genuinely missing). A within-5d off-cycle instance
        # would read fresh; a full prior-week instance does not.
        now = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        s3 = _fake_s3(objects={
            "path/2026-05-23/file.json":
                datetime(2026, 5, 23, 10, 0, tzinfo=timezone.utc),
        })
        result = check_freshness(s3, _spec(), now)
        assert result.state == "stale"
        # floor 5/25 09:00 - newest 5/23 10:00 = 47h = 2820min before floor.
        assert result.sla_violated_by_minutes == 2820
        assert result.last_modified == datetime(2026, 5, 23, 10, 0, tzinfo=timezone.utc)

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
        s3 = _fake_s3(objects={
            "recovery/2026-05-23/file.json":
                datetime(2026, 5, 23, 10, 0, tzinfo=timezone.utc),
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
    a cycle-resolution + dedup-key-label branch is the failure mode."""
    assert CADENCE_SYMBOLS == frozenset(
        {"saturday_sf", "weekday_sf", "eod_sf", "continuous"}
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
    defaults = dict(
        artifact_id="open_orders_latest",
        s3_bucket="bkt",
        s3_key_template="trades/open_orders/latest.json",
        cadence="continuous",
        interval_minutes=30,
        sla_minutes_after_cron=15,
        severity="warning",
        owner_repo="alpha-engine",
        created_at=date(2025, 1, 1),
        active_trading_days_only=True,
        active_hours_utc=[14, 21],
    )
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

    def test_hours_only_no_trading_day_gate(self):
        # active_hours_utc without active_trading_days_only: a weekend hour
        # INSIDE the window is still evaluated (not short-circuited).
        spec = _open_orders_spec(active_trading_days_only=False)
        now = datetime(2026, 6, 27, 16, 0, tzinfo=timezone.utc)  # Sat, in hours
        result = check_freshness(_fake_s3(), spec, now)
        assert result.state == "missing"


class TestActiveWindowValidation:

    def test_active_hours_list_coerced_to_tuple(self):
        s = _open_orders_spec(active_hours_utc=[14, 21])
        assert s.active_hours_utc == (14, 21)

    def test_active_hours_on_non_continuous_raises(self):
        with pytest.raises(ValueError, match="active_hours_utc"):
            _spec(active_hours_utc=[14, 21])  # default cadence saturday_sf

    def test_active_trading_days_only_on_non_continuous_raises(self):
        with pytest.raises(ValueError, match="active_trading_days_only"):
            _spec(active_trading_days_only=True)

    def test_active_hours_bad_bounds_raises(self):
        with pytest.raises(ValueError, match="0 <= start < end <= 24"):
            _open_orders_spec(active_hours_utc=[21, 14])

    def test_active_hours_wrong_length_raises(self):
        with pytest.raises(ValueError, match="2-tuple"):
            _open_orders_spec(active_hours_utc=[14])

    def test_active_hours_end_24_allowed(self):
        s = _open_orders_spec(active_hours_utc=[14, 24])
        assert s.active_hours_utc == (14, 24)
