"""Tests for the time-shift plugin (alpha-engine-config#6923)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from nousergon_lib.pytest_time_shift import shifted_types


def test_today_moves_forward_by_the_shift():
    ShiftedDate, _ = shifted_types(180)
    assert ShiftedDate.today() == date.today() + timedelta(days=180)


def test_now_moves_forward_by_the_shift():
    _, ShiftedDateTime = shifted_types(30)
    delta = ShiftedDateTime.now() - datetime.now()
    assert timedelta(days=29, hours=23) < delta < timedelta(days=30, minutes=1)


def test_utcnow_moves_too():
    _, ShiftedDateTime = shifted_types(10)
    delta = ShiftedDateTime.utcnow() - datetime.utcnow()
    assert timedelta(days=9, hours=23) < delta < timedelta(days=10, minutes=1)


def test_a_negative_shift_moves_backward():
    """Useful for the mirror question: does this test only pass because the
    corpus happens to be old enough today?"""
    ShiftedDate, _ = shifted_types(-365)
    assert ShiftedDate.today() == date.today() - timedelta(days=365)


def test_zero_is_identity():
    ShiftedDate, ShiftedDateTime = shifted_types(0)
    assert ShiftedDate.today() == date.today()


def test_construction_and_arithmetic_are_untouched():
    """A test that never asks the clock must behave identically."""
    ShiftedDate, ShiftedDateTime = shifted_types(500)

    assert ShiftedDate(2026, 4, 12) == date(2026, 4, 12)
    assert ShiftedDate(2026, 4, 12) + timedelta(days=8) == date(2026, 4, 20)
    assert ShiftedDate(2026, 4, 20) > ShiftedDate(2026, 4, 12)
    assert ShiftedDate.fromisoformat("2026-04-12") == date(2026, 4, 12)
    assert ShiftedDateTime(2026, 4, 12, 9, 30) == datetime(2026, 4, 12, 9, 30)


def test_the_shifted_types_are_still_real_dates():
    """isinstance checks and pandas/pydantic coercion must keep working."""
    ShiftedDate, ShiftedDateTime = shifted_types(1)
    assert issubclass(ShiftedDate, date)
    assert issubclass(ShiftedDateTime, datetime)
    assert isinstance(ShiftedDate.today(), date)


def test_it_reproduces_the_defect_it_exists_to_find():
    """The 2026-08-11 shape, reduced.

    A fixture pinned 8 days apart, and a window of 120 days counted from the
    clock: both rows are visible now, and only one survives once the clock
    moves past the older one.
    """
    fixture = [date.today() - timedelta(days=112), date.today() - timedelta(days=104)]

    def visible(today, lookback_days=120):
        cutoff = today - timedelta(days=lookback_days)
        return [d for d in fixture if d >= cutoff]

    assert len(visible(date.today())) == 2

    ShiftedDate, _ = shifted_types(10)
    assert len(visible(ShiftedDate.today())) == 1, (
        "the shifted clock must expose the fixture row that ages out — "
        "this is exactly the assertion that broke three tests on 2026-08-11"
    )
