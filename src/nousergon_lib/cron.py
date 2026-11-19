"""Helpers for reading GitHub Actions cron schedules."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, timedelta
from typing import cast

_CRON_LINE = re.compile(r"^\s*(?:-\s*)?cron\s*:\s*(.*)$")
_FIELD_LIMITS = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))


def _uncomment(value: str) -> str:
    quote: str | None = None
    for index, char in enumerate(value):
        if char in "'\"":
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
        elif char == "#" and quote is None:
            return value[:index].rstrip()
    return value.strip()


def schedule_crons(text: str) -> list[str]:
    """Extract cron expressions from the ``schedule`` entries in YAML text."""
    crons: list[str] = []
    for line in text.splitlines():
        match = _CRON_LINE.match(line)
        if match is None:
            continue
        value = _uncomment(match.group(1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if value:
            crons.append(value)
    return crons


def _field_values(field: str, low: int, high: int) -> set[int] | None:
    values: set[int] = set()
    for part in field.split(","):
        base, separator, step_text = part.partition("/")
        if separator:
            if not step_text.isdigit() or int(step_text) == 0:
                return None
            step = int(step_text)
        else:
            step = 1

        if base == "*":
            start, end = low, high
        elif "-" in base:
            start_text, end_text = base.split("-", 1)
            if not start_text.isdigit() or not end_text.isdigit():
                return None
            start, end = int(start_text), int(end_text)
        elif base.isdigit() and not separator:
            start = end = int(base)
        else:
            return None
        if start < low or end > high or start > end:
            return None
        values.update(range(start, end + 1, step))
    return values


def _parse_cron(cron: str) -> tuple[set[int], set[int], set[int], set[int], set[int]] | None:
    fields = cron.split()
    if len(fields) != 5:
        return None
    parsed = tuple(_field_values(field, *limits) for field, limits in zip(fields, _FIELD_LIMITS))
    if any(values is None for values in parsed):
        return None
    minute, hour, day, month, weekday = cast(tuple[set[int], set[int], set[int], set[int], set[int]], parsed)
    return minute, hour, day, month, {0 if value == 7 else value for value in weekday}


def cron_interval_minutes(crons: Iterable[str]) -> int | None:
    """Return the shortest interval between firings in a union of crons."""
    parsed = [_parse_cron(cron) for cron in crons]
    if not parsed or any(schedule is None for schedule in parsed):
        return None
    schedules = cast(list[tuple[set[int], set[int], set[int], set[int], set[int]]], parsed)

    previous: int | None = None
    shortest: int | None = None
    current = date(2024, 1, 1)
    end = date(2032, 1, 1)
    while current < end:
        midnight = (current - date(1970, 1, 1)).days * 1440
        times: set[int] = set()
        for schedule in schedules:
            minutes, hours, days, months, weekdays = schedule
            cron_weekday = (current.weekday() + 1) % 7
            day_matches = current.day in days
            weekday_matches = cron_weekday in weekdays
            day_wildcard = len(days) == 31
            weekday_wildcard = len(weekdays) == 7
            if current.month not in months:
                continue
            if not (
                day_matches and weekday_matches if day_wildcard or weekday_wildcard else day_matches or weekday_matches
            ):
                continue
            times.update(hour * 60 + minute for hour in hours for minute in minutes)
        for minute_of_day in sorted(times):
            firing = midnight + minute_of_day
            if previous is not None:
                gap = firing - previous
                shortest = gap if shortest is None else min(shortest, gap)
                if shortest == 1:
                    return 1
            previous = firing
        current += timedelta(days=1)
    return shortest


def cron_interval_hours(crons: Iterable[str]) -> float | None:
    """Compatibility wrapper returning the shortest interval in hours."""
    minutes = cron_interval_minutes(crons)
    return None if minutes is None else minutes / 60
