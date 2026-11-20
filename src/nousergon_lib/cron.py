"""Helpers for reading and comparing GitHub Actions cron schedules."""

from __future__ import annotations

import re

_CRON_LINE = re.compile(r"^\s*-?\s*cron\s*:\s*(.*?)\s*$")
_DAY_NAMES = {
    "SUN": 0,
    "MON": 1,
    "TUE": 2,
    "WED": 3,
    "THU": 4,
    "FRI": 5,
    "SAT": 6,
}


def _without_comment(value: str) -> str:
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
    """Return cron expressions from the ``schedule`` entries in workflow text."""
    crons: list[str] = []
    for line in text.splitlines():
        match = _CRON_LINE.match(line)
        if match is None:
            continue
        value = _without_comment(match.group(1)).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if value:
            crons.append(value)
    return crons


def _number(value: str, names: dict[str, int] | None = None) -> int | None:
    if names is not None and value.upper() in names:
        return names[value.upper()]
    return int(value) if value.isdigit() else None


def _field_values(
    field: str,
    minimum: int,
    maximum: int,
    *,
    names: dict[str, int] | None = None,
) -> set[int] | None:
    values: set[int] = set()
    for part in field.split(","):
        base, separator, step_text = part.partition("/")
        if separator:
            if not step_text.isdigit() or int(step_text) < 1:
                return None
            step = int(step_text)
        else:
            step = 1
        if base in ("*", "?"):
            start, end = minimum, maximum
        elif "-" in base:
            start_text, end_text = base.split("-", 1)
            start = _number(start_text, names)
            end = _number(end_text, names)
            if start is None or end is None or start > end:
                return None
        else:
            value = _number(base, names)
            if value is None:
                return None
            start = end = value
        if start < minimum or end > maximum:
            return None
        values.update(range(start, end + 1, step))
    return values or None


def _weekly_fires(expr: str) -> set[int] | None:
    fields = expr.split()
    if len(fields) != 5 or fields[2] not in ("*", "?") or fields[3] != "*":
        return None
    minutes = _field_values(fields[0], 0, 59)
    hours = _field_values(fields[1], 0, 23)
    days = _field_values(fields[4], 0, 7, names=_DAY_NAMES)
    if minutes is None or hours is None or days is None:
        return None
    days = {day % 7 for day in days}
    return {day * 1440 + hour * 60 + minute for day in days for hour in hours for minute in minutes}


def _average_interval(expr: str) -> float | None:
    fields = expr.split()
    if len(fields) != 5:
        return None
    minutes = _field_values(fields[0], 0, 59)
    hours = _field_values(fields[1], 0, 23)
    month_days = _field_values(fields[2], 1, 30)
    months = _field_values(fields[3], 1, 12)
    week_days = _field_values(fields[4], 0, 7, names=_DAY_NAMES)
    if (
        minutes is None
        or hours is None
        or month_days is None
        or months is None
        or week_days is None
    ):
        return None
    week_days = {day % 7 for day in week_days}
    fires_per_month = (
        len(minutes)
        * len(hours)
        * len(month_days)
        * len(months)
        / 12
        * len(week_days)
        / 7
    )
    return 43200 / fires_per_month if fires_per_month else None


def cron_interval_minutes(crons: list[str]) -> int | None:
    """Return the longest interval between firings in a union of schedules.

    Weekly schedules are evaluated together, including the wrap into the next
    week. Schedules that cannot be represented over a week use their average
    firing interval instead.
    """
    if not crons:
        return None
    fires: set[int] = set()
    averages: list[float] = []
    for expr in crons:
        weekly = _weekly_fires(expr)
        if weekly is None:
            average = _average_interval(expr)
            if average is None:
                return None
            averages.append(average)
        else:
            fires.update(weekly)
    if averages:
        rates = sum(1 / average for average in averages)
        if fires:
            rates += len(fires) / 10080
        return round(1 / rates)
    ordered = sorted(fires)
    if not ordered:
        return None
    gaps = [later - earlier for earlier, later in zip(ordered, ordered[1:])]
    gaps.append(10080 + ordered[0] - ordered[-1])
    return max(gaps)


def cron_interval_hours(crons: list[str]) -> float | None:
    """Return :func:`cron_interval_minutes` converted to hours."""
    minutes = cron_interval_minutes(crons)
    return minutes / 60 if minutes is not None else None
