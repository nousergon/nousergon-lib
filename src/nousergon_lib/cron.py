"""Read GitHub Actions cron schedules and derive the cadence they keep.

Lifted from nous-ergon-ops ``scripts/authority_surface.py``
(``schedule_crons`` / ``cron_interval_minutes`` and their helpers) so that
consumers can share one implementation. The behaviour is the ops contract,
unchanged: ``tests/test_cron.py`` carries that repo's parametrized cases
verbatim, and a divergence from them is a defect here, not an improvement.

The interval answers "how long until this SHOULD have fired once" — the
input to a staleness threshold. It is not a scheduler.
"""

from __future__ import annotations

import re

__all__ = ["cron_interval_hours", "cron_interval_minutes", "schedule_crons"]

# A cron LINE, with the expression left to `_cron_expr` to unwrap. "The rest of
# the line" rather than a character class: a class excluding `#` silently
# drops every cron carrying a trailing comment.
_CRON_RE = re.compile(r"^\s*-\s*cron:\s*(.+?)\s*$", re.M)
_SCHEDULE_RE = re.compile(r"^\s*schedule:\s*$", re.M)
_CRON_ENTRY_RE = re.compile(r"^\s*-\s*cron:", re.M)

_WEEK_MINUTES = 10080


def _has_schedule(text: str) -> bool:
    """True when the workflow declares a cron trigger.

    Deliberately textual: parsing the YAML resolves ``on:`` to True in some
    documents (the Norway problem), which drops the trigger block entirely.
    """
    return bool(_SCHEDULE_RE.search(text)) and bool(_CRON_ENTRY_RE.search(text))


def _cron_expr(raw: str) -> str:
    """One cron field value, unquoted and stripped of a trailing comment.

    A comment is only stripped OUTSIDE quotes: ``'0 7 * * *'  # daily`` loses
    the comment, ``"0 7 * * *"`` keeps its five fields intact, and an unquoted
    ``0 7 * * * # daily`` loses everything from the hash.
    """
    raw = raw.strip()
    if raw[:1] in ("'", '"'):
        quote = raw[0]
        end = raw.find(quote, 1)
        if end != -1:
            return raw[1:end]
        return raw[1:]
    return raw.split("#", 1)[0].strip()


def schedule_crons(text: str) -> list[str]:
    """Every cron expression a workflow file declares.

    Returns ``[]`` when the file declares no schedule at all. A file that does
    declare ``schedule:`` but whose crons are unreadable is a different
    condition; callers check for it separately.
    """
    if not _has_schedule(text):
        return []
    return [expr for expr in (_cron_expr(match.group(1)) for match in _CRON_RE.finditer(text)) if expr]


def _field_fires(field: str, period: int) -> int:
    """How many times one cron field fires across its full period (60/24/31/7).

    Handles ``*``, ``*/N``, comma lists, ranges and a bare value. Unparseable
    input counts as ONE fire, which overstates the interval; that is the answer
    only where the expression could not be read at all.
    """
    field = field.strip()
    if field == "*":
        return period
    if "," in field:
        return sum(_field_fires(part, period) for part in field.split(","))
    base, _, step_text = field.partition("/")
    step = 1
    if step_text:
        try:
            step = max(1, int(step_text))
        except ValueError:
            return 1
    if base == "*":
        span = period
    elif "-" in base:
        low, _, high = base.partition("-")
        try:
            span = max(1, int(high) - int(low) + 1)
        except ValueError:
            return 1
    else:
        return 1
    return max(1, span // step)


def _field_values(field: str, low: int, high: int) -> set[int] | None:
    """The exact values one cron field selects in ``[low, high]``, or None when
    the field cannot be read exactly (a name like ``MON``, a malformed step).

    Handles ``*``, ``*/N``, ``a``, ``a-b``, ``a-b/N`` and comma lists of those.
    None is never guessed into a value: the caller falls back to the coarse
    average for the whole workflow instead.
    """
    values: set[int] = set()
    for part in field.strip().split(","):
        base, _, step_text = part.partition("/")
        try:
            step = int(step_text) if step_text else 1
            if step < 1:
                return None
            if base == "*":
                first, last = low, high
            elif "-" in base:
                start, _, end = base.partition("-")
                first, last = int(start), int(end)
            else:
                first = int(base)
                last = high if step_text else first
        except ValueError:
            return None
        if first < low or last > high or first > last:
            return None
        values.update(range(first, last + 1, step))
    return values or None


def _week_fires(expr: str) -> list[int] | None:
    """Every fire of one cron expression as a minute-of-week (Sunday 00:00 = 0),
    or None when the expression is not exactly expandable over a week.

    Only expressions whose day-of-month and month are ``*`` repeat weekly, so
    only those expand; a monthly cron returns None and takes the average.
    """
    fields = expr.split()
    if len(fields) != 5:
        return None
    minute, hour, day_of_month, month, day_of_week = fields
    if day_of_month != "*" or month != "*":
        return None
    minutes = _field_values(minute, 0, 59)
    hours = _field_values(hour, 0, 23)
    days = _field_values(day_of_week, 0, 7)
    if minutes is None or hours is None or days is None:
        return None
    days = {day % 7 for day in days}  # cron's 7 is Sunday, same as 0
    return sorted(day * 1440 + hour_value * 60 + minute_value for day in days for hour_value in hours for minute_value in minutes)


def _cron_average_interval_minutes(crons: list[str]) -> int | None:
    """The coarse fallback: the AVERAGE interval of the fastest expression.

    Exact only for evenly spaced crons. For an irregular one it understates the
    longest gap (``0 3,6,8 * * *`` averages 480 min but goes 19 h overnight),
    so it is used only where `_week_fires` cannot expand an expression.
    """
    best: int | None = None
    for expr in crons:
        fields = expr.split()
        if len(fields) != 5:
            continue
        minute, hour, day_of_month, _month, day_of_week = fields
        per_day = _field_fires(minute, 60) * _field_fires(hour, 24)
        if day_of_week != "*":
            interval = 10080.0 / (per_day * _field_fires(day_of_week, 7))
        elif day_of_month != "*":
            interval = 43200.0 / (per_day * _field_fires(day_of_month, 31))
        else:
            interval = 1440.0 / per_day
        rounded = max(1, int(round(interval)))
        best = rounded if best is None else min(best, rounded)
    return best


def cron_interval_minutes(crons: list[str]) -> int | None:
    """The LONGEST gap between consecutive fires across a workflow's cron
    expressions, in whole minutes, or None when nothing parseable was given.

    The fires of every expression are unioned over one week (wrapping into the
    next week) before the gaps are measured, so interleaving expressions are
    read together and ``*/N`` fires every N minutes. If ANY expression does not
    repeat weekly (day-of-month or month restricted) or cannot be read exactly
    (for example a day name), the whole workflow falls back to the minimum of
    the per-expression average intervals.
    """
    fires: set[int] = set()
    for expr in crons:
        week = _week_fires(expr)
        if week is None:
            return _cron_average_interval_minutes(crons)
        fires.update(week)
    if not fires:
        return None
    ordered = sorted(fires)
    gaps = [later - earlier for earlier, later in zip(ordered, ordered[1:])]
    gaps.append(ordered[0] + _WEEK_MINUTES - ordered[-1])
    return max(gaps)


def cron_interval_hours(crons: list[str]) -> float | None:
    """Compatibility wrapper: :func:`cron_interval_minutes` in hours."""
    minutes = cron_interval_minutes(crons)
    return minutes / 60 if minutes is not None else None
