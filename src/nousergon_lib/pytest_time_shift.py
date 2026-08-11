"""Run a test suite as if it were N days from now (alpha-engine-config#6923).

The bug class this exists to find: a **fixed date literal inside a relative
window**. A fixture seeds `2026-04-12`; the code under test filters on
`lookback_days=120` counted from *today*. Both are correct in isolation, the
test passes for months, and then one morning the literal turns 121 days old and
the assertion fails — with no code change, on whoever's PR happens to run CI
that day, presenting as a row-count mismatch that reads like a logic bug.

That is not hypothetical: it happened on 2026-08-11 to three tests in
`crucible-backtester` (config#6923), and the second fixture date in the same
module was due to expire seven days later.

Static detection does not work. "A test file contains a date literal and
imports a module that mentions `timedelta`" matches hundreds of files, almost
all of them fine — a scan over seven repos on 2026-08-11 produced ~100 such
pairs, every one of which passed. The distinguishing property is not textual;
it is *what the test does when the clock moves*.

So move the clock. A test that passes today and fails at ``--time-shift-days``
is, by construction, coupled to the wall clock. There is no heuristic and no
false-positive class to tune.

Usage — as a scheduled job, never on the PR path (a red that only appears in
the future must not block a merge today)::

    pytest tests/ --time-shift-days=180

Mechanics: ``datetime.date`` and ``datetime.datetime`` are replaced with
subclasses whose ``today()`` / ``now()`` / ``utcnow()`` return the shifted
instant. Construction, arithmetic and comparison are untouched, so a test that
never asks the clock behaves identically.

**Run it with a per-test timeout.** Wall-clock coupling does not only surface
as a failed assertion: measured on `crucible-backtester` at ``+400`` days, a
module that passes in seconds today ran past ten minutes without finishing —
code that iterates or retries toward a date condition the shift pushes out of
reach. A hang is a finding, but an un-timed job cannot report it, so pair this
with ``pytest-timeout`` (``--timeout=120``) and treat a timeout as a hit.

**Known limit, stated rather than hidden:** C extensions that captured the
original types before the shift (pandas' own `Timestamp.now`, for instance)
keep their own clock. This finds Python-level wall-clock coupling, which is
where this bug class lives — not every possible source of it. A clean run is
evidence, not proof.
"""

from __future__ import annotations

import datetime as _datetime_module
from datetime import date as _RealDate
from datetime import datetime as _RealDateTime
from datetime import timedelta

__all__ = ["pytest_addoption", "pytest_configure", "shifted_types", "SHIFT_OPTION"]

SHIFT_OPTION = "--time-shift-days"


def shifted_types(days: int):
    """(date_subclass, datetime_subclass) whose clock readers are shifted."""
    delta = timedelta(days=days)

    class ShiftedDate(_RealDate):
        @classmethod
        def today(cls):
            return _RealDate.today() + delta

    class ShiftedDateTime(_RealDateTime):
        @classmethod
        def now(cls, tz=None):
            return _RealDateTime.now(tz) + delta

        @classmethod
        def utcnow(cls):
            return _RealDateTime.utcnow() + delta

        @classmethod
        def today(cls):
            return _RealDateTime.today() + delta

    return ShiftedDate, ShiftedDateTime


def pytest_addoption(parser):  # pragma: no cover - pytest hook wiring
    parser.addoption(
        SHIFT_OPTION,
        action="store",
        type=int,
        default=0,
        help=(
            "Run the suite as if it were N days from now, to surface tests "
            "coupled to the wall clock (alpha-engine-config#6923). Intended "
            "for a scheduled job, not the PR path."
        ),
    )


def pytest_configure(config):  # pragma: no cover - pytest hook wiring
    days = config.getoption(SHIFT_OPTION.lstrip("-").replace("-", "_"), 0)
    if not days:
        return
    shifted_date, shifted_datetime = shifted_types(days)
    _datetime_module.date = shifted_date
    _datetime_module.datetime = shifted_datetime
    config.addinivalue_line(
        "markers", "time_shift: suite is running under a shifted clock"
    )
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(
            f"[time-shift] clock advanced {days}d — a failure OR A HANG here is "
            f"a test coupled to the wall clock, not a regression in today's "
            f"code. Run with --timeout so a hang is reportable (config#6923)",
            yellow=True,
        )
