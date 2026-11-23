import pytest

from nousergon_lib.cron import (
    cron_interval_hours,
    cron_interval_minutes,
    schedule_crons,
)


def test_schedule_crons_strips_only_comments_outside_quotes() -> None:
    text = """
    schedule:
      - cron: '0 7 * * 1-5' # weekday run
      - cron: "15 8 * * *" # quoted expression
      - cron: '30 9 * * * # retained'
    """

    assert schedule_crons(text) == [
        "0 7 * * 1-5",
        "15 8 * * *",
        "30 9 * * * # retained",
    ]


@pytest.mark.parametrize(
    ("crons", "expected"),
    [
        (["30 7 * * 1-5"], 4320),
        (["0 3,6,8 * * *"], 1140),
        (["5,20,35,50 7-8 * * 4,5,6"], 7095),
        (["0 8-16/4 * * *"], 960),
        (["0 0 1 * *"], 43200),
        (["0 7 * * 1-5", "0 7 * * 6"], 2880),
        (["0 12 * * MON"], 10080),
        (["*/15 * * * *"], 15),
        (["0 12 * * 7"], 10080),
    ],
)
def test_cron_interval_minutes(crons: list[str], expected: int) -> None:
    assert cron_interval_minutes(crons) == expected


def test_cron_interval_rejects_unknown_expression() -> None:
    assert cron_interval_minutes(["not a cron"]) is None
    assert cron_interval_minutes([]) is None


def test_cron_interval_hours_is_compatibility_wrapper() -> None:
    assert cron_interval_hours(["*/30 * * * *"]) == 0.5


# --- The nous-ergon-ops contract, ported verbatim ---------------------------
# Source: nous-ergon-ops tests/test_publish_workflow_checks.py "cron cadence"
# (cases for scripts/authority_surface.py cron_interval_minutes). This module
# lifts that function; every case below must keep the ops answer.

DAILY = "on:\n  schedule:\n    - cron: '0 7 * * *'\n"
UNSCHEDULED = "on:\n  pull_request:\n"
NO_CRON = "on:\n  schedule:\n"


@pytest.mark.parametrize(
    ("expr", "minutes"),
    [
        ("0 7 * * *", 1440),
        ("*/15 * * * *", 15),
        ("0 */6 * * *", 360),
        ("0 9 * * 6", 10080),
        # A weekday cron sleeps Fri 07:30 -> Mon 07:30.
        ("30 7 * * 1-5", 4320),
        # Day-of-month restricted: not weekly, so the coarse average stands.
        ("0 0 1 * *", 43200),
    ],
)
def test_ops_cron_interval_minutes(expr: str, minutes: int) -> None:
    assert cron_interval_minutes([expr]) == minutes


@pytest.mark.parametrize(
    ("expr", "minutes"),
    [
        # 08:00 -> 03:00 next day is 19 h.
        ("0 3,6,8 * * *", 1140),
        # Sat 08:50 -> Thu 07:05.
        ("5,20,35,50 7-8 * * 4,5,6", 7095),
        # cron's 7 is Sunday, the same day as 0.
        ("0 9 * * 7", 10080),
        ("0 9 * * 0,7", 10080),
        ("0 8-16/4 * * *", 960),
    ],
)
def test_ops_cron_interval_is_the_longest_gap_for_an_irregular_schedule(expr: str, minutes: int) -> None:
    assert cron_interval_minutes([expr]) == minutes


def test_ops_cron_interval_unions_expressions_before_measuring_the_gap() -> None:
    # Weekdays at 07:00 plus Saturday at 07:00: the longest gap is Sat -> Mon.
    assert cron_interval_minutes(["0 7 * * 1-5", "0 7 * * 6"]) == 2880


def test_ops_an_unexpandable_expression_falls_back_to_the_average() -> None:
    assert cron_interval_minutes(["0 12 * * MON"]) == 10080
    assert cron_interval_minutes(["0 7 * * *", "0 0 1 * *"]) == 1440
    assert cron_interval_minutes(["not a cron"]) is None


def test_ops_cron_interval_takes_the_shortest_across_expressions() -> None:
    assert cron_interval_minutes(["0 7 * * *", "*/10 * * * *"]) == 10


def test_ops_schedule_crons_reads_only_scheduled_workflows() -> None:
    assert schedule_crons(UNSCHEDULED) == []
    assert schedule_crons(DAILY) == ["0 7 * * *"]


# --- Fallback is the MIN of per-expression averages, never a rate sum -------


@pytest.mark.parametrize(
    ("crons", "minutes"),
    [
        # A rate sum would give 1394 for the first and 9 for the third.
        (["0 7 * * *", "0 0 1 * *"], 1440),
        (["0 0 1 * *", "0 7 * * *"], 1440),
        (["*/10 * * * *", "0 0 1 * *"], 10),
        # A day name is not read exactly, so the whole workflow averages:
        # min(10080 / 5 = 2016, 1440) rather than an exact union.
        (["0 12 * * MON-FRI"], 10080),
        (["0 12 * * MON", "0 7 * * *"], 1440),
        # An unreadable expression alongside a weekly one: the average of the
        # readable one stands and the unreadable one is skipped.
        (["0 7 * * *", "not a cron"], 1440),
    ],
)
def test_fallback_is_the_fastest_per_expression_average(crons: list[str], minutes: int) -> None:
    assert cron_interval_minutes(crons) == minutes


def test_a_schedule_key_without_cron_entries_declares_no_crons() -> None:
    assert schedule_crons(NO_CRON) == []


def test_a_cron_line_in_a_workflow_without_a_schedule_key_is_not_a_schedule() -> None:
    assert schedule_crons("on:\n  push:\n# - cron: '0 7 * * *'\nsteps:\n  - cron: '0 7 * * *'\n") == []


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("    - cron: 0 7 * * * # daily", "0 7 * * *"),
        ("    - cron: '0 7 * * *'   # daily", "0 7 * * *"),
        ('    - cron: "0 7 * * *"', "0 7 * * *"),
        ("    - cron: '0 7 * * *", "0 7 * * *"),
    ],
)
def test_cron_line_unquoting(line: str, expected: str) -> None:
    assert schedule_crons(f"on:\n  schedule:\n{line}\n") == [expected]


def test_cron_interval_hours_tracks_the_longest_gap() -> None:
    assert cron_interval_hours(["30 7 * * 1-5"]) == 72.0
    assert cron_interval_hours([]) is None
