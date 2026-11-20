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
