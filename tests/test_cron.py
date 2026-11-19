import pytest

from nousergon_lib.cron import cron_interval_hours, cron_interval_minutes, schedule_crons


def test_schedule_crons_strips_trailing_comments() -> None:
    text = """
on:
  schedule:
    - cron: 15 3 * * *  # nightly run
"""

    assert schedule_crons(text) == ["15 3 * * *"]


def test_schedule_crons_handles_quoted_expressions() -> None:
    text = """
  schedule:
    - cron: "0 6 * * 1-5" # weekdays
    - cron: '30 12 * * *' # midday
"""

    assert schedule_crons(text) == ["0 6 * * 1-5", "30 12 * * *"]


def test_schedule_crons_preserves_comment_markers_inside_quotes() -> None:
    assert schedule_crons("- cron: '0 6 * * * # literal' # trailing") == ["0 6 * * * # literal"]


@pytest.mark.parametrize(
    ("crons", "expected"),
    [
        (["*/15 * * * *"], 15),
        (["0,10,45 * * * *"], 10),
        (["0 6 * * *", "30 6 * * *"], 30),
        (["0 6,18 * * *"], 720),
        (["0 6 * * 1"], 10080),
    ],
)
def test_cron_interval_minutes(crons: list[str], expected: int) -> None:
    assert cron_interval_minutes(crons) == expected


def test_cron_interval_hours_compatibility_wrapper() -> None:
    assert cron_interval_hours(["*/30 * * * *"]) == 0.5
