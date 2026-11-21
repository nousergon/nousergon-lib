"""Fixture tests for shared imperative alarm-authorship detection."""

from __future__ import annotations

import pytest

from nousergon_lib.alarm_authorship import SourceLanguage, find_imperative_alarm_calls


@pytest.mark.parametrize(
    ("language", "source", "expected_line"),
    [
        ("python", 'cloudwatch.put_metric_alarm(AlarmName="dead-job")', 1),
        (
            "python",
            'subprocess.run(["aws", "cloudwatch", "put-metric-alarm", "--alarm-name", "dead-job"])',
            1,
        ),
        ("shell", "#!/bin/bash\naws cloudwatch put-metric-alarm --alarm-name dead-job", 2),
        (
            "yaml",
            "steps:\n  - run: |\n      echo checking\n      aws cloudwatch put-metric-alarm --alarm-name dead-job\n",
            4,
        ),
        (
            "yaml",
            "phases:\n  build:\n    commands:\n      - aws cloudwatch put-metric-alarm --alarm-name dead-job\n",
            4,
        ),
    ],
)
def test_detects_imperative_alarm_calls(
    language: SourceLanguage, source: str, expected_line: int
) -> None:
    calls = find_imperative_alarm_calls(source, language)
    assert len(calls) == 1
    assert calls[0].line == expected_line
    expected_column = (
        1
        if language == "python"
        else source.splitlines()[expected_line - 1].find("aws") + 1
    )
    assert calls[0].column == expected_column
    assert calls[0].call in {"put_metric_alarm", "aws cloudwatch put-metric-alarm"}


@pytest.mark.parametrize(
    ("language", "source"),
    [
        ("python", '# cloudwatch.put_metric_alarm(AlarmName="example")'),
        ("python", '"""Use put_metric_alarm to create an alarm."""'),
        ("shell", "# aws cloudwatch put-metric-alarm --alarm-name example"),
        ("shell", "echo 'aws cloudwatch put-metric-alarm is prohibited'"),
        ("shell", "echo aws cloudwatch put-metric-alarm is prohibited"),
        ("yaml", "description: aws cloudwatch put-metric-alarm creates an alarm\n"),
        ("yaml", "Resources:\n  Alarm:\n    Type: AWS::CloudWatch::Alarm\n"),
        ("yaml", "run: |\n  # aws cloudwatch put-metric-alarm --alarm-name example\n"),
    ],
)
def test_ignores_comments_docstrings_and_prose(
    language: SourceLanguage, source: str
) -> None:
    assert find_imperative_alarm_calls(source, language) == []
