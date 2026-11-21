"""Tests for syntax-aware imperative CloudWatch alarm call detection."""

from __future__ import annotations

import pytest

from nousergon_lib.alarm_authorship import AlarmCall, SourceLanguage, find_imperative_alarm_calls


@pytest.mark.parametrize(
    ("source", "expected_line"),
    [
        ("cw.put_metric_alarm(AlarmName='broken')\n", 1),
        ("client = boto3.client('cloudwatch')\nclient.put_metric_alarm(AlarmName='broken')\n", 2),
        ("subprocess.run(['aws', 'cloudwatch', 'put-metric-alarm', '--alarm-name', 'broken'])\n", 1),
        ("subprocess.check_call(('aws', 'cloudwatch', 'put-metric-alarm'))\n", 1),
    ],
)
def test_python_calls_are_reported_with_locations(source: str, expected_line: int) -> None:
    calls = find_imperative_alarm_calls(source, "python")
    assert len(calls) == 1
    assert calls[0].line == expected_line
    assert calls[0].column == 1


def test_python_comments_docstrings_and_prose_are_not_calls() -> None:
    source = '''"""Do not call cw.put_metric_alarm()."""
# cw.put_metric_alarm(AlarmName="example")
guidance = "aws cloudwatch put-metric-alarm --alarm-name example"
'''
    assert find_imperative_alarm_calls(source, "python") == []


@pytest.mark.parametrize(
    "source",
    [
        "aws cloudwatch put-metric-alarm --alarm-name broken\n",
        "sudo aws cloudwatch put-metric-alarm --alarm-name broken\n",
        "prepare && aws cloudwatch put-metric-alarm --alarm-name broken\n",
    ],
)
def test_shell_calls_are_reported(source: str) -> None:
    calls = find_imperative_alarm_calls(source, "shell")
    assert calls == [AlarmCall(line=1, column=source.index("aws") + 1, text=source.strip())]


def test_shell_comments_and_printed_prose_are_not_calls() -> None:
    source = '''# aws cloudwatch put-metric-alarm --alarm-name example
echo "aws cloudwatch put-metric-alarm --alarm-name example"
printf 'Use aws cloudwatch put-metric-alarm carefully\\n'
Use aws cloudwatch put-metric-alarm only during an incident.
'''
    assert find_imperative_alarm_calls(source, "shell") == []


def test_yaml_only_scans_executable_scalar_keys() -> None:
    source = """\
name: Mention aws cloudwatch put-metric-alarm in prose
jobs:
  alarm:
    steps:
      - run: |
          # aws cloudwatch put-metric-alarm --alarm-name old
          aws cloudwatch put-metric-alarm --alarm-name real
      - name: aws cloudwatch put-metric-alarm is forbidden
"""
    calls = find_imperative_alarm_calls(source, "yaml")
    assert [(call.line, call.text) for call in calls] == [
        (7, "aws cloudwatch put-metric-alarm --alarm-name real")
    ]


def test_public_language_type_accepts_each_supported_language() -> None:
    languages: tuple[SourceLanguage, ...] = ("python", "shell", "yaml")
    assert [find_imperative_alarm_calls("", language) for language in languages] == [[], [], []]
