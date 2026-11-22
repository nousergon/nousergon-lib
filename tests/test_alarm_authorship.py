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


@pytest.mark.parametrize(
    "source",
    [
        "run aws cloudwatch put-metric-alarm --alarm-name x\n",
        "timeout 30 aws cloudwatch put-metric-alarm --alarm-name x\n",
        "xargs aws cloudwatch put-metric-alarm --alarm-name x\n",
        "env REGION=us-east-1 aws cloudwatch put-metric-alarm --alarm-name x\n",
        "command aws cloudwatch put-metric-alarm --alarm-name x\n",
        "nice -n 10 aws cloudwatch put-metric-alarm --alarm-name x\n",
        '"$AWS" cloudwatch put-metric-alarm --alarm-name x\n',
        '"${AWS}" cloudwatch put-metric-alarm --alarm-name x\n',
        'bash -c "aws cloudwatch put-metric-alarm --alarm-name x"\n',
        "sh -c 'aws cloudwatch put-metric-alarm --alarm-name x'\n",
        'eval "aws cloudwatch put-metric-alarm --alarm-name x"\n',
    ],
)
def test_shell_wrapper_forms_are_reported(source: str) -> None:
    assert len(find_imperative_alarm_calls(source, "shell")) == 1


def test_shell_command_variable_is_reported_when_executed() -> None:
    source = 'alarm_cmd="aws cloudwatch put-metric-alarm --alarm-name x"\n$alarm_cmd\n'
    assert [call.line for call in find_imperative_alarm_calls(source, "shell")] == [2]


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


# --- Cases the existing consumer guards exercise (nous-ergon-ops / nousergon-data) ---


def test_consumer_fixture_shell_line_continuation_is_a_call() -> None:
    # The commonest real form, and the exact fixture of two consumer guards:
    # the command continues onto the next line with a trailing backslash.
    source = (
        "#!/usr/bin/env bash\n"
        "# aws cloudwatch put-metric-alarm  <- a comment, not a finding\n"
        '  aws cloudwatch put-metric-alarm --alarm-name "x" \\\n'
        "    --namespace Example\n"
    )
    calls = find_imperative_alarm_calls(source, "shell")
    assert [(call.line, call.column) for call in calls] == [(3, 3)]


def test_consumer_fixture_python_multiline_docstring_and_message_are_not_calls() -> None:
    source = (
        '"""A docstring mentioning put-metric-alarm across\n'
        "more than one line.\"\"\"\n"
        "import subprocess\n"
        'MSG = "put-metric-alarm rejects an over-long description"\n'
        'subprocess.run(["aws", "cloudwatch", "put-metric-alarm"])\n'
    )
    assert [call.line for call in find_imperative_alarm_calls(source, "python")] == [5]


def test_consumer_fixture_chained_boto3_call() -> None:
    source = 'import boto3\nboto3.client("cloudwatch").put_metric_alarm(AlarmName="x")\n'
    assert [call.line for call in find_imperative_alarm_calls(source, "python")] == [2]


# --- Shell syntax the line-by-line splitter missed or mislocated ---


@pytest.mark.parametrize(
    "source",
    [
        'aws --region "$REGION" cloudwatch put-metric-alarm --alarm-name x\n',
        "aws cloudwatch --profile ops put-metric-alarm --alarm-name x\n",
        "aws --no-cli-pager --output=json cloudwatch put-metric-alarm --alarm-name x\n",
        "if ! aws cloudwatch put-metric-alarm --alarm-name x; then exit 1; fi\n",
        "out=$(aws cloudwatch put-metric-alarm --alarm-name x)\n",
        "REGION=us-east-1 aws cloudwatch put-metric-alarm --alarm-name x\n",
        "env -i aws cloudwatch put-metric-alarm --alarm-name x\n",
        "/usr/local/bin/aws cloudwatch put-metric-alarm --alarm-name x\n",
        "for a in x y; do aws cloudwatch put-metric-alarm --alarm-name \"$a\"; done\n",
    ],
)
def test_shell_command_positions_are_found(source: str) -> None:
    calls = find_imperative_alarm_calls(source, "shell")
    assert len(calls) == 1
    assert calls[0].line == 1
    assert source[calls[0].column - 1 :].split()[0].endswith("aws")


def test_shell_column_points_at_the_command_not_an_earlier_substring() -> None:
    source = "prepare_aws && aws cloudwatch put-metric-alarm --alarm-name x\n"
    calls = find_imperative_alarm_calls(source, "shell")
    assert [call.column for call in calls] == [source.index(" aws ") + 2]


def test_shell_reports_every_call_on_a_line() -> None:
    source = "aws cloudwatch put-metric-alarm --alarm-name a; aws cloudwatch put-metric-alarm --alarm-name b\n"
    calls = find_imperative_alarm_calls(source, "shell")
    assert [call.column for call in calls] == [1, source.rindex("aws") + 1]


@pytest.mark.parametrize(
    "source",
    [
        'echo "first; aws cloudwatch put-metric-alarm --alarm-name x"\n',
        "echo 'a && aws cloudwatch put-metric-alarm --alarm-name x'\n",
        "aws cloudwatch describe-alarms --alarm-names put-metric-alarm\n",
        "grep -n put-metric-alarm setup.sh\n",
        "cat <<EOF\naws cloudwatch put-metric-alarm --alarm-name x\nEOF\n",
        "cat <<-'DOC' > notes.txt\n\taws cloudwatch put-metric-alarm --alarm-name x\n\tDOC\n",
    ],
)
def test_shell_quoted_text_arguments_and_heredoc_bodies_are_not_calls(source: str) -> None:
    assert find_imperative_alarm_calls(source, "shell") == []


def test_shell_resumes_scanning_after_a_heredoc() -> None:
    source = "cat <<EOF\nprose\nEOF\naws cloudwatch put-metric-alarm --alarm-name x\n"
    assert [call.line for call in find_imperative_alarm_calls(source, "shell")] == [4]


# --- Python forms the subprocess-list check missed ---


@pytest.mark.parametrize(
    "source",
    [
        'subprocess.run(["aws", "--region", region, "cloudwatch", "put-metric-alarm"])\n',
        'subprocess.run("aws cloudwatch put-metric-alarm --alarm-name x", shell=True)\n',
        'os.system(f"aws cloudwatch put-metric-alarm --alarm-name {name}")\n',
        'subprocess.run(args=["aws", "cloudwatch", "put-metric-alarm"], check=True)\n',
    ],
)
def test_python_subprocess_forms_are_calls(source: str) -> None:
    calls = find_imperative_alarm_calls(source, "python")
    assert [(call.line, call.column) for call in calls] == [(1, 1)]


def test_python_argv_literal_bound_to_a_name_is_a_call() -> None:
    source = 'cmd = ["aws", "cloudwatch", "put-metric-alarm", "--alarm-name", name]\nsubprocess.run(cmd)\n'
    calls = find_imperative_alarm_calls(source, "python")
    assert [(call.line, call.column) for call in calls] == [(1, 7)]


def test_python_printed_command_is_not_a_call() -> None:
    source = 'print(f"run: aws cloudwatch put-metric-alarm --alarm-name {name}")\n'
    assert find_imperative_alarm_calls(source, "python") == []


@pytest.mark.parametrize(
    "source",
    [
        "create_alarm = client.put_metric_alarm\n",
        'create_alarm = getattr(client, "put_metric_alarm")\n',
    ],
)
def test_python_bound_method_references_are_calls(source: str) -> None:
    assert [call.line for call in find_imperative_alarm_calls(source, "python")] == [1]


def test_unparseable_python_fails_closed() -> None:
    # A file that does not parse cannot be proven clean; report the raw lines
    # rather than returning an empty (passing) result.
    source = "def broken(:\n    cw.put_metric_alarm(AlarmName='x')\n# put_metric_alarm in a comment\n"
    assert [call.line for call in find_imperative_alarm_calls(source, "python")] == [2]


# --- YAML locations and list-valued script keys ---


def test_yaml_block_scalar_column_is_the_source_column() -> None:
    source = """\
jobs:
  alarm:
    steps:
      - run: |
          echo start
          aws cloudwatch put-metric-alarm \\
            --alarm-name real
"""
    calls = find_imperative_alarm_calls(source, "yaml")
    assert [(call.line, call.column) for call in calls] == [(6, 11)]
    line = source.splitlines()[5]
    assert line[calls[0].column - 1 :].startswith("aws cloudwatch")


def test_yaml_flow_scalar_location() -> None:
    source = "steps:\n  - run: aws cloudwatch put-metric-alarm --alarm-name x\n"
    calls = find_imperative_alarm_calls(source, "yaml")
    assert [(call.line, call.column) for call in calls] == [(2, 10)]


def test_yaml_script_lists_are_executable() -> None:
    source = """\
phases:
  build:
    commands:
      - echo build
      - aws cloudwatch put-metric-alarm --alarm-name x
deploy:
  script:
    - aws cloudwatch put-metric-alarm --alarm-name y
"""
    calls = find_imperative_alarm_calls(source, "yaml")
    assert [(call.line, call.column) for call in calls] == [(5, 9), (8, 7)]


def test_unparseable_yaml_fails_closed() -> None:
    source = "run: [unclosed\n  aws cloudwatch put-metric-alarm --alarm-name x\n# put-metric-alarm\n"
    assert [call.line for call in find_imperative_alarm_calls(source, "yaml")] == [2]
