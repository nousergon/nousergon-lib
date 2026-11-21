"""Detect imperative CloudWatch alarm creation in source text.

The detector is deliberately syntax-aware enough to ignore examples in Python
docstrings and comments.  YAML is restricted to fields that execute shell
commands; declarative CloudFormation alarm resources are not authorship calls.
"""

from __future__ import annotations

import ast
import shlex
from dataclasses import dataclass
from typing import Literal, cast

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

__all__ = ["AlarmCall", "SourceLanguage", "find_imperative_alarm_calls"]

SourceLanguage = Literal["python", "shell", "yaml"]


@dataclass(frozen=True)
class AlarmCall:
    """One imperative alarm-creation call in source text."""

    line: int
    column: int
    language: SourceLanguage
    call: str


def _is_aws_alarm_command(command: str) -> bool:
    """Return whether *command* invokes ``aws cloudwatch put-metric-alarm``."""
    try:
        words = shlex.split(command, comments=True)
    except ValueError:
        return False
    for index, word in enumerate(words):
        if word != "aws":
            continue
        prefix = words[:index]
        if prefix and prefix[-1] not in {"&&", "||", ";", "do", "exec", "sudo", "then", "|"}:
            if not all("=" in part and not part.startswith("=") for part in prefix):
                continue
        tail = words[index + 1 :]
        if any(
            tail[position : position + 2] == ["cloudwatch", "put-metric-alarm"]
            for position in range(len(tail) - 1)
        ):
            return True
    return False


def _python_calls(source: str) -> list[AlarmCall]:
    tree = ast.parse(source)
    calls: list[AlarmCall] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "put_metric_alarm":
            calls.append(AlarmCall(node.lineno, node.col_offset + 1, "python", "put_metric_alarm"))
            continue
        is_subprocess = (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
            and node.func.attr in {"call", "check_call", "check_output", "Popen", "run"}
        )
        if not is_subprocess or not node.args:
            continue
        command_node = node.args[0]
        command: str | None = None
        if isinstance(command_node, ast.Constant) and isinstance(command_node.value, str):
            command = command_node.value
        elif isinstance(command_node, (ast.List, ast.Tuple)):
            values: list[str] = []
            for item in command_node.elts:
                if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                    break
                values.append(item.value)
            else:
                command = shlex.join(values)
        if command is not None and _is_aws_alarm_command(command):
            calls.append(AlarmCall(node.lineno, node.col_offset + 1, "python", "aws cloudwatch put-metric-alarm"))
    return calls


def _shell_calls(source: str, *, language: SourceLanguage = "shell", line_offset: int = 0) -> list[AlarmCall]:
    calls: list[AlarmCall] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        if _is_aws_alarm_command(line):
            column = line.find("aws") + 1
            calls.append(AlarmCall(line_number + line_offset, column, language, "aws cloudwatch put-metric-alarm"))
    return calls


_EXECUTABLE_YAML_KEYS = frozenset({"run", "script", "commands", "pre_build", "build", "post_build"})


def _yaml_calls(source: str) -> list[AlarmCall]:
    root = cast(Node | None, yaml.compose(source))  # pyright: ignore[reportUnknownMemberType]
    if root is None:
        return []
    calls: list[AlarmCall] = []

    def visit(node: Node, executable: bool = False) -> None:
        if isinstance(node, MappingNode):
            for key, value in node.value:
                is_executable = (
                    isinstance(key, ScalarNode) and key.value in _EXECUTABLE_YAML_KEYS
                )
                visit(value, executable or is_executable)
        elif isinstance(node, SequenceNode):
            for value in node.value:
                visit(value, executable)
        elif executable and isinstance(node, ScalarNode):
            block_scalar_offset = 1 if node.style in {"|", ">"} else 0
            calls.extend(_shell_calls(
                node.value,
                language="yaml",
                line_offset=node.start_mark.line + block_scalar_offset,
            ))

    visit(root)
    source_lines = source.splitlines()
    return [
        AlarmCall(
            call.line,
            source_lines[call.line - 1].find("aws") + 1,
            call.language,
            call.call,
        )
        for call in calls
    ]


def find_imperative_alarm_calls(source: str, language: SourceLanguage) -> list[AlarmCall]:
    """Find imperative CloudWatch alarm creation and report its locations.

    ``language`` selects Python parsing, shell command parsing, or executable
    command fields in YAML.  Malformed source raises its parser's normal error.
    """
    if language == "python":
        calls = _python_calls(source)
    elif language == "shell":
        calls = _shell_calls(source)
    elif language == "yaml":
        calls = _yaml_calls(source)
    else:
        raise ValueError(f"unsupported source language: {language}")
    return sorted(calls, key=lambda call: (call.line, call.column))
