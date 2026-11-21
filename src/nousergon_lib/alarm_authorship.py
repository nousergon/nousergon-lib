"""Locate imperative CloudWatch alarm creation in source text.

The detector is intentionally syntax-aware: names mentioned in comments,
docstrings, and explanatory prose are not executions.  It recognises the two
imperative forms used by fleet guards: boto3 ``put_metric_alarm`` calls and
AWS CLI ``cloudwatch put-metric-alarm`` commands (including Python subprocess
argument lists).
"""

from __future__ import annotations

import ast
import re
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
    text: str


def _python_calls(source: str) -> list[AlarmCall]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    calls: list[AlarmCall] = []
    source_lines = source.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        is_boto3_call = (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "put_metric_alarm"
        )
        is_subprocess_call = False
        subprocess_names = {"call", "check_call", "check_output", "Popen", "run"}
        function_name = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else node.func.id
            if isinstance(node.func, ast.Name)
            else ""
        )
        if (
            function_name in subprocess_names
            and node.args
            and isinstance(node.args[0], (ast.List, ast.Tuple))
        ):
            values = node.args[0].elts
            words = [value.value for value in values if isinstance(value, ast.Constant) and isinstance(value.value, str)]
            is_subprocess_call = _is_alarm_argv(words)
        if is_boto3_call or is_subprocess_call:
            calls.append(
                AlarmCall(
                    line=node.lineno,
                    column=node.col_offset + 1,
                    text=source_lines[node.lineno - 1].strip(),
                )
            )
    return calls


def _is_alarm_argv(words: list[str]) -> bool:
    return (
        len(words) >= 3
        and words[0].rsplit("/", 1)[-1] == "aws"
        and words[1:3] == ["cloudwatch", "put-metric-alarm"]
    )


def _shell_line_call(line: str, line_number: int, column_offset: int = 0) -> AlarmCall | None:
    stripped = line.lstrip()
    if not stripped or stripped.startswith("#"):
        return None
    column = len(line) - len(stripped)
    # A command may follow a shell control operator.  Quoted prose still does
    # not match because shlex keeps it in a single token.
    for segment in re.split(r"(?:^|[;&|]\s*)", stripped):
        candidate = segment.strip()
        if not candidate:
            continue
        try:
            words = shlex.split(candidate, comments=True, posix=True)
        except ValueError:
            continue
        while words and ("=" in words[0] and not words[0].startswith("=")):
            words.pop(0)
        while words and words[0] in {"command", "env", "exec", "sudo"}:
            words.pop(0)
        if _is_alarm_argv(words):
            start = line.find("aws", column)
            return AlarmCall(
                line=line_number,
                column=column_offset + start + 1,
                text=line.strip(),
            )
    return None


def _shell_calls(source: str, *, first_line: int = 1, column_offset: int = 0) -> list[AlarmCall]:
    return [
        call
        for offset, line in enumerate(source.splitlines())
        if (call := _shell_line_call(line, first_line + offset, column_offset)) is not None
    ]


def _yaml_calls(source: str) -> list[AlarmCall]:
    try:
        root = cast(
            Node | None,
            yaml.compose(source),  # pyright: ignore[reportUnknownMemberType]
        )
    except yaml.YAMLError:
        return []
    if root is None:
        return []

    calls: list[AlarmCall] = []

    def visit(node: Node) -> None:
        if isinstance(node, MappingNode):
            for key, value in node.value:
                if (
                    isinstance(key, ScalarNode)
                    and key.value in {"run", "script", "shell"}
                    and isinstance(value, ScalarNode)
                ):
                    is_block = value.style in {"|", ">"}
                    calls.extend(
                        _shell_calls(
                            value.value,
                            first_line=value.start_mark.line + (2 if is_block else 1),
                            column_offset=value.start_mark.column + (1 if is_block else 0),
                        )
                    )
                else:
                    visit(value)
        elif isinstance(node, SequenceNode):
            for child in node.value:
                visit(child)

    visit(root)
    return calls


def find_imperative_alarm_calls(source: str, language: SourceLanguage) -> list[AlarmCall]:
    """Return alarm-creation calls in *source*, ordered by source location."""
    if language == "python":
        calls = _python_calls(source)
    elif language == "shell":
        calls = _shell_calls(source)
    else:
        calls = _yaml_calls(source)
    return sorted(calls, key=lambda call: (call.line, call.column))
