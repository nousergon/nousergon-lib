"""Locate imperative CloudWatch alarm creation in source text.

The detector is intentionally syntax-aware: names mentioned in comments,
docstrings, and explanatory prose are not executions.  It recognises the two
imperative forms used by fleet guards: boto3 ``put_metric_alarm`` calls and
AWS CLI ``cloudwatch put-metric-alarm`` commands (including Python subprocess
argument lists, shell-string subprocess calls and argv literals).

Shell text is tokenised rather than split line by line, so a command continued
with a trailing backslash, preceded by AWS CLI global options, or placed after
a control operator, ``if``/``!``/``do`` or inside ``$(...)`` is still found,
while quoted arguments and here-document bodies are not.

Source that cannot be parsed (Python or YAML) cannot be proven clean, so it
fails closed: every non-comment line naming the call is reported.
"""

from __future__ import annotations

import ast
import bisect
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

__all__ = ["AlarmCall", "SourceLanguage", "find_imperative_alarm_calls"]

SourceLanguage = Literal["python", "shell", "yaml"]

_BOTO3_METHOD = "put_metric_alarm"
_CLI_VERB = "put-metric-alarm"

# subprocess functions that take an argv list or a shell string, plus the os
# functions that take a shell string.
_COMMAND_FUNCTIONS = frozenset(
    {"call", "check_call", "check_output", "Popen", "run", "system", "popen", "getoutput", "getstatusoutput"}
)
_COMMAND_KEYWORDS = frozenset({"args", "cmd", "command"})

# AWS CLI global options that consume the following word as their value.
_AWS_VALUE_OPTIONS = frozenset(
    {
        "--region",
        "--profile",
        "--output",
        "--query",
        "--endpoint-url",
        "--cli-read-timeout",
        "--cli-connect-timeout",
        "--ca-bundle",
        "--color",
        "--cli-binary-format",
    }
)

# Shell operators, longest first so a prefix never shadows a longer operator.
_REDIRECT_OPERATORS = ("<<<", "<<-", "<<", ">>", ">&", "<&", "<>", ">|", "<", ">")
_CONTROL_OPERATORS = ("&&", "||", ";;", "|&", "$(", ";", "&", "|", "(", ")", "`")
_OPERATORS = tuple(sorted(_REDIRECT_OPERATORS + _CONTROL_OPERATORS, key=len, reverse=True))

# Words that may precede the command name of a simple command.
_COMMAND_PREFIXES = frozenset({"!", "{", "if", "then", "else", "elif", "while", "until", "do"})
# Wrappers that run the following words as a command, after their own options.
_COMMAND_WRAPPERS = frozenset({"command", "env", "exec", "nohup", "run", "sudo", "time"})
_DURATION_WRAPPERS = frozenset({"timeout"})
_OPTION_VALUE_WRAPPERS = frozenset({"nice", "xargs"})
_WRAPPER_VALUE_OPTIONS = frozenset(
    {
        "--arg-file",
        "--eof",
        "--max-args",
        "--max-chars",
        "--max-lines",
        "--max-procs",
        "--replace",
        "-E",
        "-I",
        "-L",
        "-P",
        "-a",
        "-n",
        "-s",
    }
)
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\+?=")
_VARIABLE = re.compile(r"\$(?:{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))$")

# YAML keys whose scalar (or list of scalars) value is a shell script.
_YAML_SCRIPT_KEYS = frozenset({"run", "script", "shell", "commands", "before_script", "after_script"})


@dataclass(frozen=True)
class AlarmCall:
    """One imperative alarm-creation call in source text."""

    line: int
    column: int
    text: str


@dataclass(frozen=True)
class _Word:
    text: str
    offset: int


def _is_alarm_argv(words: Sequence[str | None]) -> bool:
    """True when *words* is ``aws [global options] cloudwatch put-metric-alarm ...``.

    ``None`` stands for an element whose value is not known statically.
    """
    if not words or words[0] is None or words[0].rsplit("/", 1)[-1] != "aws":
        return False
    positionals: list[str | None] = []
    index = 1
    while index < len(words) and len(positionals) < 2:
        word = words[index]
        if word is not None and word.startswith("-"):
            index += 2 if word in _AWS_VALUE_OPTIONS else 1
            continue
        positionals.append(word)
        index += 1
    return positionals == ["cloudwatch", _CLI_VERB]


# --- shell -----------------------------------------------------------------


def _operator_at(text: str, index: int) -> str | None:
    for operator in _OPERATORS:
        if text.startswith(operator, index):
            return operator
    return None


def _lex_word(text: str, index: int) -> tuple[str, int]:
    """Read one shell word starting at *index*; return its unquoted value and end."""
    parts: list[str] = []
    length = len(text)
    while index < length:
        char = text[index]
        if char in " \t\n":
            break
        if char == "\\":
            if text.startswith("\\\n", index):
                index += 2
            else:
                parts.append(text[index + 1 : index + 2])
                index += 2
            continue
        if char == "'":
            close = text.find("'", index + 1)
            close = length if close < 0 else close
            parts.append(text[index + 1 : close])
            index = close + 1
            continue
        if char == '"':
            index += 1
            while index < length and text[index] != '"':
                if text[index] == "\\" and index + 1 < length:
                    if text[index + 1] != "\n":
                        parts.append(text[index + 1])
                    index += 2
                    continue
                parts.append(text[index])
                index += 1
            index += 1
            continue
        if _operator_at(text, index) is not None:
            break
        parts.append(char)
        index += 1
    return "".join(parts), index


def _skip_heredocs(text: str, index: int, pending: list[tuple[str, bool]]) -> int:
    """Skip the here-document bodies that start at *index* (the line after their operator)."""
    length = len(text)
    for delimiter, strip_tabs in pending:
        while index < length:
            end = text.find("\n", index)
            end = length if end < 0 else end
            line = text[index:end]
            index = end + 1
            if (line.lstrip("\t") if strip_tabs else line) == delimiter:
                break
    return index


def _shell_simple_commands(text: str) -> list[list[_Word]]:
    """Split shell *text* into simple commands, dropping comments, redirect targets and heredoc bodies."""
    commands: list[list[_Word]] = []
    current: list[_Word] = []
    pending_heredocs: list[tuple[str, bool]] = []
    redirect: str | None = None
    index = 0
    length = len(text)

    def finish() -> None:
        if current:
            commands.append(list(current))
            current.clear()

    while index < length:
        char = text[index]
        if char in " \t":
            index += 1
        elif text.startswith("\\\n", index):
            index += 2
        elif char == "\n":
            finish()
            index += 1
            if pending_heredocs:
                index = _skip_heredocs(text, index, pending_heredocs)
                pending_heredocs.clear()
        elif char == "#":
            end = text.find("\n", index)
            index = length if end < 0 else end
        elif (operator := _operator_at(text, index)) is not None:
            index += len(operator)
            if operator in _REDIRECT_OPERATORS:
                redirect = operator
            else:
                finish()
        else:
            start = index
            word, index = _lex_word(text, index)
            if redirect is not None:
                if redirect in {"<<", "<<-"}:
                    pending_heredocs.append((word, redirect == "<<-"))
                redirect = None
            else:
                current.append(_Word(word, start))
    finish()
    return commands


def _command_name_index(words: list[_Word]) -> int:
    index = 0
    while index < len(words):
        word = words[index].text
        if _ASSIGNMENT.match(word) or word in _COMMAND_PREFIXES:
            index += 1
        elif word in _COMMAND_WRAPPERS:
            index += 1
            while index < len(words) and (words[index].text.startswith("-") or _ASSIGNMENT.match(words[index].text)):
                index += 1
        elif word in _DURATION_WRAPPERS:
            index += 1
            while index < len(words) and words[index].text.startswith("-"):
                index += 1
            if index < len(words):
                index += 1
        elif word in _OPTION_VALUE_WRAPPERS:
            index += 1
            while index < len(words) and words[index].text.startswith("-"):
                option = words[index].text
                index += 1
                if option in _WRAPPER_VALUE_OPTIONS and index < len(words):
                    index += 1
        else:
            break
    return index


def _shell_call_positions(text: str) -> list[tuple[int, int]]:
    """Return (1-based line, 0-based column) of each alarm command in shell *text*."""
    line_starts = [0] + [match.end() for match in re.finditer("\n", text)]
    positions: list[tuple[int, int]] = []
    variables: dict[str, str] = {}
    for words in _shell_simple_commands(text):
        for word in words:
            assignment = _ASSIGNMENT.match(word.text)
            if assignment is not None:
                variables[word.text[: assignment.end() - 1].rstrip("+")] = word.text[assignment.end() :]
        start = _command_name_index(words)
        if start >= len(words):
            continue
        command_words = [word.text for word in words[start:]]
        variable = _VARIABLE.fullmatch(command_words[0])
        variable_name = variable.group("braced") or variable.group("plain") if variable is not None else None
        if variable_name is not None and (value := variables.get(variable_name)) is not None:
            command_words = value.split() + command_words[1:]
        unresolved_aws = variable_name is not None and variable_name not in variables
        if _is_alarm_argv(command_words) or (
            unresolved_aws and _is_alarm_argv(["aws", *command_words[1:]])
        ):
            offset = words[start].offset
            line_index = bisect.bisect_right(line_starts, offset) - 1
            positions.append((line_index + 1, offset - line_starts[line_index]))
            continue

        command = command_words[0].rsplit("/", 1)[-1]
        string_index: int | None = None
        if command in {"bash", "sh"} and "-c" in command_words[1:]:
            option_index = command_words.index("-c", 1)
            if option_index + 1 < len(command_words):
                string_index = start + option_index + 1
        elif command == "eval" and len(command_words) > 1:
            string_index = start + 1
        if string_index is not None and _shell_call_positions(words[string_index].text):
            offset = words[string_index].offset
            line_index = bisect.bisect_right(line_starts, offset) - 1
            positions.append((line_index + 1, offset - line_starts[line_index]))
    return positions


def _shell_calls(source: str) -> list[AlarmCall]:
    lines = source.splitlines()
    return [
        AlarmCall(line=line, column=column + 1, text=lines[line - 1].strip())
        for line, column in _shell_call_positions(source)
    ]


# --- fail-closed fallback ---------------------------------------------------


def _raw_mentions(source: str, comment_prefix: str = "#") -> list[AlarmCall]:
    """Every non-comment line naming the call: the answer for source that does not parse."""
    calls: list[AlarmCall] = []
    for number, line in enumerate(source.splitlines(), 1):
        if line.lstrip().startswith(comment_prefix):
            continue
        found = [position for name in (_BOTO3_METHOD, _CLI_VERB) if (position := line.find(name)) >= 0]
        if found:
            calls.append(AlarmCall(line=number, column=min(found) + 1, text=line.strip()))
    return calls


# --- python ----------------------------------------------------------------


def _argv_words(node: ast.expr) -> list[str | None] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    return [
        element.value if isinstance(element, ast.Constant) and isinstance(element.value, str) else None
        for element in node.elts
    ]


def _shell_string(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value if isinstance(part, ast.Constant) and isinstance(part.value, str) else "_"
            for part in node.values
        )
    return None


def _command_argument(node: ast.Call) -> ast.expr | None:
    if node.args:
        return node.args[0]
    return next((keyword.value for keyword in node.keywords if keyword.arg in _COMMAND_KEYWORDS), None)


def _function_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return ""


def _python_calls(source: str) -> list[AlarmCall]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return _raw_mentions(source)

    source_lines = source.splitlines()
    positions: list[tuple[int, int]] = []
    command_arguments: set[int] = set()
    # ast.walk is breadth-first, so a call is visited before the argv literal it carries.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            is_alarm = isinstance(node.func, ast.Attribute) and node.func.attr == _BOTO3_METHOD
            is_alarm = is_alarm or (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == _BOTO3_METHOD
            )
            argument = _command_argument(node) if _function_name(node) in _COMMAND_FUNCTIONS else None
            if argument is not None:
                command_arguments.add(id(argument))
                argv = _argv_words(argument)
                shell = _shell_string(argument)
                if argv is not None:
                    is_alarm = is_alarm or _is_alarm_argv(argv)
                elif shell is not None:
                    is_alarm = is_alarm or bool(_shell_call_positions(shell))
            if is_alarm:
                positions.append((node.lineno, node.col_offset))
        elif isinstance(node, ast.Attribute) and node.attr == _BOTO3_METHOD:
            positions.append((node.lineno, node.col_offset))
        elif isinstance(node, (ast.List, ast.Tuple)) and id(node) not in command_arguments:
            # An argv literal is a command being built, wherever it is bound.
            argv = _argv_words(node)
            if argv is not None and _is_alarm_argv(argv):
                positions.append((node.lineno, node.col_offset))
    return [
        AlarmCall(line=line, column=column + 1, text=source_lines[line - 1].strip())
        for line, column in dict.fromkeys(positions)
    ]


# --- yaml ------------------------------------------------------------------


def _yaml_scalar_calls(node: ScalarNode, source_lines: list[str]) -> list[AlarmCall]:
    if node.style in {"|", ">"}:
        # Scan the block's own source lines, dedented, so locations are exact.
        first = node.start_mark.line + 1
        end = node.end_mark.line if node.end_mark.column == 0 else node.end_mark.line + 1
        block = source_lines[first:end]
        indents = [len(line) - len(line.lstrip(" ")) for line in block if line.strip()]
        indent = min(indents, default=0)
        text = "\n".join(line[indent:] for line in block)
        line_offset, column_offset = first, indent
    else:
        text = cast(str, node.value)
        line_offset = node.start_mark.line
        column_offset = node.start_mark.column + (1 if node.style in {"'", '"'} else 0)
    calls: list[AlarmCall] = []
    for line, column in _shell_call_positions(text):
        source_line = line_offset + line
        calls.append(
            AlarmCall(
                line=source_line,
                column=column + 1 + (column_offset if line == 1 or node.style in {"|", ">"} else 0),
                text=source_lines[source_line - 1].strip() if source_line <= len(source_lines) else text.strip(),
            )
        )
    return calls


def _yaml_calls(source: str) -> list[AlarmCall]:
    try:
        root = cast(
            Node | None,
            yaml.compose(source),  # pyright: ignore[reportUnknownMemberType]
        )
    except yaml.YAMLError:
        return _raw_mentions(source)
    if root is None:
        return []

    source_lines = source.splitlines()
    calls: list[AlarmCall] = []

    def visit(node: Node) -> None:
        if isinstance(node, MappingNode):
            for key, value in cast(list[tuple[Node, Node]], node.value):
                if isinstance(key, ScalarNode) and key.value in _YAML_SCRIPT_KEYS:
                    if isinstance(value, ScalarNode):
                        calls.extend(_yaml_scalar_calls(value, source_lines))
                        continue
                    if isinstance(value, SequenceNode):
                        for item in cast(list[Node], value.value):
                            if isinstance(item, ScalarNode):
                                calls.extend(_yaml_scalar_calls(item, source_lines))
                        continue
                visit(value)
        elif isinstance(node, SequenceNode):
            for child in cast(list[Node], node.value):
                visit(child)

    visit(root)
    return calls


def find_imperative_alarm_calls(source: str, language: SourceLanguage) -> list[AlarmCall]:
    """Return alarm-creation calls in *source*, ordered by source location.

    Lines and columns are 1-based.  Python or YAML source that does not parse
    fails closed: every non-comment line naming the call is reported.
    """
    if language == "python":
        calls = _python_calls(source)
    elif language == "shell":
        calls = _shell_calls(source)
    else:
        calls = _yaml_calls(source)
    return sorted(calls, key=lambda call: (call.line, call.column))
