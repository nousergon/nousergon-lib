"""Contracts for shell scripts that dispatch other scripts through heredocs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["HeredocExpansionDiagnostic", "detect_heredoc_expansions"]

_ASSIGNMENT = re.compile(r"(?:^|[;\s])(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=")
_REFERENCE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)[^}]*\}|([A-Za-z_][A-Za-z0-9_]*))")
_HEREDOC = re.compile(r"<<(-?)(?:\s*)(?:(['\"])([^'\"]+)\2|([A-Za-z_][A-Za-z0-9_]*))")
_SOURCE = re.compile(r"^\s*(?:source|\.)\s+([^\s;]+)", re.MULTILINE)


@dataclass(frozen=True)
class HeredocExpansionDiagnostic:
    """An expansion that happens before its heredoc reaches the target shell."""

    variable: str
    opening_line: int
    reference_line: int


@dataclass(frozen=True)
class _Heredoc:
    opening_line: int
    body_start: int
    body_end: int
    quoted: bool


def _heredocs(text: str) -> list[_Heredoc]:
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)

    found: list[_Heredoc] = []
    line_index = 0
    while line_index < len(lines):
        match = _HEREDOC.search(lines[line_index])
        if match is None:
            line_index += 1
            continue
        strip_tabs = bool(match.group(1))
        delimiter = match.group(3) or match.group(4)
        end_index = line_index + 1
        while end_index < len(lines):
            candidate = lines[end_index].rstrip("\r\n")
            if strip_tabs:
                candidate = candidate.lstrip("\t")
            if candidate == delimiter:
                break
            end_index += 1
        if end_index == len(lines):
            line_index += 1
            continue
        body_start = offsets[line_index] + len(lines[line_index])
        found.append(_Heredoc(
            opening_line=line_index + 1,
            body_start=body_start,
            body_end=offsets[end_index],
            quoted=match.group(2) is not None,
        ))
        line_index = end_index + 1
    return found


def _assigned(text: str) -> set[str]:
    return {match.group(1) for match in _ASSIGNMENT.finditer(text)}


def _sourced_assignments(script_path: Path, text: str) -> set[str]:
    assigned: set[str] = set()
    for match in _SOURCE.finditer(text):
        raw_path = match.group(1).strip("'\"")
        if "_spot_common.sh" not in raw_path:
            continue
        raw_path = raw_path.replace("${SCRIPT_DIR}", str(script_path.parent))
        raw_path = raw_path.replace("$SCRIPT_DIR", str(script_path.parent))
        source_path = Path(raw_path)
        if not source_path.is_absolute():
            source_path = script_path.parent / source_path
        if source_path.is_file():
            assigned.update(_assigned(source_path.read_text()))
    return assigned


def detect_heredoc_expansions(script_path: str | Path) -> list[HeredocExpansionDiagnostic]:
    """Find local variables accidentally expanded by a dispatching shell.

    Unquoted heredocs are expanded by the shell constructing them.  A variable
    assigned only in the heredoc therefore disappears before the receiving
    shell can use it unless its reference is escaped.
    """
    path = Path(script_path)
    text = path.read_text()
    heredocs = _heredocs(text)

    outer_text = text
    for heredoc in reversed(heredocs):
        outer_text = outer_text[: heredoc.body_start] + outer_text[heredoc.body_end :]
    outer_assigned = _assigned(outer_text) | _sourced_assignments(path, outer_text)

    diagnostics: list[HeredocExpansionDiagnostic] = []
    for heredoc in heredocs:
        if heredoc.quoted:
            continue
        body = text[heredoc.body_start : heredoc.body_end]
        local_only = _assigned(body) - outer_assigned
        for match in _REFERENCE.finditer(body):
            variable = match.group(1) or match.group(2)
            backslashes = 0
            cursor = match.start() - 1
            while cursor >= 0 and body[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if variable in local_only and backslashes % 2 == 0:
                diagnostics.append(HeredocExpansionDiagnostic(
                    variable=variable,
                    opening_line=heredoc.opening_line,
                    reference_line=heredoc.opening_line + body.count("\n", 0, match.start()) + 1,
                ))
    return diagnostics
