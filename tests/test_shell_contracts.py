from __future__ import annotations

from pathlib import Path

from nousergon_lib.shell_contracts import (
    HeredocExpansionDiagnostic,
    detect_heredoc_expansions,
)


def _write(tmp_path: Path, text: str) -> Path:
    script = tmp_path / "launch_backtester.sh"
    script.write_text(text)
    return script


def test_unescaped_local_reference_is_reported(tmp_path: Path) -> None:
    script = _write(tmp_path, """#!/bin/bash
aws ssm send-command <<PARAMS
WORK_DIR=/opt/backtest
cd "$WORK_DIR"
PARAMS
""")

    assert detect_heredoc_expansions(script) == [
        HeredocExpansionDiagnostic(variable="WORK_DIR", opening_line=2, reference_line=4),
    ]


def test_escaped_reference_and_quoted_delimiter_are_safe(tmp_path: Path) -> None:
    script = _write(tmp_path, """cat <<EOF
WORK_DIR=/opt/backtest
cd \"\\$WORK_DIR\"
EOF
cat <<'LITERAL'
OTHER_DIR=/opt/other
cd "$OTHER_DIR"
LITERAL
""")

    assert detect_heredoc_expansions(script) == []


def test_dispatcher_assignment_is_an_intentional_injection(tmp_path: Path) -> None:
    script = _write(tmp_path, """WORK_DIR=/mnt/dispatcher
cat <<EOF
WORK_DIR=/opt/backtest
cd "$WORK_DIR"
EOF
""")

    assert detect_heredoc_expansions(script) == []


def test_backtester_common_assignments_are_dispatcher_assignments(tmp_path: Path) -> None:
    (tmp_path / "_spot_common.sh").write_text("WORK_DIR=/mnt/dispatcher\n")
    script = _write(tmp_path, """source "$SCRIPT_DIR/_spot_common.sh"
cat <<EOF
WORK_DIR=/opt/backtest
cd "$WORK_DIR"
EOF
""")

    assert detect_heredoc_expansions(script) == []


def test_discovers_each_unquoted_heredoc(tmp_path: Path) -> None:
    script = _write(tmp_path, """cat <<FIRST
ONE=1
echo "$ONE"
FIRST
cat <<-SECOND
\tTWO=2
\techo "${TWO}"
\tSECOND
""")

    assert detect_heredoc_expansions(script) == [
        HeredocExpansionDiagnostic(variable="ONE", opening_line=1, reference_line=3),
        HeredocExpansionDiagnostic(variable="TWO", opening_line=5, reference_line=7),
    ]
