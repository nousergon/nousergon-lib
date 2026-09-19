"""CI lint for the ``universe`` field of ``signals.json`` (alpha-engine-config#5809).

``signals.schema.json``'s ``properties.universe.description`` states the field is a
**sizing envelope, not a scope**: board-width (~900 names) by construction, with the
scoped decision set living in ``universe_membership/{date}/membership.json`` and
resolved via :mod:`nousergon_lib.decision_set`. Deliverable 3 of I5809 asked for a
cardinality note *and a CI lint*; the note landed in the schema description, the lint
did not (I11142) — a rule that lives only in a docstring binds nobody, and
``crucible-predictor/model/research_features.py`` read the envelope as a scope for
almost two months before the mismeasurement surfaced (I11106/I11123/I11140).

This module is the lint. It is shipped from ``nousergon-lib`` — the one place every
signals-schema consumer repo already pins by git tag (see that repo's
``requirements.txt`` and the fleet's lockstep guards) — rather than duplicated per
consumer repo, per ``policy-shared-code``'s second-adoption rule: the schema, the
envelope helpers (:mod:`nousergon_lib.signals`) and this lint all live at the one
place a version bump reaches every reader. A lint that only exists in one consumer's
CI config is the same failure mode this lint exists to close: a rule that binds one
reader and not the rest.

Binding mechanism for the fleet: each consumer repo adds a CI step that runs the
``universe-scope-lint`` console script (``[project.scripts]`` in this package's
``pyproject.toml``) against its own source tree — the same shape as any other
nousergon-lib-shipped check. A reusable ``workflow_call`` wrapper lives in
``nous-ergon-ops/.github/workflows/universe-scope-lint.yml`` so adopting it is a
three-line ``uses:`` step, not copied logic (tracked: alpha-engine-config-I5809
follow-up, per-repo adoption is separate per-repo work).

What it flags
--------------
A read of the ``"universe"`` key off a variable whose name looks like a signals
payload (``signals``, ``signals_raw``, ``signals_payload``, ``signals_data``,
``sig_data``, ``prior_signals``, ``out_signals``, ``envelope``, or a ``payload``/
``sd``/``data`` that appears signals-flavoured via a nearby ``signals`` mention),
via either ``obj["universe"]`` or ``obj.get("universe", ...)`` — anywhere outside:

* the built-in allowlist (the executor's sizing/exit path, the envelope producer), or
* a line carrying an inline exemption comment: ``# universe-scope-lint: allow --
  reason=<non-empty reason>``.

Test files are out of scope by default (fixture assertions on the contract's own
shape are not scope-reads); pass ``--include-tests`` to include them.

This is intentionally a heuristic, not a type-checker: variable-name matching can
both over- and under-fire. That is why every hit is exemptable with a reviewed,
comment-carried reason instead of the lint being unconditionally authoritative — see
the module docstring's SOTA-vs-delta note in the PR that introduced this file for why
an unexemptable lint is worse than none (it gets disabled outright the first time it
is wrong).
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path

# The one legitimate reader of board-width universe, per the schema description:
# the executor's sizing/exit path. Exempted as a whole production module rather
# than file-by-file because every file under it participates in that one path.
_ALLOWED_PATH_PREFIXES: tuple[str, ...] = (
    "crucible-executor/executor/",
    "executor/",  # when linted from inside a crucible-executor checkout
)

# The envelope producer itself: writing/assembling ``universe`` is not the misuse
# this lint targets (misuse is *reading* the envelope as a decision set elsewhere).
_ALLOWED_EXACT_FILES: tuple[str, ...] = (
    "crucible-research/scoring/signals_envelope.py",
    "scoring/signals_envelope.py",
    "crucible-research/lambda/signals_envelope_handler.py",
    "lambda/signals_envelope_handler.py",
)

_SIGNALS_NAME_RE = re.compile(
    r"(?i)^(signals(_raw|_payload|_data)?|sig_data|prior_signals|out_signals|"
    r"envelope|synthetic_signals)$"
)

# Ambiguous base names (payload, data, sd, out, result, env) are only flagged when
# the surrounding line/assignment context also mentions "signal" or "envelope" —
# keeps a completely unrelated ``libs["universe"]`` (an ArcticDB library name, seen
# in nousergon-data) from false-firing.
_AMBIGUOUS_NAME_RE = re.compile(r"(?i)^(payload|data|sd|out|result|env)$")
_CONTEXT_HINT_RE = re.compile(r"(?i)signal|envelope")

_EXEMPT_RE = re.compile(r"universe-scope-lint:\s*allow\s*--\s*reason=(\S.*)")


@dataclass(frozen=True)
class Hit:
    path: Path
    line: int
    col: int
    snippet: str

    def format(self) -> str:
        return f"{self.path}:{self.line}:{self.col}: {self.snippet}"


def _is_allowed_path(rel_path: str) -> bool:
    norm = rel_path.replace("\\", "/")
    if norm in _ALLOWED_EXACT_FILES:
        return True
    return any(norm.startswith(p) for p in _ALLOWED_PATH_PREFIXES)


def _base_name(node: ast.AST) -> str | None:
    """Best-effort variable/attribute name a subscript/`.get` call hangs off."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _line_has_exemption(line: str) -> bool:
    m = _EXEMPT_RE.search(line)
    return bool(m and m.group(1).strip())


def _looks_like_signals(name: str, context_lines: str) -> bool:
    if _SIGNALS_NAME_RE.match(name):
        return True
    if _AMBIGUOUS_NAME_RE.match(name) and _CONTEXT_HINT_RE.search(context_lines):
        return True
    return False


def scan_file(path: Path, root: Path) -> list[Hit]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    rel_path = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    if _is_allowed_path(rel_path):
        return []

    hits: list[Hit] = []

    for node in ast.walk(tree):
        target: ast.AST | None = None
        key: str | None = None

        if isinstance(node, ast.Subscript):
            slice_node = node.slice
            if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
                key = slice_node.value
                target = node.value
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "get" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    key = first.value
                    target = node.func.value

        if key != "universe" or target is None:
            continue

        name = _base_name(target)
        if not name:
            continue

        lineno = getattr(node, "lineno", 0)
        source_line = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
        prev_line = lines[lineno - 2] if lineno > 1 else ""
        context_lines = f"{prev_line}\n{source_line}"

        if not _looks_like_signals(name, context_lines):
            continue

        # Exemption comment may sit on the hit line or the line immediately above.
        if _line_has_exemption(source_line):
            continue
        if lineno > 1 and _line_has_exemption(lines[lineno - 2]):
            continue

        hits.append(
            Hit(
                path=Path(rel_path),
                line=lineno,
                col=getattr(node, "col_offset", 0) + 1,
                snippet=source_line.strip(),
            )
        )

    return hits


def iter_python_files(root: Path, include_tests: bool) -> list[Path]:
    skip_dirs = {".venv", "venv", "build", "dist", ".git", "node_modules", "__pycache__"}
    out: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in skip_dirs for part in path.parts):
            continue
        rel = path.relative_to(root)
        if not include_tests:
            parts = rel.parts
            if any(p == "tests" or p == "test" for p in parts) or rel.name.startswith("test_"):
                continue
        out.append(path)
    return sorted(out)


def scan_tree(root: Path, include_tests: bool = False) -> list[Hit]:
    hits: list[Hit] = []
    for path in iter_python_files(root, include_tests):
        hits.extend(scan_file(path, root))
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="universe-scope-lint",
        description=(
            "Flag reads of signals.json's board-width 'universe' field as a ticker "
            "scope, outside the executor sizing/exit path and the envelope producer "
            "(alpha-engine-config#5809 deliverable 3, #11142)."
        ),
    )
    parser.add_argument(
        "roots",
        nargs="*",
        default=["."],
        help="Directories to scan (default: current directory).",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Also scan tests/ and test_*.py files (excluded by default).",
    )
    args = parser.parse_args(argv)

    all_hits: list[Hit] = []
    for root_arg in args.roots:
        root = Path(root_arg).resolve()
        all_hits.extend(scan_tree(root, include_tests=args.include_tests))

    if not all_hits:
        print("universe-scope-lint: no unexempted board-width reads found")
        return 0

    print(f"universe-scope-lint: {len(all_hits)} unexempted read(s) of universe as scope:")
    for hit in all_hits:
        print("  " + hit.format())
    print(
        "\nEach hit is a read of the sizing envelope as if it were a decision set. "
        "Either resolve the ticker list from universe_membership/{date}/membership.json "
        "via nousergon_lib.decision_set, or — if the site genuinely needs board-width — "
        "add an inline `# universe-scope-lint: allow -- reason=<why>` comment and get it "
        "reviewed."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
