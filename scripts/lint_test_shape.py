#!/usr/bin/env python3
"""Fail when a test-shaped `def`/`class` will not be collected by pytest.

WHY THIS EXISTS
----------------
`krepis` `tests/test_router.py` carried four `def test_*` functions for
`route_is_degraded` (alpha-engine-config-I10005). They were never collected:
nested inside the helper `_capture_ssm_param`, *after its* `return seen` —
function definitions written into unreachable code, referencing an
out-of-scope `self`.

    pytest --collect-only | grep degraded   ->   (no output)

Nothing was red. Nothing was skipped. The suite reported a passing count
those four tests never contributed to, and the count went up over time from
other work, so no one had a reason to look. They were the ONLY coverage of
`route_is_degraded` — the predicate `crucible/llm.py` reads to stamp
`route_degraded` on the v2 run manifest — so the fleet believed a
load-bearing predicate was tested. It was not, and the predicate was in fact
broken.

**A test that is never collected is indistinguishable from a test that
passes**, from every surface anyone looks at: the run summary, the coverage
gate, the PR check. Fixed in `krepis` by PR203; this script is the guard for
the CLASS, not that one instance.

TWO INDEPENDENT FINDINGS
-------------------------
A. NESTED TEST SHAPE — a `def test_*` (or `async def test_*`) or a
   `class Test*` whose nearest enclosing scope is anything other than
   module scope or a `class Test*` body. pytest's collector only descends
   into module scope and `Test*` classes; anything else (a plain function,
   an `if`, a `with`, a `for`, a `try`) is invisible to it. It parses as
   valid Python and raises no collection error, so nothing about a normal
   test run reveals it.

B. DEAD CODE AFTER AN UNCONDITIONAL `return` — a statement that follows a
   top-level `return` in the body of a function/method defined in a test
   module. This is the SHAPE the krepis defect actually had:
   `_capture_ssm_param` returned, then four `def test_*` were written below
   that return, inside the same body. Finding A already flags the nested
   defs; finding B catches the broader defect even when the trailing dead
   statements are not test-shaped (an assertion, a mutation, anything that
   can never run).

Scope: every `tests/**/*.py` file under the target root (default: cwd).
Second-adoption rule (policy-shared-code): this guard is needed by more than
one repo from the day it ships (krepis, crucible), so it is lifted straight
into `nousergon-lib` rather than starting as a single-repo script.

Usage: python3 scripts/lint_test_shape.py [root]
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Optional

_TEST_FUNC_PREFIX = "test_"
_TEST_CLASS_PREFIX = "Test"

_FUNC_NODE_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)

_PRUNED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "node_modules",
        "site-packages",
        "build",
        "dist",
        ".worktrees",
        "__pycache__",
    }
)


def is_pruned(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    return any(part in _PRUNED_DIRS for part in rel.parts)


def _iter_test_files(root: Path) -> list[Path]:
    found: dict[Path, None] = {}
    for path in root.glob("tests/**/*.py"):
        if path.is_file() and not is_pruned(path, root):
            found[path] = None
    return sorted(found)


def _describe(node: Optional[ast.AST]) -> str:
    if node is None:
        return "<no enclosing scope>"
    if isinstance(node, ast.Module):
        return "module scope"
    if isinstance(node, ast.ClassDef):
        return f"class `{node.name}`"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return f"function `{node.name}`"
    if isinstance(node, ast.If):
        return "an `if` block"
    if isinstance(node, (ast.With, ast.AsyncWith)):
        return "a `with` block"
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return "a `for` loop"
    if isinstance(node, ast.Try):
        return "a `try` block"
    return type(node).__name__


def _is_legal_container(node: Optional[ast.AST]) -> bool:
    if isinstance(node, ast.Module):
        return True
    if isinstance(node, ast.ClassDef) and node.name.startswith(_TEST_CLASS_PREFIX):
        return True
    return False


def _build_parent_map(tree: ast.Module) -> dict:
    parent: dict = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
    return parent


def _check_nested_shape(path: Path, tree: ast.Module, parent: dict) -> list[str]:
    findings: list[str] = []
    for node in ast.walk(tree):
        is_test_func = isinstance(node, _FUNC_NODE_TYPES) and node.name.startswith(
            _TEST_FUNC_PREFIX
        )
        is_test_class = isinstance(node, ast.ClassDef) and node.name.startswith(
            _TEST_CLASS_PREFIX
        )
        if not (is_test_func or is_test_class):
            continue
        enclosing = parent.get(node)
        if _is_legal_container(enclosing):
            continue
        kind = "class" if is_test_class else "function"
        findings.append(
            f"{path}:{node.lineno}: `{node.name}` ({kind}) is nested inside "
            f"{_describe(enclosing)}, not module scope or a `Test*` class "
            f"body — pytest will not collect this."
        )
    return findings


def _check_dead_code_after_return(path: Path, tree: ast.Module) -> list[str]:
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, _FUNC_NODE_TYPES):
            continue
        body = node.body
        for i, stmt in enumerate(body[:-1]):
            if isinstance(stmt, ast.Return):
                dead = body[i + 1]
                findings.append(
                    f"{path}:{dead.lineno}: unreachable statement after an "
                    f"unconditional `return` on line {stmt.lineno}, at the "
                    f"top level of `{node.name}` — pytest will not collect "
                    f"this if it defines a test."
                )
                break
    return findings


def check_module(path: Path, source: str) -> list[str]:
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}: SyntaxError while parsing — {exc}"]
    parent = _build_parent_map(tree)
    findings = _check_nested_shape(path, tree, parent)
    findings.extend(_check_dead_code_after_return(path, tree))
    return findings


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(".")
    root = root.resolve()

    files = _iter_test_files(root)
    if not files:
        # A checker that silently scans nothing reports "clean" for a repo
        # it never opened. That is the failure shape this whole class is
        # about (see nousergon-lib/scripts/lint_extras.py).
        print(
            f"lint_test_shape: ERROR: no test files found under {root}/tests",
            file=sys.stderr,
        )
        return 2

    problems: list[str] = []
    for path in files:
        problems.extend(check_module(path, path.read_text(encoding="utf-8")))

    if problems:
        print(
            "lint_test_shape: test-shaped def/class pytest will not collect "
            "(alpha-engine-config-I10005):",
            file=sys.stderr,
        )
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    print(f"lint_test_shape: OK — {len(files)} test file(s) scanned, all collectible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
