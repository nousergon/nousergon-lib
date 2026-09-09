"""Shared debug-only/pass exception-swallow scanner.

Lifted out of ``crucible-executor/tests/test_no_debug_only_swallows.py``
(alpha-engine-config-I10031, ``crucible-executor-PR547``), on second
adoption per ``policy-shared-code`` — ``alpha-engine-config-I10226``
measured the same class present in five sibling repos (86 sites across
``crucible-research``, ``crucible-predictor``, ``crucible-backtester``,
``nousergon-data``, ``crucible-dashboard``). This module is the detector;
each consumer repo keeps its own ``tests/test_no_debug_only_swallows.py``
(a thin call-site) and its own ``.debug-swallow-allowlist.yaml``, mirroring
``.provider-linkage-allowlist.yaml`` / ``scripts/provider_linkage_guard.py``
(alpha-engine-config-I9295) — a lib-level scanner, a per-repo allowlist.

The class this catches: ``except Exception`` (a bare ``except Exception:``
or ``except Exception as e:`` — scoped to match the fleet's own
measurement, ``grep -rn -A1 "except Exception"``; a narrower catch is a
deliberate, scoped catch and out of scope, see ``~/Development/CLAUDE.md``
"Fail loud and fast") whose ENTIRE body is a single call to
``logger.debug(...)`` or a bare ``pass``. A handler that also does
something else (re-raises, records via ``logger.error``/``logger.warning``,
returns an in-band error value, calls ``fd.report(...)``) is not in scope —
the invisible-record shape is specifically a body with nothing else in it.
A trailing ``continue``/``break``/``return`` after the debug/pass is still
in scope: that is control flow, not a second record.

Consumers call :func:`find_debug_only_swallows` over their own source
directory and diff the result against a repo-local allowlist — see any of
the five repos above (or ``crucible-executor``, the original) for the
call-site shape.
"""

from __future__ import annotations

import ast
import datetime as dt
from pathlib import Path

import yaml


def _is_debug_call(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == "debug"
        and isinstance(stmt.value.func.value, ast.Name)
        and stmt.value.func.value.id == "logger"
    )


def _is_debug_only_or_pass(body: list[ast.stmt]) -> bool:
    record_stmts = [s for s in body if isinstance(s, ast.Pass) or _is_debug_call(s)]
    if len(record_stmts) != 1:
        return False
    other_stmts = [s for s in body if s not in record_stmts]
    return all(isinstance(s, (ast.Continue, ast.Break, ast.Return)) for s in other_stmts)


def _is_bare_except_exception(handler: ast.ExceptHandler) -> bool:
    return isinstance(handler.type, ast.Name) and handler.type.id == "Exception"


def find_swallow_sites(path: Path) -> set[int]:
    """Line numbers of bare ``except Exception`` clauses in ``path`` whose
    body is a debug-only-or-pass swallow. Raises ``SyntaxError`` unchanged
    if ``path`` does not parse — a scanner that silently skipped an
    unparseable file would itself be a swallow."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    sites: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                if _is_bare_except_exception(handler) and _is_debug_only_or_pass(handler.body):
                    sites.add(handler.lineno)
    return sites


def find_debug_only_swallows(
    source_dir: Path, *, repo_root: Path | None = None
) -> dict[str, set[int]]:
    """Scan every ``*.py`` file directly under ``source_dir`` (non-recursive,
    matching the original ``executor/*.py`` shape — pass each package
    directory separately for a multi-package repo) for debug-only/pass
    exception swallows.

    Returns ``{relative_path: {line_numbers}}``, keyed relative to
    ``repo_root`` (defaults to ``source_dir``'s parent) so the result lines
    up with the ``path``/``line`` keys a consumer's allowlist file uses.
    """
    root = repo_root if repo_root is not None else source_dir.parent
    return {
        str(p.relative_to(root)): find_swallow_sites(p)
        for p in sorted(source_dir.glob("*.py"))
    }


def load_allowlist(allowlist_path: Path) -> list[dict]:
    """Load and shape-check a ``.debug-swallow-allowlist.yaml`` file
    (``schema_version: 1``, mirrors ``.provider-linkage-allowlist.yaml``)."""
    doc = yaml.safe_load(allowlist_path.read_text(encoding="utf-8"))
    assert doc.get("schema_version") == 1, "unrecognized allowlist schema_version"
    return doc["entries"]


def check_against_allowlist(
    live_sites: dict[str, set[int]], allowlist: list[dict]
) -> list[str]:
    """Diff ``live_sites`` (from :func:`find_debug_only_swallows`) against
    a loaded allowlist. Returns human-readable failure lines: an
    unallowlisted new site, an allowlist entry past its ``expires`` date,
    or an entry that no longer matches any live site (so the allowance
    cannot quietly widen after the site it covered is fixed or moves).
    Empty list means clean."""
    today = dt.date.today()

    allowed: set[tuple[str, int]] = set()
    expired: list[str] = []
    for entry in allowlist:
        key = (entry["path"], entry["line"])
        expires = dt.date.fromisoformat(entry["expires"])
        if expires < today:
            expired.append(f"{entry['path']}:{entry['line']} expired {expires} — re-justify or remove")
            continue
        allowed.add(key)

    uncovered: list[str] = []
    for path, lines in live_sites.items():
        for line in lines:
            if (path, line) not in allowed:
                uncovered.append(
                    f"{path}:{line} — new debug-only/pass swallow with no "
                    "allowlist entry. Raise it, record it at WARNING/ERROR+ "
                    "with a named recording surface, or add a justified, "
                    "expiring entry to .debug-swallow-allowlist.yaml."
                )

    stale: list[str] = []
    for entry in allowlist:
        key = (entry["path"], entry["line"])
        if entry["path"] not in live_sites or entry["line"] not in live_sites[entry["path"]]:
            stale.append(
                f"{entry['path']}:{entry['line']} no longer matches a "
                "debug-only/pass swallow — remove the stale entry so the "
                "allowance cannot quietly widen."
            )

    return expired + uncovered + stale


def check_allowlist_entries_self_contained(allowlist: list[dict]) -> list[str]:
    """Every entry must name a reason, an expiry, and a tracking issue — a
    swallow with no named recording surface is not a swallow, it is a
    deletion. Returns human-readable failure lines; empty means clean."""
    failures: list[str] = []
    for entry in allowlist:
        for field in ("path", "line", "reason", "expires", "tracking"):
            if not entry.get(field):
                failures.append(f"allowlist entry missing {field!r}: {entry}")
        if entry.get("reason") is not None and not str(entry["reason"]).strip():
            failures.append(f"empty reason: {entry}")
    return failures
