"""nousergon_lib.quant.horizon_guard — the wide-horizon-column burn-down guard,
as a shared testing primitive (EPIC config#1483 Phase 3).

WHY THIS EXISTS
---------------
The eval horizon was historically encoded in wide, horizon-suffixed
``score_performance`` column names (``beat_spy_21d``, ``spy_5d_return``, …).
An incomplete horizon rename silently starves consumers — the config#1456 bug
class. The root-cause fix (EPIC config#1483) moves outcomes to the long-format
``score_performance_outcomes`` store read via
:mod:`nousergon_lib.quant.horizons`, and each consumer repo enforces the
migration with a RATCHET test:

  * any production (non-test) read of a wide horizon-suffixed outcome column
    fails, UNLESS the file is on the repo's ``migrating`` allowlist (files
    known to still read the wide columns at ratchet-seed time);
  * a ``migrating`` file that has become CLEAN also fails — forcing the
    allowlist to burn down to ``{}`` as each cutover PR lands;
  * an optional ``exempt`` set holds files that legitimately contain the
    ambiguous literals FOREVER (e.g. readers of ``universe_returns`` or a
    repo's own feature store, whose column names collide with the
    score_performance ones) — an exempt file that stops matching entirely
    also fails, so the exemption list can't rot.

The first adoption lived in
``crucible-backtester/tests/test_wide_horizon_column_burndown_guard.py``; this
module lifts the scan/ratchet mechanics on the second adoption (the
config#1527 predictor cutover) per the "lift the invariant to a chokepoint
after the second recurrence" rule, so predictor/research/dashboard/evaluator
seed their guards as ~10-line test files instead of divergent mirrors.
Repo-specific honesty checks (e.g. backtester's "exempt files must read
universe_returns, not score_performance") stay repo-side.

Pure stdlib — no pytest dependency; failures raise ``AssertionError`` with an
actionable message, so any test runner surfaces them directly.
"""

from __future__ import annotations

import io
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

__all__ = [
    "WIDE_OUTCOME_COLUMNS",
    "DEFAULT_EXCLUDE_PREFIXES",
    "BurndownReport",
    "strip_comments",
    "wide_columns_in",
    "production_py_files",
    "scan_repo",
    "check_burndown",
    "assert_burndown",
]

# The wide horizon-suffixed score_performance OUTCOME columns the long-format
# store replaces (ground truth: the live score_performance schema — see
# nousergon_lib.quant.horizons.OutcomeColumns). ``return_{N}d``/``beat_spy_{N}d``
# are ambiguous (identical literals exist in universe_returns / feature stores)
# — repos exempt those readers; ``spy_{N}d_return`` + ``log_alpha_21d`` are
# unambiguously score_performance.
WIDE_OUTCOME_COLUMNS: tuple[str, ...] = (
    "beat_spy_5d", "beat_spy_10d", "beat_spy_21d", "beat_spy_30d",
    "spy_5d_return", "spy_10d_return", "spy_21d_return", "spy_30d_return",
    "return_5d", "return_10d", "return_21d", "return_30d",
    "log_alpha_21d",
)

# Non-production paths never scanned. ``tests/`` is excluded because the guard
# targets production reads; fixtures/parity tests legitimately spell the wide
# names. ``.claude/`` covers harness worktrees nested inside a repo checkout.
DEFAULT_EXCLUDE_PREFIXES: tuple[str, ...] = (
    "tests/", ".venv", ".claude/", ".git/",
)


def strip_comments(src: str) -> str:
    """Drop ``#`` comments (migration-explaining comments must not trip the
    guard) while KEEPING string literals + docstrings (a SQL SELECT / dict key
    / f-string building a wide-column name is a real read)."""
    try:
        return "\n".join(
            tok.string
            for tok in tokenize.generate_tokens(io.StringIO(src).readline)
            if tok.type != tokenize.COMMENT
        )
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src


def wide_columns_in(
    path: Path | str,
    columns: Iterable[str] = WIDE_OUTCOME_COLUMNS,
) -> list[str]:
    """Wide-column literals present in a file (comment-stripped)."""
    code = strip_comments(Path(path).read_text(errors="ignore"))
    return sorted({c for c in columns if c in code})


def production_py_files(
    repo_root: Path | str,
    exclude_prefixes: Iterable[str] = DEFAULT_EXCLUDE_PREFIXES,
) -> list[Path]:
    """All production ``.py`` files under ``repo_root``.

    A file is excluded when its repo-relative posix path starts with an
    exclude prefix, or contains one as an intermediate path segment
    (``pkg/.venv/…``). Prefixes may also name a single file
    (``synthetic/gate_calibration.py``).
    """
    root = Path(repo_root)
    prefixes = tuple(exclude_prefixes)
    out: list[Path] = []
    for f in root.rglob("*.py"):
        rel = f.relative_to(root).as_posix()
        if any(rel.startswith(p) or f"/{p}" in f"/{rel}" for p in prefixes):
            continue
        out.append(f)
    return out


def scan_repo(
    repo_root: Path | str,
    columns: Iterable[str] = WIDE_OUTCOME_COLUMNS,
    exclude_prefixes: Iterable[str] = DEFAULT_EXCLUDE_PREFIXES,
) -> dict[str, list[str]]:
    """{repo-relative path: wide-column hits} for every production file
    containing at least one wide-column literal."""
    root = Path(repo_root)
    result: dict[str, list[str]] = {}
    for f in production_py_files(root, exclude_prefixes):
        hits = wide_columns_in(f, columns)
        if hits:
            result[f.relative_to(root).as_posix()] = hits
    return result


@dataclass(frozen=True)
class BurndownReport:
    """Outcome of one guard evaluation. ``ok`` iff the ratchet holds."""

    violations: Mapping[str, list[str]] = field(default_factory=dict)
    stale_migrating: tuple[str, ...] = ()
    stale_exempt: tuple[str, ...] = ()
    missing_entries: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not (
            self.violations
            or self.stale_migrating
            or self.stale_exempt
            or self.missing_entries
        )

    def message(self) -> str:
        parts: list[str] = []
        if self.violations:
            parts.append(
                "NEW wide horizon-suffixed column read(s) in production files "
                "not on the migrating allowlist — read the long-format "
                "score_performance_outcomes store via "
                "nousergon_lib.quant.horizons instead (EPIC config#1483): "
                f"{dict(self.violations)}"
            )
        if self.stale_migrating:
            parts.append(
                "STALE migrating entries (files now clean) — remove them so "
                f"the allowlist burns down: {sorted(self.stale_migrating)}"
            )
        if self.stale_exempt:
            parts.append(
                "STALE exempt entries (files no longer contain any wide-column "
                f"literal) — remove the exemption: {sorted(self.stale_exempt)}"
            )
        if self.missing_entries:
            parts.append(
                "allowlist entries that do not exist on disk: "
                f"{sorted(self.missing_entries)}"
            )
        return "\n".join(parts) if parts else "burn-down guard: OK"


def check_burndown(
    repo_root: Path | str,
    migrating: Iterable[str],
    exempt: Iterable[str] = (),
    columns: Iterable[str] = WIDE_OUTCOME_COLUMNS,
    exclude_prefixes: Iterable[str] = DEFAULT_EXCLUDE_PREFIXES,
) -> BurndownReport:
    """Evaluate the burn-down ratchet. See module docstring for the rules."""
    root = Path(repo_root)
    migrating_set = frozenset(migrating)
    exempt_set = frozenset(exempt)
    overlap = migrating_set & exempt_set
    if overlap:
        raise ValueError(
            f"files cannot be both migrating and exempt: {sorted(overlap)}"
        )

    hits_by_file = scan_repo(root, columns, exclude_prefixes)

    violations = {
        rel: hits
        for rel, hits in hits_by_file.items()
        if rel not in migrating_set and rel not in exempt_set
    }
    missing = tuple(
        sorted(
            rel
            for rel in (migrating_set | exempt_set)
            if not (root / rel).is_file()
        )
    )
    stale_migrating = tuple(
        sorted(
            rel
            for rel in migrating_set
            if (root / rel).is_file() and rel not in hits_by_file
        )
    )
    stale_exempt = tuple(
        sorted(
            rel
            for rel in exempt_set
            if (root / rel).is_file() and rel not in hits_by_file
        )
    )
    return BurndownReport(
        violations=violations,
        stale_migrating=stale_migrating,
        stale_exempt=stale_exempt,
        missing_entries=missing,
    )


def assert_burndown(
    repo_root: Path | str,
    migrating: Iterable[str],
    exempt: Iterable[str] = (),
    columns: Iterable[str] = WIDE_OUTCOME_COLUMNS,
    exclude_prefixes: Iterable[str] = DEFAULT_EXCLUDE_PREFIXES,
) -> None:
    """Raise ``AssertionError`` (with an actionable message) unless the
    burn-down ratchet holds. The one-call form a consumer repo's guard test
    uses::

        from nousergon_lib.quant.horizon_guard import assert_burndown

        def test_wide_horizon_column_burndown():
            assert_burndown(REPO_ROOT, migrating=_MIGRATING, exempt=_EXEMPT)
    """
    report = check_burndown(repo_root, migrating, exempt, columns, exclude_prefixes)
    assert report.ok, report.message()
