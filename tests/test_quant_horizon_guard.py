"""Tests for nousergon_lib.quant.horizon_guard — the shared wide-horizon
burn-down ratchet primitive (EPIC config#1483 Phase 3, lifted on the second
adoption per config#1527)."""

from __future__ import annotations

import pytest

from nousergon_lib.quant.horizon_guard import (
    WIDE_OUTCOME_COLUMNS,
    BurndownReport,
    assert_burndown,
    check_burndown,
    scan_repo,
    wide_columns_in,
)


def _repo(tmp_path, files: dict[str, str]):
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return tmp_path


CLEAN = "def f():\n    return 1\n"
READS_WIDE = 'QUERY = "SELECT beat_spy_21d FROM score_performance"\n'
READS_AMBIGUOUS = 'COL = "return_5d"  # own feature store\n'
COMMENT_ONLY = "# the old beat_spy_21d column is retired\nx = 1\n"
DOCSTRING = '"""reads beat_spy_21d somewhere"""\nx = 1\n'


def test_columns_match_the_live_wide_schema():
    # Pins the guarded set to the known score_performance naming convention —
    # aligned with nousergon_lib.quant.horizons.outcome_columns.
    from nousergon_lib.quant.horizons import outcome_columns

    for h in (5, 10, 21, 30):
        oc = outcome_columns(h)
        assert oc.beat_spy in WIDE_OUTCOME_COLUMNS
        assert oc.stock_return in WIDE_OUTCOME_COLUMNS
        assert oc.spy_return in WIDE_OUTCOME_COLUMNS
    assert "log_alpha_21d" in WIDE_OUTCOME_COLUMNS


def test_comment_stripped_but_docstrings_matched(tmp_path):
    repo = _repo(tmp_path, {"a.py": COMMENT_ONLY, "b.py": DOCSTRING})
    hits = scan_repo(repo)
    assert "a.py" not in hits          # comments never trip the guard
    assert hits["b.py"] == ["beat_spy_21d"]  # docstrings DO


def test_violation_on_unlisted_production_read(tmp_path):
    repo = _repo(tmp_path, {"prod.py": READS_WIDE, "ok.py": CLEAN})
    report = check_burndown(repo, migrating=())
    assert not report.ok
    assert report.violations == {"prod.py": ["beat_spy_21d"]}
    with pytest.raises(AssertionError, match="score_performance_outcomes"):
        assert_burndown(repo, migrating=())


def test_migrating_allowlist_grandfathers_reads(tmp_path):
    repo = _repo(tmp_path, {"prod.py": READS_WIDE})
    assert check_burndown(repo, migrating=("prod.py",)).ok


def test_stale_migrating_entry_fails_the_ratchet(tmp_path):
    repo = _repo(tmp_path, {"prod.py": CLEAN})
    report = check_burndown(repo, migrating=("prod.py",))
    assert report.stale_migrating == ("prod.py",)
    with pytest.raises(AssertionError, match="burns down"):
        assert_burndown(repo, migrating=("prod.py",))


def test_exempt_is_permanent_but_honest(tmp_path):
    repo = _repo(tmp_path, {"features.py": READS_AMBIGUOUS})
    assert check_burndown(repo, migrating=(), exempt=("features.py",)).ok
    # an exempt file that stops matching must be removed from the exemption
    (repo / "features.py").write_text(CLEAN)
    report = check_burndown(repo, migrating=(), exempt=("features.py",))
    assert report.stale_exempt == ("features.py",)


def test_missing_allowlist_entry_fails(tmp_path):
    repo = _repo(tmp_path, {"ok.py": CLEAN})
    report = check_burndown(repo, migrating=("ghost.py",))
    assert report.missing_entries == ("ghost.py",)


def test_migrating_and_exempt_must_be_disjoint(tmp_path):
    repo = _repo(tmp_path, {"prod.py": READS_WIDE})
    with pytest.raises(ValueError, match="both migrating and exempt"):
        check_burndown(repo, migrating=("prod.py",), exempt=("prod.py",))


def test_tests_and_harness_dirs_excluded(tmp_path):
    repo = _repo(
        tmp_path,
        {
            "tests/test_x.py": READS_WIDE,
            ".claude/worktrees/w/inner.py": READS_WIDE,
            "prod.py": CLEAN,
        },
    )
    assert scan_repo(repo) == {}
    assert check_burndown(repo, migrating=()).ok


def test_single_file_exclude_prefix(tmp_path):
    repo = _repo(tmp_path, {"synthetic/gate_calibration.py": READS_WIDE})
    default = check_burndown(repo, migrating=())
    assert not default.ok
    scoped = check_burndown(
        repo, migrating=(),
        exclude_prefixes=("tests/", "synthetic/gate_calibration.py"),
    )
    assert scoped.ok


def test_wide_columns_in_lists_all_hits(tmp_path):
    p = tmp_path / "m.py"
    p.write_text('a = "beat_spy_5d"\nb = "log_alpha_21d"\n')
    assert wide_columns_in(p) == ["beat_spy_5d", "log_alpha_21d"]


def test_clean_repo_report_is_ok_and_messages_read_well(tmp_path):
    repo = _repo(tmp_path, {"ok.py": CLEAN})
    report = check_burndown(repo, migrating=())
    assert report.ok
    assert report.message() == "burn-down guard: OK"
    assert BurndownReport().ok
