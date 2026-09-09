"""Tests for ``nousergon_lib.testing.debug_swallow_guard``
(alpha-engine-config-I10226 — lift of ``crucible-executor``'s
``tests/test_no_debug_only_swallows.py``, alpha-engine-config-I10031).
"""

from __future__ import annotations

import textwrap

import pytest

from nousergon_lib.testing.debug_swallow_guard import (
    check_against_allowlist,
    check_allowlist_entries_self_contained,
    find_debug_only_swallows,
    find_swallow_sites,
    load_allowlist,
)


def _write(path, source: str):
    path.write_text(textwrap.dedent(source))


def test_finds_bare_pass_swallow(tmp_path):
    mod = tmp_path / "mod.py"
    _write(
        mod,
        """
        def f():
            try:
                risky()
            except Exception:
                pass
        """,
    )
    assert find_swallow_sites(mod) == {5}


def test_finds_debug_only_swallow(tmp_path):
    mod = tmp_path / "mod.py"
    _write(
        mod,
        """
        def f():
            try:
                risky()
            except Exception as e:
                logger.debug("failed: %s", e)
        """,
    )
    assert find_swallow_sites(mod) == {5}


def test_control_flow_after_debug_still_in_scope(tmp_path):
    mod = tmp_path / "mod.py"
    _write(
        mod,
        """
        def f():
            for x in items:
                try:
                    risky(x)
                except Exception:
                    logger.debug("skip")
                    continue
        """,
    )
    assert find_swallow_sites(mod) == {6}


def test_ignores_handler_with_second_statement(tmp_path):
    mod = tmp_path / "mod.py"
    _write(
        mod,
        """
        def f():
            try:
                risky()
            except Exception as e:
                logger.error("failed: %s", e)
        """,
    )
    assert find_swallow_sites(mod) == set()


def test_ignores_narrow_except(tmp_path):
    mod = tmp_path / "mod.py"
    _write(
        mod,
        """
        def f():
            try:
                risky()
            except (TypeError, ValueError):
                pass
        """,
    )
    assert find_swallow_sites(mod) == set()


def test_ignores_reraise(tmp_path):
    mod = tmp_path / "mod.py"
    _write(
        mod,
        """
        def f():
            try:
                risky()
            except Exception:
                raise
        """,
    )
    assert find_swallow_sites(mod) == set()


def test_find_debug_only_swallows_scans_directory_non_recursive(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _write(pkg / "a.py", "try:\n    risky()\nexcept Exception:\n    pass\n")
    sub = pkg / "sub"
    sub.mkdir()
    _write(sub / "b.py", "try:\n    risky()\nexcept Exception:\n    pass\n")
    result = find_debug_only_swallows(pkg)
    assert result == {"pkg/a.py": {3}}


def test_check_against_allowlist_flags_uncovered_site():
    live = {"mod.py": {4}}
    failures = check_against_allowlist(live, allowlist=[])
    assert len(failures) == 1
    assert "mod.py:4" in failures[0]


def test_check_against_allowlist_clears_covered_site():
    live = {"mod.py": {4}}
    allowlist = [
        {"path": "mod.py", "line": 4, "reason": "x", "expires": "2099-01-01", "tracking": "I1"}
    ]
    assert check_against_allowlist(live, allowlist) == []


def test_check_against_allowlist_flags_expired_entry():
    live = {"mod.py": {4}}
    allowlist = [
        {"path": "mod.py", "line": 4, "reason": "x", "expires": "2000-01-01", "tracking": "I1"}
    ]
    failures = check_against_allowlist(live, allowlist)
    assert any("expired" in f for f in failures)


def test_check_against_allowlist_flags_stale_entry():
    allowlist = [
        {"path": "mod.py", "line": 4, "reason": "x", "expires": "2099-01-01", "tracking": "I1"}
    ]
    failures = check_against_allowlist({}, allowlist)
    assert any("no longer matches" in f for f in failures)


def test_check_allowlist_entries_self_contained_requires_fields():
    failures = check_allowlist_entries_self_contained([{"path": "mod.py", "line": 4}])
    assert failures


def test_load_allowlist_rejects_wrong_schema_version(tmp_path):
    p = tmp_path / "allow.yaml"
    p.write_text("schema_version: 2\nentries: []\n")
    with pytest.raises(AssertionError):
        load_allowlist(p)
