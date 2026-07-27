"""Tests for nousergon_lib.context — the shared repo-context loader."""

from __future__ import annotations

import os
import time

import pytest

from nousergon_lib.context import _clear_cache, load_repo_context


@pytest.fixture(autouse=True)
def _clear():
    _clear_cache()
    yield
    _clear_cache()


def _write(directory: str, name: str, content: str) -> str:
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def test_returns_none_when_no_file(tmp_path):
    assert load_repo_context(str(tmp_path)) is None


def test_finds_agents_md_at_cwd(tmp_path):
    _write(str(tmp_path), "AGENTS.md", "# Project rules\nBe thorough.")
    result = load_repo_context(str(tmp_path))
    assert result is not None
    assert "Project rules" in result


def test_falls_back_to_claude_md(tmp_path):
    _write(str(tmp_path), "CLAUDE.md", "# Claude instructions\nSOTA only.")
    result = load_repo_context(str(tmp_path))
    assert result is not None
    assert "Claude instructions" in result


def test_prefers_agents_md_over_claude_md(tmp_path):
    _write(str(tmp_path), "AGENTS.md", "# canonical")
    _write(str(tmp_path), "CLAUDE.md", "# legacy")
    result = load_repo_context(str(tmp_path))
    assert result is not None
    assert "canonical" in result
    assert "legacy" not in result


def test_walks_up_from_subdirectory(tmp_path):
    _write(str(tmp_path), "AGENTS.md", "# top-level rules")
    subdir = str(tmp_path / "deep" / "nested" / "folder")
    os.makedirs(subdir, exist_ok=True)
    result = load_repo_context(subdir)
    assert result is not None
    assert "top-level rules" in result


def test_all_levels_are_concatenated_not_replaced(tmp_path):
    """A repo-level file SUPPLEMENTS the fleet-level one; it does not hide it.

    The previous implementation returned the first file found and stopped, so
    adding a small repo AGENTS.md silently shrank an agent's context from the
    full fleet conventions to whatever that file said. Claude Code
    concatenates every level, so the same repo produced different context per
    consumer.

    The old test asserted only that the nearest file was PRESENT, which is
    true under both semantics — so the behaviour it was named for was never
    actually pinned. This asserts both files survive.
    """
    _write(str(tmp_path), "AGENTS.md", "# root")
    sub = str(tmp_path / "project")
    os.makedirs(sub, exist_ok=True)
    _write(sub, "AGENTS.md", "# project-specific")
    result = load_repo_context(sub)
    assert result is not None
    assert "project-specific" in result
    assert "# root" in result, "the fleet-level file must not be hidden"


def test_root_comes_first_so_specific_instructions_win(tmp_path):
    """Ordering matches Claude Code: broadest scope first, most specific last,
    so a repo rule appears after — and therefore overrides — a fleet rule."""
    _write(str(tmp_path), "AGENTS.md", "# root")
    sub = str(tmp_path / "project")
    os.makedirs(sub, exist_ok=True)
    _write(sub, "AGENTS.md", "# project-specific")
    result = load_repo_context(sub)
    assert result.index("# root") < result.index("# project-specific")


def test_each_section_names_its_source(tmp_path):
    """An agent reading concatenated instructions has to know which repo a
    rule came from."""
    _write(str(tmp_path), "AGENTS.md", "# root")
    sub = str(tmp_path / "project")
    os.makedirs(sub, exist_ok=True)
    _write(sub, "AGENTS.md", "# project-specific")
    result = load_repo_context(sub)
    assert "Repository context:" in result
    assert os.path.join(sub, "AGENTS.md") in result


def test_single_file_is_returned_bare(tmp_path):
    """One file, no headers — the common case stays byte-identical to before,
    so nothing that already worked changes shape."""
    _write(str(tmp_path), "AGENTS.md", "# only")
    assert load_repo_context(str(tmp_path)) == "# only"


def test_one_file_per_directory_even_when_both_names_exist(tmp_path):
    """CLAUDE.md is conventionally a symlink to AGENTS.md; reading both would
    duplicate the content in the model's context."""
    _write(str(tmp_path), "AGENTS.md", "# canonical")
    _write(str(tmp_path), "CLAUDE.md", "# canonical")
    result = load_repo_context(str(tmp_path))
    assert result.count("# canonical") == 1


def test_stops_at_filesystem_root():
    result = load_repo_context("/tmp")  # noqa: S108
    # Should not crash — just returns None when no context file exists
    assert result is None


def test_caches_when_mtime_unchanged(tmp_path):
    _write(str(tmp_path), "AGENTS.md", "# cached rules")
    first = load_repo_context(str(tmp_path))
    assert first is not None
    second = load_repo_context(str(tmp_path))
    assert second == first


def test_refreshes_when_mtime_changes(tmp_path):
    _write(str(tmp_path), "AGENTS.md", "# version 1")
    first = load_repo_context(str(tmp_path))
    assert "version 1" in (first or "")

    # Force mtime to advance
    time.sleep(0.05)
    _write(str(tmp_path), "AGENTS.md", "# version 2")

    second = load_repo_context(str(tmp_path))
    assert "version 2" in (second or "")


def test_capped_walk_depth(tmp_path):
    # Build a chain deeper than _MAX_WALK_DEPTH + put the file too deep
    deep = str(tmp_path)
    for i in range(10):  # 10 > _MAX_WALK_DEPTH (8)
        deep = os.path.join(deep, f"level-{i}")
        os.makedirs(deep, exist_ok=True)
    _write(deep, "AGENTS.md", "# too deep")
    result = load_repo_context(str(tmp_path))
    assert result is None
