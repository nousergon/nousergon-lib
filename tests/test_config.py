"""Tests for ``nousergon_lib.config.resolve_experiment_config`` — the canonical
experiment-package-first config resolver lifted from the five inline copies
(alpha-engine-config#1157).

The search-order consensus under test (per config root, then repo-local):

  1. ``<root>/experiments/<exp>/<subdir>/<file>``    (experiment package)
  2. ``<root>/<subdir>/<file>``                      (legacy top-level)
  3. repo-local fallback(s)

config roots = ``~/alpha-engine-config`` then ``<repo_root>/../alpha-engine-config``
(+ ``$GITHUB_WORKSPACE/alpha-engine-config`` when opted in). ``exp`` defaults to
``$ALPHA_ENGINE_EXPERIMENT_ID`` or ``reference``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nousergon_lib.config import (
    DEFAULT_EXPERIMENT_ID,
    resolve_experiment_config,
)


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Point ``Path.home()`` (via $HOME) at an isolated dir with no config repo."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    # Path.home() consults USERPROFILE on Windows; harmless to set on POSIX.
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


@pytest.fixture
def repo_root(tmp_path):
    """A consumer repo root with a parent dir available for the sibling clone."""
    root = tmp_path / "parent" / "crucible-executor"
    root.mkdir(parents=True)
    return root


@pytest.fixture(autouse=True)
def _clear_experiment_env(monkeypatch):
    monkeypatch.delenv("ALPHA_ENGINE_EXPERIMENT_ID", raising=False)
    monkeypatch.delenv("GITHUB_WORKSPACE", raising=False)


def _write(path: Path, content: str = "x: 1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


# ── candidate-list shape (resolve=False) ──────────────────────────────────────


def test_candidate_list_order_default_experiment(fake_home, repo_root):
    """Default (resolve=False) returns the full ordered candidate list with the
    ``reference`` experiment baked in, experiment-package first."""
    cands = resolve_experiment_config("executor", "risk.yaml", repo_root=repo_root)
    sibling = repo_root.parent / "alpha-engine-config"
    assert cands == [
        fake_home / "alpha-engine-config" / "experiments" / "reference" / "executor" / "risk.yaml",
        sibling / "experiments" / "reference" / "executor" / "risk.yaml",
        fake_home / "alpha-engine-config" / "executor" / "risk.yaml",
        sibling / "executor" / "risk.yaml",
        repo_root / "executor" / "risk.yaml",
    ]


def test_candidate_list_does_not_touch_filesystem(fake_home, repo_root):
    """resolve=False must not require any candidate to exist."""
    cands = resolve_experiment_config("data", "config.yaml", repo_root=repo_root)
    assert all(not p.exists() for p in cands)
    assert len(cands) == 5


# ── default experiment id ─────────────────────────────────────────────────────


def test_reference_is_the_default_experiment(fake_home, repo_root):
    assert DEFAULT_EXPERIMENT_ID == "reference"
    cands = resolve_experiment_config("research", "universe.yaml", repo_root=repo_root)
    assert "reference" in str(cands[0])
    assert cands[0].parts[-3] == "reference"


def test_experiment_id_from_env(fake_home, repo_root, monkeypatch):
    monkeypatch.setenv("ALPHA_ENGINE_EXPERIMENT_ID", "exp_42")
    cands = resolve_experiment_config("research", "universe.yaml", repo_root=repo_root)
    assert cands[0].parts[-3] == "exp_42"
    # legacy + local layers are unaffected by the experiment id
    assert "experiments" not in str(cands[-1])


def test_experiment_id_explicit_overrides_env(fake_home, repo_root, monkeypatch):
    monkeypatch.setenv("ALPHA_ENGINE_EXPERIMENT_ID", "from_env")
    cands = resolve_experiment_config(
        "research", "universe.yaml", repo_root=repo_root, experiment_id="explicit"
    )
    assert cands[0].parts[-3] == "explicit"


# ── resolution: package-first wins ────────────────────────────────────────────


def test_package_first_wins_over_legacy_and_local(fake_home, repo_root):
    """When all three layers exist, the experiment-package copy resolves."""
    sibling = repo_root.parent / "alpha-engine-config"
    pkg = _write(fake_home / "alpha-engine-config" / "experiments" / "reference" / "executor" / "risk.yaml")
    _write(fake_home / "alpha-engine-config" / "executor" / "risk.yaml")
    _write(repo_root / "executor" / "risk.yaml")
    _write(sibling / "experiments" / "reference" / "executor" / "risk.yaml")
    got = resolve_experiment_config("executor", "risk.yaml", repo_root=repo_root, resolve=True)
    assert got == pkg


def test_sibling_root_searched_when_home_absent(fake_home, repo_root):
    """The ``<repo_root>/../alpha-engine-config`` sibling root resolves when the
    home clone is absent."""
    sibling = repo_root.parent / "alpha-engine-config"
    pkg = _write(sibling / "experiments" / "reference" / "executor" / "risk.yaml")
    got = resolve_experiment_config("executor", "risk.yaml", repo_root=repo_root, resolve=True)
    assert got == pkg


# ── resolution: legacy fallback ───────────────────────────────────────────────


def test_legacy_fallback_when_no_experiment_package(fake_home, repo_root):
    """Falls through to the legacy top-level copy when no experiment package
    exists anywhere."""
    legacy = _write(fake_home / "alpha-engine-config" / "executor" / "risk.yaml")
    _write(repo_root / "executor" / "risk.yaml")  # local also present, must lose
    got = resolve_experiment_config("executor", "risk.yaml", repo_root=repo_root, resolve=True)
    assert got == legacy


def test_legacy_sibling_before_local(fake_home, repo_root):
    sibling = repo_root.parent / "alpha-engine-config"
    legacy = _write(sibling / "executor" / "risk.yaml")
    _write(repo_root / "executor" / "risk.yaml")
    got = resolve_experiment_config("executor", "risk.yaml", repo_root=repo_root, resolve=True)
    assert got == legacy


# ── resolution: repo-local fallback ───────────────────────────────────────────


def test_repo_local_fallback_when_no_config_repo(fake_home, repo_root):
    """With no config-repo copies at all, the repo-local default resolves."""
    local = _write(repo_root / "executor" / "risk.yaml")
    got = resolve_experiment_config("executor", "risk.yaml", repo_root=repo_root, resolve=True)
    assert got == local


def test_repo_local_fallback_override(fake_home, repo_root):
    """``repo_local_fallback`` overrides the default ``<repo_root>/<subdir>/<file>``
    — e.g. research/executor's subdir-flattened ``<repo_root>/config/<file>``."""
    flattened = _write(repo_root / "config" / "risk.yaml")
    got = resolve_experiment_config(
        "executor",
        "risk.yaml",
        repo_root=repo_root,
        repo_local_fallback=repo_root / "config" / "risk.yaml",
        resolve=True,
    )
    assert got == flattened
    # And the default location is NOT searched once overridden.
    cands = resolve_experiment_config(
        "executor",
        "risk.yaml",
        repo_root=repo_root,
        repo_local_fallback=repo_root / "config" / "risk.yaml",
    )
    assert repo_root / "config" / "risk.yaml" in cands
    assert repo_root / "executor" / "risk.yaml" not in cands


# ── extra_fallbacks ───────────────────────────────────────────────────────────


def test_extra_fallbacks_appended_after_consensus(fake_home, repo_root, tmp_path):
    extra = tmp_path / "extra" / "risk.yaml"
    cands = resolve_experiment_config(
        "executor", "risk.yaml", repo_root=repo_root, extra_fallbacks=(extra,)
    )
    assert cands[-1] == extra
    # consensus order preserved ahead of it
    assert cands[-2] == repo_root / "executor" / "risk.yaml"


def test_extra_fallbacks_resolve_after_default_local(fake_home, repo_root, tmp_path):
    """An extra fallback resolves only when earlier layers (incl. the default
    repo-local) are absent."""
    extra = _write(tmp_path / "extra" / "risk.yaml")
    got = resolve_experiment_config(
        "executor",
        "risk.yaml",
        repo_root=repo_root,
        extra_fallbacks=(extra,),
        resolve=True,
    )
    assert got == extra

    # but the default repo-local still wins over the extra when present
    local = _write(repo_root / "executor" / "risk.yaml")
    got2 = resolve_experiment_config(
        "executor",
        "risk.yaml",
        repo_root=repo_root,
        extra_fallbacks=(extra,),
        resolve=True,
    )
    assert got2 == local


def test_extra_fallbacks_string_paths_accepted(fake_home, repo_root, tmp_path):
    extra = _write(tmp_path / "extra" / "risk.yaml")
    got = resolve_experiment_config(
        "executor",
        "risk.yaml",
        repo_root=repo_root,
        extra_fallbacks=(str(extra),),
        resolve=True,
    )
    assert got == extra


# ── github_workspace option (research's CI root) ──────────────────────────────


def test_github_workspace_root_opt_in_explicit_path(fake_home, repo_root, tmp_path):
    ws = tmp_path / "ws"
    cands = resolve_experiment_config(
        "research", "universe.yaml", repo_root=repo_root, github_workspace=str(ws)
    )
    ws_pkg = ws / "alpha-engine-config" / "experiments" / "reference" / "research" / "universe.yaml"
    assert ws_pkg in cands
    # Default (None) omits the CI root → 5 candidates, not 7.
    default = resolve_experiment_config("research", "universe.yaml", repo_root=repo_root)
    assert ws_pkg not in default
    assert len(default) == 5
    # research has three roots → 3 pkg + 3 legacy + 1 local = 7
    assert len(cands) == 7


def test_github_workspace_true_reads_env(fake_home, repo_root, tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    monkeypatch.setenv("GITHUB_WORKSPACE", str(ws))
    cands = resolve_experiment_config(
        "research", "universe.yaml", repo_root=repo_root, github_workspace=True
    )
    ws_pkg = ws / "alpha-engine-config" / "experiments" / "reference" / "research" / "universe.yaml"
    assert ws_pkg in cands


def test_github_workspace_true_with_unset_env_is_noop(fake_home, repo_root, monkeypatch):
    monkeypatch.delenv("GITHUB_WORKSPACE", raising=False)
    cands = resolve_experiment_config(
        "research", "universe.yaml", repo_root=repo_root, github_workspace=True
    )
    assert len(cands) == 5  # no CI root added when env unset


# ── exclude_suffixes option (executor's .example guard) ───────────────────────


def test_exclude_suffix_filters_from_candidate_list(fake_home, repo_root):
    cands = resolve_experiment_config(
        "executor",
        "risk.yaml.example",
        repo_root=repo_root,
        exclude_suffixes=(".example",),
    )
    assert cands == []  # every candidate ends in .example → all excluded


def test_exclude_suffix_never_resolves_example(fake_home, repo_root):
    """The ``.example`` template must never auto-resolve even when present —
    executor's silent-fallthrough guard (placeholder bucket names)."""
    # Put a real repo-local fallback under the .example name; with the guard it
    # must NOT resolve, and resolution raises.
    _write(repo_root / "executor" / "risk.yaml.example")
    with pytest.raises(FileNotFoundError):
        resolve_experiment_config(
            "executor",
            "risk.yaml.example",
            repo_root=repo_root,
            exclude_suffixes=(".example",),
            resolve=True,
        )


def test_exclude_suffix_leaves_normal_files_resolvable(fake_home, repo_root):
    """The guard only excludes ``.example`` tails; the real ``risk.yaml`` still
    resolves with the guard active."""
    local = _write(repo_root / "executor" / "risk.yaml")
    got = resolve_experiment_config(
        "executor",
        "risk.yaml",
        repo_root=repo_root,
        exclude_suffixes=(".example",),
        resolve=True,
    )
    assert got == local


# ── resolve_symlinks option (executor realpath/isfile semantics) ──────────────


def test_resolve_symlinks_follows_link_to_real_file(fake_home, repo_root, tmp_path):
    target = _write(tmp_path / "real" / "risk.yaml")
    link_dir = repo_root / "executor"
    link_dir.mkdir(parents=True)
    link = link_dir / "risk.yaml"
    link.symlink_to(target)
    got = resolve_experiment_config(
        "executor", "risk.yaml", repo_root=repo_root, resolve=True, resolve_symlinks=True
    )
    # Returns the realpath (executor semantics), i.e. the link target.
    assert got == Path(os.path.realpath(link)) == target.resolve()


def test_resolve_symlinks_rejects_directory(fake_home, repo_root):
    """os.path.isfile semantics: a directory at the candidate path is not a
    match (Path.exists() would wrongly accept it)."""
    # Make the repo-local candidate a DIRECTORY rather than a file.
    (repo_root / "executor" / "risk.yaml").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        resolve_experiment_config(
            "executor", "risk.yaml", repo_root=repo_root, resolve=True, resolve_symlinks=True
        )
    # Without symlink-resolution, Path.exists() accepts the directory — proving
    # the option changes behavior as intended.
    got = resolve_experiment_config(
        "executor", "risk.yaml", repo_root=repo_root, resolve=True
    )
    assert got == repo_root / "executor" / "risk.yaml"


# ── error path ────────────────────────────────────────────────────────────────


def test_resolve_raises_with_searched_paths_listed(fake_home, repo_root):
    with pytest.raises(FileNotFoundError) as exc:
        resolve_experiment_config("executor", "risk.yaml", repo_root=repo_root, resolve=True)
    msg = str(exc.value)
    assert "executor/risk.yaml" in msg
    assert "experiments" in msg  # the experiment-package candidate is named


def test_custom_error_message(fake_home, repo_root):
    with pytest.raises(FileNotFoundError, match="copy the template"):
        resolve_experiment_config(
            "executor",
            "risk.yaml",
            repo_root=repo_root,
            resolve=True,
            error_message="risk.yaml not found — copy the template",
        )


# ── string repo_root accepted ─────────────────────────────────────────────────


def test_repo_root_accepts_str(fake_home, repo_root):
    local = _write(repo_root / "executor" / "risk.yaml")
    got = resolve_experiment_config(
        "executor", "risk.yaml", repo_root=str(repo_root), resolve=True
    )
    assert got == local


# ── consumer-shape regression: each inline copy's call reproduces its order ────


def test_backtester_shape(fake_home, repo_root):
    """backtester: load_config(path) → repo-local default is Path(path)."""
    cands = resolve_experiment_config(
        "backtester",
        "config.yaml",
        repo_root=repo_root,
        repo_local_fallback="some/given/path.yaml",
    )
    assert cands[0].parts[-4:] == ("experiments", "reference", "backtester", "config.yaml")
    assert cands[-1] == Path("some/given/path.yaml")


def test_executor_shape(fake_home, repo_root):
    """executor: config/risk.yaml local + .example guard + symlink realpath."""
    cands = resolve_experiment_config(
        "executor",
        "risk.yaml",
        repo_root=repo_root,
        repo_local_fallback=repo_root / "config" / "risk.yaml",
        exclude_suffixes=(".example",),
    )
    assert cands[0] == fake_home / "alpha-engine-config" / "experiments" / "reference" / "executor" / "risk.yaml"
    assert cands[-1] == repo_root / "config" / "risk.yaml"
