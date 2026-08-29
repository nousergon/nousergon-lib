"""Tests for scripts/anthropic_guard.py (alpha-engine-config-I9263).

Uses a real throwaway git repo per test (the script shells out to
`git ls-files` to enumerate the scan surface), same convention as
test_openrouter_guard.py — this guard mirrors openrouter_guard.py
structurally and this test file mirrors that one.
"""

from __future__ import annotations

import datetime as _dt
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "anthropic_guard.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("anthropic_guard", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["anthropic_guard"] = mod
    spec.loader.exec_module(mod)
    return mod


ag = _load_module()


_GIT = shutil.which("git")


def _git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run([_GIT, "init", "-q"], cwd=repo, check=True)
    subprocess.run([_GIT, "config", "user.email", "test@test"], cwd=repo, check=True)
    subprocess.run([_GIT, "config", "user.name", "test"], cwd=repo, check=True)
    return repo


def _add(repo: Path, rel: str, content: str) -> None:
    fp = repo / rel
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content)
    subprocess.run([_GIT, "-C", str(repo), "add", rel], check=True)


def _commit(repo: Path) -> None:
    subprocess.run([_GIT, "-C", str(repo), "commit", "-q", "-m", "test"], cwd=repo, check=True)


TODAY = _dt.date(2026, 8, 29)
FUTURE = "2026-12-31"
PAST = "2026-01-01"


def test_clean_repo_passes(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "src/app.py", "print('hello world')\n")
    _commit(repo)

    matches = ag.scan(repo, ag.DEFAULT_EXTENSIONS)
    report = ag.evaluate(matches, [], TODAY)
    assert report.ok
    assert ag.render(report, len(matches)) == 0


# --------------------------------------------------------------------------- #
# The regression case this guard exists for: a fresh, unallowlisted direct
# Anthropic SDK call site must fail CI. This is the "fails before the script
# exists / passes after" case — before this module existed, nothing in the
# fleet's CI caught this shape at all.
# --------------------------------------------------------------------------- #


def test_new_sdk_client_construction_fails(tmp_path):
    repo = _git_repo(tmp_path)
    _add(
        repo, "src/client.py",
        "import anthropic\n\nclient = anthropic.Anthropic(api_key=KEY)\n",
    )
    _commit(repo)

    matches = ag.scan(repo, ag.DEFAULT_EXTENSIONS)
    report = ag.evaluate(matches, [], TODAY)
    assert not report.ok
    assert any(m.pattern_class == ag.PATTERN_SDK_CLIENT for m in report.unallowlisted)
    assert ag.render(report, len(matches)) == 1


def test_new_async_sdk_client_construction_fails(tmp_path):
    repo = _git_repo(tmp_path)
    _add(
        repo, "src/client.py",
        "import anthropic\n\nclient = anthropic.AsyncAnthropic(api_key=KEY)\n",
    )
    _commit(repo)

    matches = ag.scan(repo, ag.DEFAULT_EXTENSIONS)
    assert any(m.pattern_class == ag.PATTERN_SDK_CLIENT for m in matches)


def test_new_from_import_fails(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "src/client.py", "from anthropic import Anthropic\n")
    _commit(repo)

    matches = ag.scan(repo, ag.DEFAULT_EXTENSIONS)
    assert any(m.pattern_class == ag.PATTERN_SDK_CLIENT for m in matches)


def test_new_env_key_read_fails(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "src/client.py", 'key = os.environ["ANTHROPIC_API_KEY"]\n')
    _commit(repo)

    matches = ag.scan(repo, ag.DEFAULT_EXTENSIONS)
    report = ag.evaluate(matches, [], TODAY)
    assert not report.ok
    assert report.unallowlisted[0].pattern_class == ag.PATTERN_ENV_KEY


def test_new_base_url_literal_fails(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "src/client.py", 'BASE_URL = "https://api.anthropic.com"\n')
    _commit(repo)

    matches = ag.scan(repo, ag.DEFAULT_EXTENSIONS)
    report = ag.evaluate(matches, [], TODAY)
    assert not report.ok
    assert report.unallowlisted[0].pattern_class == ag.PATTERN_BASE_URL


def test_new_base_url_env_literal_fails(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "src/client.py", 'base = os.environ.get("ANTHROPIC_BASE_URL")\n')
    _commit(repo)

    matches = ag.scan(repo, ag.DEFAULT_EXTENSIONS)
    report = ag.evaluate(matches, [], TODAY)
    assert not report.ok
    assert report.unallowlisted[0].pattern_class == ag.PATTERN_BASE_URL_ENV


def test_docs_excluded_by_default(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "README.md", "We deliberately never call api.anthropic.com directly.\n")
    _commit(repo)

    matches = ag.scan(repo, ag.DEFAULT_EXTENSIONS)
    assert matches == []

    matches_with_docs = ag.scan(repo, ag.DEFAULT_EXTENSIONS | ag.DOC_EXTENSIONS)
    assert len(matches_with_docs) == 1


def test_allowlisted_match_passes(tmp_path):
    repo = _git_repo(tmp_path)
    _add(
        repo, "src/client.py",
        "import anthropic\n\nclient = anthropic.Anthropic(api_key=KEY)\n",
    )
    _commit(repo)

    matches = ag.scan(repo, ag.DEFAULT_EXTENSIONS)
    allowlist = [
        ag.AllowlistEntry(
            path="src/client.py", pattern_class=ag.PATTERN_SDK_CLIENT,
            reason="pre-existing, tracked", expires=_dt.date.fromisoformat(FUTURE),
            tracking="alpha-engine-config-I9999", line_index=0,
        )
    ]
    report = ag.evaluate(matches, allowlist, TODAY)
    assert report.ok
    assert report.covered == 1


def test_expired_allowlist_entry_fails_even_though_match_is_covered(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "src/client.py", 'BASE_URL = "https://api.anthropic.com"\n')
    _commit(repo)

    matches = ag.scan(repo, ag.DEFAULT_EXTENSIONS)
    allowlist = [
        ag.AllowlistEntry(
            path="src/client.py", pattern_class=ag.PATTERN_BASE_URL,
            reason="stale exemption", expires=_dt.date.fromisoformat(PAST),
            tracking=None, line_index=0,
        )
    ]
    report = ag.evaluate(matches, allowlist, TODAY)
    assert not report.ok
    assert len(report.expired) == 1
    assert report.expired[0].path == "src/client.py"


def test_stale_allowlist_entry_with_no_matching_content_fails(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "src/client.py", "print('clean now')\n")
    _commit(repo)

    matches = ag.scan(repo, ag.DEFAULT_EXTENSIONS)
    allowlist = [
        ag.AllowlistEntry(
            path="src/client.py", pattern_class=ag.PATTERN_BASE_URL,
            reason="used to reference api.anthropic.com, now removed",
            expires=_dt.date.fromisoformat(FUTURE),
            tracking=None, line_index=0,
        )
    ]
    report = ag.evaluate(matches, allowlist, TODAY)
    assert not report.ok
    assert len(report.stale) == 1


def test_load_allowlist_rejects_bad_pattern_class(tmp_path):
    path = tmp_path / ".anthropic-allowlist.yaml"
    path.write_text(
        "entries:\n"
        "  - path: x.py\n"
        "    pattern: not-a-real-class\n"
        "    reason: r\n"
        "    expires: '2099-01-01'\n"
    )
    with pytest.raises(ag.GuardError, match="not in"):
        ag.load_allowlist(path)


def test_load_allowlist_requires_all_keys(tmp_path):
    path = tmp_path / ".anthropic-allowlist.yaml"
    path.write_text("entries:\n  - path: x.py\n    pattern: base_url\n")
    with pytest.raises(ag.GuardError, match="missing required keys"):
        ag.load_allowlist(path)


def test_missing_allowlist_file_is_empty(tmp_path):
    assert ag.load_allowlist(tmp_path / "does-not-exist.yaml") == []


def test_main_end_to_end_clean(tmp_path, capsys):
    repo = _git_repo(tmp_path)
    _add(repo, "src/app.py", "print('hello')\n")
    _commit(repo)
    rc = ag.main(["--repo", str(repo)])
    assert rc == 0


def test_main_end_to_end_finds_new_reference(tmp_path, capsys):
    """The regression case, run through the CLI entry point end to end: a
    fixture repo containing `client = anthropic.Anthropic(api_key=KEY)` is
    detected and exits non-zero."""
    repo = _git_repo(tmp_path)
    _add(
        repo, "src/app.py",
        "import anthropic\n\nclient = anthropic.Anthropic(api_key=KEY)\n",
    )
    _commit(repo)
    rc = ag.main(["--repo", str(repo)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "anthropic" in out.lower() or "unallowlisted" in out.lower()


def test_main_end_to_end_with_valid_allowlist(tmp_path, capsys):
    """Same fixture as above, but allowlisted — exits zero."""
    repo = _git_repo(tmp_path)
    _add(
        repo, "src/app.py",
        "import anthropic\n\nclient = anthropic.Anthropic(api_key=KEY)\n",
    )
    _add(
        repo, ".anthropic-allowlist.yaml",
        "entries:\n"
        "  - path: src/app.py\n"
        "    pattern: sdk_client\n"
        "    reason: pre-existing, tracked in alpha-engine-config-I9999\n"
        "    expires: '2099-01-01'\n"
        "    tracking: alpha-engine-config-I9999\n",
    )
    _commit(repo)
    rc = ag.main(["--repo", str(repo)])
    assert rc == 0


# --- the scanner does not scan itself -------------------------------------- #


def test_the_scanner_excludes_its_own_source(tmp_path):
    """A guard flagging its own source is a defect in its own right: every
    pattern it detects is necessarily written out in it, so self-scanning
    costs one allowlist entry per pattern class forever. Handled as a
    property, not as allowlist lines (mirrors openrouter_guard.py, I9111)."""
    repo = _git_repo(tmp_path)
    _add(repo, "scripts/anthropic_guard.py", _SCRIPT.read_text())
    _commit(repo)
    assert ag.scan(repo, ag.DEFAULT_EXTENSIONS) == []


def test_the_scanner_excludes_the_tests_that_load_it(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "tests/test_anthropic_guard.py", Path(__file__).read_text())
    _commit(repo)
    assert ag.scan(repo, ag.DEFAULT_EXTENSIONS) == []


def test_self_exclusion_does_not_cover_an_ordinary_call_site(tmp_path):
    """The exclusion is structural, not a name people can borrow: a file must
    BE the guard or LOAD the guard."""
    repo = _git_repo(tmp_path)
    _add(repo, "app/anthropic_client.py", 'BASE = "https://api.anthropic.com"\n')
    _commit(repo)
    assert ag.scan(repo, ag.DEFAULT_EXTENSIONS) != []


def test_sdk_client_does_not_match_unrelated_anthropic_prose(tmp_path):
    """A comment merely mentioning 'anthropic' as a topic (no SDK construction
    shape) must not fire — mirrors the OpenRouter guard's docs-are-not-linkage
    stance, applied within scanned code files too."""
    repo = _git_repo(tmp_path)
    _add(repo, "src/notes.py", "# we used to use anthropic models here\n")
    _commit(repo)

    matches = ag.scan(repo, ag.DEFAULT_EXTENSIONS)
    assert not any(m.pattern_class == ag.PATTERN_SDK_CLIENT for m in matches)
