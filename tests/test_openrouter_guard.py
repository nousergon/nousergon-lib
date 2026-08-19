"""Tests for scripts/openrouter_guard.py (alpha-engine-config-I6564).

Uses a real throwaway git repo per test (the script shells out to
`git ls-files` to enumerate the scan surface, same convention as
validate_llm_callsite_registry.py's presence check).
"""

from __future__ import annotations

import datetime as _dt
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "openrouter_guard.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("openrouter_guard", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["openrouter_guard"] = mod
    spec.loader.exec_module(mod)
    return mod


og = _load_module()


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


TODAY = _dt.date(2026, 8, 19)
FUTURE = "2026-12-31"
PAST = "2026-01-01"


def test_clean_repo_passes(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "src/app.py", "print('hello world')\n")
    _commit(repo)

    matches = og.scan(repo, og.DEFAULT_EXTENSIONS)
    report = og.evaluate(matches, [], TODAY)
    assert report.ok
    assert og.render(report, len(matches)) == 0


def test_new_base_url_literal_fails(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "src/client.py", 'BASE_URL = "https://openrouter.ai/api/v1"\n')
    _commit(repo)

    matches = og.scan(repo, og.DEFAULT_EXTENSIONS)
    report = og.evaluate(matches, [], TODAY)
    assert not report.ok
    assert len(report.unallowlisted) == 1
    assert report.unallowlisted[0].pattern_class == og.PATTERN_BASE_URL
    assert og.render(report, len(matches)) == 1


def test_new_env_key_read_fails(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "src/client.py", 'key = os.environ["OPENROUTER_API_KEY"]\n')
    _commit(repo)

    matches = og.scan(repo, og.DEFAULT_EXTENSIONS)
    report = og.evaluate(matches, [], TODAY)
    assert not report.ok
    assert report.unallowlisted[0].pattern_class == og.PATTERN_ENV_KEY


def test_new_provider_literal_fails(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "config/models.yaml", "provider: openrouter\nmodel: glm-5.2\n")
    _commit(repo)

    matches = og.scan(repo, og.DEFAULT_EXTENSIONS)
    report = og.evaluate(matches, [], TODAY)
    assert not report.ok
    assert report.unallowlisted[0].pattern_class == og.PATTERN_PROVIDER_LITERAL


def test_provider_literal_does_not_match_other_values(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "config/models.yaml", "provider: openrouter_shadow_retired\nprovider: anthropic\n")
    _commit(repo)

    matches = og.scan(repo, og.DEFAULT_EXTENSIONS)
    assert not any(m.pattern_class == og.PATTERN_PROVIDER_LITERAL for m in matches)


def test_docs_excluded_by_default(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "README.md", "We deliberately never call openrouter.ai directly.\n")
    _commit(repo)

    matches = og.scan(repo, og.DEFAULT_EXTENSIONS)
    assert matches == []

    matches_with_docs = og.scan(repo, og.DEFAULT_EXTENSIONS | og.DOC_EXTENSIONS)
    assert len(matches_with_docs) == 1


def test_allowlisted_match_passes(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "src/client.py", 'BASE_URL = "https://openrouter.ai/api/v1"\n')
    _commit(repo)

    matches = og.scan(repo, og.DEFAULT_EXTENSIONS)
    allowlist = [
        og.AllowlistEntry(
            path="src/client.py", pattern_class=og.PATTERN_BASE_URL,
            reason="pre-existing, tracked", expires=_dt.date.fromisoformat(FUTURE),
            tracking="alpha-engine-config-I9999", line_index=0,
        )
    ]
    report = og.evaluate(matches, allowlist, TODAY)
    assert report.ok
    assert report.covered == 1


def test_expired_allowlist_entry_fails_even_though_match_is_covered(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "src/client.py", 'BASE_URL = "https://openrouter.ai/api/v1"\n')
    _commit(repo)

    matches = og.scan(repo, og.DEFAULT_EXTENSIONS)
    allowlist = [
        og.AllowlistEntry(
            path="src/client.py", pattern_class=og.PATTERN_BASE_URL,
            reason="stale exemption", expires=_dt.date.fromisoformat(PAST),
            tracking=None, line_index=0,
        )
    ]
    report = og.evaluate(matches, allowlist, TODAY)
    assert not report.ok
    assert len(report.expired) == 1
    assert report.expired[0].path == "src/client.py"


def test_stale_allowlist_entry_with_no_matching_content_fails(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "src/client.py", "print('clean now')\n")
    _commit(repo)

    matches = og.scan(repo, og.DEFAULT_EXTENSIONS)
    allowlist = [
        og.AllowlistEntry(
            path="src/client.py", pattern_class=og.PATTERN_BASE_URL,
            reason="used to reference openrouter.ai, now removed", expires=_dt.date.fromisoformat(FUTURE),
            tracking=None, line_index=0,
        )
    ]
    report = og.evaluate(matches, allowlist, TODAY)
    assert not report.ok
    assert len(report.stale) == 1


def test_load_allowlist_rejects_bad_pattern_class(tmp_path):
    path = tmp_path / ".openrouter-allowlist.yaml"
    path.write_text(
        "entries:\n"
        "  - path: x.py\n"
        "    pattern: not-a-real-class\n"
        "    reason: r\n"
        "    expires: '2099-01-01'\n"
    )
    with pytest.raises(og.GuardError, match="not in"):
        og.load_allowlist(path)


def test_load_allowlist_requires_all_keys(tmp_path):
    path = tmp_path / ".openrouter-allowlist.yaml"
    path.write_text("entries:\n  - path: x.py\n    pattern: base_url\n")
    with pytest.raises(og.GuardError, match="missing required keys"):
        og.load_allowlist(path)


def test_missing_allowlist_file_is_empty(tmp_path):
    assert og.load_allowlist(tmp_path / "does-not-exist.yaml") == []


def test_main_end_to_end_clean(tmp_path, capsys):
    repo = _git_repo(tmp_path)
    _add(repo, "src/app.py", "print('hello')\n")
    _commit(repo)
    rc = og.main(["--repo", str(repo)])
    assert rc == 0


def test_main_end_to_end_finds_new_reference(tmp_path, capsys):
    repo = _git_repo(tmp_path)
    _add(repo, "src/app.py", 'BASE_URL = "https://openrouter.ai/api/v1"\n')
    _commit(repo)
    rc = og.main(["--repo", str(repo)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "openrouter.ai" in out.lower() or "unallowlisted" in out.lower()


def test_main_end_to_end_with_valid_allowlist(tmp_path, capsys):
    repo = _git_repo(tmp_path)
    _add(repo, "src/app.py", 'BASE_URL = "https://openrouter.ai/api/v1"\n')
    _add(
        repo, ".openrouter-allowlist.yaml",
        "entries:\n"
        "  - path: src/app.py\n"
        "    pattern: base_url\n"
        "    reason: pre-existing, tracked in alpha-engine-config-I9999\n"
        "    expires: '2099-01-01'\n"
        "    tracking: alpha-engine-config-I9999\n",
    )
    _commit(repo)
    rc = og.main(["--repo", str(repo)])
    assert rc == 0
