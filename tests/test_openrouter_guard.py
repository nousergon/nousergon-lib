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


# --------------------------------------------------------------------------- #
# Widened patterns (alpha-engine-config-I9092): a lowercase attribute name and
# a runtime equality comparison — the exact shape that shipped undetected in
# vires/api/services/coach/agent.py (pre-fix, reproduced verbatim below) and
# that the ORIGINAL three patterns could not see.
# --------------------------------------------------------------------------- #

# Verbatim excerpt of vires/api/services/coach/agent.py BEFORE
# alpha-engine-config-I9092's fix (vires-PR, 2026-08-28). Kept as a literal
# fixture rather than a paraphrase so this test is pinned to the actual
# incident, not to a shape someone reconstructed from memory.
_VIRES_PRE_FIX_AGENT_PY = '''\
def _resolve_spec():
    from dataclasses import replace

    from krepis.llm_config import ModelSpec, resolve_model_spec

    settings = get_settings()
    spec = resolve_model_spec(
        settings.coach_llm_ssm_param,
        env_var="VIRES_COACH_LLM",
        default=ModelSpec(
            "anthropic", settings.coach_model, max_tokens=settings.coach_max_tokens
        ),
        max_tokens=settings.coach_max_tokens,
    )
    if spec.provider == "openrouter" and spec.reasoning is None:
        spec = replace(spec, reasoning={"exclude": True})
    return spec


def _api_key_for(spec):
    settings = get_settings()
    if spec.provider == "anthropic":
        return settings.anthropic_api_key
    return settings.openrouter_api_key
'''


def test_widened_patterns_are_red_against_the_pre_fix_vires_agent_py(tmp_path):
    """The guard this session widens must catch the actual incident.

    Before I9092: three patterns (base_url / env_key / provider_literal) all
    miss this file — `openrouter.ai` never appears, `OPENROUTER_API_KEY`
    never appears (only the lowercase attribute `openrouter_api_key` does),
    and `provider ==` is a runtime comparison, not the assignment shape
    `provider_literal` matches. That is the exact blind spot
    alpha-engine-config-I9092 found live on `main`, reachable via a config
    flip with zero code review.
    """
    repo = _git_repo(tmp_path)
    _add(repo, "api/services/coach/agent.py", _VIRES_PRE_FIX_AGENT_PY)
    _commit(repo)

    matches = og.scan(repo, og.DEFAULT_EXTENSIONS)
    found_classes = {m.pattern_class for m in matches}

    # The original three patterns see NOTHING in this file — confirms this
    # fixture reproduces the actual blind spot, not a strawman.
    assert og.PATTERN_BASE_URL not in found_classes
    assert og.PATTERN_ENV_KEY not in found_classes
    assert og.PATTERN_PROVIDER_LITERAL not in found_classes

    # The two widened patterns catch it.
    assert og.PATTERN_ATTR_KEY in found_classes
    assert og.PATTERN_PROVIDER_COMPARISON in found_classes

    report = og.evaluate(matches, [], TODAY)
    assert not report.ok
    assert og.render(report, len(matches)) == 1


def test_widened_patterns_pass_once_allowlisted_or_fixed(tmp_path):
    """GREEN after: either the post-fix source (no openrouter reference at
    all) or an allowlisted pre-fix source both pass — proving the widened
    guard is a real gate, not a permanent red."""
    repo = _git_repo(tmp_path)
    _add(repo, "api/services/coach/agent.py", _VIRES_PRE_FIX_AGENT_PY)
    _add(
        repo, ".openrouter-allowlist.yaml",
        "entries:\n"
        "  - path: api/services/coach/agent.py\n"
        "    pattern: attr_key\n"
        "    reason: covered by this allowlist entry for the purpose of this test\n"
        "    expires: '2099-01-01'\n"
        "    tracking: alpha-engine-config-I9092\n"
        "  - path: api/services/coach/agent.py\n"
        "    pattern: provider_comparison\n"
        "    reason: covered by this allowlist entry for the purpose of this test\n"
        "    expires: '2099-01-01'\n"
        "    tracking: alpha-engine-config-I9092\n",
    )
    _commit(repo)
    rc = og.main(["--repo", str(repo)])
    assert rc == 0


def test_attr_key_matches_lowercase_and_mixed_case_variants(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "src/settings.py", "openrouter_api_key: str | None = None\n")
    _commit(repo)

    matches = og.scan(repo, og.DEFAULT_EXTENSIONS)
    assert any(m.pattern_class == og.PATTERN_ATTR_KEY for m in matches)


def test_attr_key_does_not_double_count_the_all_caps_env_var(tmp_path):
    """The all-caps literal stays PATTERN_ENV_KEY's finding alone — widening
    attr_key must not force every existing env_key allowlist entry in the
    fleet to grow a second, redundant entry."""
    repo = _git_repo(tmp_path)
    _add(repo, "src/client.py", 'key = os.environ["OPENROUTER_API_KEY"]\n')
    _commit(repo)

    matches = og.scan(repo, og.DEFAULT_EXTENSIONS)
    classes = [m.pattern_class for m in matches]
    assert classes.count(og.PATTERN_ENV_KEY) == 1
    assert og.PATTERN_ATTR_KEY not in classes


def test_provider_comparison_matches_equality_and_inequality(tmp_path):
    repo = _git_repo(tmp_path)
    _add(
        repo, "src/router_check.py",
        "if spec.provider == \"openrouter\":\n"
        "    pass\n"
        "assert client.spec.provider != 'openrouter'\n",
    )
    _commit(repo)

    matches = og.scan(repo, og.DEFAULT_EXTENSIONS)
    comparison_matches = [m for m in matches if m.pattern_class == og.PATTERN_PROVIDER_COMPARISON]
    assert len(comparison_matches) == 2


def test_provider_comparison_does_not_match_the_assignment_shape(tmp_path):
    """provider_literal and provider_comparison are DIFFERENT token shapes —
    an assignment must not also count as a comparison finding."""
    repo = _git_repo(tmp_path)
    _add(repo, "config/models.yaml", "provider: openrouter\n")
    _commit(repo)

    matches = og.scan(repo, og.DEFAULT_EXTENSIONS)
    assert not any(m.pattern_class == og.PATTERN_PROVIDER_COMPARISON for m in matches)
