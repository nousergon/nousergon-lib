"""Tests for scripts/provider_linkage_guard.py (alpha-engine-config-I9263).

Brian's 2026-08-29 ruling: everything funnels through the krepis router, with
no parallel setups. This guard is the detector for the negative case.

Every test builds a real throwaway git repo, because the script enumerates its
scan surface with `git ls-files` -- same convention as test_openrouter_guard.py.

**The regression cases that matter** are the ones the predecessor guards could
not see, each of which shipped to production undetected:

  * ``anthropic.Anthropic(api_key=...)`` in a Lambda handler -- three call
    sites in crucible-research's eval-judge path. The OpenRouter guard was
    scoped to a different vendor, so it was structurally blind to this.
  * ``openai.OpenAI`` / ``ChatOpenAI`` and every other vendor with no guard
    at all.
  * A launchd plist or systemd unit exporting a provider credential -- a live
    call site with no source file. Neither predecessor guard read those
    extensions.
  * A TypeScript surface importing the npm SDK.
"""

from __future__ import annotations

import datetime as _dt
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "provider_linkage_guard.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("provider_linkage_guard", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["provider_linkage_guard"] = mod
    spec.loader.exec_module(mod)
    return mod


plg = _load_module()

_GIT = shutil.which("git")

TODAY = _dt.date(2026, 8, 29)
FUTURE = "2026-12-31"
PAST = "2026-01-01"

ALL_PATTERNS = plg.compile_patterns(plg.ALL_PROVIDER_NAMES)


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


def _scan(repo: Path):
    return plg.scan(repo, plg.DEFAULT_EXTENSIONS, ALL_PATTERNS)


def _classes(matches) -> set[str]:
    return {m.pattern_class for m in matches}


# -- the regressions the predecessor guards were blind to --------------------


def test_detects_direct_anthropic_sdk_client(tmp_path):
    """The exact shape that shipped in crucible-research's eval-judge handlers.

    This is the whole reason the guard is generalized: the fleet had a
    direct-linkage guard, and it was scoped to OpenRouter, so this was
    invisible to it.
    """
    repo = _git_repo(tmp_path)
    _add(repo, "lambda/eval_judge_submit_handler.py", (
        "import anthropic\n"
        "client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)\n"
    ))
    matches = _scan(repo)
    assert "anthropic:sdk_client" in _classes(matches)
    assert "anthropic:env_key" in _classes(matches)


def test_detects_async_anthropic_and_from_import(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "a.py", "c = anthropic.AsyncAnthropic()\n")
    _add(repo, "b.py", "from anthropic import Anthropic\n")
    hits = [m for m in _scan(repo) if m.pattern_class == "anthropic:sdk_client"]
    assert {m.path for m in hits} == {"a.py", "b.py"}


def test_detects_openai_sdk_and_langchain_bindings(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "x.py", "client = openai.OpenAI(api_key=k)\n")
    _add(repo, "y.py", "llm = ChatOpenAI(model=m)\n")
    _add(repo, "z.py", "llm = ChatAnthropic(model=m)\n")
    classes = _classes(_scan(repo))
    assert "openai:sdk_client" in classes
    assert "anthropic:sdk_client" in classes


def test_detects_npm_sdk_in_typescript(tmp_path):
    """A TypeScript call site bypasses the router exactly as much as a Python one."""
    repo = _git_repo(tmp_path)
    _add(repo, "src/agent.ts", "import Anthropic from '@anthropic-ai/sdk';\n")
    assert "anthropic:sdk_client" in _classes(_scan(repo))


def test_detects_vercel_ai_sdk_as_a_parallel_abstraction(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "src/route.ts", "import { openai } from '@ai-sdk/openai';\n")
    assert "vercel_ai_sdk:sdk_client" in _classes(_scan(repo))


def test_detects_credential_in_a_launchd_plist(tmp_path):
    """A non-code execution surface is a call site with no source file.

    Neither predecessor guard scanned .plist / .service, so a scheduled job
    holding its own provider credential was undetectable.
    """
    repo = _git_repo(tmp_path)
    _add(repo, "launchd/ai.nousergon.thing.plist", (
        "<key>DEEPSEEK_API_KEY</key><string>sk-x</string>\n"
    ))
    assert "deepseek:env_key" in _classes(_scan(repo))


def test_detects_credential_in_a_systemd_unit(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "infra/thing.service", 'Environment="OPENAI_API_KEY=sk-x"\n')
    assert "openai:env_key" in _classes(_scan(repo))


def test_detects_base_url_repointing(tmp_path):
    """Addressing a model by base URL is a principle-8 violation on its own."""
    repo = _git_repo(tmp_path)
    _add(repo, "run.sh", 'export ANTHROPIC_BASE_URL="https://openrouter.ai/api"\n')
    classes = _classes(_scan(repo))
    assert "anthropic:base_url_env" in classes
    assert "openrouter:base_url" in classes


def test_retains_openrouter_parity_with_the_predecessor_guard(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "svc.py", 'URL = "https://openrouter.ai/api/v1"\nK = os.environ["OPENROUTER_API_KEY"]\n')
    classes = _classes(_scan(repo))
    assert {"openrouter:base_url", "openrouter:env_key"} <= classes


@pytest.mark.parametrize(
    ("rel", "content", "expected"),
    [
        ("a.py", 'H = "api.deepseek.com"\n', "deepseek:base_url"),
        ("a.py", 'H = "api.x.ai"\n', "xai:base_url"),
        ("a.py", 'H = "generativelanguage.googleapis.com"\n', "google:base_url"),
        ("a.py", 'H = "api.groq.com"\n', "groq:base_url"),
        ("a.py", 'H = "api.mistral.ai"\n', "mistral:base_url"),
        ("a.py", 'H = "open.bigmodel.cn"\n', "zhipu:base_url"),
        ("a.py", 'c = boto3.client("bedrock-runtime")\n', "bedrock:sdk_client"),
        ("a.py", "import google.generativeai as genai\n", "google:sdk_client"),
    ],
)
def test_every_vendor_in_the_table_is_detected(tmp_path, rel, content, expected):
    repo = _git_repo(tmp_path)
    _add(repo, rel, content)
    assert expected in _classes(_scan(repo))


# -- a compliant call site is NOT a finding ----------------------------------


def test_router_addressed_call_site_is_clean(tmp_path):
    """The shape the fleet is migrating TO must not be flagged.

    A guard that reds the compliant pattern trains everyone to ignore it.
    """
    repo = _git_repo(tmp_path)
    _add(repo, "svc.py", (
        "from krepis.router import resolve_group_spec\n"
        "from krepis.llm import LLMClient\n"
        "spec = resolve_group_spec('high', exec_context='ec2')\n"
        "result = LLMClient(spec).complete(prompt)\n"
    ))
    assert _scan(repo) == []


# -- allowlist semantics -----------------------------------------------------


def _allowlist(repo: Path, entries: str) -> Path:
    p = repo / ".provider-linkage-allowlist.yaml"
    p.write_text(entries)
    return p


def test_allowlisted_match_is_covered_not_reported(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "lambda/h.py", "c = anthropic.Anthropic(api_key=k)\n")
    p = _allowlist(repo, (
        "entries:\n"
        "  - path: lambda/h.py\n"
        "    pattern: anthropic:sdk_client\n"
        "    reason: baselined\n"
        f"    expires: {FUTURE}\n"
    ))
    report = plg.evaluate(
        _scan(repo), plg.load_allowlist(p, plg.all_pattern_classes()), TODAY
    )
    assert report.ok
    assert report.covered == 1


def test_unallowlisted_match_fails(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "lambda/h.py", "c = anthropic.Anthropic(api_key=k)\n")
    report = plg.evaluate(_scan(repo), [], TODAY)
    assert not report.ok
    assert len(report.unallowlisted) == 1


def test_expired_entry_fails_loudly(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "lambda/h.py", "c = anthropic.Anthropic(api_key=k)\n")
    p = _allowlist(repo, (
        "entries:\n"
        "  - path: lambda/h.py\n"
        "    pattern: anthropic:sdk_client\n"
        "    reason: baselined\n"
        f"    expires: {PAST}\n"
    ))
    report = plg.evaluate(
        _scan(repo), plg.load_allowlist(p, plg.all_pattern_classes()), TODAY
    )
    assert not report.ok
    assert len(report.expired) == 1


def test_stale_entry_fails(tmp_path):
    """An allowance that outlives what it covered silently widens."""
    repo = _git_repo(tmp_path)
    _add(repo, "clean.py", "x = 1\n")
    p = _allowlist(repo, (
        "entries:\n"
        "  - path: gone.py\n"
        "    pattern: anthropic:sdk_client\n"
        "    reason: baselined\n"
        f"    expires: {FUTURE}\n"
    ))
    report = plg.evaluate(
        _scan(repo), plg.load_allowlist(p, plg.all_pattern_classes()), TODAY
    )
    assert not report.ok
    assert len(report.stale) == 1


def test_unmatched_entry_for_a_disabled_class_is_not_stale(tmp_path):
    """alpha-engine-config-I10225: a `dist` allowlist entry added ahead of
    `nousergon-lib-PR397`'s enforcement flip must not fail the pre-flip
    reusable workflow run (no `--include-dist`) that never scans `dist` at
    all -- an entry for a class this invocation did not look at is
    unknowable, not stale. Unmatched, disabled, no `raw_matches` covering it
    at all (mirrors the reusable workflow's real pre-flip shape)."""
    repo = _git_repo(tmp_path)
    _add(repo, "clean.py", "x = 1\n")
    p = _allowlist(repo, (
        "entries:\n"
        "  - path: pyproject.toml\n"
        "    pattern: anthropic:dist\n"
        "    reason: description prose, not a dependency\n"
        f"    expires: {FUTURE}\n"
    ))
    report = plg.evaluate(
        _scan(repo), plg.load_allowlist(p, plg.all_pattern_classes()), TODAY,
        enabled_classes=plg.DEFAULT_PATTERN_CLASSES,
    )
    assert report.ok
    assert report.stale == []
    assert len(report.disabled) == 1
    assert report.disabled[0].pattern_class == "anthropic:dist"


def test_unmatched_entry_for_an_enabled_class_is_still_stale(tmp_path):
    """The disabled-class carve-out must not weaken the guard for a class
    this invocation DOES scan -- an unused entry for an ENABLED class still
    fails, unchanged."""
    repo = _git_repo(tmp_path)
    _add(repo, "clean.py", "x = 1\n")
    p = _allowlist(repo, (
        "entries:\n"
        "  - path: gone.py\n"
        "    pattern: anthropic:sdk_client\n"
        "    reason: baselined\n"
        f"    expires: {FUTURE}\n"
    ))
    report = plg.evaluate(
        _scan(repo), plg.load_allowlist(p, plg.all_pattern_classes()), TODAY,
        enabled_classes=plg.DEFAULT_PATTERN_CLASSES,
    )
    assert not report.ok
    assert len(report.stale) == 1
    assert report.disabled == []


def test_allowlist_entry_is_scoped_to_its_pattern_class(tmp_path):
    """Clearing one vendor shape must not clear a different one in the same file."""
    repo = _git_repo(tmp_path)
    _add(repo, "h.py", "c = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)\n")
    p = _allowlist(repo, (
        "entries:\n"
        "  - path: h.py\n"
        "    pattern: anthropic:sdk_client\n"
        "    reason: baselined\n"
        f"    expires: {FUTURE}\n"
    ))
    report = plg.evaluate(
        _scan(repo), plg.load_allowlist(p, plg.all_pattern_classes()), TODAY
    )
    assert not report.ok
    assert {m.pattern_class for m in report.unallowlisted} == {"anthropic:env_key"}


def test_unknown_pattern_class_is_a_hard_error(tmp_path):
    repo = _git_repo(tmp_path)
    p = _allowlist(repo, (
        "entries:\n"
        "  - path: h.py\n"
        "    pattern: anthropic:not_a_class\n"
        "    reason: r\n"
        f"    expires: {FUTURE}\n"
    ))
    with pytest.raises(plg.GuardError, match="not a known"):
        plg.load_allowlist(p, plg.all_pattern_classes())


@pytest.mark.parametrize("key", ["path", "pattern", "reason", "expires"])
def test_missing_required_allowlist_key_is_a_hard_error(tmp_path, key):
    repo = _git_repo(tmp_path)
    fields = {
        "path": "h.py",
        "pattern": "anthropic:sdk_client",
        "reason": "r",
        "expires": FUTURE,
    }
    del fields[key]
    body = "entries:\n  - " + "\n    ".join(f"{k}: {v}" for k, v in fields.items()) + "\n"
    p = _allowlist(repo, body)
    with pytest.raises(plg.GuardError, match="missing required keys"):
        plg.load_allowlist(p, plg.all_pattern_classes())


def test_empty_reason_is_a_hard_error(tmp_path):
    repo = _git_repo(tmp_path)
    p = _allowlist(repo, (
        "entries:\n"
        "  - path: h.py\n"
        "    pattern: anthropic:sdk_client\n"
        '    reason: "  "\n'
        f"    expires: {FUTURE}\n"
    ))
    with pytest.raises(plg.GuardError, match="reason must be non-empty"):
        plg.load_allowlist(p, plg.all_pattern_classes())


def test_missing_allowlist_file_is_treated_as_empty(tmp_path):
    repo = _git_repo(tmp_path)
    assert plg.load_allowlist(repo / "nope.yaml", plg.all_pattern_classes()) == []


# -- scope and shape ---------------------------------------------------------


def test_markdown_is_not_scanned_by_default(tmp_path):
    """Policy prose discusses every vendor constantly; prose is not linkage."""
    repo = _git_repo(tmp_path)
    _add(repo, "POLICY.md", "We must never call anthropic.Anthropic( directly.\n")
    assert _scan(repo) == []
    with_docs = plg.scan(repo, plg.DEFAULT_EXTENSIONS | plg.DOC_EXTENSIONS, ALL_PATTERNS)
    assert "anthropic:sdk_client" in _classes(with_docs)


def test_untracked_file_is_not_scanned(tmp_path):
    repo = _git_repo(tmp_path)
    (repo / "untracked.py").write_text("c = anthropic.Anthropic()\n")
    assert _scan(repo) == []


def test_scanner_never_scans_itself(tmp_path):
    """Structural, not allowlisted: 33 self-entries would otherwise be permanent."""
    repo = _git_repo(tmp_path)
    _add(repo, "scripts/provider_linkage_guard.py", "PAT = 'anthropic.Anthropic('\n")
    _add(repo, "tests/t.py", "import provider_linkage_guard\nX = 'ANTHROPIC_API_KEY'\n")
    assert _scan(repo) == []


def test_providers_subset_selects_only_those(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "h.py", "c = anthropic.Anthropic(api_key=OPENAI_API_KEY)\n")
    only_openai = plg.scan(repo, plg.DEFAULT_EXTENSIONS, plg.compile_patterns(("openai",)))
    assert _classes(only_openai) == {"openai:env_key"}


def test_unknown_provider_selection_is_a_hard_error():
    with pytest.raises(plg.GuardError, match="unknown provider"):
        plg._selected_providers("anthropic,nosuchvendor")


def test_empty_provider_selection_is_a_hard_error():
    with pytest.raises(plg.GuardError, match="selected nothing"):
        plg._selected_providers(",")


def test_default_selection_is_every_provider():
    assert plg._selected_providers(None) == plg.ALL_PROVIDER_NAMES


def test_provider_names_are_unique():
    assert len(plg.PROVIDERS_BY_NAME) == len(plg.PROVIDERS)


def test_every_provider_declares_at_least_one_shape():
    for p in plg.PROVIDERS:
        assert any(getattr(p, k) is not None for k in plg.PATTERN_CLASSES), p.name


def test_every_declared_pattern_compiles():
    assert len(ALL_PATTERNS) == len(plg.all_pattern_classes())


def test_the_two_retired_vendors_of_todays_rulings_are_covered():
    """OpenRouter (2026-08-03 ruling) and Anthropic (2026-08-29 ruling)."""
    classes = plg.all_pattern_classes()
    assert "openrouter:base_url" in classes
    assert "openrouter:env_key" in classes
    assert "anthropic:sdk_client" in classes
    assert "anthropic:env_key" in classes


# -- CLI ---------------------------------------------------------------------


def test_cli_exit_codes(tmp_path, capsys):
    repo = _git_repo(tmp_path)
    _add(repo, "h.py", "c = anthropic.Anthropic()\n")
    assert plg.main(["--repo", str(repo)]) == 1

    _allowlist(repo, (
        "entries:\n"
        "  - path: h.py\n"
        "    pattern: anthropic:sdk_client\n"
        "    reason: baselined\n"
        "    expires: 2099-01-01\n"
    ))
    assert plg.main(["--repo", str(repo)]) == 0


def test_cli_returns_2_when_the_check_cannot_complete(tmp_path):
    """A broken check is an infrastructure fault, never a silent pass."""
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    assert plg.main(["--repo", str(not_a_repo)]) == 2


def test_cli_list_providers(capsys):
    assert plg.main(["--list-providers"]) == 0
    out = capsys.readouterr().out
    for name in plg.ALL_PROVIDER_NAMES:
        assert name in out


# -- comments are not linkage (alpha-engine-config-I9295) --------------------
#
# Measured 2026-08-29 before this change: alpha-engine-config 469 findings,
# crucible-research 77, nous-ergon-ops 58. Wiring the guard fleet-wide on that
# baseline means an allowlist of hundreds of entries, which is the guard being
# switched off one entry at a time rather than a baseline. A comment is not an
# execution surface -- the same reason DOC_EXTENSIONS are excluded by default.


def test_a_python_comment_is_not_linkage(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "h.py", "# reads ANTHROPIC_API_KEY on the box\nx = 1\n")
    assert plg.main(["--repo", str(repo)]) == 0


def test_a_hash_comment_in_shell_and_yaml_is_not_linkage(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "s.sh", "# export ANTHROPIC_API_KEY=...\necho hi\n")
    _add(repo, "w.yml", "steps:\n  # ANTHROPIC_API_KEY is no longer set here\n  - run: true\n")
    assert plg.main(["--repo", str(repo)]) == 0


def test_a_slash_comment_in_typescript_is_not_linkage(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "a.ts", "// import '@ai-sdk/openai'\n/* ANTHROPIC_API_KEY */\nexport const x = 1;\n")
    assert plg.main(["--repo", str(repo)]) == 0


def test_a_string_literal_is_still_linkage(tmp_path):
    """Comments are blanked; STRINGS are not. A credential in a string executes."""
    repo = _git_repo(tmp_path)
    _add(repo, "h.py", 'k = os.environ["ANTHROPIC_API_KEY"]\n')
    assert plg.main(["--repo", str(repo)]) == 1


def test_a_hash_inside_a_python_string_is_not_treated_as_a_comment(tmp_path):
    """The real tokenizer, not a `#`-split: the linkage after it must still fire."""
    repo = _git_repo(tmp_path)
    _add(repo, "h.py", 'url = "#anchor"; k = os.environ["ANTHROPIC_API_KEY"]\n')
    assert plg.main(["--repo", str(repo)]) == 1


def test_a_trailing_comment_does_not_hide_linkage_earlier_on_the_line(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "h.py", 'k = os.environ["ANTHROPIC_API_KEY"]  # legacy\n')
    assert plg.main(["--repo", str(repo)]) == 1


def test_reported_line_numbers_survive_comment_blanking(tmp_path, capsys):
    """Blanking preserves numbering, so a finding still points at the real line."""
    repo = _git_repo(tmp_path)
    _add(repo, "h.py", "# a\n# b\n# c\nk = os.environ['ANTHROPIC_API_KEY']\n")
    assert plg.main(["--repo", str(repo)]) == 1
    assert "line=4" in capsys.readouterr().out


def test_a_file_that_does_not_tokenize_is_still_scanned(tmp_path):
    """A syntactically broken .py file must never be silently skipped."""
    repo = _git_repo(tmp_path)
    _add(repo, "h.py", "def broken(:\nk = os.environ['ANTHROPIC_API_KEY']\n")
    assert plg.main(["--repo", str(repo)]) == 1


def test_json_has_no_comment_syntax_and_is_left_intact(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "c.json", '{"env": {"ANTHROPIC_API_KEY": "x"}}\n')
    assert plg.main(["--repo", str(repo)]) == 1


def test_ignoring_comments_does_not_make_a_live_allowlist_entry_stale(tmp_path):
    """A guard-side RELAXATION must not be able to redden a consumer repo.

    The reusable workflow checks this script out unpinned, so a change here
    re-verdicts every consumer's `main` with no commit in that repo. Staleness
    is therefore evaluated against RAW text: the entry still describes
    something real, it just no longer fails.
    """
    repo = _git_repo(tmp_path)
    _add(repo, "h.py", "# reads ANTHROPIC_API_KEY on the box\nx = 1\n")
    _allowlist(repo, (
        "entries:\n"
        "  - path: h.py\n"
        "    pattern: anthropic:env_key\n"
        "    reason: baselined comment\n"
        "    expires: 2099-01-01\n"
    ))
    assert plg.main(["--repo", str(repo)]) == 0


def test_an_entry_covering_nothing_at_all_is_still_stale(tmp_path):
    """The stale check is not weakened — only comment-only coverage survives."""
    repo = _git_repo(tmp_path)
    _add(repo, "h.py", "x = 1\n")
    _allowlist(repo, (
        "entries:\n"
        "  - path: h.py\n"
        "    pattern: anthropic:env_key\n"
        "    reason: nothing here any more\n"
        "    expires: 2099-01-01\n"
    ))
    assert plg.main(["--repo", str(repo)]) == 1


# -- a declared registry is not a call site (alpha-engine-config-I9295) -----
#
# A registry of provider linkage necessarily NAMES providers -- that is its
# job. The remaining 360 findings measured on alpha-engine-config's main
# after comment-stripping were ONE class: the declared registry files
# themselves, plus the validators/tests whose entire subject is one of them.
# Generalized the same way comment-stripping was: a STRUCTURAL property, not
# a path list.


def test_a_declared_registry_file_is_not_a_call_site(tmp_path):
    repo = _git_repo(tmp_path)
    _add(
        repo,
        "private-docs/LLM_MODEL_REGISTRY.yaml",
        "# provider-linkage-registry: declared\n"
        "models:\n"
        "  - provider: anthropic\n"
        "    env_key: ANTHROPIC_API_KEY\n"
        "    base_url: api.anthropic.com\n",
    )
    assert plg.scan(repo, plg.DEFAULT_EXTENSIONS, ALL_PATTERNS) == []


def test_a_yaml_file_without_the_marker_is_still_scanned(tmp_path):
    """The marker is required -- a YAML file does not get a free pass just
    because it happens to be named ``*_REGISTRY.yaml``... except when the
    filename itself carries the naming convention, which is covered by
    ``test_a_registry_named_by_convention_is_recognized`` below. This test
    covers a YAML file that is neither marked nor conventionally named."""
    repo = _git_repo(tmp_path)
    _add(repo, "config/settings.yaml", "anthropic_key: ANTHROPIC_API_KEY\n")
    matches = plg.scan(repo, plg.DEFAULT_EXTENSIONS, ALL_PATTERNS)
    assert _classes(matches) == {"anthropic:env_key"}


def test_a_validator_naming_the_registry_is_exempt(tmp_path):
    """The same technique as `_is_scanner_source`: a file whose subject is a
    declared registry names that registry's filename literally, exactly how
    a test of this module names ``provider_linkage_guard.py``."""
    repo = _git_repo(tmp_path)
    _add(
        repo,
        "private-docs/LLM_MODEL_REGISTRY.yaml",
        "# provider-linkage-registry: declared\nmodels: []\n",
    )
    _add(
        repo,
        "scripts/validate_llm_model_registry.py",
        '"""Validate private-docs/LLM_MODEL_REGISTRY.yaml."""\n'
        'KNOWN_HOSTS = ["api.anthropic.com", "openrouter.ai"]\n',
    )
    matches = plg.scan(repo, plg.DEFAULT_EXTENSIONS, ALL_PATTERNS)
    assert matches == []


def test_an_unrelated_file_naming_the_registry_stays_exempt_too(tmp_path):
    """Deliberate: the exemption is FILE-WIDE, same shape as the scanner's own
    self-exemption. A real bypass in an unrelated file does not happen to
    mention a registry filename by construction -- it would have to add that
    reference itself, which is visible in the diff."""
    repo = _git_repo(tmp_path)
    _add(
        repo,
        "private-docs/LLM_MODEL_REGISTRY.yaml",
        "# provider-linkage-registry: declared\nmodels: []\n",
    )
    _add(
        repo,
        "scripts/unrelated.py",
        '# see private-docs/LLM_MODEL_REGISTRY.yaml for context\n'
        'x = "ANTHROPIC_API_KEY"\n',
    )
    matches = plg.scan(repo, plg.DEFAULT_EXTENSIONS, ALL_PATTERNS)
    assert matches == []


def test_no_declared_registry_means_no_subject_exemption(tmp_path):
    """Without any file carrying the marker, naming a filename that merely
    LOOKS like a registry buys nothing -- the exemption is keyed off an
    actually-declared registry, not off vocabulary."""
    repo = _git_repo(tmp_path)
    _add(
        repo,
        "scripts/mentions_a_registry.py",
        '# see private-docs/LLM_MODEL_REGISTRY.yaml for context\n'
        'x = "ANTHROPIC_API_KEY"\n',
    )
    matches = plg.scan(repo, plg.DEFAULT_EXTENSIONS, ALL_PATTERNS)
    assert _classes(matches) == {"anthropic:env_key"}


def test_declared_registry_exemption_does_not_redden_a_consumer(tmp_path):
    """Same safety property as the comment-stripping relaxation: applying
    uniformly to both the findings scan and the raw staleness scan means an
    existing allowlist entry inside a (newly) exempt file cannot flip to
    unallowlisted -- it simply stops appearing at all, on both sides."""
    repo = _git_repo(tmp_path)
    _add(
        repo,
        "private-docs/LLM_MODEL_REGISTRY.yaml",
        "# provider-linkage-registry: declared\n"
        "models:\n  - provider: anthropic\n    env_key: ANTHROPIC_API_KEY\n",
    )
    _allowlist(repo, (
        "entries:\n"
        "  - path: private-docs/LLM_MODEL_REGISTRY.yaml\n"
        "    pattern: anthropic:env_key\n"
        "    reason: pre-existing baseline entry, now superseded by the marker\n"
        "    expires: 2099-01-01\n"
    ))
    # No verdict change: the file is fully exempt, so the entry is neither a
    # finding nor considered stale (it is simply out of scan scope).
    assert plg.main(["--repo", str(repo)]) == 0


# -- a docstring is prose, not a call site (alpha-engine-config-I9263) ------
#
# Five allowlist entries EXPIRED 2026-09-05 on crucible-research covering the
# exact same shape comment-stripping already fixed once: a migration's own
# module docstring narrating the retired `anthropic.Anthropic(api_key=...)`
# construction it replaced, and a test asserting that pattern's absence. A
# docstring is a real syntax position (first statement of a module, class or
# function), identified structurally -- never a text heuristic over what the
# string says -- exactly the discipline `_blank_python_comments` and
# `_is_declared_registry` already use.


def test_a_module_docstring_naming_a_retired_pattern_is_not_linkage(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "h.py", (
        '"""This module replaced `anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)`.\n'
        '"""\n'
        "x = 1\n"
    ))
    assert plg.main(["--repo", str(repo)]) == 0


def test_a_function_docstring_naming_a_retired_pattern_is_not_linkage(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "h.py", (
        "def f():\n"
        '    """Used to read ANTHROPIC_API_KEY here."""\n'
        "    return 1\n"
    ))
    assert plg.main(["--repo", str(repo)]) == 0


def test_a_real_anthropic_construction_still_trips_the_guard(tmp_path):
    """The regression this guard exists for must still fire -- a docstring
    exemption must not become a way to hide a real call site."""
    repo = _git_repo(tmp_path)
    _add(repo, "h.py", (
        '"""Docstring with nothing to say about providers."""\n'
        "import anthropic\n"
        "client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)\n"
    ))
    matches = _scan(repo)
    assert "anthropic:sdk_client" in _classes(matches)
    assert "anthropic:env_key" in _classes(matches)


def test_a_string_not_in_first_statement_position_is_not_a_docstring(tmp_path):
    """A bare string expression AFTER other statements is dead code, not
    documentation -- it is not exempted, because it is not structurally a
    docstring at all, the same distinction `_docstring_spans` draws."""
    repo = _git_repo(tmp_path)
    _add(repo, "h.py", (
        "x = 1\n"
        '"anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)"\n'
    ))
    matches = _scan(repo)
    assert "anthropic:sdk_client" in _classes(matches)


def test_a_string_literal_assigned_to_a_variable_is_still_linkage(tmp_path):
    """Docstring-stripping must not widen to ordinary string-literal code --
    a base URL or credential name assigned to a variable is still linkage,
    because it never occupies the first-statement-of-a-scope position."""
    repo = _git_repo(tmp_path)
    _add(repo, "h.py", 'BASE_URL = "api.anthropic.com"\n')
    assert plg.main(["--repo", str(repo)]) == 1


def test_a_python_file_that_does_not_parse_still_gets_docstring_pass_skipped(tmp_path):
    """`_docstring_spans` must degrade to "nothing stripped", never crash, on
    a syntactically broken file -- the comment-blanking fallback already
    guarantees the file is still scanned in full."""
    repo = _git_repo(tmp_path)
    _add(repo, "h.py", "def broken(:\nk = os.environ['ANTHROPIC_API_KEY']\n")
    assert plg.main(["--repo", str(repo)]) == 1


def test_docstring_exemption_does_not_redden_a_consumer(tmp_path):
    """Same safety property already proven for comment-stripping and the
    declared-registry marker: this is a RELAXATION evaluated only in the
    findings scan, so an existing allowlist entry inside a docstring cannot
    flip from covered to unallowlisted -- it simply stops appearing."""
    repo = _git_repo(tmp_path)
    _add(repo, "h.py", (
        '"""Reads ANTHROPIC_API_KEY historically."""\n'
        "x = 1\n"
    ))
    _allowlist(repo, (
        "entries:\n"
        "  - path: h.py\n"
        "    pattern: anthropic:env_key\n"
        "    reason: baselined docstring prose\n"
        "    expires: 2099-01-01\n"
    ))
    assert plg.main(["--repo", str(repo)]) == 0


# -- a declared "asserts-absence" file is not a call site (alpha-engine-
# config-I9263) --------------------------------------------------------------
#
# A test's literal tuple of forbidden strings, compared with `not in` against
# a module's source to prove a retired pattern is GONE, is ordinary CODE, not
# a docstring -- `tests/test_eval_judge_batch_transport.py`'s
# `test_no_anthropic_sdk_construction` is the exact shape. Exempted the same
# way `_is_declared_registry` exempts a registry: an explicit, opt-in,
# structural marker the file declares about itself, narrower than "any
# string in a test file is exempt" so a genuine accidental construction
# elsewhere in a test is still caught.


def test_a_file_marked_asserts_absence_is_not_a_call_site(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "tests/test_no_anthropic.py", (
        "# provider-linkage-guard: asserts-absence\n"
        "FORBIDDEN = (\"anthropic.Anthropic(\", \"ANTHROPIC_API_KEY\")\n"
        "def test_gone():\n"
        "    assert all(f not in read_source() for f in FORBIDDEN)\n"
    ))
    assert plg.main(["--repo", str(repo)]) == 0


def test_a_test_file_without_the_marker_still_trips_the_guard(tmp_path):
    """The marker is opt-in, not automatic for anything under tests/ -- a
    genuine accidental construction in an unmarked test must still fire."""
    repo = _git_repo(tmp_path)
    _add(repo, "tests/test_something.py", (
        "def test_x():\n"
        "    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)\n"
    ))
    matches = _scan(repo)
    assert "anthropic:sdk_client" in _classes(matches)


# -- dist: dependency/lock files (alpha-engine-config-I10032) ---------------
#
# I7723: the `anthropic` SDK sat in the trading box's venv for three months,
# pulled transitively through krepis's old `flow-doctor[diagnosis]` extra,
# with zero `import anthropic` anywhere in `crucible-executor`'s tree --
# undetectable by every call-site-shaped class above. `dist` closes that gap
# by reading dependency/lock files for a bare PEP 503 distribution name.
# Off by default (mirrors `--include-docs`) -- see DEPENDENCY_FILENAME_RE.


def _dist_patterns():
    return plg.compile_patterns(plg.ALL_PROVIDER_NAMES, (plg.CLASS_DIST,))


def test_dist_is_excluded_from_the_default_cli_run(tmp_path):
    """The class exists, but a plain `--repo .` run must not see it -- the
    fleet's reusable workflow calls the script without `--include-dist`
    until each caller's baseline is measured and allowlisted."""
    repo = _git_repo(tmp_path)
    _add(repo, "requirements.txt", "anthropic==0.34.0\n")
    assert plg.main(["--repo", str(repo)]) == 0


def test_dist_catches_a_pinned_requirements_txt_line(tmp_path):
    """The literal acceptance scenario: a planted `anthropic==<version>` pin
    in a consuming repo's requirements.txt fails the guard, once enabled."""
    repo = _git_repo(tmp_path)
    _add(repo, "requirements.txt", "boto3==1.34.0\nanthropic==0.34.0\n")
    assert plg.main(["--repo", str(repo), "--include-dist"]) == 1


def test_dist_catches_requirements_in_and_extras(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "requirements.in", "anthropic[bedrock]>=0.30\n")
    matches = plg.scan(
        repo, plg.DEFAULT_EXTENSIONS, _dist_patterns(),
        extra_filename_re=plg.DEPENDENCY_FILENAME_RE,
    )
    assert "anthropic:dist" in _classes(matches)


def test_dist_catches_a_pyproject_dependency_string(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "pyproject.toml", '[project]\ndependencies = ["openai>=1.0"]\n')
    matches = plg.scan(
        repo, plg.DEFAULT_EXTENSIONS, _dist_patterns(),
        extra_filename_re=plg.DEPENDENCY_FILENAME_RE,
    )
    assert "openai:dist" in _classes(matches)


def test_dist_catches_a_uv_lock_package_table(tmp_path):
    """A TRANSITIVE dependency never named in requirements.in at all -- only
    the compiled lock names it. Skipping lock files would recreate I7723's
    exact blind spot."""
    repo = _git_repo(tmp_path)
    _add(repo, "uv.lock", '[[package]]\nname = "anthropic"\nversion = "0.34.0"\n')
    matches = plg.scan(
        repo, plg.DEFAULT_EXTENSIONS, _dist_patterns(),
        extra_filename_re=plg.DEPENDENCY_FILENAME_RE,
    )
    assert "anthropic:dist" in _classes(matches)


def test_dist_catches_a_pipfile_lock_json_key(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "Pipfile.lock", '{"default": {"anthropic": {"version": "==0.34.0"}}}\n')
    matches = plg.scan(
        repo, plg.DEFAULT_EXTENSIONS, _dist_patterns(),
        extra_filename_re=plg.DEPENDENCY_FILENAME_RE,
    )
    assert "anthropic:dist" in _classes(matches)


def test_dist_matches_underscore_and_dot_variants_of_a_hyphenated_name(tmp_path):
    """PEP 503 normalizes -/_/. to one separator; a requirement author can
    spell the google SDK any of the three ways."""
    repo = _git_repo(tmp_path)
    _add(repo, "requirements.txt", "google_generativeai==0.3.0\n")
    matches = plg.scan(
        repo, plg.DEFAULT_EXTENSIONS, _dist_patterns(),
        extra_filename_re=plg.DEPENDENCY_FILENAME_RE,
    )
    assert "google:dist" in _classes(matches)


def test_dist_does_not_false_positive_on_an_unrelated_bare_word(tmp_path):
    """The regex is a real word, not a substring: mentioning "an anthropic
    argument" as a requirements comment is prose, and an unrelated package
    name is not a match either."""
    repo = _git_repo(tmp_path)
    _add(repo, "requirements.txt", "# anthropic-adjacent tooling notes\nsomeotherpkg==1.0\n")
    matches = plg.scan(
        repo, plg.DEFAULT_EXTENSIONS, _dist_patterns(),
        extra_filename_re=plg.DEPENDENCY_FILENAME_RE,
    )
    assert matches == []


def test_dependency_filename_matches_a_nested_requirements_file(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "backend/requirements-dev.txt", "anthropic==0.34.0\n")
    matches = plg.scan(
        repo, plg.DEFAULT_EXTENSIONS, _dist_patterns(),
        extra_filename_re=plg.DEPENDENCY_FILENAME_RE,
    )
    assert "anthropic:dist" in _classes(matches)


def test_a_bare_requirements_txt_stays_out_of_scope_without_the_filename_pattern(tmp_path):
    """requirements.txt is `.txt`, a DOC_EXTENSION -- confirms it never enters
    scope through the ordinary extension path, only through the filename
    match. Regression guard for "cleaner, preferred" per the issue."""
    repo = _git_repo(tmp_path)
    _add(repo, "requirements.txt", "anthropic==0.34.0\n")
    assert plg.scan(repo, plg.DEFAULT_EXTENSIONS, _dist_patterns()) == []
    assert plg.scan(repo, plg.DEFAULT_EXTENSIONS | plg.DOC_EXTENSIONS, _dist_patterns()) == []


def test_dist_is_a_known_allowlist_pattern_class():
    assert "anthropic:dist" in plg.all_pattern_classes()


def test_every_provider_with_a_known_pypi_sdk_declares_dist():
    for name in ("anthropic", "openai", "google", "mistral", "groq", "zhipu", "xai"):
        assert plg.PROVIDERS_BY_NAME[name].dist is not None, name


def test_cli_list_providers_shows_dist_shape(capsys):
    assert plg.main(["--list-providers"]) == 0
    out = capsys.readouterr().out
    assert "anthropic: sdk_client, env_key, base_url, base_url_env, dist" in out


def test_asserts_absence_exemption_does_not_redden_a_consumer(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "tests/test_no_anthropic.py", (
        "# provider-linkage-guard: asserts-absence\n"
        "FORBIDDEN = \"ANTHROPIC_API_KEY\"\n"
    ))
    _allowlist(repo, (
        "entries:\n"
        "  - path: tests/test_no_anthropic.py\n"
        "    pattern: anthropic:env_key\n"
        "    reason: pre-existing baseline entry, now superseded by the marker\n"
        "    expires: 2099-01-01\n"
    ))
    assert plg.main(["--repo", str(repo)]) == 0


# -- the ROUTER's own extras (alpha-engine-config-I11017) -------------------
#
# `krepis` IS the router (Brian's 2026-08-29 ruling, -I9294/-I9295). The
# `[openai]` extra installs the openai SDK for KREPIS's own use, so the router
# can speak the OpenAI WIRE FORMAT to its endpoint -- transport-plane, inside
# the one declared adapter. A dist spec whose PACKAGE is the router cannot, by
# construction, be a bypass of the router. The guard's 0.124.128 bump began
# matching `krepis[openai]==0.59.62` at nousergon-data's requirements.txt:208
# and held that repo's `main` red for four consecutive runs with no commit in
# it. The rule below is keyed on the package being the router -- not on the
# literal string `krepis[openai]`, and not on a per-consumer allowlist entry.
# The narrowing must not become a hole: the last tests here prove a genuine
# direct linkage still fails.


def _router_scan(repo: Path):
    return plg.scan(
        repo, plg.DEFAULT_EXTENSIONS, _dist_patterns(),
        extra_filename_re=plg.DEPENDENCY_FILENAME_RE,
    )


@pytest.mark.parametrize("line", [
    "krepis[openai]==0.59.62",
    "krepis[openai,flow-doctor]>=0.59.0",
    "krepis[flow-doctor,openai]>=0.59.0",
    "krepis[anthropic]>=0.60",            # tomorrow's extra, same rule, no edit
    "krepis[openai, anthropic]==0.59.62",
    "krepis [openai] == 0.59.62",
    "krepis==0.59.62",                    # bare router pin
    "krepis",
])
def test_router_dist_spec_is_never_a_direct_linkage(tmp_path, line):
    """The literal I11017 case and its general form: any extras, any pin."""
    repo = _git_repo(tmp_path)
    _add(repo, "requirements.txt", f"boto3==1.34.0\n{line}\n")
    assert _router_scan(repo) == []


def test_router_extra_in_a_pyproject_dependency_string(tmp_path):
    """crucible's shape (-I10473): the pin lives in pyproject, quoted."""
    repo = _git_repo(tmp_path)
    _add(repo, "pyproject.toml", '[project]\ndependencies = ["krepis[openai]>=0.59.62"]\n')
    assert _router_scan(repo) == []


def test_router_extra_in_a_toml_inline_table(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "pyproject.toml", 'krepis = { extras = ["openai"], version = ">=0.59" }\n')
    assert _router_scan(repo) == []


def test_router_package_spelling_variants_are_covered(tmp_path):
    """PEP 503 normalization and case-insensitivity, same as every other
    dist name -- the rule is the PACKAGE, not one spelling of it."""
    repo = _git_repo(tmp_path)
    _add(repo, "requirements.txt", "KREPIS[OpenAI]==0.59.62\n")
    assert _router_scan(repo) == []


def test_the_nousergon_data_case_is_green_end_to_end(tmp_path):
    """The closes-when, as the consumer repo sees it: `krepis[openai]` pinned
    in requirements.txt, NO allowlist entry, exit 0 with `--include-dist`."""
    repo = _git_repo(tmp_path)
    _add(repo, "requirements.txt", "boto3==1.34.0\nkrepis[openai]==0.59.62\n")
    assert plg.main(["--repo", str(repo), "--include-dist"]) == 0


# -- and the narrowing is not a hole ---------------------------------------


def test_a_direct_openai_dist_still_fails(tmp_path):
    """The genuine bypass: the SDK as a FIRST-CLASS requirement of the
    consumer, not an extra of the router."""
    repo = _git_repo(tmp_path)
    _add(repo, "requirements.txt", "krepis[openai]==0.59.62\nopenai==1.40\n")
    matches = _router_scan(repo)
    assert _classes(matches) == {"openai:dist"}
    assert [m.line for m in matches] == [2]
    assert plg.main(["--repo", str(repo), "--include-dist"]) == 1


def test_a_direct_dist_on_the_same_line_as_the_router_spec_still_fails(tmp_path):
    """Suppression is SPAN-scoped, not line-scoped: only the characters
    belonging to the router spec are exempt."""
    repo = _git_repo(tmp_path)
    _add(repo, "requirements.txt", "krepis[openai]==0.59.62; openai==1.40\n")
    assert _classes(_router_scan(repo)) == {"openai:dist"}


def test_a_call_site_constructing_an_openai_client_still_fails(tmp_path):
    """The other half of the narrowing test: a repo may pin `krepis[openai]`
    AND still be caught for addressing the provider itself."""
    repo = _git_repo(tmp_path)
    _add(repo, "requirements.txt", "krepis[openai]==0.59.62\n")
    _add(repo, "src/judge.py", (
        "from openai import OpenAI\n"
        "client = OpenAI(base_url='https://api.openai.com/v1', api_key=OPENAI_API_KEY)\n"
    ))
    classes = _classes(_scan(repo))
    assert "openai:sdk_client" in classes
    assert "openai:base_url" in classes
    assert plg.main(["--repo", str(repo), "--include-dist"]) == 1


def test_an_unrelated_package_whose_name_contains_the_router_is_not_exempt(tmp_path):
    """`krepis-shim[openai]` is not the router; the boundary must hold."""
    repo = _git_repo(tmp_path)
    _add(repo, "requirements.txt", "krepis-shim[openai]==1.0\n")
    assert _classes(_router_scan(repo)) == {"openai:dist"}


def test_the_router_exemption_never_stales_an_existing_allowlist_entry(tmp_path):
    """The guard is checked out UNPINNED by the reusable workflow, so this
    relaxation re-verdicts every consumer `main` with no commit there. A
    consumer holding the interim `krepis[openai]` entry (nousergon-data-PR1789)
    must go GREEN on the finding AND not be reddened by a stale entry -- the
    entry is deleted on their next PR, not by this script landing."""
    repo = _git_repo(tmp_path)
    _add(repo, "requirements.txt", "krepis[openai]==0.59.62\n")
    _allowlist(repo, (
        "entries:\n"
        "  - path: requirements.txt\n"
        "    pattern: openai:dist\n"
        "    reason: interim entry for the router's own wire-format extra (I11017)\n"
        "    expires: 2099-01-01\n"
    ))
    assert plg.main(["--repo", str(repo), "--include-dist"]) == 0


def test_router_aware_is_off_in_the_raw_staleness_scan(tmp_path):
    """The mechanism behind the test above, asserted directly."""
    repo = _git_repo(tmp_path)
    _add(repo, "requirements.txt", "krepis[openai]==0.59.62\n")
    raw = plg.scan(
        repo, plg.DEFAULT_EXTENSIONS, _dist_patterns(),
        strip=False, registry_aware=False, absence_aware=False, router_aware=False,
        extra_filename_re=plg.DEPENDENCY_FILENAME_RE,
    )
    assert _classes(raw) == {"openai:dist"}


# -- the other half of the I11017 narrowing: it must not over-apply ----------
#
# nousergon-lib-PR422 exempts a `dist` match inside the ROUTER's own
# requirement spec. The exemption is keyed on the package being the router,
# and the cases below pin the two consequences of that wording which nothing
# else asserts: a provider extra belonging to some OTHER distribution is not
# covered by it, and a repo may DELETE its now-unnecessary allowlist entry
# without the staleness check reddening it.


def test_a_provider_extra_of_a_NON_router_distribution_is_still_a_finding(tmp_path):
    """The narrowing is not keyed on the SHAPE `name[extra]`. Any other
    package's provider extra installs that SDK with nothing between it and the
    call site -- the I7723 failure mode the `dist` class exists for -- so it
    must still fail. Distinct from the `krepis-shim` case above, which tests
    the NAME boundary; this tests that an unrelated, plausible, real-world
    package spelling does not inherit the router's exemption."""
    repo = _git_repo(tmp_path)
    _add(
        repo, "requirements.txt",
        "krepis[openai]==0.59.62\n"
        "langchain[openai]==0.2.0\n"
        "some-agent-sdk[anthropic]>=1.0\n",
    )
    matches = _router_scan(repo)
    assert _classes(matches) == {"openai:dist", "anthropic:dist"}
    assert sorted(m.line for m in matches) == [2, 3]


def test_a_repo_may_delete_its_now_unnecessary_router_entry(tmp_path):
    """The second half of the no-lockstep property. The test above this block
    proves an EXISTING `krepis[openai]` entry is not staled by the narrowing
    landing; this proves the cleanup that removes it is green too. Neither
    order of merges can redden anyone, which is why the guard fix and the
    per-repo allowlist cleanups are independent PRs with no merge order
    (crucible-evaluator, crucible-backtester, crucible-dashboard,
    alpha-engine-config-I11017)."""
    repo = _git_repo(tmp_path)
    _add(repo, "requirements.txt", "krepis[flow-doctor, openai]==0.59.62\n")
    _allowlist(repo, "schema_version: 1\nentries: []\n")
    assert plg.main(["--repo", str(repo), "--include-dist"]) == 0


# -- the scan prefilters are exact (alpha-engine-config-I9938) ---------------
#
# `scan` skips a pattern for a file when `required_literals` or
# `body_prefilter_safe` proves it cannot hit any line. These pin that the
# proofs hold for every pattern in the table, at every line edge and under
# every case-fold the matcher applies, and that the scan's result is the same
# as the plain per-line loop it replaced.

#: Every line separator `str.splitlines` honours.
_SEPARATORS = ("\n", "\r", "\r\n", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", " ", " ")

#: Lines that hit, between them, every pattern class in the table, including
#: the case-fold oddities `re.IGNORECASE` accepts (U+0130/U+0131/U+017F/U+212A).
_HIT_LINES = (
    "anthropic.Anthropic(", "anthropic.AsyncAnthropic(", "from  anthropic import AsyncAnthropic",
    "ChatAnthropic(", "@anthropic-ai/sdk", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
    "API.ANTHROPIC.COM", "ANTHROPIC_BASE_URL", "anthropic", "openai.OpenAI(", "openai.AsyncOpenAI(",
    "from openai import OpenAI", "ChatOpenAI(", "OPENAI_API_KEY", "apİ.openai.com",
    "OPENAI_BASE_URL", "OpenAI", "OPENROUTER_API_KEY", "openrouter.aı", "DEEPSEEK_API_KEY",
    "api.deepſeek.com", "XAI_API_KEY", "GROK_API_KEY", "api.x.ai", "xai_sdk",
    "google.generativeai", "genai.GenerativeModel(", "ChatGoogleGenerativeAI(",
    "@google/generative-ai", "GEMINI_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY",
    "generativelanguage.googleapis.com", "google-genai", "GROQ_API_KEY", "api.groq.com", "groq",
    "MISTRAL_API_KEY", "api.mistral.ai", "mistralai", "GLM_API_KEY", "ZHIPU_API_KEY",
    "open.bigmodel.cn", "zhipuai", "bedrock-runtime", "@ai-sdk/", "api.Kimi.example",
)


def _all_patterns_all_classes():
    return plg.compile_patterns(plg.ALL_PROVIDER_NAMES, plg.PATTERN_CLASSES)


def test_every_hit_line_contains_a_required_literal_of_each_pattern_it_hits():
    covered = set()
    for klass, regex in _all_patterns_all_classes():
        lits = plg.required_literals(regex)
        for line in _HIT_LINES:
            if regex.search(line) is None:
                continue
            covered.add(klass)
            if lits is None:
                continue
            casefold, needles = lits
            if casefold and not line.isascii():
                continue  # scan never applies a casefold literal check to non-ASCII text
            hay = line.lower() if casefold else line
            assert any(n in hay for n in needles), (klass, line, needles)
    assert covered == {k for k, _ in _all_patterns_all_classes()}, "a pattern class has no hit line"


@pytest.mark.parametrize("sep", _SEPARATORS)
def test_a_safe_pattern_hitting_a_line_hits_the_whole_text_at_every_edge(sep):
    for klass, regex in _all_patterns_all_classes():
        if not plg.body_prefilter_safe(regex):
            continue
        for line in _HIT_LINES:
            if regex.search(line) is None:
                continue
            for body in (line + sep + "x", "x" + sep + line, "x" + sep + line + sep + "y"):
                assert regex.search(body) is not None, (klass, repr(sep), line)


@pytest.mark.parametrize("pattern,flags,safe", [
    (r"OPENAI_API_KEY", 0, True),
    (r"\bChatOpenAI\(", 0, True),
    (r"api\.openai\.com", "I", True),
    (r"^openai", 0, False),
    (r"openai$", 0, False),
    (r"\Aopenai", 0, False),
    (r"(?<![\w.-])openai", 0, False),
    (r"openai(?=\s)", 0, False),
    (r"\Bopenai", 0, False),
    (r"(\w)openai\1", 0, False),
    (r"(?m)openai", 0, False),
    (r"(?s)open.ai", 0, False),
    (r"(?a)\bopenai", 0, False),
    (r"openai", "M", False),
])
def test_body_prefilter_safe_refuses_line_edge_syntax(pattern, flags, safe):
    import re as _re
    f = {"I": _re.IGNORECASE, "M": _re.MULTILINE}.get(flags, 0) if flags else 0
    assert plg.body_prefilter_safe(_re.compile(pattern, f)) is safe


@pytest.mark.parametrize("pattern,flags,expected", [
    (r"abc(?:def|ghi)", 0, (False, {"abc"})),
    (r"(?:abcd|efgh)", 0, (False, {"abcd", "efgh"})),
    (r"(?:abcd|xy)", 0, None),
    (r"x*abcd", 0, (False, {"abcd"})),
    (r"(?:abcd)?xy", 0, None),
    (r"API\.X\.ai", "I", (True, {"api.x.ai"})),
    (r"(?i)API\.X\.ai", 0, (True, {"api.x.ai"})),
    (r"café", "I", None),
    (r"abcd", "M", None),
])
def test_required_literals_names_only_what_every_match_consumes(pattern, flags, expected):
    import re as _re
    f = {"I": _re.IGNORECASE, "M": _re.MULTILINE}.get(flags, 0) if flags else 0
    got = plg.required_literals(_re.compile(pattern, f))
    assert (got if got is None else (got[0], set(got[1]))) == expected


def _reference_scan(repo: Path, patterns, *, strip: bool = True):
    """The per-line loop `scan` ran before its prefilters, over the same file
    selection and exemptions -- the oracle the prefiltered scan must equal."""
    dist_suffix = f":{plg.CLASS_DIST}"
    out = []
    for fp in plg._tracked_files(repo, plg.DEFAULT_EXTENSIONS, plg.DEPENDENCY_FILENAME_RE):
        rel = str(fp.relative_to(repo))
        text = fp.read_text(errors="replace")
        is_dep = plg.DEPENDENCY_FILENAME_RE.match(fp.name) is not None
        applicable = [(k, r) for k, r in patterns if is_dep or not k.endswith(dist_suffix)]
        body = plg.strip_comments(rel, text) if strip else text
        for lineno, line in enumerate(body.splitlines(), 1):
            for klass, regex in applicable:
                if regex.search(line):
                    out.append((rel, lineno, klass, line.strip()))
    return out


def test_prefiltered_scan_equals_the_plain_per_line_loop(tmp_path):
    repo = _git_repo(tmp_path)
    patterns = _all_patterns_all_classes()
    for i, sep in enumerate(_SEPARATORS):
        _add(repo, f"src/m{i}.py", sep.join(("x = 1",) + _HIT_LINES + ("y = 2",)) + "\n")
        _add(repo, f"cfg/c{i}.yaml", sep.join(_HIT_LINES) + "\n")
    _add(repo, "requirements.txt", "\n".join(("openai==1.0", "anthropic>=0.3", "xai-sdk", "groq",
                                              "google-genai", "mistralai", "zhipuai", "requests")) + "\n")
    _add(repo, "web/a.ts", "// ANTHROPIC_API_KEY in a comment\nconst k = 'OPENAI_API_KEY';\n"
                           "/* from openai import OpenAI */ from /*c*/ openai import OpenAI\n")
    _add(repo, "quiet.py", "print('nothing to see')\n" * 50)
    got = plg.scan(repo, plg.DEFAULT_EXTENSIONS, patterns, registry_aware=False,
                   absence_aware=False, router_aware=False,
                   extra_filename_re=plg.DEPENDENCY_FILENAME_RE)
    raw = plg.scan(repo, plg.DEFAULT_EXTENSIONS, patterns, strip=False, registry_aware=False,
                   absence_aware=False, router_aware=False,
                   extra_filename_re=plg.DEPENDENCY_FILENAME_RE)
    as_tuples = lambda ms: [(m.path, m.line, m.pattern_class, m.text) for m in ms]  # noqa: E731
    assert as_tuples(got) == _reference_scan(repo, patterns)
    assert as_tuples(raw) == _reference_scan(repo, patterns, strip=False)
    assert len(got) > 100
