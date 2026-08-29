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
