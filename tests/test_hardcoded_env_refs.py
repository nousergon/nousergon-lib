"""Tests for scripts/hardcoded_env_refs.py (plan S-04, alpha-engine-config-I12492).

The guard is shrink-only: a count that rises is ``new``, a baseline entry that
no longer reproduces is ``stale``, and ``--update`` can only lower the ceiling.
Observe mode reports both and exits 0; enforce mode exits 1.

Every test builds a real throwaway git repo, because the scan surface is
``git ls-files`` (same convention as test_provider_linkage_guard.py). The
literals are assembled from pieces so this file is not itself a hit.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hardcoded_env_refs.py"
_GIT = shutil.which("git") or "git"

ACCOUNT = "7113" + "98986525"
RESEARCH = "alpha-engine-" + "research"
CRUCIBLE = "alpha-engine-" + "crucible-v2"


def _load():
    spec = importlib.util.spec_from_file_location("hardcoded_env_refs", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hardcoded_env_refs"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


g = _load()


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    subprocess.run([_GIT, "init", "-q"], cwd=repo, check=True)
    _write(repo, files)
    return repo


def _write(repo: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        fp = repo / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(text)
        subprocess.run([_GIT, "-C", str(repo), "add", rel], check=True)


def _rm(repo: Path, rel: str) -> None:
    subprocess.run([_GIT, "-C", str(repo), "rm", "-q", "-f", rel], check=True)


# --- scope -----------------------------------------------------------------


def test_counts_code_and_infra_literals(tmp_path):
    repo = _repo(
        tmp_path,
        {
            "app/io.py": f'B = "{RESEARCH}"\nC = "s3://{CRUCIBLE}/x"\n',
            "infra/role.json": f'{{"Resource": "arn:aws:iam::{ACCOUNT}:role/r"}}\n',
            ".github/workflows/deploy.yml": f"role: arn:aws:iam::{ACCOUNT}:role/x\n",
            "bin/tool": f"#!/bin/sh\naws s3 ls s3://{RESEARCH}/\n",
            "conf/app.yaml.example": f"bucket: {RESEARCH}\n",
        },
    )
    counts = g.scan(repo)
    assert counts == {
        "app/io.py": {"bucket": 2},
        "infra/role.json": {"account": 1},
        ".github/workflows/deploy.yml": {"account": 1},
        "bin/tool": {"bucket": 1},
        "conf/app.yaml.example": {"bucket": 1},
    }


def test_docs_changelogs_history_and_registry_are_exempt(tmp_path):
    repo = _repo(
        tmp_path,
        {
            "README.md": f"{ACCOUNT} {RESEARCH}\n",
            "notes.txt": f"{RESEARCH}\n",
            "docs/example.py": f'"{RESEARCH}"\n',
            "changelog.d/20261021.yaml": f"bucket: {RESEARCH}\n",
            "audits/2026-10/evidence.json": f'"{ACCOUNT}"\n',
            "CHANGELOG.json": f'"{ACCOUNT}"\n',
            "governance/environments.yaml": f'account: "{ACCOUNT}"\n',
            "src/nousergon_lib/environments.yaml": f"research: {RESEARCH}\n",
            "bin/noshebang": f"{RESEARCH}\n",
        },
    )
    assert g.scan(repo) == {}


def test_changelog_workflow_is_not_mistaken_for_a_changelog(tmp_path):
    repo = _repo(tmp_path, {".github/workflows/changelog.yml": f"arn:aws:iam::{ACCOUNT}:role/x\n"})
    assert g.scan(repo) == {".github/workflows/changelog.yml": {"account": 1}}


def test_boundaries_do_not_match_other_names(tmp_path):
    repo = _repo(
        tmp_path,
        {
            "a.py": (
                f'X = "{RESEARCH}-logs"\n'  # a different bucket
                f'Y = "my-{RESEARCH}"\n'
                f'Z = "{ACCOUNT}9"\n'  # a longer number
                f'W = "s3://{RESEARCH}/k"\n'  # counts
            )
        },
    )
    assert g.scan(repo) == {"a.py": {"bucket": 1}}


def test_untracked_files_are_ignored(tmp_path):
    repo = _repo(tmp_path, {"a.py": "x = 1\n"})
    (repo / "b.py").write_text(f'"{RESEARCH}"\n')
    assert g.scan(repo) == {}


# --- shrink-only -----------------------------------------------------------


def _init(repo: Path) -> dict:
    assert g.main(["--repo", str(repo), "--init"]) == 0
    return json.loads((repo / g.BASELINE_NAME).read_text())


def test_existing_hits_pass_in_both_modes(tmp_path):
    repo = _repo(tmp_path, {"a.py": f'"{RESEARCH}" "{ACCOUNT}"\n'})
    doc = _init(repo)
    assert doc["files"] == {"a.py": {"account": 1, "bucket": 1}}
    assert doc["totals"]["account_hits"] == 1 and doc["totals"]["files"] == 1
    assert g.main(["--repo", str(repo), "--mode", "enforce"]) == 0
    assert g.main(["--repo", str(repo)]) == 0


def test_a_new_hit_is_reported_and_fails_only_when_enforcing(tmp_path, capsys):
    repo = _repo(tmp_path, {"a.py": f'"{RESEARCH}"\n'})
    _init(repo)
    _write(repo, {"a.py": f'"{RESEARCH}"\n"{RESEARCH}"\n', "b.sh": f"echo {ACCOUNT}\n"})
    assert g.main(["--repo", str(repo)]) == 0  # observe
    out = capsys.readouterr().out
    assert "NEW   a.py: bucket literal count 1 -> 2" in out
    assert "NEW   b.sh: account literal count 0 -> 1" in out
    assert "nousergon_lib.env" in out
    assert "observe mode" in out
    assert g.main(["--repo", str(repo), "--mode", "enforce"]) == 1


def test_a_baseline_entry_that_no_longer_reproduces_is_stale(tmp_path, capsys):
    repo = _repo(tmp_path, {"a.py": f'"{RESEARCH}"\n"{RESEARCH}"\n', "b.py": f'"{ACCOUNT}"\n'})
    _init(repo)
    _write(repo, {"a.py": f'"{RESEARCH}"\n'})
    _rm(repo, "b.py")
    assert g.main(["--repo", str(repo), "--mode", "enforce"]) == 1
    out = capsys.readouterr().out
    assert "STALE a.py: bucket baseline 2 but 1 now" in out
    assert "STALE b.py: account baseline 1 but 0 now" in out


def test_update_only_lowers_the_ceiling(tmp_path):
    repo = _repo(tmp_path, {"a.py": f'"{RESEARCH}"\n"{RESEARCH}"\n', "b.py": f'"{ACCOUNT}"\n'})
    _init(repo)
    # a.py shrinks, b.py grows, c.py is new: only the shrink is written.
    _write(
        repo,
        {"a.py": f'"{RESEARCH}"\n', "b.py": f'"{ACCOUNT}" "{ACCOUNT}"\n', "c.py": f'"{CRUCIBLE}"\n'},
    )
    assert g.main(["--repo", str(repo), "--update"]) == 0
    doc = json.loads((repo / g.BASELINE_NAME).read_text())
    assert doc["files"] == {"a.py": {"bucket": 1}, "b.py": {"account": 1}}
    new, stale = g.compare(g.scan(repo), g.load_baseline(repo / g.BASELINE_NAME))
    assert stale == []
    assert {(r["path"], r["kind"]) for r in new} == {("b.py", "account"), ("c.py", "bucket")}


def test_init_refuses_to_overwrite(tmp_path):
    repo = _repo(tmp_path, {"a.py": f'"{RESEARCH}"\n'})
    _init(repo)
    assert g.main(["--repo", str(repo), "--init"]) == 2


def test_missing_baseline_is_a_zero_ceiling(tmp_path):
    repo = _repo(tmp_path, {"a.py": f'"{RESEARCH}"\n'})
    assert g.main(["--repo", str(repo), "--mode", "enforce"]) == 1
    clean = _repo(tmp_path / "c", {"a.py": "x = 1\n"})
    assert g.main(["--repo", str(clean), "--mode", "enforce"]) == 0


@pytest.mark.parametrize(
    "entry",
    [{"account": 0}, {"bucket": "2"}, {"region": 1}, ["account"]],
)
def test_malformed_baseline_entries_are_rejected(tmp_path, entry):
    repo = _repo(tmp_path, {"a.py": "x = 1\n"})
    (repo / g.BASELINE_NAME).write_text(json.dumps({"schema_version": 1, "guard": g.GUARD, "files": {"a.py": entry}}))
    with pytest.raises(SystemExit):
        g.load_baseline(repo / g.BASELINE_NAME)


def test_github_annotations_and_summary(tmp_path, capsys, monkeypatch):
    repo = _repo(tmp_path, {"a.py": "x = 1\n"})
    _init(repo)
    _write(repo, {"a.py": f'x = 1\nB = "{RESEARCH}"\n'})
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert g.main(["--repo", str(repo)]) == 0
    out = capsys.readouterr().out
    assert "::warning file=a.py,line=2,title=hardcoded-env-refs::" in out
    assert "alpha-engine-config-I12492" in summary.read_text()
    assert g.main(["--repo", str(repo), "--mode", "enforce"]) == 1
    assert "::error file=a.py,line=2" in capsys.readouterr().out
