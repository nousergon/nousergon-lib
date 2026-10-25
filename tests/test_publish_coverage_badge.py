"""scripts/publish_coverage_badge.sh under concurrent pushes to main.

Two pushes seconds apart (a merge, then its merge-time version bump) ran the
badge step at the same moment on 2026-10-24 18:16Z. Both read one blob sha;
the first PUT won and the second failed on the stale sha, which turned main
red on a run whose tests had passed. The script now publishes only for the
current head of main and retries a sha conflict. These tests run the real
script with `gh` and `python -m coverage` stubbed on PATH / PYTHONPATH.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BASH = shutil.which("bash") or "/bin/bash"
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "publish_coverage_badge.sh"

FAKE_GH = r"""#!/usr/bin/env bash
echo "$*" >> "$FAKE_LOG"
case "$*" in
  *"commits/main"*) echo "$FAKE_HEAD" ;;
  *"--method PUT"*)
    n=$(grep -c -- '--method PUT' "$FAKE_LOG")
    if [ "$n" -le "${FAKE_PUT_FAILS:-0}" ]; then echo "gh: sha does not match (HTTP 409)" >&2; exit 1; fi
    echo newcommit ;;
  *"--jq .content"*) echo "$FAKE_CONTENT_B64" ;;
  *"--jq .sha"*) echo blobsha ;;
  *) echo "unexpected gh call: $*" >&2; exit 2 ;;
esac
"""

FAKE_COVERAGE = r"""
import json, sys
out = sys.argv[sys.argv.index("-o") + 1]
json.dump({"totals": {"percent_covered": 93.5}}, open(out, "w"))
"""


def _endpoint(pct: str = "93.50") -> str:
    return json.dumps({"schemaVersion": 1, "label": "coverage", "message": f"{pct}%", "color": "brightgreen"}, indent=2)


@pytest.fixture
def env(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    gh = bindir / "gh"
    gh.write_text(FAKE_GH)
    gh.chmod(0o755)
    py = bindir / "python"
    py.symlink_to(sys.executable)
    pkg = tmp_path / "pylib" / "coverage"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "__main__.py").write_text(FAKE_COVERAGE)
    log = tmp_path / "gh.log"
    log.write_text("")
    base = {
        "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": str(tmp_path / "pylib"),
        "GH_TOKEN": "x",
        "GITHUB_REPOSITORY": "nousergon/nousergon-lib",
        "GITHUB_SHA": "aaaa",
        "FAKE_HEAD": "aaaa",
        "FAKE_LOG": str(log),
        "FAKE_CONTENT_B64": base64.b64encode(_endpoint("90.00").encode()).decode(),
        "BADGE_PUBLISH_BACKOFF_S": "0",
        "HOME": str(tmp_path),
    }
    return tmp_path, base, log


def _run(tmp_path: Path, env_vars: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([BASH, str(SCRIPT)], cwd=tmp_path, env=env_vars, capture_output=True, text=True)


def _puts(log: Path) -> int:
    return log.read_text().count("--method PUT")


def test_publishes_for_the_current_head(env: tuple[Path, dict[str, str], Path]) -> None:
    tmp, e, log = env
    r = _run(tmp, e)
    assert r.returncode == 0, r.stderr
    assert _puts(log) == 1


def test_an_older_run_does_not_publish_over_a_newer_push(env: tuple[Path, dict[str, str], Path]) -> None:
    tmp, e, log = env
    r = _run(tmp, {**e, "FAKE_HEAD": "bbbb"})
    assert r.returncode == 0, r.stderr
    assert "the newer push publishes" in r.stdout
    assert _puts(log) == 0


def test_a_sha_conflict_is_retried(env: tuple[Path, dict[str, str], Path]) -> None:
    tmp, e, log = env
    r = _run(tmp, {**e, "FAKE_PUT_FAILS": "1"})
    assert r.returncode == 0, r.stderr
    assert _puts(log) == 2
    assert "attempt 1/3" in r.stderr


def test_persistent_failure_still_fails_loud(env: tuple[Path, dict[str, str], Path]) -> None:
    tmp, e, log = env
    r = _run(tmp, {**e, "FAKE_PUT_FAILS": "9"})
    assert r.returncode == 1
    assert _puts(log) == 3
    assert "after 3 attempts" in r.stderr


def test_unchanged_coverage_publishes_nothing(env: tuple[Path, dict[str, str], Path]) -> None:
    tmp, e, log = env
    same = base64.b64encode(_endpoint("93.50").encode()).decode()
    r = _run(tmp, {**e, "FAKE_CONTENT_B64": same})
    assert r.returncode == 0, r.stderr
    assert "nothing to publish" in r.stdout
    assert _puts(log) == 0
