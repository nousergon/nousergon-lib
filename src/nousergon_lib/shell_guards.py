"""Static checks on spot-bootstrap shell scripts (alpha-engine-config#6916).

Every EC2-spot repo writes its instance bootstrap as a bash heredoc: a systemd
unit, a package install, a git clone, a staged config fetch. Those heredocs are
**invisible to every linter we run** — shellcheck sees a string, no systemd
tooling ever parses the unit, and the failure only appears on a freshly
launched instance minutes into a pipeline stage.

Three such defects hit production within 24 hours on 2026-08-10/11, each in a
separate copy of the same helper, and each hidden behind the one ahead of it:

1. a watchdog unit declaring ``Type=oneshot`` around an endless ExecStart, so
   ``systemctl start`` blocked until SSM killed the command
   (nousergon-data#1294, crucible-predictor#461);
2. ``command -v python3.12 || exit 1`` with nothing that installs it — an AMI
   contract nothing provided (nousergon-data#1296, crucible-predictor#462);
3. a single-quoted heredoc reading ``${REPO_URL}`` that the launcher never
   exported, so ``git clone`` ran against the empty string
   (crucible-predictor#463).

Each was first fixed in one repo and re-discovered in its twin. This module is
the second-adoption lift (`shared-code-policy` §2): the checks live here once,
and each consuming repo's test calls them against its own script.

Every function is **pure text analysis** — no subprocess, no filesystem beyond
what the caller reads — and returns a list of human-readable violation strings,
empty when the script is clean. The caller owns the assertion, so each repo
keeps its own failure message and its own provenance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "EmbeddedUnit",
    "embedded_units",
    "endless_execstart_violations",
    "unprovided_binary_violations",
    "unsupplied_variable_violations",
    "BLOCKING_SERVICE_TYPES",
    "BINARY_PROVIDERS",
]

#: Service types whose ``systemctl start`` waits for ExecStart to exit. For
#: ``oneshot`` in particular, ``TimeoutStartSec`` defaults to infinity — so an
#: endless ExecStart blocks the caller until the caller's own timeout fires.
BLOCKING_SERVICE_TYPES = frozenset({"oneshot"})

#: Shapes that never return on their own.
_ENDLESS = (
    re.compile(r"while\s+true"),
    re.compile(r"while\s+:\s*;"),
    re.compile(r"for\s*\(\(\s*;\s*;\s*\)\)"),
)

#: How a bootstrap may legitimately provide a binary it later asserts.
BINARY_PROVIDERS = ("dnf install", "yum install", "apt-get install", "curl -", "pip install")

#: Set by the SSM/Lambda/EC2 runtime, so a launcher need not export them.
_AMBIENT_VARS = frozenset({
    "HOME", "PATH", "PWD", "USER", "SHELL", "TMPDIR",
    "AWS_REGION", "AWS_DEFAULT_REGION", "XDG_CACHE_HOME",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI", "AWS_EXECUTION_ENV",
})


@dataclass(frozen=True)
class EmbeddedUnit:
    """A systemd unit written inside a shell heredoc."""

    body: str
    service_type: str
    execstart: str | None


def embedded_units(script_text: str) -> list[EmbeddedUnit]:
    """Every ``[Unit] … [Install]`` block written inside *script_text*."""
    units = []
    for match in re.finditer(r"\[Unit\].*?\[Install\]", script_text, re.S):
        body = match.group(0)
        type_match = re.search(r"^Type=(\S+)", body, re.M)
        exec_match = re.search(r"^ExecStart=(\S+)", body, re.M)
        units.append(EmbeddedUnit(
            body=body,
            service_type=type_match.group(1) if type_match else "simple",
            execstart=exec_match.group(1) if exec_match else None,
        ))
    return units


def _inline_script_body(script_path: str, script_text: str) -> str:
    """The heredoc body written to *script_path* within the same file.

    Bootstraps write a unit's ExecStart target in a sibling heredoc, e.g.
    ``cat > /usr/local/bin/x.sh <<'WDSH' … WDSH``. Returns "" when the target
    is written elsewhere — nothing to analyse, so nothing to assert.
    """
    pattern = rf"cat\s*>\s*{re.escape(script_path)}\s*<<'?(\w+)'?\n(.*?)\n\1"
    match = re.search(pattern, script_text, re.S)
    return match.group(2) if match else ""


def endless_execstart_violations(script_text: str) -> list[str]:
    """Units whose ExecStart never returns under a blocking service type.

    A genuinely one-shot job (a boot-time pull, a drift check) is correctly
    ``Type=oneshot`` and is not flagged: the check keys on the ExecStart
    script's own body containing an unbounded loop, not on the type alone.
    """
    violations = []
    for unit in embedded_units(script_text):
        if not unit.execstart:
            continue
        body = _inline_script_body(unit.execstart, script_text)
        if not body:
            continue
        if not any(rx.search(body) for rx in _ENDLESS):
            continue
        if unit.service_type in BLOCKING_SERVICE_TYPES:
            violations.append(
                f"{unit.execstart} runs an unbounded loop but its unit declares "
                f"Type={unit.service_type}. `systemctl start` blocks until ExecStart "
                f"exits — which it never does — and the caller dies when its own "
                f"timeout kills it. Use Type=simple."
            )
    return violations


def unprovided_binary_violations(block_text: str) -> list[str]:
    """Fatal ``command -v X … exit 1`` guards with nothing that provides X.

    An assertion that a tool exists is a PRECONDITION on the image. A
    bootstrap's job is to establish preconditions, not to require them — so
    every fatal guard must be preceded, in the same block, by an install.
    """
    violations = []
    for match in re.finditer(r"command -v (\S+)[^\n]*\|\|[^\n]*exit 1", block_text):
        binary = match.group(1)
        preceding = block_text[: match.start()]
        base = binary.split(".")[0]
        provided = any(
            provider in preceding and (binary in preceding or base in preceding)
            for provider in BINARY_PROVIDERS
        )
        if not provided:
            violations.append(
                f"the block asserts `{binary}` is present but never installs it "
                f"first — a bootstrap establishes preconditions, it does not "
                f"require them."
            )
    return violations


def unsupplied_variable_violations(
    heredoc_body: str,
    exported: set[str] | frozenset[str],
    *,
    ambient: frozenset[str] = _AMBIENT_VARS,
) -> list[str]:
    """Variables a single-quoted heredoc reads that nobody supplies.

    A single-quoted heredoc is literal on the target, so its variables arrive
    only via the launcher's export prefix or an assignment inside the body. A
    bare ``${X}`` that is neither resolves to the EMPTY STRING on the instance
    — silently, which is how a ``git clone`` ran against ``''``.

    Reads carrying a ``${X:-default}`` fallback are safe by construction and
    are not flagged.
    """
    assigned = set(re.findall(r"^\s*(?:export\s+)?(\w+)=", heredoc_body, re.M))
    for line in re.findall(r"^\s*export\s+([^\n]+)", heredoc_body, re.M):
        assigned |= set(re.findall(r"(\w+)=", line))

    read: set[str] = set()
    for match in re.finditer(r"\$\{(\w+)([^}]*)\}", heredoc_body):
        name, tail = match.group(1), match.group(2)
        if tail.startswith("-") or tail.startswith(":-"):
            continue  # defaulted
        read.add(name)
    read |= set(re.findall(r"\$(\w+)\b", heredoc_body))

    return [
        f"the heredoc reads ${{{name}}} but it is neither exported by the "
        f"launcher nor assigned in the body. A single-quoted heredoc is literal "
        f"on the instance, so this resolves to the EMPTY STRING there."
        for name in sorted(read - set(exported) - assigned - ambient)
    ]
