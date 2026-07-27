"""Shared harness-neutral repository-context loader (nous-ergon-ops-I47).

Walks up from *cwd* collecting every ``AGENTS.md`` (or ``CLAUDE.md``) on the
way to the filesystem root, and returns them concatenated root-first — the
same resolution Claude Code performs for its own instruction files. Any
non-Claude-Code-CLI caller — a raw DeepSeek/xAI/OpenAI API integration, a
script building its own `role: "system"` message, a non-Anthropic agent
harness — calls :func:`load_repo_context` and injects the returned string
verbatim as a system-prompt prefix.

Usage::

    from nousergon_lib.context import load_repo_context

    ctx = load_repo_context("/path/to/repo")
    if ctx:
        system_message = f"{ctx}\\n\\n{original_system_message}"

Why concatenate rather than take the nearest
--------------------------------------------
This used to return the FIRST file found and stop. That made a repo-level
``AGENTS.md`` *replace* the fleet-level one rather than supplement it, so
adding a small repo file silently shrank an agent's context from the full
fleet conventions down to whatever that file happened to say.

Claude Code does the opposite — it concatenates every level, root-first — so
the same repo produced materially different context depending on which
consumer read it, while ``context-delivery-policy.md`` §6 asserted the two
implementations behaved identically. They now do.

Ordering matches Claude Code's: broadest scope first, most specific last, so
a repo-level instruction appears after (and therefore overrides) a
fleet-level one when they conflict.

Caveats:
- The cache lives for the lifetime of the process.  If instructions are
  edited live on a long-running worker, the next call picks up the new
  mtime automatically.
- Walk depth is capped at 8 levels — a repo root shouldn't be deeper
  than that from any plausible working directory.
"""

from __future__ import annotations

import os

_MAX_WALK_DEPTH = 8
_CANDIDATES = ("AGENTS.md", "CLAUDE.md")

_cache: dict[str, tuple[float, str]] = {}  # path -> (mtime, content)


def _read_cached(path: str) -> str | None:
    """Return *path*'s content, using the ``(path, mtime)`` cache."""
    try:
        stat = os.stat(path)
    except OSError:
        return None  # ENOENT / EACCES

    cached = _cache.get(path)
    if cached is not None and cached[0] == stat.st_mtime:
        return cached[1]

    try:
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
    except OSError:
        return None

    _cache[path] = (stat.st_mtime, content)
    return content


def load_repo_context(cwd: str) -> str | None:
    """Return every ``AGENTS.md`` from *cwd* up to the root, concatenated.

    Walks up from *cwd* (an absolute path). At each level ``AGENTS.md`` is
    preferred over ``CLAUDE.md``; at most one file is taken per directory,
    since ``CLAUDE.md`` is conventionally a symlink to ``AGENTS.md`` and
    reading both would duplicate the content.

    Sections are joined **root-first**, matching Claude Code's ordering, so
    the most specific instructions come last and win on conflict. Each is
    preceded by a header naming its source, because an agent reading tens of
    kilobytes of concatenated instructions needs to know which repo a rule
    came from.

    Returns ``None`` when no instruction file exists anywhere up the tree.
    """
    sections: list[tuple[str, str]] = []
    directory = cwd

    for _ in range(_MAX_WALK_DEPTH):
        for name in _CANDIDATES:
            candidate = os.path.join(directory, name)
            content = _read_cached(candidate)
            if content is not None:
                sections.append((candidate, content))
                break  # one file per directory — CLAUDE.md is usually a symlink

        parent = os.path.dirname(directory)
        if parent == directory:  # filesystem root
            break
        directory = parent

    if not sections:
        return None

    # Collected nearest-first; emit root-first so specificity increases.
    sections.reverse()

    if len(sections) == 1:
        return sections[0][1]

    return "\n\n".join(
        f"── Repository context: {path} ──\n\n{content}" for path, content in sections
    )


def _clear_cache() -> None:
    """Purge the in-process cache (exposed for tests only)."""
    _cache.clear()
