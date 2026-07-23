"""Shared harness-neutral repository-context loader (nous-ergon-ops-I47).

Walks up from *cwd* looking for ``AGENTS.md`` (preferred) or ``CLAUDE.md``
(fallback), caching by (path, mtime). Any non-Claude-Code-CLI caller —
a raw DeepSeek/xAI/OpenAI API integration, a script building its own
`role: "system"` message, a non-Anthropic agent harness — calls
:func:`load_repo_context` and injects the returned string verbatim
as a system-prompt prefix.

Usage::

    from nousergon_lib.context import load_repo_context

    ctx = load_repo_context("/path/to/repo")
    if ctx:
        system_message = f"{ctx}\\n\\n{original_system_message}"

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


def load_repo_context(cwd: str) -> str | None:
    """Return the nearest ``AGENTS.md`` / ``CLAUDE.md`` content, or *None*.

    Walks up from *cwd* (an absolute path) looking for ``AGENTS.md`` first,
    then ``CLAUDE.md`` as fallback.  The first file found is returned; if both
    exist at the same level ``AGENTS.md`` wins (the harness-neutral canonical
    name).

    The result is cached keyed on ``(path, mtime)`` — a subsequent call with
    the same file unchanged returns the cached string without a disk read.
    """
    directory = cwd
    for _ in range(_MAX_WALK_DEPTH):
        for name in _CANDIDATES:
            candidate = os.path.join(directory, name)
            try:
                stat = os.stat(candidate)
            except OSError:
                continue  # ENOENT / EACCES → try next candidate or parent

            mtime = stat.st_mtime
            cached = _cache.get(candidate)
            if cached is not None and cached[0] == mtime:
                return cached[1]

            try:
                with open(candidate, encoding="utf-8") as fh:
                    content = fh.read()
            except OSError:
                continue

            _cache[candidate] = (mtime, content)
            return content

        parent = os.path.dirname(directory)
        if parent == directory:  # filesystem root
            break
        directory = parent

    return None


def _clear_cache() -> None:
    """Purge the in-process cache (exposed for tests only)."""
    _cache.clear()
