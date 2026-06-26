"""
config.py — canonical experiment-package-first config resolution.

The experiment-package-first config-resolution pattern is mirrored
**inline** across five Alpha Engine entrypoints (alpha-engine-config#1157):

  - alpha-engine-research   ``config.py::_find_config``
  - alpha-engine-data       ``weekly_collector.py::load_config`` +
                            ``features/feature_engineer.py``
  - crucible-executor       ``executor/config_loader.py``
  - crucible-backtester     ``pipeline_common.py::load_config``
  - crucible-predictor      ``config.py`` (reads ``ALPHA_ENGINE_EXPERIMENT_ID``)

Every copy resolves a config file by searching, in order:

  1. the **experiment-package** copy —
     ``<config-root>/experiments/$ALPHA_ENGINE_EXPERIMENT_ID/<subdir>/<file>``
     (default experiment ``reference``) — for each config root,
  2. the **legacy top-level** copy —
     ``<config-root>/<subdir>/<file>`` — for each config root,
  3. a **repo-local** fallback supplied by the caller.

The config roots are the sibling clone at ``~/alpha-engine-config`` and the
repo-parent clone at ``<repo_root>/../alpha-engine-config``. The experiment id
is read from ``ALPHA_ENGINE_EXPERIMENT_ID`` (default ``reference``).

Origin (HARNESS_EXPERIMENT_CLASSIFICATION §3 / config#1042): experiment
"beliefs" load from the experiment package ahead of the legacy top-level
location, which is retained as a fallback through the transition.

This module is the single source of truth the five inline copies can adopt.
It is IO-agnostic: it returns *paths*, not parsed config. Each consumer keeps
its own parse/validate tail (backtester's ``_validate_config``, executor's
``yaml.safe_load``, data's ``Path(path)`` repo-local default, etc.).

Per-repo divergences are exposed as OPTIONS with sane defaults so a consumer
can adopt the helper without losing its specifics:

  - ``github_workspace`` (research): also search
    ``$GITHUB_WORKSPACE/alpha-engine-config`` as a config root (CI checkout).
  - ``resolve_symlinks`` (executor): resolve each candidate to its real path
    and test it as a file rather than ``Path.exists()`` — executor uses
    ``os.path.realpath`` + ``os.path.isfile``.
  - ``exclude_suffixes`` (executor's ``.example``-never-searched guard):
    never treat a candidate whose name ends in one of these suffixes as a
    match. Executor ships ``risk.yaml.example`` with placeholder bucket names
    that would silently point downstream at nonexistent S3 buckets; the
    template must be copyable but never auto-resolved.
  - ``extra_fallbacks``: additional repo-local candidates appended after the
    consensus search order (for consumers with more than one local fallback).

Two entry points::

    from nousergon_lib.config import resolve_experiment_config

    # The ordered candidate list (what every inline copy builds internally):
    candidates = resolve_experiment_config(
        "backtester", "config.yaml", repo_root=Path(__file__).parent
    )

    # Or resolve the first existing one, raising if none match:
    path = resolve_experiment_config(
        "executor", "risk.yaml",
        repo_root=Path(__file__).parent.parent,
        resolve=True,
        resolve_symlinks=True,
        exclude_suffixes=(".example",),
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

__all__ = ["resolve_experiment_config", "DEFAULT_EXPERIMENT_ID"]

# The experiment slot used when ``ALPHA_ENGINE_EXPERIMENT_ID`` is unset.
# Baked in identically across all five inline copies (config#1042).
DEFAULT_EXPERIMENT_ID = "reference"

_CONFIG_REPO_DIRNAME = "alpha-engine-config"

PathLike = Union[str, os.PathLike]


def _config_roots(repo_root: Path, github_workspace: Optional[str]) -> List[Path]:
    """Build the ordered config-repo roots searched for the experiment package.

    Consensus across all five inline copies:
      1. ``~/alpha-engine-config``                    (sibling clone, local dev)
      2. ``<repo_root>/../alpha-engine-config``       (repo-parent clone)

    Research additionally appends ``$GITHUB_WORKSPACE/alpha-engine-config`` for
    its CI checkout; this is opt-in via ``github_workspace`` so the default
    matches the four-consumer majority.
    """
    roots = [
        Path.home() / _CONFIG_REPO_DIRNAME,
        repo_root.parent / _CONFIG_REPO_DIRNAME,
    ]
    if github_workspace:
        roots.append(Path(github_workspace) / _CONFIG_REPO_DIRNAME)
    return roots


def _candidate_paths(
    subdir: str,
    filename: str,
    *,
    repo_root: Path,
    experiment_id: str,
    github_workspace: Optional[str],
    repo_local_fallbacks: Sequence[Path],
) -> List[Path]:
    """Assemble the full ordered candidate list (experiment → legacy → local)."""
    roots = _config_roots(repo_root, github_workspace)
    # 1. experiment-package copy, per root
    candidates: List[Path] = [
        root / "experiments" / experiment_id / subdir / filename for root in roots
    ]
    # 2. legacy top-level copy, per root
    candidates += [root / subdir / filename for root in roots]
    # 3. repo-local fallback(s), in caller order
    candidates += list(repo_local_fallbacks)
    return candidates


def _is_excluded(path: Path, exclude_suffixes: Sequence[str]) -> bool:
    """True if ``path`` ends with any excluded suffix (e.g. ``.example``).

    Matches against the full name so ``risk.yaml.example`` is excluded by the
    suffix ``.example``. Comparison is on the string tail, not ``Path.suffix``,
    because ``Path("risk.yaml.example").suffix`` is ``.example`` but
    ``Path("config.example.yaml").suffix`` is ``.yaml`` — the inline guard the
    executor ships keys off the literal ``.example`` tail.
    """
    name = path.name
    return any(name.endswith(suffix) for suffix in exclude_suffixes)


def _exists(path: Path, *, resolve_symlinks: bool) -> bool:
    """Existence test, optionally symlink-resolving (executor semantics).

    Default uses ``Path.exists()`` (the research/data/backtester majority).
    With ``resolve_symlinks=True`` the candidate is resolved to its real path
    and tested with ``os.path.isfile`` — matching executor's
    ``os.path.realpath`` + ``os.path.isfile`` (rejects directories and dangling
    symlinks, follows valid ones to the real file).
    """
    if resolve_symlinks:
        return os.path.isfile(os.path.realpath(path))
    return path.exists()


def resolve_experiment_config(
    subdir: str,
    filename: str,
    *,
    repo_root: PathLike,
    extra_fallbacks: Iterable[PathLike] = (),
    repo_local_fallback: Optional[PathLike] = None,
    experiment_id: Optional[str] = None,
    github_workspace: Union[str, bool, None] = None,
    resolve: bool = False,
    resolve_symlinks: bool = False,
    exclude_suffixes: Sequence[str] = (),
    error_message: Optional[str] = None,
) -> Union[List[Path], Path]:
    """Resolve a config file experiment-package-first (config#1042 consensus).

    The canonical lift of the inline ``_find_config`` / ``load_config`` /
    ``config_loader`` copies mirrored across the five Alpha Engine entrypoints
    (alpha-engine-config#1157). Search order, per config root, then local:

      1. ``<root>/experiments/<experiment_id>/<subdir>/<filename>``
      2. ``<root>/<subdir>/<filename>``                  (legacy top-level)
      3. repo-local fallback(s)

    where the config roots are ``~/alpha-engine-config`` and
    ``<repo_root>/../alpha-engine-config`` (plus ``$GITHUB_WORKSPACE/
    alpha-engine-config`` when ``github_workspace`` is supplied), and
    ``experiment_id`` defaults to ``$ALPHA_ENGINE_EXPERIMENT_ID`` or
    ``reference``.

    Args:
        subdir: Config subdirectory inside the config repo (e.g. ``executor``,
            ``backtester``, ``data``, ``research``).
        filename: Config file name (e.g. ``risk.yaml``, ``config.yaml``).
        repo_root: The consumer repo's root. ``<repo_root>/..`` is searched for
            a sibling ``alpha-engine-config`` clone, and ``repo_root`` anchors
            the default repo-local fallback (``<repo_root>/<subdir>/<filename>``)
            when none is given explicitly.
        extra_fallbacks: Extra repo-local candidates appended after the
            consensus order (for consumers with multiple local fallbacks).
        repo_local_fallback: Override the single repo-local fallback. Defaults
            to ``<repo_root>/<subdir>/<filename>`` — matches the
            ``Path(path)``-style tail the inline copies use. Pass an explicit
            path (e.g. ``<repo_root>/config/<filename>``) to match a consumer
            whose local layout flattens the subdir (research, executor).
        experiment_id: Override the experiment slot. Defaults to
            ``os.environ["ALPHA_ENGINE_EXPERIMENT_ID"]`` or
            :data:`DEFAULT_EXPERIMENT_ID` (``reference``).
        github_workspace: Add ``$GITHUB_WORKSPACE/alpha-engine-config`` as a
            config root (research's CI behavior). Pass ``True`` to read
            ``$GITHUB_WORKSPACE`` from the environment, or a path string to set
            it explicitly. Default (``None``/``False``) omits the CI root —
            matching the four-consumer majority.
        resolve: If ``True``, return the first existing candidate (a ``Path``)
            and raise ``FileNotFoundError`` if none exist. If ``False``
            (default), return the full ordered candidate list without touching
            the filesystem.
        resolve_symlinks: When resolving, follow symlinks via ``os.path.realpath``
            and test with ``os.path.isfile`` (executor semantics) instead of
            ``Path.exists()``. With ``resolve=True`` the returned path is the
            resolved real path.
        exclude_suffixes: Candidate name suffixes that are never a match even
            if present (executor's ``(".example",)`` guard — the template ships
            placeholder values and must never auto-resolve). Excluded
            candidates are filtered out of the returned candidate list too, so
            the listed candidates and the resolution always agree.
        error_message: Override the ``FileNotFoundError`` message when
            ``resolve=True`` and nothing matches.

    Returns:
        With ``resolve=False`` (default): the ordered ``list[Path]`` of
        candidates (excluded-suffix candidates filtered out). With
        ``resolve=True``: the first existing candidate ``Path``.

    Raises:
        FileNotFoundError: When ``resolve=True`` and no candidate exists.
    """
    repo_root_path = Path(repo_root)

    if experiment_id is None:
        experiment_id = os.environ.get("ALPHA_ENGINE_EXPERIMENT_ID", DEFAULT_EXPERIMENT_ID)

    # Resolve the GITHUB_WORKSPACE opt-in: True → read env; str → use as-is;
    # None/False → omit the CI root.
    if github_workspace is True:
        ws: Optional[str] = os.environ.get("GITHUB_WORKSPACE")
    elif github_workspace in (None, False):
        ws = None
    else:
        ws = str(github_workspace)

    # Repo-local fallback chain: the single default (or override) first, then
    # any extra_fallbacks, in caller order.
    if repo_local_fallback is not None:
        repo_local_fallbacks: List[Path] = [Path(repo_local_fallback)]
    else:
        repo_local_fallbacks = [repo_root_path / subdir / filename]
    repo_local_fallbacks += [Path(p) for p in extra_fallbacks]

    candidates = _candidate_paths(
        subdir,
        filename,
        repo_root=repo_root_path,
        experiment_id=experiment_id,
        github_workspace=ws,
        repo_local_fallbacks=repo_local_fallbacks,
    )

    if exclude_suffixes:
        candidates = [c for c in candidates if not _is_excluded(c, exclude_suffixes)]

    if not resolve:
        return candidates

    for candidate in candidates:
        if _exists(candidate, resolve_symlinks=resolve_symlinks):
            return Path(os.path.realpath(candidate)) if resolve_symlinks else candidate

    if error_message is None:
        error_message = (
            f"Config {subdir}/{filename} not found. Searched: "
            f"{[str(c) for c in candidates]}"
        )
    raise FileNotFoundError(error_message)
