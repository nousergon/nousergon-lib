"""Pin ``nousergon_lib.__version__`` to ``pyproject.toml::version``.

Doc-string drift: 2026-05-08 we shipped v0.5.6 by bumping
``pyproject.toml::version`` 0.5.5 → 0.5.6 but forgot to bump
``src/nousergon_lib/__init__.py::__version__``. The wheel built
from v0.5.6 had package metadata 0.5.6 but the runtime
``__version__`` string lagged at 0.5.5 — confusing for any consumer
that reads the runtime attribute (operator scripts, dashboards,
log lines).

Functional impact: zero (load_inventory + every other code path
reads the YAML or the actual code, not the version string).
But the drift makes "what version is deployed?" harder to answer.

This test pins the two together so future bumps that miss one
side fail at CI.
"""

from __future__ import annotations

import re
from pathlib import Path

import nousergon_lib

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _pyproject_version() -> str:
    """Read ``version = "X.Y.Z"`` from pyproject.toml.

    Avoid a tomllib import (3.11+) — kept stdlib-free + Python-3.9-safe
    via a single regex match. The line we want is the only top-level
    ``version = "..."`` in the [project] block; the file ships one of
    these per release.
    """
    text = _PYPROJECT.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match is not None, "version not found in pyproject.toml"
    return match.group(1)


def test_init_version_matches_pyproject():
    """Lockstep pin: bumping one without the other fails here.

    If you're updating this test to a new version, you almost
    certainly want to bump BOTH ``pyproject.toml`` and
    ``src/nousergon_lib/__init__.py`` in the same commit. Search
    for ``__version__`` and ``version =`` to find them.
    """
    assert nousergon_lib.__version__ == _pyproject_version(), (
        f"nousergon_lib.__version__={nousergon_lib.__version__!r} "
        f"!= pyproject.toml::version={_pyproject_version()!r} — bump both "
        f"in lockstep or this test fails. Doc-string drift cause was a "
        f"pyproject-only bump on 2026-05-08 (v0.5.6) that left __init__.py "
        f"saying 0.5.5 for ~one day."
    )


# PyPI core-metadata spec caps the ``summary`` field (sourced from
# pyproject.toml::project.description) at 512 characters. Twine accepts
# longer values locally + at build time; PyPI rejects the upload with
# HTTP 400 only after the auto-tag has already cut the git tag.
# Regression: 2026-05-27 v0.38.0 — description grew to 550 chars when
# the locks submodule was added; auto-tag.yml succeeded (tagged v0.38.0)
# but publish.yml failed at the PyPI upload step. The git tag still lets
# ``git+https://github.com/...@v0.38.0`` consumer pins resolve, but
# PyPI is out of sync until a fresh patch release.
# Spec: https://packaging.python.org/specifications/core-metadata/#summary
_PYPI_SUMMARY_MAX = 512


def _pyproject_description() -> str:
    text = _PYPROJECT.read_text()
    match = re.search(r'^description\s*=\s*"((?:[^"\\]|\\.)*)"', text, re.MULTILINE)
    assert match is not None, "description not found in pyproject.toml"
    return match.group(1)


def test_pyproject_description_under_pypi_summary_limit():
    """``pyproject.toml::project.description`` MUST fit PyPI's 512-char
    summary limit. Caught at PR time prevents the v0.38.0 recurrence
    (tag cut, PyPI publish failed, git+https pins work but PyPI is
    out of sync)."""
    desc = _pyproject_description()
    assert len(desc) <= _PYPI_SUMMARY_MAX, (
        f"pyproject.toml::description is {len(desc)} characters; "
        f"PyPI's core-metadata 'summary' field caps at {_PYPI_SUMMARY_MAX}. "
        f"Trim before merging — auto-tag.yml will cut the git tag BEFORE "
        f"publish.yml hits PyPI, so a too-long description ships an "
        f"orphan tag that consumer git+https pins resolve but PyPI does "
        f"not have. See https://packaging.python.org/specifications/"
        f"core-metadata/#summary"
    )
