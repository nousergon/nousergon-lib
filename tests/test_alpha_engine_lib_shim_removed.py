"""The deprecated ``alpha_engine_lib`` import alias has been REMOVED.

config#1172 (fleet-wide alpha_engine_lib -> nousergon_lib conversion): the
package was renamed to ``nousergon_lib`` at v0.60.0, with a meta-path-finder
compat shim (``src/alpha_engine_lib/__init__.py``) keeping the old import
name working during a gradual, fleet-wide migration (see the now-deleted
``tests/test_compat_alias.py``, which pinned the shim's behavior).

All 7 dependent repos (nousergon-data, crucible-predictor, crucible-research,
crucible-backtester, crucible-evaluator, crucible-executor, crucible-dashboard)
were verified clean of live functional ``alpha_engine_lib`` references before
this PR retired the shim in this v0.113.0 breaking change. This test flips
the old "shim still importable" guard into the opposite invariant: the shim
must NOT come back.
"""

from __future__ import annotations

import subprocess
import sys


def test_alpha_engine_lib_is_not_importable():
    """A fresh interpreter must fail to import the retired alias."""
    result = subprocess.run(
        [sys.executable, "-c", "import alpha_engine_lib"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, (
        "alpha_engine_lib imported successfully — the deprecated alias shim "
        "was supposed to be removed at v0.113.0 (config#1172). If it's back, "
        "that's a regression of the shim retirement, not a legitimate re-add."
    )
    assert "ModuleNotFoundError" in result.stderr or "No module named" in result.stderr


def test_shim_source_directory_is_gone():
    """The shim's package directory must not exist in the source tree."""
    from pathlib import Path

    shim_dir = Path(__file__).resolve().parent.parent / "src" / "alpha_engine_lib"
    assert not shim_dir.exists(), (
        f"{shim_dir} exists — the alpha_engine_lib alias shim was supposed to "
        "be deleted at v0.113.0 (config#1172)."
    )
