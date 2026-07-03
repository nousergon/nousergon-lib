"""Every re-export shim wrapping a krepis CLI must delegate under `python -m`.

Bug class (config#1646, 2026-07-03): the ``nousergon_lib.<name>`` shims rebind
``sys.modules[__name__]`` to their krepis target. Under ``python -m`` (runpy)
the shim itself runs as ``__main__`` — the krepis target is imported under its
own name, so ITS ``if __name__ == "__main__"`` guard never fires, and without
a guard in the shim the process exits 0 having executed nothing. The weekly
SF wrapped all 11 EC2 workloads in ``python -m nousergon_lib.ssm_log_capture
run …`` and ran ZERO of them while reporting SUCCESS.

These tests enumerate the shims dynamically (any future shim is covered) and
prove, via subprocess, that ``python -m nousergon_lib.<name>`` reaches the
krepis CLI rather than silently no-opping. A guard-less shim fails these on
the "produced no output" assertion — exactly the silent-failure signature.
"""

from __future__ import annotations

import re
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest

_SHIM_DIR = Path(__file__).resolve().parents[1] / "src" / "nousergon_lib"
_TARGET_RE = re.compile(r"^import (krepis\.[\w]+) as _mod$", re.MULTILINE)


def _cli_shims() -> list[tuple[str, str]]:
    """(shim module, krepis target) for every shim whose target is a CLI."""
    out = []
    for path in sorted(_SHIM_DIR.glob("*.py")):
        m = _TARGET_RE.search(path.read_text())
        if not m:
            continue
        target = m.group(1)
        if hasattr(import_module(target), "main"):
            out.append((f"nousergon_lib.{path.stem}", target))
    return out


def test_cli_shim_enumeration_is_not_empty():
    """An empty enumeration would make the delegation tests vacuous."""
    shims = dict(_cli_shims())
    assert "nousergon_lib.ssm_log_capture" in shims


@pytest.mark.parametrize(
    "shim,target", _cli_shims(), ids=[s for s, _ in _cli_shims()]
)
def test_shim_delegates_main_under_dash_m(shim, target):
    """`python -m <shim> --help` must reach the krepis argparse CLI.

    The guard-less form exits 0 with EMPTY stdout (runpy falls off the end
    of the shim) — asserting on usage output distinguishes real delegation
    from the silent no-op.
    """
    proc = subprocess.run(
        [sys.executable, "-m", shim, "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"`python -m {shim} --help` rc={proc.returncode}, "
        f"stderr={proc.stderr[-300:]!r}"
    )
    assert "usage" in proc.stdout.lower(), (
        f"`python -m {shim} --help` produced no usage text "
        f"(stdout={proc.stdout!r}) — the shim is not delegating to "
        f"{target}.main(); add the `if __name__ == '__main__'` guard "
        "(config#1646 silent-no-op bug class)."
    )


def test_ssm_log_capture_shim_executes_inner_command(tmp_path):
    """End-to-end: the SF's exact invocation shape must run the inner cmd."""
    log = tmp_path / "wrapper.log"
    sentinel = "SENTINEL_SHIM_DELEGATION_1646"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nousergon_lib.ssm_log_capture",
            "run",
            "--slug",
            "shim-delegation-test",
            "--log",
            str(log),
            "--",
            sys.executable,
            "-c",
            f"print('{sentinel}')",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert sentinel in proc.stdout
    assert log.exists() and sentinel in log.read_text()
    assert proc.returncode == 0
