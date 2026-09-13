"""The test-shape guard, and proof it fails when the fix is reverted.

`overseer-policy.md` #13: a guard is not a guard until it has been observed
failing. The reverted-tree cases below reconstruct the exact `krepis` shape
(alpha-engine-config-I10005: four `def test_*` written after a `return`
inside a helper, `tests/test_router.py`, fixed by `krepis` PR203) and assert
the checker rejects it.

Most cases call ``main()`` in-process so the checker's own lines are measured
by coverage; one case shells out, because the CLI exit-code contract is what
CI actually depends on and an in-process call would not exercise it.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LINTER = REPO_ROOT / "scripts" / "lint_test_shape.py"

# Some repos run this suite INSIDE a built image that copies the application
# package and not `scripts/` — see tests/test_lint_extras_guard.py for the
# full rationale, mirrored here.
if not LINTER.exists():  # pragma: no cover - image-context guard
    pytest.skip(
        f"{LINTER} absent — running inside a packaged image, not a checkout. "
        "The linter is exercised against the real tree by its own CI job.",
        allow_module_level=True,
    )

_spec = importlib.util.spec_from_file_location("lint_test_shape", LINTER)
assert _spec and _spec.loader
lint_test_shape = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lint_test_shape)


def _rc(target: Path) -> int:
    """Run the checker in-process against `target`."""
    return lint_test_shape.main(["lint_test_shape.py", str(target)])


def _write(tmp_path: Path, name: str, body: str) -> None:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_this_repo_is_clean():
    """The real tree passes — nousergon-lib's own tests collect cleanly."""
    assert _rc(REPO_ROOT) == 0


def test_cli_contract_holds_out_of_process():
    """CI invokes this as a subprocess and gates on its exit code."""
    result = subprocess.run([sys.executable, str(LINTER), str(REPO_ROOT)], capture_output=True, text=True)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_scanning_nothing_is_an_error_not_a_pass(tmp_path):
    """A checker that opened no files must not report clean."""
    assert _rc(tmp_path) == 2, "a check with nothing to check is not a passing check"


# ── Finding A: nested test-shaped def/class ──────────────────────────────────


def test_rejects_the_krepis_shape_test_after_return(tmp_path, capsys):
    """The exact I10005 defect: `def test_*` written after a `return`,
    inside a helper function, in `tests/test_router.py`."""
    _write(
        tmp_path,
        "tests/test_router.py",
        "def _capture_ssm_param(self):\n"
        "    seen = []\n"
        "    return seen\n"
        "\n"
        "    def test_litellm_proxy_route_is_never_degraded_at_resolve_time(self):\n"
        "        assert True\n"
        "\n"
        "    def test_per_provider_fallback_is_degraded(self):\n"
        "        assert True\n",
    )
    assert _rc(tmp_path) == 1, "the guard PASSED on the shape that caused I10005"
    err = capsys.readouterr().err
    assert "pytest will not collect this" in err
    assert "test_litellm_proxy_route_is_never_degraded_at_resolve_time" in err


def test_rejects_a_test_class_nested_inside_a_function(tmp_path):
    _write(
        tmp_path,
        "tests/test_nested_class.py",
        "def build_fixture():\n"
        "    class TestSomething:\n"
        "        def test_a(self):\n"
        "            assert True\n"
        "    return TestSomething\n",
    )
    assert _rc(tmp_path) == 1


def test_rejects_a_test_def_nested_inside_an_if(tmp_path):
    _write(
        tmp_path,
        "tests/test_nested_if.py",
        "import sys\n\nif sys.version_info >= (3, 0):\n    def test_only_on_py3():\n        assert True\n",
    )
    assert _rc(tmp_path) == 1


def test_rejects_a_test_def_nested_inside_a_with(tmp_path):
    _write(
        tmp_path,
        "tests/test_nested_with.py",
        "import contextlib\n\nwith contextlib.suppress(Exception):\n    def test_never_runs():\n        assert True\n",
    )
    assert _rc(tmp_path) == 1


# ── Legal shapes: module scope and Test* class methods pass clean ───────────


def test_accepts_module_scope_test_function(tmp_path):
    _write(
        tmp_path,
        "tests/test_clean_module.py",
        "def test_module_scope():\n    assert True\n",
    )
    assert _rc(tmp_path) == 0


def test_accepts_test_class_method(tmp_path):
    _write(
        tmp_path,
        "tests/test_clean_class.py",
        "class TestGroup:\n"
        "    def test_one(self):\n"
        "        assert True\n"
        "\n"
        "    def test_two(self):\n"
        "        assert True\n",
    )
    assert _rc(tmp_path) == 0


def test_accepts_unittest_testcase_class_not_named_test_star(tmp_path):
    """pytest collects ANY unittest.TestCase subclass regardless of its
    name -- the `Test*` prefix is a pytest-native heuristic, not a unittest
    rule. Reproduces nousergon-data
    tests/test_sf_pipeline_status_console_link_wiring.py's
    `PipelineStatusConsoleLinkWiringTest(unittest.TestCase)`, which a first
    revision of this guard false-flagged."""
    _write(
        tmp_path,
        "tests/test_unittest_style.py",
        "import unittest\n"
        "\n"
        "\n"
        "class PipelineStatusConsoleLinkWiringTest(unittest.TestCase):\n"
        "    def test_each_template_has_terminal_notify_states(self):\n"
        "        self.assertTrue(True)\n",
    )
    assert _rc(tmp_path) == 0


def test_accepts_bare_testcase_import_not_named_test_star(tmp_path):
    """`from unittest import TestCase` spelling, same rule."""
    _write(
        tmp_path,
        "tests/test_bare_testcase.py",
        "from unittest import TestCase\n"
        "\n"
        "\n"
        "class ConsoleLinkWiring(TestCase):\n"
        "    def test_it(self):\n"
        "        self.assertTrue(True)\n",
    )
    assert _rc(tmp_path) == 0


def test_accepts_nested_test_class_inside_test_class(tmp_path):
    """`Test*` classes may legally nest inside another `Test*` class."""
    _write(
        tmp_path,
        "tests/test_nested_ok.py",
        "class TestOuter:\n    class TestInner:\n        def test_it(self):\n            assert True\n",
    )
    assert _rc(tmp_path) == 0


def test_accepts_a_helper_function_with_no_dead_code(tmp_path):
    """A plain non-test helper with an early return is not itself flagged."""
    _write(
        tmp_path,
        "tests/test_helper_ok.py",
        "def _capture(self):\n"
        "    seen = []\n"
        "    return seen\n"
        "\n"
        "\n"
        "def test_uses_helper():\n"
        "    assert _capture(None) == []\n",
    )
    assert _rc(tmp_path) == 0


# ── Finding B: dead code after an unconditional return ───────────────────────


def test_rejects_dead_code_after_return_even_when_not_test_shaped(tmp_path, capsys):
    """The broader defect: ANY statement after a top-level `return` in a
    test-module function is unreachable, not only a nested `def test_*`."""
    _write(
        tmp_path,
        "tests/test_dead_code.py",
        "def test_something():\n    x = 1\n    return x\n    x = 2\n    assert x == 2\n",
    )
    assert _rc(tmp_path) == 1
    err = capsys.readouterr().err
    assert "unreachable statement" in err


def test_return_as_the_last_statement_is_not_flagged(tmp_path):
    """A `return` as the final statement in the body is normal control flow."""
    _write(
        tmp_path,
        "tests/test_return_last.py",
        "def _helper():\n    return 1\n\n\ndef test_uses_it():\n    assert _helper() == 1\n",
    )
    assert _rc(tmp_path) == 0


def test_return_inside_an_if_branch_does_not_flag_code_after_the_function(tmp_path):
    """A `return` nested inside an `if` is conditional, not top-level — code
    after the `if` block (at the function's own top level) is reachable."""
    _write(
        tmp_path,
        "tests/test_conditional_return.py",
        "def _helper(flag):\n"
        "    if flag:\n"
        "        return 1\n"
        "    return 2\n"
        "\n"
        "\n"
        "def test_uses_it():\n"
        "    assert _helper(True) == 1\n",
    )
    assert _rc(tmp_path) == 0
