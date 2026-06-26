"""The deprecated ``alpha_engine_lib`` import alias maps onto ``nousergon_lib``.

Guards the rename compat shim (alpha-engine-lib -> nousergon-lib): old imports
must keep working, resolve to the SAME module objects (no split module-level
state), and emit a DeprecationWarning on a fresh import.
"""
import subprocess
import sys


def test_old_submodule_is_same_object_as_new():
    import alpha_engine_lib.alerts as old_alerts
    import nousergon_lib.alerts as new_alerts

    assert old_alerts is new_alerts
    assert old_alerts.publish is new_alerts.publish


def test_old_toplevel_resolves_to_new_package():
    import alpha_engine_lib
    import nousergon_lib

    assert alpha_engine_lib is nousergon_lib
    assert alpha_engine_lib.__version__ == nousergon_lib.__version__


def test_from_import_under_old_name_works():
    from alpha_engine_lib.alerts import publish  # noqa: F401
    from alpha_engine_lib import dates as old_dates
    import nousergon_lib.dates as new_dates

    assert old_dates is new_dates


def test_alias_submodule_preserves_resource_loading():
    """Importing a resource-bearing sub-package via the OLD alias must not
    corrupt the real module's __spec__/loader.

    Regression: ``_AliasLoader`` returned the real module, but
    ``module_from_spec`` then force-reinitialised its ``__spec__``/``__loader__``
    from the alias spec — replacing the ``SourceFileLoader`` (+ package search
    locations) with the no-op ``_AliasLoader``. That broke
    ``importlib.resources.files(__package__)`` for ``nousergon_lib.contracts``'s
    ``*.schema.json`` (consumed by the executor's [contracts] slot validation).
    Run in a clean interpreter so the alias path is the FIRST importer.
    """
    import importlib.util

    if importlib.util.find_spec("jsonschema") is None:
        import pytest

        pytest.skip("needs the [contracts] extra (jsonschema)")

    code = (
        "import warnings; warnings.simplefilter('ignore'); "
        "import alpha_engine_lib.contracts as c; "
        "import sys; s = sys.modules['nousergon_lib.contracts'].__spec__; "
        "assert s.name == 'nousergon_lib.contracts', s.name; "
        "assert type(s.loader).__name__ == 'SourceFileLoader', type(s.loader).__name__; "
        "assert s.submodule_search_locations is not None; "
        "schema = c.load_schema('signals'); "
        "assert 'properties' in schema or 'type' in schema; "
        "print('RESOURCE_OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "RESOURCE_OK" in result.stdout


def test_python_dash_m_under_old_alias_runs():
    """``python -m alpha_engine_lib.<submodule>`` must execute the real module.

    Regression (Saturday-SF MorningEnrich, 2026-06-25): the compat shim handled
    the ``import`` path but ``_AliasLoader`` implemented none of the legacy
    code-access protocol. ``runpy._get_module_details`` calls
    ``loader.get_code(mod_name)`` directly, so ``python -m
    alpha_engine_lib.ssm_log_capture`` died with ``AttributeError:
    '_AliasLoader' object has no attribute 'get_code'`` — taking down every SSM
    step that drives a lib CLI under the deprecated name. Run in a clean
    interpreter so the alias path is the first importer.
    """
    code = (
        "import warnings; warnings.simplefilter('ignore'); "
        "import runpy, sys; sys.argv = ['m', '--help']; "
        "runpy.run_module('alpha_engine_lib.ssm_log_capture', "
        "run_name='__main__', alter_sys=True)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    # argparse --help exits 0 via SystemExit; the point is it REACHED the real
    # module's code (printed its usage) instead of dying in runpy on get_code.
    assert result.returncode == 0, result.stderr
    assert "get_code" not in result.stderr, result.stderr
    assert "ssm_log_capture" in result.stdout, result.stdout


def test_old_alias_get_code_matches_new():
    """The alias loader's get_code returns the SAME code as the real loader.

    Proves the proxy resolves to the real module's source (not a duplicate /
    stale object) and that get_source is wired up for traceback fidelity.
    """
    import importlib.util

    old = importlib.util.find_spec("alpha_engine_lib.ssm_log_capture")
    new = importlib.util.find_spec("nousergon_lib.ssm_log_capture")

    old_code = old.loader.get_code("alpha_engine_lib.ssm_log_capture")
    new_code = new.loader.get_code("nousergon_lib.ssm_log_capture")
    assert old_code.co_filename == new_code.co_filename

    src = old.loader.get_source("alpha_engine_lib.ssm_log_capture")
    assert "def main(" in src


def test_fresh_import_emits_deprecation_warning():
    # Order-independent: the in-process finder swallows re-imports, so assert
    # the warning in a clean interpreter with DeprecationWarning escalated.
    code = (
        "import warnings; warnings.simplefilter('error', DeprecationWarning); "
        "import alpha_engine_lib"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode != 0, "fresh `import alpha_engine_lib` should warn"
    assert "renamed to 'nousergon_lib'" in result.stderr
