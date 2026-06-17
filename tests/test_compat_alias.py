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
