"""Deprecated import alias for :mod:`nousergon_lib`.

The package was renamed from ``alpha-engine-lib`` / ``alpha_engine_lib`` to
``nousergon-lib`` / ``nousergon_lib`` (brand coherence for the public MIT
foundation lib — its repo is ``nousergon/nousergon-lib``). Importing under
the **old** name still works: a meta-path finder transparently maps
``alpha_engine_lib`` and every ``alpha_engine_lib.<submodule>`` onto the
corresponding ``nousergon_lib`` module — the *same* module object, so there
is no duplicated module-level state — and a single :class:`DeprecationWarning`
is emitted on first import.

This shim is intentionally gradual: consumers (including downstream products
and external adopters) can keep their existing ``import alpha_engine_lib``
statements and migrate to ``nousergon_lib`` at their own pace. The alias will
be removed in a future major release.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

_OLD = "alpha_engine_lib"
_NEW = "nousergon_lib"


class _AliasLoader(importlib.abc.Loader):
    """Loads ``alpha_engine_lib[.x]`` by returning the ``nousergon_lib[.x]`` module."""

    def create_module(self, spec):
        new_name = _NEW + spec.name[len(_OLD):]
        module = importlib.import_module(new_name)
        # Alias under the old fullname so `is` identity holds and the import
        # system caches the same object (no second execution / no split state).
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module):  # already executed under its real name
        pass


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Redirect any ``alpha_engine_lib`` / ``alpha_engine_lib.*`` import to ``nousergon_lib``."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == _OLD or fullname.startswith(_OLD + "."):
            return importlib.util.spec_from_loader(fullname, _AliasLoader())
        return None


# Install the finder exactly once.
if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

warnings.warn(
    "'alpha_engine_lib' has been renamed to 'nousergon_lib'. The old import "
    "name still works but is deprecated and will be removed in a future "
    "release; update your imports to 'nousergon_lib'.",
    DeprecationWarning,
    stacklevel=2,
)

# Replace this freshly-imported shim module with the real top-level package so
# attribute access (e.g. ``alpha_engine_lib.__version__``) resolves correctly.
sys.modules[_OLD] = importlib.import_module(_NEW)
