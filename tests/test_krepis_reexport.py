"""Contract test: the v0.66.0 krepis relocation preserves nousergon_lib's
back-compat import surface.

The generic, edge-free primitives moved to the MIT ``krepis`` package. Each
``nousergon_lib.<module>`` is now a re-export shim that rebinds itself to
``krepis.<module>`` via ``sys.modules``, so the *entire* public surface
(including private helpers like the flow-doctor secret-seeding chokepoint)
resolves unchanged through the legacy path. This test pins that invariant so
a downstream ``from nousergon_lib.X import ...`` consumer is never silently
broken by the relocation.
"""

import importlib

import pytest

# The 15 modules relocated to krepis in v0.66.0.
RELOCATED = [
    "secrets",
    "trading_calendar",
    "http_retry",
    "anthropic_payload",
    "ec2_spot",
    "ssm_dispatcher",
    "ssm_log_capture",
    "metrics",
    "locks",
    "telegram",
    "email_sender",
    "logging",
    "alerts",
    "dates",
    "cost",
    "yfinance_quiet",
]


@pytest.mark.parametrize("mod", RELOCATED)
def test_legacy_path_is_the_krepis_module(mod):
    """``nousergon_lib.<mod>`` resolves to the exact ``krepis.<mod>`` object."""
    legacy = importlib.import_module(f"nousergon_lib.{mod}")
    relocated = importlib.import_module(f"krepis.{mod}")
    assert legacy is relocated, (
        f"nousergon_lib.{mod} must re-export krepis.{mod} (same module object)"
    )


def test_flow_doctor_seed_helper_resolves_via_legacy_path():
    """The private flow-doctor secret-seeding chokepoint must stay importable
    from the legacy path — every deployed repo seeds secrets through it."""
    from nousergon_lib.logging import _seed_flow_doctor_secrets  # noqa: F401


def test_model_metadata_is_the_shared_krepis_object():
    """ModelMetadata was lifted into krepis; decision_capture (which stays in
    nousergon_lib) re-imports it, so both paths yield the same class."""
    from krepis.model_metadata import ModelMetadata as via_krepis
    from nousergon_lib.decision_capture import ModelMetadata as via_decision_capture

    assert via_decision_capture is via_krepis
