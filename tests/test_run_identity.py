"""`nousergon_lib.run_identity` — the contract-neutral run-id/code-sha module.

`alpha-engine-config-I10831` deliverable 2. `resolve_code_sha` and
`new_run_id` were lifted out of `run_manifest` (bound to
`data_run_manifest.v1`) so a second writer (crucible's `run_manifest.v2`,
crucible-PR305) can import them without coupling to the data contract's
module. The tests here assert the two things that make the lift safe:

1. The primitives are importable from `run_identity` directly — the point of
   the lift.
2. `run_manifest` still exports the SAME objects (not a copy) — every
   existing importer of `nousergon_lib.run_manifest.{new_run_id,
   resolve_code_sha, CodeShaError, CODE_SHA_ENV}` keeps working unchanged.
"""

from __future__ import annotations

import datetime as dt

from nousergon_lib import run_identity, run_manifest


def test_run_identity_exports_new_run_id_and_resolve_code_sha_standalone():
    """The contract-neutral module is importable with no `run_manifest`
    dependency in the picture — a second writer's own contract module can
    import this one without dragging `data_run_manifest.v1` along."""
    assert run_identity.new_run_id is not None
    assert run_identity.resolve_code_sha is not None
    assert callable(run_identity.new_run_id)
    assert callable(run_identity.resolve_code_sha)

    base = dt.datetime(2026, 9, 14, tzinfo=dt.timezone.utc)
    run_id = run_identity.new_run_id(base)
    assert len(run_id) == 26


def test_run_manifest_reexports_the_identical_objects_not_copies():
    """`run_manifest.new_run_id is run_identity.new_run_id` — a re-export, not
    a second definition that could drift from the one in `run_identity`."""
    assert run_manifest.new_run_id is run_identity.new_run_id
    assert run_manifest.resolve_code_sha is run_identity.resolve_code_sha
    assert run_manifest.CodeShaError is run_identity.CodeShaError
    assert run_manifest.CODE_SHA_ENV == run_identity.CODE_SHA_ENV


def test_both_names_are_declared_in_run_manifests_public_api():
    """Existing importers reach these through `run_manifest.__all__` — the
    contract this lift promises not to break."""
    assert "new_run_id" in run_manifest.__all__
    assert "resolve_code_sha" in run_manifest.__all__
    assert "CodeShaError" in run_manifest.__all__
    assert "CODE_SHA_ENV" in run_manifest.__all__


def test_env_declared_code_sha_wins_from_either_import_path(monkeypatch):
    """The behavior is identical from either module — a caller migrating from
    `run_manifest.resolve_code_sha` to `run_identity.resolve_code_sha` changes
    nothing."""
    monkeypatch.setenv("NE_DATA_CODE_SHA", "c" * 40)
    assert run_identity.resolve_code_sha() == "c" * 40
    assert run_manifest.resolve_code_sha() == "c" * 40
