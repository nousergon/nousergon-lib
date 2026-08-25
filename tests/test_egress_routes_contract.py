"""Contract tests for the published LLM egress-proxy route table.

At birth, per the M0 contract discipline: a cross-repo artifact gets a versioned
schema and a producer/consumer contract test in the same change that creates it.

The producer-side half lives in the private producer repo
(``claude-code-config/llm-routing/test_published_egress_routes.py``, which
re-derives this file from the proxy launchers and fails on divergence). The
consumer-side half lives in ``alpha-engine-config/scripts/test_egress_proxy_routes.py``.
This module owns the artifact's own invariants: it validates, it says what
version it is, and it carries nothing it must not carry.

alpha-engine-config-I8337.
"""
from __future__ import annotations

import json

import pytest

from nousergon_lib.egress import routes

jsonschema = pytest.importorskip("jsonschema")


def test_the_published_contract_validates_against_its_own_schema():
    jsonschema.validate(instance=routes.load_contract(), schema=routes.load_schema())


def test_schema_is_itself_a_valid_json_schema():
    schema = routes.load_schema()
    jsonschema.Draft202012Validator.check_schema(schema)


def test_version_is_declared_and_matches_the_reader():
    assert routes.load_contract()["schema_version"] == routes.SCHEMA_VERSION


def test_an_unknown_version_raises_rather_than_being_read_partially(monkeypatch):
    doc = routes.load_contract()
    doc["schema_version"] = 99
    monkeypatch.setattr(routes, "_read", lambda _n: doc)
    with pytest.raises(routes.UnsupportedContractVersion):
        routes.load_contract()


def test_both_deployments_are_published_and_are_not_the_same_table():
    contract = routes.load_contract()
    assert set(contract["tables"]) == {"box", "laptop"}
    # The laptop table is deliberately WIDER. Unioning them restores the exact
    # blindness invariants 18/19 exist to remove (alpha-engine-config-I7897):
    # a row naming a laptop-only host would read as servable by the router.
    assert routes.laptop_upstream_hosts() > routes.box_upstream_hosts()


def test_an_unknown_table_raises_rather_than_reading_as_empty():
    with pytest.raises(KeyError, match="published tables"):
        routes.table("nope")


def test_hosts_are_returned_as_a_set_of_names():
    box = routes.box_upstream_hosts()
    assert "api.deepseek.com" in box
    assert all(isinstance(h, str) and h for h in box)


def test_the_contract_carries_no_credential_material():
    """Rule 1 and rule 2 of repository-tiering-policy, asserted rather than assumed.

    This artifact is PUBLIC and is regenerated from a private launcher that does
    hold key-environment names, an SSM path map and a listen port. The generator
    projects those away; this is what fails if a future generator change stops
    projecting them away.
    """
    raw = json.dumps(routes.load_contract())
    for forbidden in ("_API_KEY", "api_key", "SSM", "/nous-ergon/", "/symposion/", "/alpha-engine/"):
        assert forbidden not in raw, f"published route contract leaks {forbidden!r}"
    for record in routes.table("box") + routes.table("laptop"):
        assert set(record) == {"upstream_host", "auth_mode", "path_prefix"}
