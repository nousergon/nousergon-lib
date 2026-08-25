"""Published route contract of the multi-tenant LLM egress proxy.

The proxy (:mod:`nousergon_lib.egress.proxy`) selects an upstream per request by
reading the ``X-Upstream-Host`` header against a route table it is handed at
startup. A request naming a host absent from that table is refused with
``unknown upstream host '<h>'`` — it cannot be served, ever. Anything that
decides *which model rows are servable* therefore has to know the table.

The table is configured in a launcher that lives in a private repo. This module
publishes the part of it that is a **contract** — which upstream hosts a
deployment serves, and how each is authenticated — so a consumer can read it
with **no credential**. Nothing here is a secret: no key-environment names, no
listen ports, no host of ours (see ``routes.schema.v1.json``).

Producer / consumer
-------------------
* Producer: ``claude-code-config/llm-routing/publish_egress_routes.py`` derives
  :data:`CONTRACT_PATH` from the launchers and fails that repo's CI when the two
  diverge. A stale published contract is worse than none, because it is green
  while wrong.
* Reference consumer: ``alpha-engine-config/scripts/egress_proxy_routes.py``,
  which backs the LLM model registry's servability invariants 18/19.

Why it is published at all: alpha-engine-config-I8337. The consumer used to
``actions/checkout`` the private producer repo with a PAT, which made every
Dependabot pull_request run there permanently red — Dependabot runs cannot see
repository Actions secrets. Brian ruled 2026-08-25 to remove the need for the
private read rather than widen the credential or narrow the guard.

Deployments are never unioned. ``box`` and ``laptop`` are different tables; a
host the laptop serves is not thereby servable by anything routed through the
dashboard box, which is where the LiteLLM router walks every fallback chain.
"""
from __future__ import annotations

import json
from typing import Any

try:  # Python 3.9 has importlib.resources but not .files
    from importlib.resources import files as _files
except ImportError:  # pragma: no cover - 3.8 and below are unsupported
    _files = None  # type: ignore[assignment]

__all__ = [
    "SCHEMA_VERSION",
    "CONTRACT_FILENAME",
    "SCHEMA_FILENAME",
    "UnsupportedContractVersion",
    "load_contract",
    "load_schema",
    "table",
    "upstream_hosts",
    "box_upstream_hosts",
    "laptop_upstream_hosts",
]

#: The only contract version this module can read.
SCHEMA_VERSION = 1

CONTRACT_FILENAME = "routes.v1.json"
SCHEMA_FILENAME = "routes.schema.v1.json"


class UnsupportedContractVersion(RuntimeError):
    """The artifact declares a ``schema_version`` this module cannot read.

    Raised rather than read partially: a consumer that reads a v2 artifact with
    v1 assumptions produces a host set that is wrong in a direction nothing
    checks.
    """


def _read(name: str) -> dict[str, Any]:
    if _files is None:  # pragma: no cover
        raise RuntimeError("importlib.resources.files is unavailable")
    # The package name as a literal, not ``__package__``: that is typed
    # ``str | None`` and pyright rejects it as an ``anchor`` argument.
    return json.loads(_files("nousergon_lib.egress").joinpath(name).read_text(encoding="utf-8"))


def load_contract() -> dict[str, Any]:
    """Return the whole published contract, version-checked."""
    doc = _read(CONTRACT_FILENAME)
    declared = doc.get("schema_version")
    if declared != SCHEMA_VERSION:
        raise UnsupportedContractVersion(
            "egress route contract declares schema_version "
            f"{declared!r}, this reader understands {SCHEMA_VERSION}"
        )
    return doc


def load_schema() -> dict[str, Any]:
    """Return the JSON Schema the published contract conforms to."""
    return _read(SCHEMA_FILENAME)


def table(name: str) -> list[dict[str, Any]]:
    """Return one deployment's route records.

    :raises KeyError: naming the tables that DO exist — an unknown deployment
        name must never read as "this deployment serves nothing".
    """
    tables = load_contract()["tables"]
    if name not in tables:
        raise KeyError(
            f"no published egress route table named {name!r}; "
            f"published tables: {sorted(tables)}"
        )
    return list(tables[name]["routes"])


def upstream_hosts(name: str) -> frozenset[str]:
    """Hosts one deployment's proxy will serve."""
    return frozenset(r["upstream_host"] for r in table(name))


def box_upstream_hosts() -> frozenset[str]:
    """Hosts the DASHBOARD BOX's proxy serves.

    The set that decides servability for the LiteLLM router: the router process
    runs on that box, so a model group's fallback chain is walked there no
    matter which execution context the caller runs in.
    """
    return upstream_hosts("box")


def laptop_upstream_hosts() -> frozenset[str]:
    """Hosts the LAPTOP shim serves. Never union this with the box's."""
    return upstream_hosts("laptop")
