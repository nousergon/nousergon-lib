"""Slot boundary contracts — versioned JSON Schemas + conformance validation (M0).

The harness's experiment slots exchange artifacts across repo boundaries; those
artifacts are PRODUCT CONTRACTS, not internal conveniences (M0 contract
discipline, ratified 2026-06-11 — config#638, ne_product_architecture_plan
v2.0). This module is the single source of truth for the slot schemas:

- ``signals``        — Slot R (research orchestration) → ``signals/{date}/signals.json``
- ``predictions``    — Slot M (model/prediction)       → ``predictor/predictions/{date}.json``
- ``research_intel`` — Slot R neutral product intel     → ``research_intel/{date}.json``

Producers validate a representative emitted artifact in CI; consumers validate
the fixtures their readers are tested against; external slot implementations
("bring your own R/M") validate their output with :func:`conformance_errors` /
:func:`validate` — the same check the future ``ne validate`` CLI verb fronts.
CLI today: ``python -m nousergon_lib.contracts validate <slot> <path.json>``.

Contract evolution is ADDITIVE-ONLY (S3 Contract Safety): new optional fields
may appear at any time; removing/renaming a required field requires a new
schema version + a dual-write window.

Requires the ``contracts`` extra (``nousergon-lib[contracts]`` → jsonschema).
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any, Iterator

__all__ = [
    "SLOT_SCHEMAS",
    "SCHEMA_VERSIONS",
    "ContractViolation",
    "load_schema",
    "iter_errors",
    "conformance_errors",
    "validate",
]

# slot artifact name -> (schema resource filename, slot id)
SLOT_SCHEMAS: dict[str, tuple[str, str]] = {
    "signals": ("signals.schema.json", "R"),
    "predictions": ("predictions.schema.json", "M"),
    "research_intel": ("research_intel.schema.json", "R"),
}

# Current contract version per artifact. Bump ONLY on a breaking change,
# alongside a new .schema.json and a dual-write window.
SCHEMA_VERSIONS: dict[str, int] = {
    "signals": 1,
    "predictions": 1,
    "research_intel": 1,
}


class ContractViolation(Exception):
    """A payload failed its slot contract. ``errors`` carries the full list."""

    def __init__(self, name: str, errors: list[str]):
        self.name = name
        self.errors = errors
        preview = "; ".join(errors[:5]) + (" …" if len(errors) > 5 else "")
        super().__init__(
            f"{name} payload violates contract v{SCHEMA_VERSIONS[name]} "
            f"({len(errors)} error(s)): {preview}"
        )


@lru_cache(maxsize=None)
def load_schema(name: str) -> dict[str, Any]:
    """Load the JSON Schema for a slot artifact (``signals`` | ``predictions``)."""
    if name not in SLOT_SCHEMAS:
        raise KeyError(f"unknown contract {name!r}; known: {sorted(SLOT_SCHEMAS)}")
    fname, _slot = SLOT_SCHEMAS[name]
    with resources.files(__package__).joinpath(fname).open("r", encoding="utf-8") as f:
        return json.load(f)


def _validator(name: str):
    # Deferred import so the lib core stays importable without the extra.
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "nousergon_lib.contracts needs the 'contracts' extra: "
            "pip install 'nousergon-lib[contracts]'"
        ) from exc
    return Draft202012Validator(load_schema(name))


def iter_errors(name: str, payload: Any) -> Iterator[Any]:
    """Yield raw jsonschema ValidationError objects for ``payload``."""
    return _validator(name).iter_errors(payload)


def conformance_errors(name: str, payload: Any) -> list[str]:
    """Human-readable contract errors for ``payload`` (empty list = conforms).

    The conformance-kit primitive: producer CI, consumer fixture tests, and
    external slot implementations all assert ``conformance_errors(...) == []``.
    """
    out = []
    for e in iter_errors(name, payload):
        path = "/".join(str(p) for p in e.absolute_path) or "<root>"
        out.append(f"{path}: {e.message}")
    return sorted(out)


def validate(name: str, payload: Any) -> None:
    """Validate ``payload`` against its slot contract; raise :class:`ContractViolation`.

    Fail-loud by design (no boolean-return variant): a contract break should
    surface at the earliest callsite, not be swallowed into a falsy branch.
    """
    errors = conformance_errors(name, payload)
    if errors:
        raise ContractViolation(name, errors)
