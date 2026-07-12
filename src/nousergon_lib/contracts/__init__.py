"""Cross-repo contracts — versioned JSON Schemas + conformance validation (M0).

The fleet exchanges artifacts across repo boundaries; those artifacts are
PRODUCT CONTRACTS, not internal conveniences (M0 contract discipline, ratified
2026-06-11 — config#638, ne_product_architecture_plan v2.0). This module is the
single source of truth for those schemas:

- ``signals``        — Slot R (research orchestration) → ``signals/{date}/signals.json``
- ``predictions``    — Slot M (model/prediction)       → ``predictor/predictions/{date}.json``
- ``research_intel`` — Slot R neutral product intel     → ``research_intel/{date}.json``
- ``outcome_record`` — long-format eval outcome (one row per signal/date/horizon;
  the config#1483 replacement for wide horizon-suffixed ``score_performance``
  columns). NOT a product SLOT (R/M/S) — it's a cross-repo eval-storage contract,
  so it lives in :data:`CONTRACT_SCHEMAS` but not :data:`SLOT_SCHEMAS`.
- ``attractiveness_eval`` — universe-board attractiveness vs realized forward
  alpha (crucible-backtester → crucible-evaluator) →
  ``backtest/{date}/attractiveness_eval.json``. Eval-storage contract, not a
  slot. v2 renames ``counterfactual.*.mean_alpha_21d`` → ``mean_alpha``
  (horizon-is-a-parameter, config#1861).
- ``apply_audit`` — one record per auto-apply loop, emitted every evaluate run
  so a blocked/starved apply loop is impossible to miss (crucible-backtester →
  crucible-evaluator) → ``config/apply_audit/{date}.json``. Eval-storage
  contract, not a slot.
- ``report_card`` — the evaluator's unified system report card (RC v2: v1
  letter grades for research/predictor/executor + the 9-tile MetricRecord
  substrate + provenance), emitted by ``grading/aggregate.py::build_report_card``
  (crucible-evaluator) → ``evaluator/{date}/report_card.json`` and consumed by
  the dashboard. Eval-storage contract, not a slot (config#2343).
- ``experiment`` — the Crucible declarative experiment MANIFEST (input
  envelope: per-slot stock|artifact|command|entry_point binding, window,
  universe, evaluation config). Phase A of crucible_ux_output_plan_260708.md
  (config#1966). Envelope contract, not a slot.
- ``experiment_record`` — the per-run INDEX a Crucible run emits (manifest
  hash + slot fingerprints + typed artifact link table with honest absences);
  the results renderer consumes this record, never a directory listing.
  Envelope contract, not a slot.

Producers validate a representative emitted artifact in CI; consumers validate
the fixtures their readers are tested against; external slot implementations
("bring your own R/M") validate their output with :func:`conformance_errors` /
:func:`validate` — the same check the future ``ne validate`` CLI verb fronts.
CLI today: ``python -m nousergon_lib.contracts validate <name> <path.json>``.

Contract evolution is ADDITIVE-ONLY (S3 Contract Safety): new optional fields
may appear at any time (``additionalProperties`` stays open); removing/renaming
a required field requires a new schema version + a dual-write window.

Requires the ``contracts`` extra (``nousergon-lib[contracts]`` → jsonschema).
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any, Iterator

__all__ = [
    "CONTRACT_SCHEMAS",
    "SLOT_SCHEMAS",
    "SCHEMA_VERSIONS",
    "ContractViolation",
    "load_schema",
    "iter_errors",
    "conformance_errors",
    "validate",
]

# ALL cross-repo contracts: name -> schema resource filename. The superset that
# backs load_schema / validate / the CLI.
CONTRACT_SCHEMAS: dict[str, str] = {
    "signals": "signals.schema.json",
    "predictions": "predictions.schema.json",
    "outcome_record": "outcome_record.schema.json",
    "research_intel": "research_intel.schema.json",
    "attractiveness_eval": "attractiveness_eval.schema.json",
    "apply_audit": "apply_audit.schema.json",
    # Evaluator report card (RC v2): grading/aggregate.py::build_report_card →
    # ``evaluator/{date}/report_card.json``; consumed by the dashboard System
    # Report Card + 9-tile substrate. Eval-storage contract, not a slot
    # (config#2343, sibling of config#692).
    "report_card": "report_card.schema.json",
    # Crucible harness envelope contracts (Phase A of the ratified
    # crucible_ux_output_plan_260708.md — config#1966). Not slot boundaries:
    # ``experiment`` DECLARES which slot implementations run (the manifest);
    # ``experiment_record`` indexes what a run emitted (the results renderer
    # consumes the record, never a directory listing).
    "experiment": "experiment.schema.json",
    "experiment_record": "experiment_record.schema.json",
}

# The subset of CONTRACT_SCHEMAS that are product SLOT boundaries (R/M/S), with
# their slot id. ``outcome_record`` is a cross-repo eval-storage contract, not a
# slot boundary, so it is deliberately absent here.
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
    "outcome_record": 1,
    "research_intel": 1,
    "attractiveness_eval": 2,
    "apply_audit": 1,
    "report_card": 1,
    "experiment": 1,
    "experiment_record": 1,
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
    """Load the JSON Schema for a contract (``signals`` | ``predictions`` | ``outcome_record``)."""
    if name not in CONTRACT_SCHEMAS:
        raise KeyError(f"unknown contract {name!r}; known: {sorted(CONTRACT_SCHEMAS)}")
    fname = CONTRACT_SCHEMAS[name]
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
