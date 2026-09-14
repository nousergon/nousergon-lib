"""One gate's reading: every clause, and whether the phase may exit."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from nousergon_lib.gates.clause import Clause, clause_member_status

__all__ = ["GATE_SCHEMA_VERSION", "GateResult", "gate_state_for"]

GATE_SCHEMA_VERSION = "gate.v1"


@dataclass
class GateResult:
    """Every clause of one gate, and whether the phase may exit."""

    gate: str
    trading_day: dt.date
    window: list[dt.date] = field(default_factory=list)
    clauses: list[Clause] = field(default_factory=list)
    #: How much of the phase issue this clause list grades. Not a clause: it is
    #: true by construction, so counting it in ``met_ratio`` would inflate the
    #: one figure the gate exists to keep honest.
    coverage: str | None = None
    #: What the reading read, as a caller wants to cite it: the store URI and
    #: the commit the clause list came from. Carried on the artifact so a
    #: pasted reading names its own provenance.
    store: str | None = None
    code_sha: str | None = None

    @property
    def met(self) -> bool:
        return bool(self.clauses) and all(c.met for c in self.clauses)

    @property
    def met_ratio(self) -> float | None:
        """Met clauses over total, or ``None`` when nothing was measured.

        Zero clauses is ``None``, never ``0.0`` and never ``1.0`` — an empty
        gate measured nothing, and ``0.0`` there is a false measurement: it
        reads as "measured, and none passed" when the truth is "never
        measured" (principle 7).

        ``None`` too when ANY clause is unmeasurable: a store access failure on
        one clause, with the rest genuinely measured, renders as e.g. ``0.5``,
        a specific number that reads as "half the requirement was checked and
        failed" when the truth is "one of two checks could not even run".
        Excluding the unmeasurable clause from the denominator instead would
        still publish a number computed from a PARTIAL read — the same shape of
        overclaim in miniature.
        """
        if not self.clauses or any(c.unmeasurable for c in self.clauses):
            return None
        return sum(1 for c in self.clauses if c.met) / len(self.clauses)

    @property
    def unmeasurable(self) -> int:
        return sum(1 for c in self.clauses if c.unmeasurable)

    @property
    def members(self) -> list[dict[str, Any]]:
        """This gate's clauses as member rows — a VIEW over ``self.clauses``.

        ``met`` and ``met_ratio`` are already pure functions of the same list,
        so the roll-up and the members cannot drift apart.
        """
        return [{"id": c.name, "value": c.met, "status": clause_member_status(c)} for c in self.clauses]

    @property
    def earliest_satisfiable(self) -> dt.date | None:
        """The MAX over every clause's own derived date, never re-derived here.

        The max, never the min: the gate cannot exit before its SLOWEST dated
        clause clears.
        """
        dated = [c.earliest_satisfiable for c in self.clauses if c.earliest_satisfiable is not None]
        return max(dated) if dated else None

    @property
    def earliest_satisfiable_clause(self) -> str | None:
        dated = [c for c in self.clauses if c.earliest_satisfiable is not None]
        if not dated:
            return None
        return max(dated, key=lambda c: c.earliest_satisfiable).name  # type: ignore[arg-type,return-value]

    def to_dict(self) -> dict[str, Any]:
        ratio = self.met_ratio
        earliest = self.earliest_satisfiable
        return {
            "schema_version": GATE_SCHEMA_VERSION,
            "gate": self.gate,
            "trading_day": self.trading_day.isoformat(),
            "window": [d.isoformat() for d in self.window],
            "met": self.met,
            "met_ratio": None if ratio is None else round(ratio, 6),
            "clauses_total": len(self.clauses),
            "clauses_met": sum(1 for c in self.clauses if c.met),
            "clauses_unmeasurable": self.unmeasurable,
            "coverage": self.coverage,
            "store": self.store,
            "code_sha": self.code_sha,
            "clauses": [c.to_dict() for c in self.clauses],
            "members": list(self.members),
            "earliest_satisfiable": None if earliest is None else earliest.isoformat(),
            "earliest_satisfiable_clause": self.earliest_satisfiable_clause,
        }

    def render(self) -> str:
        lines = [
            "gate {}: {} ({}/{} clauses met, {} unmeasurable)".format(
                self.gate,
                "MET" if self.met else "NOT MET",
                sum(1 for c in self.clauses if c.met),
                len(self.clauses),
                self.unmeasurable,
            )
        ]
        if self.store:
            lines.append(f"store: {self.store}")
        if self.code_sha:
            lines.append(f"commit: {self.code_sha}")
        if self.coverage:
            lines.append(f"coverage: {self.coverage}")
        earliest = self.earliest_satisfiable
        if earliest is not None:
            lines.append(f"exits no earlier than: {earliest.isoformat()} (set by {self.earliest_satisfiable_clause})")
        for clause in self.clauses:
            marker = "x" if clause.met else ("?" if clause.unmeasurable else " ")
            lines.append(f"  [{marker}] {clause.name}: {clause.detail}")
        return "\n".join(lines)


def gate_state_for(reading: GateResult) -> str:
    """``MET``, ``UNMET`` or ``UNMEASURABLE`` for one gate reading.

    The single derivation of a gate's own state from its clauses.
    ``UNMEASURABLE`` outranks both others: a reading with one unreadable clause
    is not "we checked and it fell short". A gate with NO clauses is
    ``UNMEASURED`` on the ladder — that case is the ladder's, because it is a
    fact about registration rather than about a reading.
    """
    if any(c.unmeasurable for c in reading.clauses):
        return "UNMEASURABLE"
    return "MET" if reading.met else "UNMET"
