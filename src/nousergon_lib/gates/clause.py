"""The clause state machine: MET, UNMET, UNMEASURABLE — and containment.

Lifted from `crucible/crucible/gate.py` on its second adoption
(`shared-code-policy`; `data_collection_plan_260914.md` §4.1, item P-02). The
reasoning below is the original's, kept verbatim in meaning because it is the
part that was paid for in incidents rather than in design.

**A clause is met, unmet, or UNMEASURABLE, and UNMEASURABLE is never met.**
``unmeasurable`` is a fact about OUR READING, never about the system being
graded, and it renders distinctly rather than being folded into "unmet" so an
operator does not go fix a producer when the fault is on our side of the
boundary. Two things make a reading unmeasurable: a store read that raised (a
denial, a throttle — we could not obtain the artifact), and an artifact whose
schema PREDATES the question the clause now asks (we obtained it and it cannot
answer). Neither is "the system failed", and neither is ever met.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, MutableMapping
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable

__all__ = [
    "CLAUSE_FUNCTION_PREFIX",
    "CLAUSE_MEMBER_RANK",
    "Clause",
    "ClauseMisconfiguredError",
    "clause_member_status",
    "contain_clause_exceptions",
    "contained",
    "unmeasurable",
]


@dataclass(frozen=True)
class Clause:
    """One gate condition and what the store said about it."""

    name: str
    requirement: str
    met: bool
    detail: str
    evidence: tuple[str, ...] = ()
    unmeasurable: bool = False
    #: The first day on which THIS clause could next read MET, derived by the
    #: clause itself and never restated by a caller. ``None`` on every MET
    #: clause (nothing to wait for) and on every clause whose requirement
    #: carries no calendar floor. A date here is a statement about WHEN, never
    #: a softened statement about WHETHER.
    earliest_satisfiable: dt.date | None = None
    #: The phase by whose exit this clause must read MET. Free-form so an
    #: adopter names its own ladder's rungs; the ladder validates the set.
    phase: str | None = None
    #: Where the reading came from — the registry file, the descriptor, the
    #: adapter. `console-policy` §5.1's fourth field ("source"); ``evidence``
    #: carries the keys, this carries who read them.
    source: str | None = None
    #: When the evidence itself was stamped, as the evidence stamped it — not
    #: the moment of the read. `console-policy` §5.1's third field.
    as_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "requirement": self.requirement,
            "met": self.met,
            "unmeasurable": self.unmeasurable,
            "detail": self.detail,
            # Ordered, not de-duplicated: a repeated key in this list is
            # evidence of a real collision rather than an artifact of a bare
            # read that de-duplication would mask.
            "evidence": sorted(self.evidence),
            "earliest_satisfiable": (
                None if self.earliest_satisfiable is None else self.earliest_satisfiable.isoformat()
            ),
            "phase": self.phase,
            "source": self.source,
            "as_of": self.as_of,
        }


#: The rank a clause's member status is reduced under. HIGHER IS WORSE.
#: ``UNMEASURABLE`` ranks worse than ``UNMET`` for the same reason every
#: ``N/A-*`` status ranks worse than ``RED`` — "we could not read this" must
#: never render better than "we read it and it said no".
CLAUSE_MEMBER_RANK: dict[str, int] = {"MET": 0, "UNMET": 1, "UNMEASURABLE": 2}


def clause_member_status(clause: Clause) -> str:
    """``MET`` / ``UNMET`` / ``UNMEASURABLE`` for one clause."""
    if clause.unmeasurable:
        return "UNMEASURABLE"
    return "MET" if clause.met else "UNMET"


def unmeasurable(
    name: str,
    requirement: str,
    detail: str,
    evidence: Iterable[str] = (),
    *,
    phase: str | None = None,
    source: str | None = None,
) -> Clause:
    """One clause that could not be read, with the reason.

    ``met=False`` and ``unmeasurable=True`` together, always: ``met`` is what
    the ladder and ``met_ratio`` count, and an unmeasurable clause that set
    ``met`` True would be *no data* painted green — the single failure mode the
    gate design exists to forbid.
    """
    return Clause(
        name,
        requirement,
        False,
        detail,
        tuple(evidence),
        unmeasurable=True,
        phase=phase,
        source=source,
    )


#: The prefix a clause function carries so :func:`contain_clause_exceptions`
#: can find it. The containment wrapper is applied by walking a module's
#: globals for this prefix, so a clause added tomorrow is contained without
#: anybody remembering to decorate it — the difference between a rule and a
#: habit.
CLAUSE_FUNCTION_PREFIX = "_clause_"


class ClauseMisconfiguredError(ValueError):
    """A clause was CALLED wrongly — the arguments cannot describe any system.

    The one thing :func:`contained` re-raises, and the distinction is between a
    fact about the environment and a bug in the clause list. Every other
    exception a clause can raise is a failed READING: a client that would not
    build, a bucket that would not list, a document that would not parse. Those
    are UNMEASURABLE, because the system was not observed. This one is different
    in kind — the only place such a call can come from is a clause assembler,
    never an operator, a box, or a credential. Rendering it as UNMEASURABLE
    would put a bug in our own clause list behind a message that reads like an
    AWS problem, and it would render that way forever.
    """


def contained(fn: Callable[..., Clause]) -> Callable[..., Clause]:
    """``fn``, with any escaped exception turned into an UNMEASURABLE clause.

    **No gate clause may raise into its caller.** A clause is one reading among
    hundreds; a reading that could not be taken is that clause's UNMEASURABLE,
    never the death of the whole ladder. Measured in crucible on 2026-09-09: one
    clause gained a call whose CLIENT CONSTRUCTION raised ``NoRegionError``
    outside the try block the read itself had, and every arc failed at the stage
    that renders the ladder. One unguarded line in one clause darkened every
    phase gate in the system and turned a gate READ into a system FAILURE.

    Guarding each clause individually is what failed there; so the containment
    lives HERE and a clause author cannot forget it.

    **This is a deliberate swallow**, so, explicitly: the failure mode swallowed
    is "a clause raised instead of returning a reading"; the primary deliverable
    — a gate result carrying every other clause — survives; and the recording
    surface is the returned clause's own UNMEASURABLE detail, which names the
    exception type and message and is rendered on the ladder and the board.
    ``met=False`` always, so an uncontainable clause can never be counted as
    passing. ``Exception``, not ``BaseException``: a KeyboardInterrupt or a spot
    reclamation must still stop the process.
    """

    @wraps(fn)
    def _guarded(*args: Any, **kwargs: Any) -> Clause:
        try:
            return fn(*args, **kwargs)
        except ClauseMisconfiguredError:
            # Re-raised, not contained: see the class.
            raise
        except Exception as exc:  # noqa: BLE001 - the whole point of this wrapper
            name = fn.__name__
            if name.startswith(CLAUSE_FUNCTION_PREFIX):
                name = name[len(CLAUSE_FUNCTION_PREFIX) :]
            return unmeasurable(
                name,
                f"{name} could be evaluated at all",
                f"the clause raised {type(exc).__name__}: {exc}. A clause that raises has learned nothing "
                "about the system, so this is UNMEASURABLE — it is not a finding about "
                "the system, and it does not darken the other clauses",
            )

    _guarded._contained = True  # type: ignore[attr-defined]
    return _guarded


def contain_clause_exceptions(namespace: MutableMapping[str, Any], *, prefix: str = CLAUSE_FUNCTION_PREFIX) -> int:
    """Wrap every ``prefix``-named callable in ``namespace`` with :func:`contained`.

    Called once at import time from the bottom of a clause module, with
    ``globals()``. Rebinding the module global is what makes it reach assemblers
    that look their clauses up by name at call time: they get the contained
    version without being edited.

    Raises if it wraps nothing — a containment pass that silently matched no
    clause is the shape of a guard that grades an empty set.
    """
    wrapped = 0
    for name, value in list(namespace.items()):
        if not name.startswith(prefix) or not callable(value):
            continue
        if getattr(value, "_contained", False):
            continue
        namespace[name] = contained(value)
        wrapped += 1
    if wrapped == 0:
        raise ClauseMisconfiguredError(
            f"contain_clause_exceptions wrapped no function under prefix {prefix!r}. A "
            "containment pass that matches nothing is a guard over an empty set: it "
            "reports success having protected no clause at all."
        )
    return wrapped
