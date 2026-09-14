"""The gate engine: clauses, gate readings, and the phase ladder.

**A gate reads; it never runs.** Every clause is evaluated against artifacts
already written, so a merge can never satisfy a gate. **A clause is MET, UNMET
or UNMEASURABLE, and UNMEASURABLE is never met.** **The gate's job succeeds when
the MEASUREMENT succeeds** — the ladder is written and the process exits
non-zero unless the gate is met, so nobody reads "not there yet" as "done".

Lifted from `crucible/crucible/gate.py` on its second adoption
(`shared-code-policy`), commissioned by `alpha-engine-config-I10748`'s data
collector plan §4.1 item P-02. What moved is the ENGINE — the clause state
machine, the containment wrapper, the gate reading, the ladder document and its
schema, and fault exclusion. What did NOT move is any clause definition: those
stay in the repo that owns the thing being graded (`architecture.d/146` rule 1),
which is why this package has no notion of a phase 0 or of crucible's arcs.

One engine, two ladders, one schema. `crucible`'s re-import onto this package
is a tracked follow-up; until it lands crucible keeps its own copy, and the two
schemas are kept compatible on purpose — this one is a strict generalization
(`phase`, `number` and the tracker refs are no longer bounded to crucible's
own), so a document crucible produces validates here unchanged.

Usage sketch::

    from nousergon_lib.gates import Clause, GateResult, Phase, build_ladder

    def _clause_thing_exists(store):
        read = read_store_document(store, "some/key.json")
        if read.problem is not None:
            return unmeasurable("thing_exists", "...", read.problem)
        return Clause("thing_exists", "the thing is published", not read.absent, "...")

    contain_clause_exceptions(globals())
"""

from __future__ import annotations

from nousergon_lib.gates.clause import (
    CLAUSE_FUNCTION_PREFIX,
    CLAUSE_MEMBER_RANK,
    Clause,
    ClauseMisconfiguredError,
    clause_member_status,
    contain_clause_exceptions,
    contained,
    unmeasurable,
)
from nousergon_lib.gates.faults import (
    FAULT_OUTCOME_INDUCED,
    fault_excused_run_ids,
)
from nousergon_lib.gates.keys import LADDER_KEY, gate_key, gate_prefix, ladder_key
from nousergon_lib.gates.ladder import (
    LADDER_CONSOLE_STATE,
    LADDER_SCHEMA_PATH,
    LADDER_SCHEMA_VERSION,
    LADDER_STATES,
    Ladder,
    Phase,
    PhaseRow,
    build_ladder,
    ladder_payload,
    ladder_schema,
    last_read,
    validate_ladder_document,
)
from nousergon_lib.gates.result import GATE_SCHEMA_VERSION, GateResult, gate_state_for
from nousergon_lib.gates.store import (
    DocumentRead,
    GateStore,
    KeysRead,
    list_store_keys,
    read_store_document,
)

__all__ = [
    "CLAUSE_FUNCTION_PREFIX",
    "CLAUSE_MEMBER_RANK",
    "DocumentRead",
    "FAULT_OUTCOME_INDUCED",
    "GATE_SCHEMA_VERSION",
    "GateResult",
    "GateStore",
    "KeysRead",
    "LADDER_CONSOLE_STATE",
    "LADDER_KEY",
    "LADDER_SCHEMA_PATH",
    "LADDER_SCHEMA_VERSION",
    "LADDER_STATES",
    "Clause",
    "ClauseMisconfiguredError",
    "Ladder",
    "Phase",
    "PhaseRow",
    "build_ladder",
    "clause_member_status",
    "contain_clause_exceptions",
    "contained",
    "fault_excused_run_ids",
    "gate_key",
    "gate_prefix",
    "gate_state_for",
    "ladder_key",
    "ladder_payload",
    "ladder_schema",
    "last_read",
    "list_store_keys",
    "read_store_document",
    "unmeasurable",
    "validate_ladder_document",
]
