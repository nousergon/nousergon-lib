"""Fault exclusion: which run ids an induced-fault record excuses.

A commissioning programme deliberately breaks things (`observability-policy`
§9: a detector that has never fired is not in service). Those induced failures
must not be counted against the SLO clauses that grade real operation — and the
mechanism that arranges that is the one mechanism in a gate capable of turning
an arbitrary red clause green, so its refusals matter more than its behaviour.

Two refusals, both lifted from crucible's original:

* **Matched on ``run_id`` alone, never on the day.** A failed run is excused
  only when SOME fault record names its exact ``run_id``, so a genuine failure
  on a day a fault was once induced still fails.
* **Only an ``induced`` record excuses anything.** A record filed for a fault
  the system absorbed, or for one that never reached a run at all, must not be
  a second route to the excusal the ``run_id`` refusal guards.

A listing that could not be read returns ``(None, problem)`` — never an empty
exclusion set — so the caller folds it into UNMEASURABLE rather than grading
with zero exclusions it never confirmed.
"""

from __future__ import annotations

from nousergon_lib.gates.store import GateStore, list_store_keys, read_store_document

__all__ = [
    "FAULT_OUTCOME_INDUCED",
    "FAULT_RECORD_OUTCOME_FIELD",
    "FAULT_RECORD_RUN_ID_FIELD",
    "fault_excused_run_ids",
]

FAULT_RECORD_OUTCOME_FIELD = "outcome"
FAULT_RECORD_RUN_ID_FIELD = "run_id"

#: The only outcome that excuses a failure. ``absorbed`` names a run that read
#: ``ok`` (never counted against anyone) and ``unreachable`` names no run at all.
FAULT_OUTCOME_INDUCED = "induced"


def fault_excused_run_ids(store: GateStore, prefix: str) -> tuple[frozenset[str] | None, str | None]:
    """Every ``run_id`` an induced-fault record under ``prefix`` excuses.

    A record that fails to parse, or names no ``run_id``, is SKIPPED rather
    than raised: a malformed or hand-broken record excuses nothing, which is
    the only safe default for a mechanism whose entire point is that it must
    never be able to turn an arbitrary red clause green.
    """
    listed = list_store_keys(store, prefix)
    if listed.problem is not None:
        return None, listed.problem
    run_ids = set()
    for key in listed.keys or []:
        if not key.endswith(".json"):
            continue
        read = read_store_document(store, key)
        if read.problem is not None or read.absent:
            continue
        document = read.document or {}
        if document.get(FAULT_RECORD_OUTCOME_FIELD) != FAULT_OUTCOME_INDUCED:
            continue
        run_id = str(document.get(FAULT_RECORD_RUN_ID_FIELD) or "").strip()
        if run_id:
            run_ids.add(run_id)
    return frozenset(run_ids), None
