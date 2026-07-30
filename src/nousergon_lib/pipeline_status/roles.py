"""The ``pipeline_role`` vocabulary — one classification, fleet-wide.

Every Step Functions execution in the fleet carries a ``pipeline_role`` on
its input JSON (the "Option-D taxonomy", 2026-05-25). Consumers resolve
"what is this pipeline's current state" by filtering executions to a set of
roles: the cadence run, plus whatever overlays may legitimately stand in for
it. That filter is the reason this module exists.

**The failure this closes** (alpha-engine-config#5590, 2026-07-29): roles are
minted in PRODUCER repos (``nousergon-data``'s SF JSONs, ``crucible-executor``'s
daemon, EventBridge inputs) and were allow-listed by hand in CONSUMER repos.
An unknown role does not raise anywhere — the newest-first role walk simply
skips it — so adding ``exercise`` to the post-close SF left the dashboard's
weekly row reporting a FAILED execution from four days earlier while an
exercise run was live. Silent, and indistinguishable from a healthy read.

With the vocabulary here, a producer repo's contract test fails at MINT
time (see ``nousergon-data/tests/test_pipeline_role_vocabulary.py``) and
consumers import these sets instead of re-declaring them.

**The buckets, and why the split is load-bearing:**

``CADENCE_ROLES``
    Cron-triggered first-try runs. These ARE the cycle.
``RECOVERY_ROLES``
    Ad-hoc executions that legitimately COMPLETE the current cadence cycle
    after its scheduled run failed. A consumer may let one of these clear a
    failed cadence verdict.
``EXERCISE_ROLES``
    Synthetic runs of a real pipeline for iteration/debugging — currently
    the post-close-chained daily exercise run (alpha-engine-config#5489).
    **Never a member of RECOVERY_ROLES**, and the separation is a binding
    invariant, not a taste call: an exercise run is not the cycle's
    deliverable, so letting a Tuesday exercise SUCCESS mark the week's
    cadence COMPLETE would hide a failed Saturday belief refresh behind a
    green dot. Exercise runs get their own surface.
``ADHOC_ROLES``
    Everything operator-initiated that must never displace a cadence
    verdict: smoke tests, shell runs, backfills, operator replays.

An execution with NO role is legitimate and common (untagged manual runs) —
absence is not a vocabulary violation. Only a non-empty role outside
:data:`ALL_ROLES` is.

**Not modelled here:** the Saturday-SF-watch dispatcher's "fast-path rerun"
is NOT its own role — it re-submits the failed execution's own input
verbatim, so it carries whatever cadence role was already there and is
identified by its execution-name prefix instead.
"""

from __future__ import annotations

# Cron-triggered cadence runs: ne-weekly-freshness (weekly),
# ne-preopen-trading (daily), ne-postclose-trading (eod).
CADENCE_ROLES = frozenset({"weekly", "daily", "eod"})

# Overlays that may complete a cadence cycle after its scheduled run failed.
# ``watch-rerun`` is nousergon-data ``scripts/weekly_sf_rerun.py``'s emitted
# role; ``recovery`` is the general operator/agent convention.
RECOVERY_ROLES = frozenset({"watch-rerun", "recovery"})

# Synthetic exercise runs — real pipeline, real writes, but a debugging
# cadence rather than the cycle's deliverable. See the module docstring for
# why this must never merge into RECOVERY_ROLES.
EXERCISE_ROLES = frozenset({"exercise"})

# Operator-initiated one-offs. Never cycle-completing.
ADHOC_ROLES = frozenset({"smoke", "shell-run", "backfill", "operator-replay"})

#: Every role a fleet producer may legitimately mint. A literal outside this
#: set is a contract violation, caught by the producer repo's own CI.
ALL_ROLES = CADENCE_ROLES | RECOVERY_ROLES | EXERCISE_ROLES | ADHOC_ROLES

_BUCKETS: tuple[tuple[str, frozenset[str]], ...] = (
    ("cadence", CADENCE_ROLES),
    ("recovery", RECOVERY_ROLES),
    ("exercise", EXERCISE_ROLES),
    ("adhoc", ADHOC_ROLES),
)


def classify(role: str | None) -> str | None:
    """Bucket name for ``role``, or None when it is untagged or unknown.

    Returns one of ``"cadence"``/``"recovery"``/``"exercise"``/``"adhoc"``.
    ``None`` for a missing role (legitimate — untagged manual run) and for a
    role outside :data:`ALL_ROLES` (a contract violation the caller should
    SURFACE, never silently drop). Callers that need to tell those two apart
    check ``role`` for emptiness themselves.
    """
    if not role:
        return None
    for name, bucket in _BUCKETS:
        if role in bucket:
            return name
    return None


def cadence_filter(cadence_role: str) -> frozenset[str]:
    """Role filter for a cadence row: the cadence role plus recovery overlays.

    The canonical consumer filter. Deliberately excludes exercise and ad-hoc
    roles — neither may displace or complete the cadence verdict.
    """
    if cadence_role not in CADENCE_ROLES:
        raise ValueError(
            f"{cadence_role!r} is not a cadence role; expected one of "
            f"{sorted(CADENCE_ROLES)}"
        )
    return frozenset({cadence_role}) | RECOVERY_ROLES


__all__ = [
    "ADHOC_ROLES",
    "ALL_ROLES",
    "CADENCE_ROLES",
    "EXERCISE_ROLES",
    "RECOVERY_ROLES",
    "cadence_filter",
    "classify",
]
