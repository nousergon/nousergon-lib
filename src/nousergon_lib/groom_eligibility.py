"""Groom eligibility + demand-driven slot-dispatch decision (config#1933).

Single source of truth for the label semantics BOTH groom consumers apply:

- ``alpha-engine-config/scripts/groom_driver.py`` — on-box enumeration (its
  inline constants are contract-tested against this module; the spot box does
  not install nousergon-lib at runtime).
- ``nousergon-data`` ``scheduled-groom-dispatcher`` Lambda — pre-boot
  enumerate-then-decide: at each daily slot, count actionable issues per
  complexity tier and decide launch/skip/bundle BEFORE any spot spend.

The 2026-07-07 nousergon-data-PR683 incident (dispatcher silently downgraded
``gated-reverify`` to ``mid-only`` because its private filter set drifted from
the driver's) is the bug class this module closes: one constant set, two
consumers, contract tests on both sides.

Everything here is PURE — no I/O, no GitHub calls. Consumers fetch issues
their own way (gh CLI on-box, urllib in the Lambda) and pass plain label
lists / counts in.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Optional, Sequence

# ── Label semantics (mirrors groom_driver.py — contract-tested both sides) ──

LOW_LABEL = "complexity:low"
MID_LABEL = "complexity:mid"
HIGH_LABEL = "complexity:high"
ULTRA_LABEL = "complexity:ultra"

#: Issues carrying any of these never enter ANY groom queue.
BASE_EXCLUDE_LABELS = frozenset({"groom-digest", "in-progress", "do-not-groom", ULTRA_LABEL})

#: config#1805 gate exclusion: HARD — no automated re-entry path exists.
GATE_HARD_EXCLUDE_LABELS = frozenset({"gate:operator", "gate:decision", "gate:device"})
#: SOFT — excluded unless the issue also carries ``gate-due``. ``gate:live-run``
#: retired 2026-07-09, split by named pipeline into the three gate:*-sf labels
#: (config#2057) so gate_sf_run_sweep.py can deterministically re-admit them.
GATE_SOFT_EXCLUDE_LABELS = frozenset({
    "gate:date", "gate:data", "gate:dependency",
    "gate:weekly-sf", "gate:preopen-sf", "gate:postclose-sf",
})
GATE_DUE_LABEL = "gate-due"

#: Tier order, cheapest first. Unlabeled issues default to "mid".
TIERS = ("low", "mid", "high")

#: Tier → model that works it. A bundled run uses the model of the HIGHEST
#: tier actually present in its queue; high-tier issues never run below Opus.
TIER_MODELS = {
    "low": "claude-haiku-4-5",
    "mid": "claude-sonnet-5",
    "high": "claude-opus-4-8",
}

#: Every issue_filter value the driver accepts. Single-tier forms keep the
#: legacy names; bundled forms are "+"-joined highest-first (config#1933).
SINGLE_TIER_FILTERS = {"low": "low-only", "mid": "mid-only", "high": "high-only"}
BUNDLED_FILTERS = frozenset({"mid+low", "high+mid", "high+low", "high+mid+low"})
VALID_ISSUE_FILTERS = frozenset(
    {"default", "gated-reverify", *SINGLE_TIER_FILTERS.values(), *BUNDLED_FILTERS}
)

#: config#1933 dispatch parameters.
DEFAULT_FLOOR = 8            # min queue size worth a spot boot (= driver floor)
DEFAULT_MAX_WAIT_HOURS = 72  # anti-starvation escape valve (ARCH §66)


def filter_tiers(issue_filter: str) -> tuple[str, ...]:
    """Tiers a driver run with ``issue_filter`` works, cheapest first.

    ``default`` is the historical alias for ``mid-only``; ``gated-reverify``
    works no complexity queue (returns ()).
    """
    if issue_filter == "default":
        return ("mid",)
    if issue_filter == "gated-reverify":
        return ()
    for tier, name in SINGLE_TIER_FILTERS.items():
        if issue_filter == name:
            return (tier,)
    if issue_filter in BUNDLED_FILTERS:
        return tuple(sorted(issue_filter.split("+"), key=TIERS.index))
    raise ValueError(f"unknown issue_filter: {issue_filter!r}")


def filter_for_tiers(tiers: Iterable[str]) -> str:
    """The canonical issue_filter string for a set of tiers (highest-first)."""
    ordered = sorted(set(tiers), key=TIERS.index, reverse=True)
    if not ordered:
        raise ValueError("no tiers")
    if len(ordered) == 1:
        return SINGLE_TIER_FILTERS[ordered[0]]
    name = "+".join(ordered)
    if name not in BUNDLED_FILTERS:
        raise ValueError(f"unsupported tier bundle: {name}")
    return name


def tier_of(labels: Iterable[str]) -> Optional[str]:
    """Complexity tier for a label set, or None if excluded from grooming.

    Unlabeled ⇒ "mid" (the standing default). ``complexity:ultra`` and the
    base excludes ⇒ None.
    """
    label_set = set(labels)
    if label_set & BASE_EXCLUDE_LABELS:
        return None
    if HIGH_LABEL in label_set:
        return "high"
    if LOW_LABEL in label_set:
        return "low"
    return "mid"


def is_gate_excluded(labels: Iterable[str]) -> bool:
    """config#1805 gate exclusion: hard gates always; soft unless gate-due."""
    label_set = set(labels)
    if label_set & GATE_HARD_EXCLUDE_LABELS:
        return True
    return bool(label_set & GATE_SOFT_EXCLUDE_LABELS) and GATE_DUE_LABEL not in label_set


def is_actionable(labels: Iterable[str]) -> Optional[str]:
    """The tier this issue is actionable in, or None (excluded/gated)."""
    tier = tier_of(labels)
    if tier is None or is_gate_excluded(labels):
        return None
    return tier


@dataclass(frozen=True)
class SlotDecision:
    """Outcome of a slot's enumerate-then-decide (config#1933).

    ``partition_index``/``partition_count`` (config#2129) identify this
    decision's slice of the PR merge-readiness sweep when it co-launches
    with OTHER launching decisions from the SAME trigger. Every open PR is
    tier-agnostic, so N concurrently-launched boxes racing the identical
    sweep would double-push the same PR — these two fields let each box
    claim a deterministic, disjoint slice instead (assigned by
    ``decide_trigger`` below) so all co-launched boxes can sweep
    concurrently without a race, rather than only the first-launched one
    running the sweep at all (the prior starvation bug: ``high`` always
    sorted first in ``decide_trigger``'s pool ordering, so ``low``/``mid``
    never got a sweep turn). Defaults (0, 1) mean "no partition — sweep
    every open PR", the correct value for any solo launch (``decide_slot``,
    single-tier manual dispatch, ``gated-reverify``).
    """

    launch: bool
    tiers: tuple[str, ...]      # tiers in the queue, cheapest first ((), if skip)
    issue_filter: str           # driver filter to export ("" if skip)
    model: str                  # model for the run ("" if skip)
    reason: str                 # human-readable, rendered in the decision record
    partition_index: int = 0   # this decision's slice index among co-launched decisions
    partition_count: int = 1   # total co-launched decisions this trigger (1 = unpartitioned)

    def as_record(self) -> dict:
        return {
            "launch": self.launch, "tiers": list(self.tiers),
            "issue_filter": self.issue_filter, "model": self.model,
            "reason": self.reason, "partition_index": self.partition_index,
            "partition_count": self.partition_count,
        }


def decide_slot(
    slot_tier: str,
    counts: Mapping[str, int],
    oldest_wait_hours: Mapping[str, float] | None = None,
    has_actionable_p0: bool = False,
    *,
    floor: int = DEFAULT_FLOOR,
    max_wait_hours: float = DEFAULT_MAX_WAIT_HOURS,
) -> SlotDecision:
    """Decide what (if anything) this slot launches.

    Rules (config#1933, Brian-ratified 2026-07-07):
    - The slot considers its OWN tier plus every LOWER tier whose own count
      is below ``floor`` (those were/will be skipped at their own slot —
      they bundle upward; higher tiers never bundle down).
    - Launch iff the combined queue >= ``floor``, OR the escape valve fires:
      an actionable P0 exists, or any considered tier's oldest actionable
      issue has waited >= ``max_wait_hours`` (anti-starvation, ARCH §66).
    - The run's model = highest tier actually PRESENT in the queue — a
      bundle of only low+mid issues runs on Sonnet even at the Opus slot;
      high-tier issues never run below Opus (COMPLEXITY GUARDRAIL).
    """
    if slot_tier not in TIERS:
        raise ValueError(f"unknown slot tier: {slot_tier!r}")
    oldest_wait_hours = oldest_wait_hours or {}
    slot_idx = TIERS.index(slot_tier)
    considered = [slot_tier] + [t for t in TIERS[:slot_idx] if counts.get(t, 0) < floor]
    present = sorted(
        (t for t in considered if counts.get(t, 0) > 0), key=TIERS.index,
    )
    total = sum(counts.get(t, 0) for t in present)
    if total == 0:
        return SlotDecision(False, (), "", "", f"queue empty at {slot_tier} slot")

    overdue = [t for t in present if oldest_wait_hours.get(t, 0.0) >= max_wait_hours]
    if total >= floor:
        reason = f"{total} actionable across {'+'.join(present)} >= floor {floor}"
    elif has_actionable_p0:
        reason = f"escape valve: actionable P0 present (queue {total} < floor {floor})"
    elif overdue:
        reason = (f"escape valve: {'+'.join(overdue)} oldest waited >= "
                  f"{max_wait_hours:g}h (queue {total} < floor {floor})")
    else:
        return SlotDecision(
            False, tuple(present), "", "",
            f"queue {total} < floor {floor}, no P0, none waited {max_wait_hours:g}h — deferred upward",
        )
    model_tier = present[-1]  # highest present
    return SlotDecision(
        True, tuple(present), filter_for_tiers(present), TIER_MODELS[model_tier], reason,
    )


# ── Symmetric-trigger decision (config#1933 scope correction, 2026-07-07) ────
# All daily triggers are IDENTICAL: each evaluates the full backlog and
# launches a run per tier that clears the floor. decide_slot() above remains
# for single-tier manual dispatches; scheduled triggers use decide_trigger().

FRESH_SKIP_HOURS = 72.0
# config#2038: was 900.0, silently drifted from groom_driver.py's
# _FRESH_SKIP_SLACK_SEC=1800 (the box does not install nousergon-lib at
# runtime, so nothing caught the two constants diverging). Harmonized to the
# driver's value — 1800s better absorbs the real gap between a chunk's
# nominal end (elapsed_min) and its groom comment actually landing. The drift
# made the pre-boot Lambda's fresh-skip-aware enumeration systematically
# UNDER-skip relative to the on-box driver's, so a dispatcher decision like
# "19 actionable, launch" could deflate to "8 actually actionable" once the
# box's own (correct) fresh-skip ran three minutes later — a spot box boots
# for a queue that was never really that big.
FRESH_SKIP_SLACK_SEC = 1800.0

# config#2038: every disposition that counts as ENGAGED for fresh-skip
# purposes — SSoT for groom_driver.py's ENGAGED_DISPOSITIONS (contract-tested,
# the box doesn't import this module at runtime) and the scheduled-groom-
# dispatcher Lambda's engagement scan (which DOES import this module and must
# use this constant directly, never a local hardcoded tuple — that hardcode is
# exactly how this and FRESH_SKIP_SLACK_SEC drifted in the first place).
ENGAGED_DISPOSITIONS = ("closed", "pr_opened", "commented", "labeled")

# config#2038: how many trailing daily S3 prefixes (``groom/{date}/``) to scan
# when building the engagement map that feeds fresh-skip. Must be >= 4 to
# safely cover a 72h (FRESH_SKIP_HOURS) rolling window against calendar-day
# buckets: a run that started just before UTC midnight 3 calendar days ago is
# still inside the 72h window, but a 3-day lookback (today/yesterday/day-
# before) can miss its bucket entirely. The dispatcher Lambda hardcoded
# ``range(3)`` — under-covering the window and, combined with the slack drift
# above, undercounting fresh-skips relative to the driver's own (correct)
# 4-day lookback.
ENGAGEMENT_LOOKBACK_DAYS = 4


def fresh_skip_active(engaged_epoch: float, updated_epoch: float,
                      now_epoch: float, *, skip_hours: float = FRESH_SKIP_HOURS,
                      slack_sec: float = FRESH_SKIP_SLACK_SEC) -> bool:
    """config#1893 semantics, pure: an issue engaged by a groom < skip_hours
    ago with no NEW activity since (updated_at within the engagement window +
    slack) is skipped. Any later activity re-admits it immediately."""
    if now_epoch - engaged_epoch >= skip_hours * 3600.0:
        return False
    return updated_epoch <= engaged_epoch + slack_sec


def decide_trigger(
    counts: Mapping[str, int],
    oldest_wait_hours: Mapping[str, float] | None = None,
    p0_tiers: Iterable[str] = (),
    *,
    floor: int = DEFAULT_FLOOR,
    max_wait_hours: float = DEFAULT_MAX_WAIT_HOURS,
) -> list[SlotDecision]:
    """Full-backlog trigger decision: 0..3 launches.

    - Every tier with count >= floor gets its OWN run (its tier's model).
    - Each thin tier (0 < count < floor) attaches to the NEAREST standalone
      tier ABOVE it (upward only — high never rides below Opus).
    - Thin tiers with no standalone tier above pool together; the pool
      launches at the highest-present tier's model iff its combined count
      >= floor OR the escape valve fires for the pool (an actionable P0 in
      a pooled tier, or a pooled tier's oldest waited >= max_wait_hours).

    config#2129: every LAUNCHING decision in the returned list gets a
    ``partition_index``/``partition_count`` assignment (0-based index among
    ONLY the launching decisions, count = how many are launching this
    trigger) so co-launched boxes can each sweep a disjoint slice of the
    org's open PRs instead of racing the full set. Non-launching (skip)
    decisions keep the default (0, 1) — meaningless since they never launch
    a box.
    """
    def _assign_partitions(decisions: list[SlotDecision]) -> list[SlotDecision]:
        """config#2129: index the LAUNCHING decisions 0..N-1 (N = how many
        launch this trigger); non-launching decisions are left untouched
        (their partition fields are never read)."""
        n = sum(1 for d in decisions if d.launch)
        out: list[SlotDecision] = []
        idx = 0
        for d in decisions:
            if d.launch:
                out.append(replace(d, partition_index=idx, partition_count=n))
                idx += 1
            else:
                out.append(d)
        return out

    oldest_wait_hours = oldest_wait_hours or {}
    p0 = set(p0_tiers)
    standalone = [t for t in TIERS if counts.get(t, 0) >= floor]
    thin = [t for t in TIERS if 0 < counts.get(t, 0) < floor]
    pools: dict[str, list[str]] = {t: [t] for t in standalone}
    leftover: list[str] = []
    for t in thin:
        above = [st for st in standalone if TIERS.index(st) > TIERS.index(t)]
        if above:
            pools[min(above, key=TIERS.index)].append(t)
        else:
            leftover.append(t)
    launches: list[SlotDecision] = []
    for anchor in sorted(pools, key=TIERS.index, reverse=True):
        tiers = sorted(pools[anchor], key=TIERS.index)
        total = sum(counts.get(t, 0) for t in tiers)
        launches.append(SlotDecision(
            True, tuple(tiers), filter_for_tiers(tiers), TIER_MODELS[tiers[-1]],
            f"{total} actionable across {'+'.join(tiers)} (anchor {anchor} >= floor {floor})",
        ))
    if leftover:
        tiers = sorted(leftover, key=TIERS.index)
        total = sum(counts.get(t, 0) for t in tiers)
        overdue = [t for t in tiers if oldest_wait_hours.get(t, 0.0) >= max_wait_hours]
        pool_p0 = sorted(p0 & set(tiers), key=TIERS.index)
        if total >= floor:
            reason = f"{total} actionable pooled across {'+'.join(tiers)} >= floor {floor}"
        elif pool_p0:
            reason = f"escape valve: actionable P0 in {'+'.join(pool_p0)} (pool {total} < floor {floor})"
        elif overdue:
            reason = (f"escape valve: {'+'.join(overdue)} oldest waited >= "
                      f"{max_wait_hours:g}h (pool {total} < floor {floor})")
        else:
            launches.append(SlotDecision(
                False, tuple(tiers), "", "",
                f"thin pool {'+'.join(tiers)} ({total}) < floor {floor}, no P0, "
                f"none waited {max_wait_hours:g}h — deferred to a later trigger",
            ))
            return _assign_partitions(launches)
        launches.append(SlotDecision(
            True, tuple(tiers), filter_for_tiers(tiers), TIER_MODELS[tiers[-1]], reason,
        ))
    return _assign_partitions(launches)
