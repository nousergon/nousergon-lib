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

config#2146 (Brian ruling 2026-07-10): eligibility is DISPOSITION-STRUCTURAL,
not time-window. The 72h ``fresh_skip_active`` cooldown this module used to
export is retired — an issue is eligible unless closed / ``in-progress`` /
gated / PR-covered, all already enforced by ``BASE_EXCLUDE_LABELS`` +
``is_gate_excluded`` below. The one real anti-thrash gap that cooldown
covered (repeated comment-only, no-progress engagement) is now its own
disposition-structural rule: ``comment_only_strikes_exceeded``.

Everything here is PURE — no I/O, no GitHub calls. Consumers fetch issues
their own way (gh CLI on-box, urllib in the Lambda) and pass plain label
lists / counts in.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

# ── Label semantics (mirrors groom_driver.py — contract-tested both sides) ──

LOW_LABEL = "complexity:low"
MID_LABEL = "complexity:mid"
HIGH_LABEL = "complexity:high"
ULTRA_LABEL = "complexity:ultra"

#: config#2146 flap-breaker output: an issue that already oscillated complexity
#: judgment (>= FLAP_BREAKER_ADD_THRESHOLD relabels in the trailing window, see
#: groom_driver.py) or was 2-strike comment-only-stalled gets routed to the
#: human Decision Queue instead of the machine. Structurally excluding both
#: labels here (not just relying on the agent declining solo) is the actual
#: fix for the 2026-07-11 alpha-engine-config#688 floor-breach: gate:weekly-sf's
#: gate-due re-entry does not check these, so a stalled issue kept re-entering
#: the autonomous queue every gate cycle, burning a shared chunk's turn budget
#: and then a wasted solo retry, until it fell just short of the (correctly
#: pool-capped, config#1947) floor and paged as a false CRITICAL. Cleared only
#: by a human ruling in the /backlog-triage session (or new activity resets
#: the flap window) — never by re-admission via a gate.
GROOM_STALLED_LABEL = "groom:stalled"
TRIAGE_SESSION_LABEL = "triage:session"

#: Issues carrying any of these never enter ANY groom queue.
BASE_EXCLUDE_LABELS = frozenset({
    "groom-digest", "in-progress", "do-not-groom", ULTRA_LABEL,
    GROOM_STALLED_LABEL, TRIAGE_SESSION_LABEL,
})

#: config#1805 gate exclusion: HARD — no automated re-entry path exists.
GATE_HARD_EXCLUDE_LABELS = frozenset({"gate:operator", "gate:decision", "gate:device"})
#: SOFT — excluded unless the issue also carries ``gate-due``. ``gate:live-run``
#: retired 2026-07-09, split by named pipeline into the three gate:*-sf labels
#: (config#2057) so gate_sf_run_sweep.py can deterministically re-admit them.
#: ``gate:milestone`` (config#2519) is event-driven — no calendar Re-exam at
#: all — auto-cleared directly by alpha-engine-config's gate_milestone_sweep.py
#: (same posture as gate:data's config#2431 promoted auto-clear) the moment
#: MILESTONE_REGISTRY.yaml marks the referenced milestone ``reached``.
GATE_SOFT_EXCLUDE_LABELS = frozenset({
    "gate:date", "gate:data", "gate:dependency",
    "gate:weekly-sf", "gate:preopen-sf", "gate:postclose-sf",
    "gate:milestone",
})
GATE_DUE_LABEL = "gate-due"

#: config#3199 (2026-07-21): applied by the console Decision Queue
#: (crucible-dashboard ``decision_queue_loader.post_ruling``) when an operator
#: ruling leaves follow-on work behind (any ``gate:*`` label still on the item
#: after de-gating). It marks "a binding ruling is awaiting EXECUTION" — the
#: opposite of blocked — so it overrides the SOFT gate exclusion: the item
#: must enter the groom queue even though a gate label remains (executing the
#: ruling is exactly what resolves that gate). Root cause it closes: ~20
#: Option-A rulings on 2026-07-20 were recorded, nothing executed them
#: (gate-labeled items are excluded at enumeration; PRs are skipped entirely),
#: and gate_sf_run_sweep re-escalated every one back into the Decision Queue.
#: Hard excludes still win — a re-escalated item (``gate:decision``) is owned
#: by a human again, marker or not.
RULING_PENDING_LABEL = "ruling:pending-exec"

#: Non-blocking informational label for expected-CI-red PRs (config#TBD).
#: Applied by the Haiku end-of-SF sweep when every failing CI check
#: is a known expected-failure (drift check, pre-existing broken test).
#: Intentionally OUTSIDE the gate:* namespace -- gate means "blocking",
#: ci:expected-red means "merge is fine despite the red."
CI_EXPECTED_RED_LABEL = "ci:expected-red"

#: CI check names (from gh pr checks --json name) that are expected to
#: fail on PR branches for structural reasons. Updated when new drift
#: guards or known-failing check patterns are added.
_KNOWN_EXPECTED_RED_CHECKS: frozenset[str] = frozenset({
    "iam-drift",
    "drift-detection",
})

#: Tier order, cheapest first. Unlabeled issues default to "mid".
TIERS = ("low", "mid", "high")

#: Tier → model that works it. A bundled run uses the model of the HIGHEST
#: tier actually present in its queue; high-tier issues never run below the
#: high tier's own model. config#2409: high moved Opus -> Sonnet (Brian-
#: ratified cutover, 2026-07-13) — the tier split is now schedule/budget/
#: dedicated-attention only, not a model-capability step up from mid. See
#: the module docstring update and the groom prompt rewrites in
#: alpha-engine-config for the full rationale.
#: groom-primary-deepseek (2026-07-23): low/mid now use DeepSeek V4 Flash as
#: PRIMARY backend (Brian ruling, superseding the 7/22 AMENDED note). High
#: stays on Sonnet. Bundles containing high use the highest tier's model
#: = Sonnet, so high is never served by DeepSeek. Thinking/effort params
#: for DeepSeek are resolved on-box by groom_eligibility_fallback.py.
TIER_MODELS = {
    "low": "deepseek-v4-flash",
    "mid": "deepseek-v4-flash",
    "high": "claude-sonnet-5",
}


@dataclass(frozen=True)
class FallbackModelConfig:
    """One tier's DeepSeek-direct fallback config, used when Brian's Claude
    Max subscription usage runs out mid-groom-run and the groomer falls
    back to calling DeepSeek's own API directly (native API, not
    OpenRouter — DeepSeek's own prompt-caching discount is not reliably
    preserved through OpenRouter for this high-repetition workload).

    A dataclass rather than a bare ``{"model": ..., "thinking": ...}`` dict
    (unlike ``TIER_MODELS``, which is str -> str) because this carries
    extra per-tier fields beyond the model id — named attribute access
    beats string-keyed lookups once there's more than one field per tier.

    ``effort`` is DeepSeek's reasoning-effort request parameter. Verified:
    DeepSeek only implements two real levels server-side — "low"/"medium"
    both collapse to "high", and "xhigh" maps to "max" — so there is no
    meaningful "low effort" setting to ask for. The low complexity tier
    therefore runs with thinking disabled entirely (``thinking=False``,
    ``effort=None``) rather than requesting a low/medium effort value that
    would be silently upgraded server-side to "high".
    """

    model: str
    thinking: bool
    effort: str | None = None


#: Tier -> DeepSeek fallback config. Parallel to TIER_MODELS above (same
#: three tier keys, same "high tier never runs a lesser model than mid"
#: shape) but for the DeepSeek-direct fallback path, not Claude. See
#: FallbackModelConfig for the per-field rationale, including why the low
#: tier has no effort level.
FALLBACK_TIER_MODELS: dict[str, FallbackModelConfig] = {
    "low": FallbackModelConfig(model="deepseek-v4-flash", thinking=False),
    "mid": FallbackModelConfig(model="deepseek-v4-flash", thinking=True, effort="high"),
    "high": FallbackModelConfig(model="deepseek-v4-pro", thinking=True, effort="max"),
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


def tier_of(labels: Iterable[str]) -> str | None:
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
    """config#1805 gate exclusion: hard gates always; soft unless gate-due
    or a pending operator ruling (config#3199 — see RULING_PENDING_LABEL)."""
    label_set = set(labels)
    if label_set & GATE_HARD_EXCLUDE_LABELS:
        return True
    if RULING_PENDING_LABEL in label_set:
        return False
    return bool(label_set & GATE_SOFT_EXCLUDE_LABELS) and GATE_DUE_LABEL not in label_set


def is_actionable(labels: Iterable[str]) -> str | None:
    """The tier this issue is actionable in, or None (excluded/gated)."""
    tier = tier_of(labels)
    if tier is None or is_gate_excluded(labels):
        return None
    return tier


def expected_red_labels_for_checks(failing_checks: Iterable[str]) -> list[str]:
    """Return [CI_EXPECTED_RED_LABEL] if every failing check is
    known-expected, or [] if any check is NOT known-expected
    (genuine CI failure).
    """
    fail_set = set(failing_checks)
    if not fail_set:
        return []
    unknown = fail_set - _KNOWN_EXPECTED_RED_CHECKS
    if unknown:
        return []
    return [CI_EXPECTED_RED_LABEL]


@dataclass(frozen=True)
class SlotDecision:
    """Outcome of a slot's enumerate-then-decide (config#1933)."""

    launch: bool
    tiers: tuple[str, ...]      # tiers in the queue, cheapest first ((), if skip)
    issue_filter: str           # driver filter to export ("" if skip)
    model: str                  # model for the run ("" if skip)
    reason: str                 # human-readable, rendered in the decision record

    def as_record(self) -> dict:
        return {
            "launch": self.launch, "tiers": list(self.tiers),
            "issue_filter": self.issue_filter, "model": self.model,
            "reason": self.reason,
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
      bundle of only low+mid issues runs on Sonnet even at the high-tier
      slot; high-tier issues never run below the high tier's own model
      (COMPLEXITY GUARDRAIL — config#2409: that model is Sonnet as of the
      2026-07-13 cutover, same as mid; the guardrail now protects the
      dedicated queue/budget, not a model-capability step up).
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

# config#2146 (Brian ruling 2026-07-10): the 72h fresh-skip TIME-WINDOW
# cooldown (``fresh_skip_active``, retired below) is gone. It was a
# compensating patch for incomplete dispositions, not a real eligibility
# rule: every legitimate "don't re-touch" state is ALREADY structurally
# excluded elsewhere in this module (``BASE_EXCLUDE_LABELS`` — closed issues
# don't enumerate at all; ``in-progress`` — PR-covered; ``is_gate_excluded``
# — gated). The two dispositions the cooldown actually suppressed were
# ``commented`` (a comment-only engagement is a DEFECT under the
# complete-or-gate contract, config#2135 — rate-limiting churn to ~1/72h
# instead of fixing it was the wrong layer) and ``labeled`` (a tier
# downgrade SHOULD make an issue immediately eligible in its new tier —
# that handoff is the point of the downgrade, and the cooldown blocked it
# for up to 72h). Measured 2026-07-04→07-10: 76% of queued-issue
# dispositions were `commented`; the cooldown merely rate-limited that
# churn rather than preventing it. See ``COMMENT_ONLY_STRIKE_LIMIT`` below
# for the disposition-structural replacement: 2 no-progress comment-only
# engagements route the issue to the human Decision Queue instead of a
# machine cooldown.
# config#2038: harmonized to groom_driver.py's own value (was 900.0) — the
# real gap between a chunk's nominal end (elapsed_min) and its groom comment
# actually landing. Retained post-config#2146: still used to compute each
# S3 run artifact's "horizon" epoch for the comment-only-strike scan below
# (a groom's own comment bumping ``updated_at`` must not itself look like
# new activity that resets the strike count).
FRESH_SKIP_SLACK_SEC = 1800.0

# config#2038: every disposition that counts as an ENGAGEMENT for the
# comment-only-strike scan — SSoT for groom_driver.py's ENGAGED_DISPOSITIONS
# (contract-tested, the box doesn't import this module at runtime) and the
# scheduled-groom-dispatcher Lambda's engagement scan (which DOES import this
# module and must use this constant directly, never a local hardcoded tuple
# — that hardcode is exactly how this and FRESH_SKIP_SLACK_SEC drifted in
# the first place).
ENGAGED_DISPOSITIONS = ("closed", "pr_opened", "commented", "labeled")

# config#2038/#2146: how many trailing daily S3 prefixes (``groom/{date}/``)
# to scan when building the engagement map the comment-only-strike scan
# (and, historically, fresh-skip) reads. Kept at 4 post-retirement of the
# 72h fresh-skip window: it remains a safe cover for the strike scan's own
# short lookback (``COMMENT_ONLY_STRIKE_LOOKBACK_DAYS`` below) against
# calendar-day bucket boundaries.
ENGAGEMENT_LOOKBACK_DAYS = 4

# config#2146 (deliverable 2 — the anti-thrash replacement for the retired
# 72h cooldown): after this many CONSECUTIVE comment-only (``commented``)
# engagements on an issue with NO intervening state change (closed /
# pr_opened / labeled reset the count), the issue is a groom DEFECT that
# machine grooming cannot resolve — route it to the human Decision Queue
# (``GROOM_STALLED_LABEL`` + ``TRIAGE_SESSION_LABEL``) instead of a third
# bare comment. 2, not 3: a single comment-only pass is normal (e.g. an
# exempt priority-drift/tier-reservation/metadata note, or the first
# config#2135 defect-flag); a SECOND comment-only pass with nothing having
# changed is the thrash pattern (171 issues comment-groomed >=3x with zero
# completions, measured 2026-07-04->07-10) — never let it reach a third.
COMMENT_ONLY_STRIKE_LIMIT = 2

# config#2146: trailing days of S3 run artifacts scanned for the strike
# count. Wider than ENGAGEMENT_LOOKBACK_DAYS (4) because 2 strikes can
# legitimately span more than 3 tier-run cycles for a P2/P3 issue that
# isn't re-groomed every day — this must cover the realistic gap between
# two comment-only passes on a lower-priority issue, not just a 72h window.
COMMENT_ONLY_STRIKE_LOOKBACK_DAYS = 21


def comment_only_strikes_exceeded(dispositions_desc: Iterable[str], *,
                                  limit: int = COMMENT_ONLY_STRIKE_LIMIT) -> bool:
    """config#2146 pure predicate: given an issue's disposition history in
    REVERSE chronological order (most recent run first, restricted to runs
    that actually engaged it — i.e. already filtered to
    ``ENGAGED_DISPOSITIONS``), does it have >= ``limit`` CONSECUTIVE
    ``"commented"`` entries counting back from the most recent?

    Any ``"closed"``/``"pr_opened"``/``"labeled"`` entry is a real state
    change and stops the count (it resets the thrash streak — a downgrade,
    a PR, or a close means the machine made progress, even if a later pass
    goes comment-only again). An empty history is 0 strikes."""
    streak = 0
    for disposition in dispositions_desc:
        if disposition != "commented":
            break
        streak += 1
        if streak >= limit:
            return True
    return False


def decide_trigger(
    counts: Mapping[str, int],
    oldest_wait_hours: Mapping[str, float] | None = None,
    p0_tiers: Iterable[str] = (),
    *,
    floor: int = DEFAULT_FLOOR,
    max_wait_hours: float = DEFAULT_MAX_WAIT_HOURS,
) -> list[SlotDecision]:
    """Full-backlog trigger decision: 0..3 launches.

    groom-primary-deepseek (2026-07-23): EVERY tier with actionable issues
    launches its own dedicated box — no floor check, no thin-tier bundling.
    Each tier runs independently at its own model. A tier with zero
    actionable issues is skipped. The ``floor`` and ``max_wait_hours`` params
    are accepted for signature compatibility but unused (no tier is ever
    deferred or bundled). ``p0_tiers`` is likewise unused (every tier launches
    regardless of P0 presence).

    The ``oldest_wait_hours`` param is accepted for signature compatibility
    but unused. Returns decisions in highest-first order (high, mid, low)
    for consistent SF Map iteration ordering.
    """
    launches: list[SlotDecision] = []
    for tier in reversed(TIERS):
        count = counts.get(tier, 0)
        if count <= 0:
            continue
        reason = f"{count} actionable — unconditional launch (no floor gate)"
        launches.append(SlotDecision(
            True, (tier,), SINGLE_TIER_FILTERS[tier], TIER_MODELS[tier], reason,
        ))
    return launches
