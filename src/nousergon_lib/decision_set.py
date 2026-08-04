"""The fleet's shared decision-set resolver (alpha-engine-config#5809).

``signals/{date}/signals.json::universe`` is a **sizing envelope** — one row
per name on the whole scanner board (measured 2026-07-29: 903 rows, all
HOLD) — so the executor can size and exit anything it might hold. It is not
a scope, and reading it as one silently operates on the entire board instead
of whatever set the caller actually meant to act on.

The correct artifact is already published, canonically, by the scanner:

    universe_membership/{date}/membership.json :: cuts.<cut_name>
        {"basis": "scanner_gate", "size": 60, "tickers": [...],
         "source": "candidates/{date}/candidates.json::scanner_tickers"}

Named cuts as of 2026-07-30 include ``scanner_candidates`` (the scanner gate,
@60) and ``attractiveness_top_20`` (cross-sectional rank top 20). See
``champion-challenger-policy.md`` §2 (the universe-cut registry) and
``crucible-research/scoring/universe_membership.py`` (the producer) for the
full, current set — this module does not hardcode which cuts exist; the
caller names the one it needs.

This is a lift of ``nousergon-data/rag/pipelines/_rag_scope.py``
(config-I5700, 2026-07-30 — "the more complete of the two" per config#5809),
generalized from a single hardcoded ``scanner_candidates`` scope to any named
cut, so a second independent copy
(``alpha-engine-predictor/inference/stages/load_universe.py``) does not
become a third. Two adopters already existed at the time this module was
written; per ``shared-code-policy`` that is past the second-adoption trigger.
Repointing those two call sites is tracked follow-up (config#5809), not done
by this change — this module lands standalone first.

WHAT IT PRESERVES FROM THE SOURCE, UNCONDITIONALLY:

- **O(1) pointer read.** ``run_date=None`` reads the ``latest.json`` sidecar
  the producer writes alongside every dated artifact, rather than listing
  the bucket and taking the max — the prior generation of this pattern paged
  ``list_objects_v2`` with ``Delimiter="/"` and silently returned the
  1000th-OLDEST date once the partition count crossed the 1000-CommonPrefixes
  page limit.
- **The equity-ticker regex.** The held-position artifact is Metron's, and
  Metron holds more than equities — measured 2026-07-30, the live held set
  contained ``912828YK0`` (a US Treasury CUSIP). Every identifier that fails
  ``_TICKER_RE`` is dropped from the returned ticker list (and named in the
  logged ``rejected`` count) rather than sent to an equity-only ingestion
  source as a guaranteed wasted request.
- **The held-position union.** A position needs evidence whether or not it
  ranks this cycle — an EXIT still needs a rationale. Callers whose output
  feeds an exit or sizing decision should keep ``include_held=True``
  (the default); a caller with no such obligation may pass
  ``include_held=False`` to scope strictly to the named cut.
- **Hard fail, no fallback.** :class:`DecisionSetUnavailable` is raised on a
  missing/unparseable membership artifact or an empty/absent named cut.
  There is deliberately NO fallback to ``signals.json::universe`` — that
  fallback IS the defect this module exists to kill, and it stays invisible
  until the next timeout or vendor bill. Do not add one.

The holdings union itself IS fail-soft (see :func:`_load_holdings`) — a
missing Metron artifact narrows coverage (held names lose fresh evidence)
rather than blocking the whole resolution, which is the correct asymmetry:
the membership cut is the load-bearing input, holdings are an enrichment.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# An equity ticker every downstream vendor can actually resolve: 1-5 letters,
# optionally a share-class suffix (BRK.B, BF-B). Deliberately strict — see
# the module docstring for why (Metron holds non-equity identifiers too).
_TICKER_RE = re.compile(r"^[A-Z]{1,5}([.-][A-Z]{1,2})?$")

DEFAULT_BUCKET = "alpha-engine-research"

# O(1) pointer objects — see module docstring.
MEMBERSHIP_LATEST_KEY = "universe_membership/latest.json"
MEMBERSHIP_DATED_TPL = "universe_membership/{date}/membership.json"

# Metron's held-position artifact.
HOLDINGS_UNIVERSE_KEY = "metron/holdings_universe.json"

# Convenience constants for the two cuts named in config#5809 as of
# 2026-07-30. Not exhaustive — ``crucible-research/scoring/
# universe_membership.py::cuts`` is the live source of truth for what
# exists; pass any cut name as a plain string.
CUT_SCANNER_CANDIDATES = "scanner_candidates"
CUT_ATTRACTIVENESS_TOP_20 = "attractiveness_top_20"


class DecisionSetUnavailable(RuntimeError):
    """The requested decision set could not be resolved.

    Raised rather than degraded-to-wider: a caller that quietly falls back to
    the whole board is exactly the 903-ticker regression config-I5700 removed
    from the RAG corpus and config#5809 exists to kill everywhere else. It
    surfaces only as a vendor bill or a timeout hours later otherwise.
    """


def _get_json(s3_client: Any, bucket: str, key: str) -> dict | None:
    """Return a parsed S3 JSON object, or None when absent/unparseable.

    Callers decide the fail-loud policy — this helper deliberately does not,
    because the two artifacts it serves (membership vs. holdings) have
    different criticality.
    """
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
    except Exception as e:
        logger.warning("[decision_set] could not read s3://%s/%s: %s", bucket, key, e)
        return None
    try:
        return json.loads(obj["Body"].read())
    except Exception as e:
        logger.warning("[decision_set] unparseable JSON at s3://%s/%s: %s", bucket, key, e)
        return None


def _load_holdings(s3_client: Any, bucket: str) -> list[str]:
    """Held tickers from Metron's holdings artifact.

    NON-FATAL on absence, matching ``nousergon-data/collectors/
    daily_news.py``'s posture: a missing Metron artifact narrows coverage
    (held names lose fresh evidence) but cannot corrupt it, and blocking the
    entire resolution on a cross-product artifact is the worse failure. The
    degradation is logged at WARNING naming exactly what is not covered,
    never swallowed silently.
    """
    data = _get_json(s3_client, bucket, HOLDINGS_UNIVERSE_KEY)
    if not data:
        logger.warning(
            "[decision_set] holdings artifact s3://%s/%s unavailable — HELD "
            "NAMES WILL NOT BE COVERED this run (cut only). Not fatal, but "
            "positions go without fresh evidence until it returns.",
            bucket, HOLDINGS_UNIVERSE_KEY,
        )
        return []
    tickers = [str(t).strip().upper() for t in (data.get("tickers") or []) if t]
    logger.info("[decision_set] holdings: %d held ticker(s) (as_of=%s)",
                len(tickers), data.get("as_of"))
    return tickers


def load_decision_set(
    *,
    cut: str,
    bucket: str = DEFAULT_BUCKET,
    s3_client: Any = None,
    run_date: str | None = None,
    include_held: bool = True,
) -> dict:
    """Resolve the decision set for a named ``universe_membership`` cut.

    Returns ``{tickers, counts, run_date, cut, source}``.

    ``cut`` names the ``cuts.<cut>`` entry to resolve (e.g.
    :data:`CUT_SCANNER_CANDIDATES`, :data:`CUT_ATTRACTIVENESS_TOP_20`, or any
    other cut the producer emits). Cut width is not a free choice for every
    caller — e.g. RAG scopes to ``scanner_candidates`` (60) rather than
    ``attractiveness_top_20`` (20) because a downstream challenger arm
    consumes the wider cut and scoping narrower would confound the
    champion/challenger comparison on breadth
    (``champion-challenger-policy.md`` §4). Callers should state, in their
    own docstring, which cut they need and why.

    ``run_date`` pins the dated membership artifact; omitted reads the
    ``latest.json`` pointer. Pin it whenever the caller is part of a dated
    pipeline run, so a resolution and the run it serves cannot disagree
    about which cycle they are in.

    ``include_held`` unions Metron's held-position tickers into the result
    (default ``True``) — see the module docstring for when to disable it.

    Raises :class:`DecisionSetUnavailable` when the membership artifact is
    missing/unparseable or the named cut is missing/empty. Never widens to
    ``signals.json::universe`` — see the module docstring.
    """
    if not cut:
        raise ValueError("cut is required (e.g. 'scanner_candidates')")

    if s3_client is None:
        import boto3
        s3_client = boto3.client("s3")

    key = (
        MEMBERSHIP_DATED_TPL.format(date=run_date) if run_date
        else MEMBERSHIP_LATEST_KEY
    )
    membership = _get_json(s3_client, bucket, key)
    if not membership:
        raise DecisionSetUnavailable(
            f"universe membership artifact s3://{bucket}/{key} is missing or "
            f"unparseable. The scanner writes it every run upstream of every "
            f"producer arm, so absence is a real upstream failure. Refusing "
            f"to fall back to signals.json::universe — that is the "
            f"903-ticker path this resolver exists to remove "
            f"(alpha-engine-config#5809)."
        )

    cuts = membership.get("cuts") or {}
    cut_entry = cuts.get(cut) or {}
    cut_tickers = [str(t).strip().upper() for t in (cut_entry.get("tickers") or []) if t]
    if not cut_tickers:
        raise DecisionSetUnavailable(
            f"membership artifact s3://{bucket}/{key} carries no non-empty "
            f"cuts.{cut!r}. Available cuts: {sorted(cuts.keys())}. Refusing "
            f"to widen (alpha-engine-config#5809)."
        )

    held = _load_holdings(s3_client, bucket) if include_held else []
    candidates = sorted(set(cut_tickers) | set(held))

    # Drop anything no equity source can resolve (see _TICKER_RE). Named in
    # the log rather than silently dropped — a non-equity holding is a real
    # position that will carry no coverage, which a reader of the coverage
    # numbers has to be able to account for.
    tickers = [t for t in candidates if _TICKER_RE.match(t)]
    rejected = [t for t in candidates if not _TICKER_RE.match(t)]
    if rejected:
        logger.warning(
            "[decision_set] dropped %d non-equity identifier(s) no equity "
            "source can resolve: %s — these carry NO coverage",
            len(rejected), rejected,
        )

    logger.info(
        "[decision_set] resolved %d ticker(s) for cut=%r run_date=%s — cut "
        "%d ∪ held %d (source: %s)",
        len(tickers), cut, membership.get("run_date"), len(set(cut_tickers)),
        len(set(held)), cut_entry.get("source") or key,
    )
    return {
        "tickers": tickers,
        "counts": {
            "total": len(tickers),
            cut: len(set(cut_tickers)),
            "held": len(set(held)),
            "rejected_non_equity": len(rejected),
        },
        "run_date": membership.get("run_date"),
        "cut": cut,
        "source": cut_entry.get("source") or key,
    }


def load_decision_set_tickers(
    *,
    cut: str,
    bucket: str = DEFAULT_BUCKET,
    s3_client: Any = None,
    run_date: str | None = None,
    include_held: bool = True,
) -> list[str]:
    """Ticker-list convenience wrapper over :func:`load_decision_set`."""
    return load_decision_set(
        cut=cut, bucket=bucket, s3_client=s3_client, run_date=run_date,
        include_held=include_held,
    )["tickers"]
