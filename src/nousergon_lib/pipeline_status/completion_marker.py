"""The weekly completion marker — and the cycle shape it could not express.

``alpha-engine-config-I8186``.

## What the marker says, and what it cannot say

``s3://alpha-engine-research/_sf_completion/<pipeline>/<run_date>.json`` is
written by the state machine itself, from a ``States.Format`` body::

    {"sf": "...", "execution_arn": "...", "status": "SUCCEEDED",
     "started_at": "...", "completed_at": "...", "cycle_key": "...",
     "substrate_relaunches": 0}

Measured 2026-08-22, that object records ``status: SUCCEEDED`` and **one**
``execution_arn`` — ``watch-rerun-2026-08-22-3``, which entered 14 states and
dispatched ``Director``, ``ScannerLeaderboard`` and two health checks. The
other ~21 stages of the weekly graph were skipped because the *scheduled*
execution had already completed them hours earlier.

**The cycle genuinely completed most of the way.** That is what a mechanical
recovery IS, and the marker is not lying about the cycle. What it cannot
express is that the cycle ran across FOUR executions — so no consumer can tell
a full run from a recovery tail, ``sf_success_rate`` counts one execution where
four ran, and the ``gate:*`` ``Verified-when:`` predicates written against the
marker's mere existence clear on whichever execution happened to reach
``WriteCompletionMarker``.

## Why the marker is AUGMENTED rather than withheld

Two rules pull in opposite directions and both are real:

- ``sf-pipeline-policy.md`` §2.3a — every surface presenting a run's results
  carries its verdict state, and a missing verdict propagates as UNKNOWN,
  never as a pass.
- ``alpha-engine-config-I8186``'s explicit prohibition — *do not "fix" this by
  making ``WriteCompletionMarker`` unreachable from recovery reruns.* A genuine
  recovery that DOES complete the spine must still write the marker; the
  discriminator is the work verdict, never the execution's role or name.

Withholding the marker would call a legitimate recovery a failure. Rewriting
it from a Lambda would replace the SF envelope's *independent* completion
signal — deliberate since ``config-I1724`` — with one that depends on the same
Lambda substrate the pipeline does.

So the state machine keeps writing its envelope claim, now labelled as the
narrow claim it is (``claim: sf_execution_terminal``, ``cycle_verdict:
unknown``), and the coverage sweep — which already reads the cycle to
establish its own denominator — merges the real shape in afterwards under
``cycle``. A consumer that reads only ``status`` is unaffected. A consumer that
wants the verdict reads ``cycle.verdict``, and a marker with **no** ``cycle``
block resolves to UNKNOWN rather than to a pass, which is what
:func:`marker_verdict` returns for it.

## The double-encoding, handled rather than rediscovered

Every object in this prefix is written by ``States.Format`` and comes back off
the wire as a **JSON string containing the JSON object**, not the object. One
``json.loads`` yields a ``str``. Consumers that guard with
``isinstance(marker, dict)`` after a single decode fall through to their
"no usable status field" path on every read, silently, forever.
:func:`decode_marker` unwraps up to two levels and says which it needed.

Public surface:

- :func:`marker_key` / :func:`decode_marker` / :func:`read_marker`
- :func:`merge_cycle_shape` — pure ``(marker, shape) -> marker``.
- :func:`marker_verdict` — the ONE reader every consumer should use.
- :func:`augment_marker` — read, merge, write back.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from .cycle_shape import CycleShape

logger = logging.getLogger(__name__)

__all__ = [
    "CLAIM_CYCLE_VERDICT",
    "CLAIM_SF_EXECUTION_TERMINAL",
    "MARKER_PREFIX",
    "VERDICT_UNKNOWN",
    "augment_marker",
    "decode_marker",
    "marker_key",
    "marker_verdict",
    "merge_cycle_shape",
    "read_marker",
]

MARKER_PREFIX = "_sf_completion"

#: The claim the STATE MACHINE makes by writing the object: this execution
#: reached a real terminal. Narrower than the file's name, and now said so.
CLAIM_SF_EXECUTION_TERMINAL = "sf_execution_terminal"

#: The claim the augmented marker makes: the cycle's shape has been read and
#: is recorded in the ``cycle`` block.
CLAIM_CYCLE_VERDICT = "cycle_verdict"

#: What a marker with no cycle block resolves to. Never ``completed``.
VERDICT_UNKNOWN = "unknown"


def marker_key(pipeline: str, run_date: str, *, prefix: str = MARKER_PREFIX) -> str:
    return f"{prefix}/{pipeline}/{run_date}.json"


def decode_marker(body: bytes | str) -> tuple[dict[str, Any], int]:
    """Return ``(marker, levels_unwrapped)``.

    Two levels, because ``States.Format`` writes a JSON string literal whose
    content is the JSON object. Raises rather than returning ``{}``: an
    unreadable marker is UNREADABLE, and an empty dict would resolve to
    "no verdict recorded", which reads the same as a run that never wrote one.
    """
    parsed: Any = json.loads(body)
    levels = 1
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
        levels = 2
    if not isinstance(parsed, dict):
        raise ValueError(
            f"completion marker decoded to {type(parsed).__name__}, not an object "
            f"(after {levels} json.loads) — the object is malformed"
        )
    return parsed, levels


def read_marker(s3_client: Any, *, bucket: str, key: str) -> dict[str, Any] | None:
    """Return the decoded marker, or ``None`` if it does not exist.

    Only a genuine 404 returns ``None``. Any other failure raises: a denial
    or a transport error rendered as "the marker is absent" is a verdict
    manufactured from a denial.
    """
    try:
        body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as exc:  # noqa: BLE001 — classified, then re-raised
        code = ""
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            code = str((response.get("Error") or {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    marker, levels = decode_marker(body)
    if levels == 2:
        logger.debug("completion marker %s was double-encoded (States.Format)", key)
    return marker


def merge_cycle_shape(marker: dict[str, Any], shape: CycleShape) -> dict[str, Any]:
    """Return a copy of ``marker`` carrying the cycle's real shape. Pure.

    Every field the state machine wrote is preserved untouched — a consumer
    reading ``status`` keeps reading exactly what it read before. What is
    ADDED is the part the envelope could never know:

    - ``claim`` — what the SF's own write asserted, named rather than implied.
    - ``cycle_verdict`` — the three-way verdict, promoted to the top level so
      a consumer needs one field lookup, not a nested walk.
    - ``cycle`` — the full shape: every contributing execution, its role, its
      status, what it entered, and the union.
    """
    merged = dict(marker)
    merged.setdefault("claim", CLAIM_SF_EXECUTION_TERMINAL)
    merged["cycle_verdict"] = shape.verdict.value
    merged["cycle"] = shape.to_dict()
    return merged


def marker_verdict(marker: dict[str, Any] | None) -> str:
    """The ONE reader every consumer of this prefix should use.

    Returns the cycle verdict, or :data:`VERDICT_UNKNOWN` for an absent marker
    and for a marker written before the augmentation existed. Deliberately
    does NOT fall back to ``status``: a bare ``SUCCEEDED`` is the claim that
    could not tell a four-hour full run from a fourteen-minute recovery tail,
    and returning it here would reintroduce that ambiguity behind a function
    whose name promises a verdict (``sf-pipeline-policy.md`` §2.3a: a missing
    verdict propagates as UNKNOWN, never as a pass).
    """
    if not marker:
        return VERDICT_UNKNOWN
    verdict = marker.get("cycle_verdict")
    if isinstance(verdict, str) and verdict:
        return verdict
    return VERDICT_UNKNOWN


def augment_marker(
    shape: CycleShape,
    *,
    s3_client: Any,
    bucket: str = "alpha-engine-research",
    prefix: str = MARKER_PREFIX,
    also_dates: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """Merge the cycle shape into the marker in place. Never raises.

    Returns the merged marker, or ``None`` when there was no marker to
    augment — which is itself information: a cycle with no marker and a
    COMPLETED verdict means the state machine's own write did not happen.

    Fail-soft and LOUD, matching ``krepis.stage_coverage.record_verdict``:
    the sweep's primary deliverable is its own artifact and its own alert, so
    a failure to enrich the marker must not destroy them — but a silent
    swallow would leave every consumer reading a marker that says nothing
    about the cycle, which is the whole defect.
    """
    result = _augment_one(shape, s3_client=s3_client, bucket=bucket, prefix=prefix,
                          run_date=shape.run_date)
    # alpha-engine-config-I8809 migration window: the state machine dual-writes
    # the envelope marker to the legacy calendar partition too, so a consumer
    # still reading that family must not see an un-augmented marker beside an
    # augmented one — an UNKNOWN verdict where the cycle verdict is known is
    # exactly the ambiguity the augmentation exists to remove.
    for extra in also_dates or ():
        if extra and extra != shape.run_date:
            _augment_one(shape, s3_client=s3_client, bucket=bucket, prefix=prefix,
                         run_date=extra)
    return result


def _augment_one(
    shape: CycleShape,
    *,
    s3_client: Any,
    bucket: str,
    prefix: str,
    run_date: str,
) -> dict[str, Any] | None:
    key = marker_key(shape.pipeline, run_date, prefix=prefix)
    try:
        marker = read_marker(s3_client, bucket=bucket, key=key)
    except Exception:  # noqa: BLE001 — fail-soft, recorded at ERROR
        logger.error(
            "completion marker: could not READ s3://%s/%s to augment it", bucket, key,
            exc_info=True,
        )
        return None

    if marker is None:
        logger.warning(
            "completion marker: no object at s3://%s/%s for a cycle whose verdict is "
            "%s — the state machine's own write did not happen",
            bucket,
            key,
            shape.verdict.value,
        )
        return None

    merged = merge_cycle_shape(marker, shape)
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(merged, indent=2, default=str).encode(),
            ContentType="application/json",
        )
    except Exception:  # noqa: BLE001 — fail-soft, recorded at ERROR
        logger.error(
            "completion marker: FAILED to write the augmented marker to s3://%s/%s "
            "— every consumer keeps reading a marker with no cycle verdict",
            bucket,
            key,
            exc_info=True,
        )
    return merged
