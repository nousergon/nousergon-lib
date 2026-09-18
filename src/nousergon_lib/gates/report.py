"""The system-agnostic core of a daily accountability report.

Lifted from `crucible/crucible/morning.py` on its second adoption
(`shared-code-policy`, commissioned by `alpha-engine-config-I10951`), so a
second system — the data collector in `nousergon-data` — can publish a daily
report without a second copy of the defect classes crucible already paid for
in incidents.

What moved is the part that is TRUE OF ANY daily report that reads artifacts
it does not own and pushes a pointer at an operator: the staleness headline,
the three-way present/absent/**denied** classifier, the previous-reading
diff, the clause-withholding mechanism, the wire budget, the delivery call,
the trigger resolution, and the rolling history index. What did NOT move is
any caller's vocabulary — no phase list, no ladder key, no board key, no
forbidden-token set, no tracker repository. Every one of those is a
parameter, which is the whole test of whether the lift was honest.

The reasoning in the docstrings below is the original's, kept in meaning
rather than in words, because it is the part that was paid for in outages
rather than in design:

* **Absent and DENIED are different facts.** The first live crucible run
  (GHA 33766008781, 2026-09-03T14:18Z) died `AccessDenied` reading an
  OPTIONAL artifact, and S3 answers 403 for a MISSING key when the caller
  also lacks `s3:ListBucket` on the prefix. Rendering a denial as absence
  hides an IAM gap behind a normal-looking report.
* **The wire budget is spent on the wire body, and the body is not what a
  caller hands over.** `krepis.alerts.publish` prepends
  ``[SEV] source: `` and the transport truncates on ``len()`` BEFORE
  escaping — measured, a 4117-character body plus a 37-character prefix
  POSTed at 4152 and was refused ``400 message is too long``, which is not
  an entity-parse error, so krepis' plain-text retry never fired and the
  report was not delivered at all. Over budget RAISES; it never truncates.
* **Delivery is not silent** (Brian ruling, `alpha-engine-config-I9916`).
  The first crucible delivery went out ``silent=True``, the manifest read
  ``ok``, and he never saw it. A report the recipient does not see is the
  accountability gap the job exists to close, one layer down.
* **Withheld, never silently dropped.** A clause removed by a caller's
  policy is REPLACED by a marker counting it, so a reader can see that the
  source said something this surface refused to repeat.

**Ordering invariant, owned by the caller and tested on both sides.** The
tracker comment is posted BEFORE the headline is rendered, because the
headline's indispensable content is that comment's permalink. Both tracker
calls raise, so a failed post means the job's manifest is ``failed`` and NO
message is sent.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "TRIGGER_RE",
    "TRIGGER_UNKNOWN",
    "TRIGGER_VARS",
    "WITHHELD_MARKER",
    "MessageTooLongError",
    "DeferPredicate",
    "Move",
    "MovedResult",
    "Read",
    "UndeliveredError",
    "assert_within_budget",
    "deliver",
    "escape",
    "filter_withheld_clauses",
    "history_row",
    "moved_since",
    "read_optional",
    "read_required",
    "render_history_index",
    "resolve_trigger",
    "staleness",
    "transport_prefix",
    "wire_budget",
    "wire_length",
]


#: A trigger name that is safe as one segment of a store key. The value is
#: written to the store as a KEY (``trigger.<name>``), so a predicate over it
#: is an ``exists`` op with no body parse — which is only sound if the name
#: cannot carry a separator.
TRIGGER_RE = re.compile(r"^[a-z][a-z0-9_]*$")

#: What a run whose starter said nothing is recorded as. A DECLARED value,
#: not a swallow: it is written like any other trigger, so a run whose starter
#: is unknown is VISIBLE as unknown rather than missing, and it satisfies no
#: predicate asking for a schedule.
TRIGGER_UNKNOWN = "unknown"

#: Where the trigger is read from, in priority order, before a caller's own
#: override variable is appended by :func:`resolve_trigger`.
#:
#: ``GITHUB_EVENT_NAME`` FIRST, and that ordering is the whole provenance
#: argument: GitHub Actions sets it itself on every run, and ``GITHUB_*`` is a
#: reserved prefix a workflow's own ``env:`` block cannot override. A human
#: who dispatches the report therefore cannot produce evidence saying a
#: schedule did.
TRIGGER_VARS: tuple[str, ...] = ("GITHUB_EVENT_NAME",)

#: What replaces the clauses a caller's policy withholds. The COUNT is
#: rendered in their place: a clause that vanished without a trace is
#: indistinguishable from a source that never carried it, which is the defect
#: the whole instrument exists to remove one layer down.
WITHHELD_MARKER = "[{n} clause{s} withheld — {note}]"

#: What a history index renders where a cell was not filed.
_MISSING_CELL = "—"

#: Recognises the trading-day segment of a history-row key, so a row that
#: could not be PARSED still lands under the day it belongs to rather than
#: under an opaque key. Deliberately a segment match, not a substring search
#: anywhere in the key: a bucket or prefix carrying a date would otherwise
#: capture every row in the listing.
_DAY_IN_KEY_RE = re.compile(r"(?:^|/)(\d{4}-\d{2}-\d{2})(?:/|$)")


class MessageTooLongError(RuntimeError):
    """The rendered body does not fit the wire budget.

    A distinct type because the only correct response is to render LESS, and
    that is a decision for the caller who knows which section is expendable.
    Truncating here would ship a report that reads complete and is not.
    """


class UndeliveredError(RuntimeError):
    """The report was rendered and did not reach the operator.

    A distinct type so a manifest's ``reason`` names the failure rather than a
    transport's stringified internals, and so a caller cannot confuse it with
    a failure to READ the inputs — those two want different responses.
    """


def wire_length(text: str) -> int:
    """How many characters ``text`` occupies on the wire.

    Under ``parse_mode="HTML"`` the caller owns the markup and the transport
    escapes nothing for it, so what a caller renders IS the wire body and
    ``len()`` is exact rather than a lower bound. Kept as a named function,
    not inlined as ``len()`` at every call site, so the budget's unit of
    measure stays one declared thing if a future parse mode needs a different
    rule again.
    """
    return len(text)


def escape(text: str) -> str:
    """Escape ``& < >`` so ``text`` renders literally under HTML parse mode.

    `krepis.telegram.escape_html` is the ONE escaper: a second implementation
    here would let this module's idea of "escaped" drift from what
    `send_message` assumes the caller already did. Imported lazily, like every
    krepis reference in this package — a caller that only RENDERS a report
    should not need an HTTP client on the import path.
    """
    from krepis.telegram import escape_html  # noqa: PLC0415 - lazy on purpose

    return escape_html(text)


def transport_prefix(*, severity: str, source: str) -> str:
    """What `krepis.alerts.publish` puts in FRONT of a body, unescaped.

    Composed from the same two values :func:`deliver` publishes under rather
    than restated as a literal, so the two cannot drift. krepis exposes no
    public formatter (`_format_message` is private, and importing a private
    symbol would make this budget silently wrong the day it is renamed), so
    `tests/test_gates_report.py` PINS this against krepis' real output by
    calling that formatter from the test — a disagreement fails a test
    instead of dropping a report.
    """
    return f"[{severity.upper()}] {source}: "


def wire_budget(*, severity: str, source: str, max_chars: int) -> int:
    """How many characters of BODY fit, once the transport prefix is paid for.

    See the module docstring: the 4152-character refusal that made this
    arithmetic exist was a body of 4117 plus a prefix of 37, and the failure
    mode is total — the report was not delivered at all, and no retry fired.
    """
    return max_chars - wire_length(transport_prefix(severity=severity, source=source))


def assert_within_budget(body: str, *, budget: int) -> None:
    """RAISE when ``body`` does not fit ``budget``. Never truncate.

    A truncated report is worse than an absent one: it arrives, it reads
    complete, and the section it silently dropped is the one nobody knows to
    go look for. Raising makes the job's manifest ``failed``, which the
    absence/failure sweeps already watch.
    """
    length = wire_length(body)
    if length > budget:
        raise MessageTooLongError(
            f"the rendered body is {length} characters and the budget is {budget} — "
            f"{length - budget} over. Render less; truncating here would deliver a "
            "report that reads complete and is not."
        )


@dataclass(frozen=True)
class Read:
    """One optional-artifact read: present, absent/corrupt, or DENIED.

    Three outcomes because absent and denied are different facts about an
    artifact this job does not own and cannot write. Reporting a denied read
    as absent hides an IAM gap behind a normal-looking report, and it is
    exactly what happened live on 2026-09-03 before the fix this lift carries
    — S3 returns 403 for a missing key when the caller also lacks
    ``s3:ListBucket`` on the prefix, so absence is a LISTING, not a get.
    """

    document: dict[str, Any] | None
    reason: str | None
    denied_code: str | None

    @property
    def denied(self) -> bool:
        """True only when the read FAILED because it could not be reached."""
        return self.denied_code is not None

    @property
    def absent(self) -> bool:
        """True when there is no document and the read was not denied."""
        return self.document is None and not self.denied


def _client_error_code(exc: BaseException) -> str | None:
    """The service failure code for a botocore read failure, or ``None``.

    ``None`` means "not a botocore failure at all" — the caller re-raises in
    that case, so this stays a classifier and never widens what
    :func:`read_optional` swallows. Read off the exception's own ``response``
    mapping rather than by importing botocore, so a caller whose store is not
    S3-backed never pulls the SDK onto the import path and a test can build
    the shape without moto.
    """
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        if isinstance(error, dict):
            return str(error.get("Code") or "") or type(exc).__name__
        return type(exc).__name__
    if type(exc).__module__.startswith("botocore."):
        return type(exc).__name__
    return None


def _fetch_document(store: Any, key: str) -> dict[str, Any]:
    """``key`` as a JSON object, however this store spells "fetch".

    The contract names ``get_json``; `nousergon_lib.gates.store.GateStore` —
    the protocol this package's engine already reads through — names
    ``get_bytes``. Both are accepted so the two halves of the same package
    agree about what a store is, and neither adopter has to carry an adapter
    whose only job is renaming one method.
    """
    getter = getattr(store, "get_json", None)
    if callable(getter):
        document = getter(key)
    else:
        raw = store.get_bytes(key)
        document = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    if not isinstance(document, dict):
        raise ValueError(
            f"{key} parsed as {type(document).__name__}, not an object — a reading whose "
            "shape is not a document cannot be quoted as one."
        )
    return document


def read_optional(store: Any, key: str) -> Read:
    """Read ``key``. Never raises for absent, corrupt or denied.

    FAILURE MODES SWALLOWED: a missing key, a malformed document, and a
    botocore read failure that is not a not-found (``AccessDenied``, a
    throttle, a network failure before the request reached the service) are
    all REPORTED rather than raised, because a corrupt or unreachable
    optional artifact must not stop today's report from going out — the
    report is the thing that would tell somebody it is broken. RECORDING
    SURFACE: ``.reason`` and ``.denied_code``, which a caller renders verbatim
    into the delivered message, so every swallow here is visible on the
    operator's phone rather than only in a log.

    A ``NoSuchKey``/``404``-coded failure is still ABSENT, not denied: a store
    that normalizes that shape to ``KeyError`` and one that re-raises the
    client error reach the same honest answer rather than a spurious "denied".

    Anything that is not a recognised absence and not a botocore failure
    PROPAGATES: a classifier that swallowed every exception would report a
    defect in this job's own configuration as a fact about somebody else's
    artifact.
    """
    try:
        document = _fetch_document(store, key)
    except (KeyError, FileNotFoundError):
        return Read(None, f"absent at {key}", None)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        return Read(None, f"unreadable at {key}: {exc}", None)
    except Exception as exc:  # reclassified below; re-raised unless botocore named it
        code = _client_error_code(exc)
        if code is None:
            raise
        if code in ("NoSuchKey", "NotFound", "404"):
            return Read(None, f"absent at {key}", None)
        return Read(None, f"unreadable at {key}: {code}", code)
    return Read(document, None, None)


def read_required(store: Any, key: str) -> dict[str, Any]:
    """Read ``key``, or RAISE.

    The artifact the whole report is ABOUT is not optional: a missing, corrupt
    or denied read of it must reach the caller's manifest as ``failed``,
    naming why, rather than becoming a report that renders around the hole it
    was supposed to describe.
    """
    return _fetch_document(store, key)


def staleness(
    *,
    generated_at: str | None,
    now: dt.datetime,
    stale_after: dt.timedelta,
    label: str,
    prefix: str = "STALE",
) -> str | None:
    """The bolded first-line staleness headline, or ``None``. Never a footnote.

    An UNPARSEABLE — or absent — ``generated_at`` is STALE. Assuming freshness
    from a timestamp nobody could read is a positive claim asserted on no
    evidence, and it is the one direction of this error that leaves a reader
    confident about an age that was never measured.

    ``prefix`` is the alarm word the line opens with. It is a parameter and
    not a constant because a caller migrating onto this function has a live
    surface whose first line an operator already recognises, and changing that
    word as a side effect of consolidating two implementations is exactly the
    un-shipped behaviour `shared-code-policy` §3.1 is about. It is the ALARM,
    not a label: a caller may spell it for its own artifact, never soften it.
    """
    if generated_at is None or not str(generated_at).strip():
        return f"{prefix}: {label} carries no generated_at, so the age of every reading below is unknown."
    try:
        generated = dt.datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
    except ValueError:
        return (
            f"{prefix}: {label} generated_at={generated_at!r} is not a timestamp, so the "
            "age of every reading below is unknown."
        )
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    age = now - generated
    if age <= stale_after:
        return None
    hours = age.total_seconds() / 3600
    return f"{prefix}: {label} was generated {generated_at} — {hours:.0f}h ago. Every reading below is that old."


@dataclass(frozen=True)
class Move:
    """One row's difference between two readings, with both states named.

    ``was``/``now`` carry the sentinels ``ABSENT`` and ``VANISHED`` rather
    than ``None``, so a row that entered or left the board is a stated fact
    and never an empty cell a reader skims past.
    """

    row_id: Any
    was: str
    now: str

    def render(self) -> str:
        """The one-line rendering both the diff and the deferred list use."""
        return f"- {self.row_id}: {self.was} -> {self.now}"


#: The signature of the predicate :func:`moved_since` sets rows aside with.
#: It is handed the FULL before and after rows — ``None`` where that side has
#: none — rather than the two states, because the field a caller classifies on
#: (a scheduling state, a lifecycle marker) is usually not the field the diff
#: is about.
DeferPredicate = Callable[["dict[str, Any] | None", "dict[str, Any] | None"], bool]


@dataclass(frozen=True)
class MovedResult:
    """What changed between two readings — or why that cannot be said.

    ``cannot_say`` is set when the PREVIOUS reading could not be read, and in
    that case ``count`` is ``None`` rather than ``0``: reporting no movement
    over a failed comparison is a positive claim asserted on no evidence, and
    "nothing moved" is exactly the sentence a reader would act on.

    ``deferred`` holds the differences a caller's ``defer`` predicate set
    aside — counted and rendered separately, never dropped. A deferred row is
    a row this surface has decided is not NEWS; it is not a row it has decided
    is not THERE.
    """

    lines: list[str]
    cannot_say: str | None
    count: int | None
    moves: tuple[Move, ...] = ()
    deferred: tuple[Move, ...] = ()

    @property
    def deferred_lines(self) -> list[str]:
        """The set-aside differences, rendered like the moves they are not."""
        return [move.render() for move in self.deferred]

    @property
    def deferred_count(self) -> int | None:
        """How many differences were set aside, or ``None`` when cannot_say.

        ``None`` for the same reason ``count`` is: over a comparison that did
        not happen, zero is a claim rather than a measurement.
        """
        return None if self.cannot_say is not None else len(self.deferred)


def moved_since(
    *,
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    previous_reason: str | None,
    row_id_key: str = "id",
    state_key: str = "state",
    defer: DeferPredicate | None = None,
) -> MovedResult:
    """The previous-reading diff, old -> new. Never an absolute-state retelling.

    A row present before and gone now reads ``VANISHED``; a row that appeared
    reads ``ABSENT`` on the left. Both are named rather than skipped, because
    a row LEAVING a board is the change most worth seeing and a diff that only
    reports rows present in both would be silent about it.

    **``defer`` separates a state change from a schedule phase.**
    `alpha-engine-config-I10872`: a blind state diff counted 30 component rows
    that were RUNNING at one render as having "moved" against a board rendered
    after the work finished — a number that is arithmetically correct and
    reads as news when none of it is. A caller passes the predicate for its
    own vocabulary (crucible's is
    ``crucible.console.classify.not_yet_due`` over ``component_state``), and
    the rows it names go to :attr:`MovedResult.deferred` instead of being
    counted as moves. They are RENDERED, separately and with their count — a
    deferred row dropped silently would be the same blindness pointed the
    other way.

    The predicate is handed the whole rows, so it can classify on a field the
    diff itself does not read; a caller with no such field passes nothing and
    gets the flat diff unchanged.
    """
    if previous is None:
        reason = previous_reason or "the previous reading could not be read"
        sentence = f"cannot say — {reason}"
        return MovedResult([sentence], sentence, None)
    before = {row.get(row_id_key): row for row in previous.get("rows", []) if isinstance(row, dict)}
    after = {row.get(row_id_key): row for row in current.get("rows", []) if isinstance(row, dict)}
    moves: list[Move] = []
    deferred: list[Move] = []
    for row_id in sorted(set(before) | set(after), key=str):
        was_row, now_row = before.get(row_id), after.get(row_id)
        was = str(was_row.get(state_key)) if was_row is not None else "ABSENT"
        now = str(now_row.get(state_key)) if now_row is not None else "VANISHED"
        if was_row is not None and now_row is not None and was == now:
            continue
        move = Move(row_id, was, now)
        (deferred if defer is not None and defer(was_row, now_row) else moves).append(move)
    return MovedResult(
        [move.render() for move in moves],
        None,
        len(moves),
        tuple(moves),
        tuple(deferred),
    )


def filter_withheld_clauses(
    text: str,
    *,
    tokens: frozenset[str],
    separator: str = "; ",
    note: str = "policy",
) -> str:
    """Remove the clauses of ``text`` carrying a forbidden token, visibly.

    **Withheld, never silently dropped.** The count of removed clauses is
    rendered in their place (:data:`WITHHELD_MARKER`), so a reader can see
    that the source said something this surface refused to repeat and go read
    the source. A clause that vanished without a trace is indistinguishable
    from a source that never carried it.

    **Clean text is passed through byte for byte.** A scrubber that rewrote
    detail it had no objection to would make the report an unfaithful copy of
    the thing it reports on, which is worse than the defect it fixes. Escaping
    is NOT done here: the same filtered text is rendered into a Markdown
    comment body and into an HTML wire body, and only the second one escapes
    — a shared escape here would print literal ``&amp;`` in the first.

    ``note`` names WHOSE rule withheld it, because a marker that says only
    "withheld" tells a reader nothing about where to go and ask.
    """
    clauses = text.split(separator)
    kept = [c for c in clauses if not any(t in f" {c.lower()} " for t in tokens)]
    removed = len(clauses) - len(kept)
    if not removed:
        return text
    marker = WITHHELD_MARKER.format(n=removed, s="" if removed == 1 else "s", note=note)
    return separator.join([*kept, marker]) if kept else marker


def resolve_trigger(
    *,
    override_var: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """What started this run, as a key-safe name.

    Never inferred from the clock or the calendar: a dispatched run is just as
    live as a scheduled one, so nothing already in a manifest can answer "did
    a human start this".

    ``GITHUB_EVENT_NAME`` wins, then ``override_var`` — see
    :data:`TRIGGER_VARS` for why that order is the provenance argument.
    Returns :data:`TRIGGER_UNKNOWN` when the invocation said nothing. A value
    that does not match :data:`TRIGGER_RE` RAISES: it becomes a store-key
    segment, and refusing is the only reading that is not a guess.

    ``environ`` substitutes the environment wholesale. A test that mutates the
    real ``os.environ`` leaks into whatever runs next in the same process, and
    the leak is invisible in exactly the direction that matters — a stray
    ``GITHUB_EVENT_NAME`` makes a later assertion about ``unknown`` pass for
    the wrong reason.
    """
    source = os.environ if environ is None else environ
    names = TRIGGER_VARS + ((override_var,) if override_var else ())
    for var in names:
        value = (source.get(var) or "").strip()
        if not value:
            continue
        if not TRIGGER_RE.match(value):
            raise ValueError(
                f"{var}={value!r} is not a usable trigger name (must match "
                f"{TRIGGER_RE.pattern}). It becomes a store-key segment; refusing is the "
                "only reading that is not a guess about what started this run."
            )
        return value
    return TRIGGER_UNKNOWN


def _krepis_publish(*args: Any, **kwargs: Any) -> Any:
    """The real transport, imported at call time.

    Lazy so that rendering a report — which every unit test and every
    ``--dry-run`` does — never pulls an SNS/HTTP client onto the import path.
    Named as a module attribute so a test substitutes the transport here
    rather than through an argument no production caller would ever pass.
    """
    from krepis.alerts import publish  # noqa: PLC0415 - lazy on purpose

    return publish(*args, **kwargs)


def _destination(*, destination: str | None, destination_override_var: str | None) -> str:
    """The destination :func:`deliver` publishes to. EXPLICIT, never resolved.

    `krepis.alerts.resolve_destination` routes a non-``error`` severity to the
    LOG chat when one is configured, and reaches the operator chat only
    through its documented "no log chat configured" fallback. A report lands
    on the operator's chat today by the ABSENCE of a fleet-wide setting, and
    the day someone configures one it would move off that chat with the job's
    manifest still reading ``ok`` — a delivery that silently changed audience.
    So the destination is passed by name, from krepis' own constant rather
    than a literal here, and the routing is a decision in the diff instead of
    a property of the environment.
    """
    if destination:
        return destination
    if destination_override_var:
        override = (os.environ.get(destination_override_var) or "").strip()
        if override:
            return override
    from krepis.alerts import DESTINATION_OPERATOR_CHAT  # noqa: PLC0415 - lazy on purpose

    return DESTINATION_OPERATOR_CHAT


def deliver(
    message: str,
    *,
    severity: str,
    source: str,
    console_artifact: str,
    destination: str | None = None,
    parse_mode: str = "HTML",
    destination_override_var: str | None = None,
    transport: Callable[..., Any] | None = None,
) -> str:
    """Send ``message`` on the operator channel, or RAISE. Returns where it went.

    **The return value is the destination the transport says it reached**, not
    the one this function asked for. A caller records it on its run manifest,
    and the two are not the same claim: krepis resolves what it was handed,
    and a manifest that recorded the REQUEST would keep reading ``ok`` on the
    day the resolution changed underneath it — which is the silent
    audience-change this function's explicit ``destination`` already refuses
    one step earlier. It falls back to the transport's own name when the
    result does not say, never to the requested value.

    ``transport`` substitutes the publish call for one test or one tier. The
    module attribute :func:`_krepis_publish` is substitutable too; the
    argument exists so a CALLER — not only a monkeypatching test — can route
    one delivery without mutating module state a sibling test then inherits.

    **Telegram only, ``sns=False``.** A daily digest is not a page, and
    routing one through the page path is how a page channel becomes the
    channel someone mutes.

    **It notifies** (Brian ruling, `alpha-engine-config-I9916`). The first
    crucible delivery went out ``silent=True`` into the operator chat under
    the day's CI-failure alerts; the manifest read ``ok`` and Brian said he
    had not received it. One push per day is the accepted cost.

    **No dedup key.** `krepis.alerts.publish` suppresses a repeat within its
    window, and this message is SUPPOSED to arrive every day even when it is
    byte-identical to yesterday's — a report that stops arriving when nothing
    changed is indistinguishable from a report that stopped arriving.

    **Undelivered raises.** ``raise_on_total_failure=True`` covers a transport
    that reached nothing; the explicit check below covers the case krepis
    calls a success — ``any_ok`` is True on a muted or dedup-suppressed
    publish by its documented contract, and neither of those put the report in
    anybody's hands.
    """
    publish = _krepis_publish if transport is None else transport
    result = publish(
        message,
        severity=severity,
        source=source,
        sns=False,
        telegram=True,
        silent=False,
        dedup_key=None,
        destination=_destination(
            destination=destination,
            destination_override_var=destination_override_var,
        ),
        console_artifact=console_artifact,
        raise_on_total_failure=True,
        parse_mode=parse_mode,
    )
    if not hasattr(result, "any_ok"):
        raise TypeError(
            f"{type(result).__name__} carries no `any_ok`; a transport result that cannot "
            "say whether the report was delivered cannot be recorded as delivered."
        )
    dedup_skipped = bool(getattr(result, "dedup_skipped", False))
    muted = bool(getattr(result, "muted", False))
    if not result.any_ok or dedup_skipped or muted:
        raise UndeliveredError(
            f"the {source} report reached nobody (any_ok={result.any_ok}, "
            f"dedup_skipped={dedup_skipped}, muted={muted}). The delivery IS the "
            "deliverable; a rendered report nobody received is the accountability gap "
            "this job closes."
        )
    return str(getattr(result, "telegram_destination", None) or "telegram")


def history_row(
    *,
    trading_day: str,
    delivered_local: str,
    columns: dict[str, str],
    update_url: str,
) -> dict[str, Any]:
    """The compact facts one delivery contributes to the history index.

    Built by a caller from the SAME inputs the full update renders from —
    never a second read — so the index cannot disagree with the comment it
    links to about what that day's reading said. ``columns`` is already
    RENDERED text keyed by column title: the index is a view, and a view that
    re-derived a cell would be a second opinion.
    """
    return {
        "trading_day": str(trading_day),
        "delivered_local": str(delivered_local),
        "columns": {str(k): str(v) for k, v in columns.items()},
        "update_url": str(update_url),
    }


def _day_of(key: str, document: dict[str, Any] | None) -> str:
    """Which trading day a history row belongs to, readable or not.

    A row that could not be PARSED still has to land under its day, or a
    corrupt row would sit beside the good row it replaced and the index would
    show a day twice. The key is the only evidence left in that case, so a
    ``YYYY-MM-DD`` path SEGMENT is read out of it; a key carrying none falls
    back to the key itself, which is ugly and true rather than tidy and
    invented.
    """
    if document is not None:
        day = document.get("trading_day")
        if day:
            return str(day)
    match = _DAY_IN_KEY_RE.search(key)
    return match.group(1) if match else key


def render_history_index(
    *,
    rows: Iterable[tuple[str, dict[str, Any] | None, str | None]],
    column_titles: list[str],
    title: str = "Daily update history",
) -> str:
    """The rolling issue's regenerated BODY: a newest-first history index.

    One row per trading day — the LAST entry for that day wins, so a
    re-delivery replaces its row rather than growing the table — carrying that
    day's rendered cells and a link to the update holding the detail.

    ``rows`` is ``(key, parsed document or None, problem or None)`` in the
    listing's own ASCENDING key order, which is what makes "last wins" mean
    "most recent delivery".

    **A corrupt row gets its own ``unreadable:`` row, never a drop.** The
    index is a VIEW over each day's durable update; a fault in one day's row
    must not blank the rest of the history, and must not be invisible either.
    """
    latest: dict[str, dict[str, Any]] = {}
    faulted: dict[str, str] = {}
    for key, document, problem in rows:
        day = _day_of(key, document)
        if document is not None:
            latest[day] = document
            faulted.pop(day, None)
        else:
            faulted[day] = problem or "unreadable for an unstated reason"
            latest.pop(day, None)

    header = ["trading day", "delivered", *column_titles, "update"]
    lines = [f"# {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    days = sorted(set(latest) | set(faulted), reverse=True)
    for day in days:
        if day in faulted:
            filler = " | ".join([_MISSING_CELL] * (len(header) - 2))
            lines.append(f"| {day} | unreadable: {faulted[day]} | {filler} |")
            continue
        row = latest[day]
        columns = row.get("columns") or {}
        cells = [str(columns.get(name, _MISSING_CELL)) for name in column_titles]
        url = row.get("update_url")
        link = f"[update]({url})" if url else _MISSING_CELL
        lines.append(f"| {day} | {row.get('delivered_local', _MISSING_CELL)} | " + " | ".join(cells) + f" | {link} |")
    if not days:
        filler = " | ".join([_MISSING_CELL] * (len(header) - 1))
        lines.append(f"| _no delivery has filed a history row yet_ | {filler} |")
    return "\n".join(lines)
