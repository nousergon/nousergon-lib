"""One run record per execution of one data-collection unit.

`data_collection_plan_260914.md` §2 row 7 and §4.4; `alpha-engine-config-I10773`.

**What this is.** :func:`run_unit` runs a callable as one named unit and writes a
``data_run_manifest.v1`` record for it — on the success path, on the failure
path, and on the path where the unit correctly had nothing to do. The record
carries what the run read, what it published, how many rows landed, what it
spent, what compute it burned, and where its full logs are.

**Why it is here rather than in the collector.** It is the second adoption of
crucible's ``crucible.runner.run_job`` + ``run_manifest.v2``
(`shared-code-policy` §2: the second adoption is the signal, not the third).
The data collector is the second consumer, so the pattern is lifted into the
shared library rather than copied. It is a **sibling** of crucible's schema, not
a copy: the collector keys its runs by ``unit_id`` (the audit's D01–D46) rather
than by a crucible ``job``, identifies published objects by S3 ETag rather than
by content hash, and carries a third status crucible genuinely does not have —
``not_applicable``, for a cycle in which the unit was correctly asked to do
nothing. Crucible's re-import onto this module is a tracked follow-up; until it
lands crucible is unchanged, and for the data component this module is the only
implementation.

**The rule the whole record exists for** (`observability-policy` §3.1): *the
failure path writes the same telemetry as the success path, except the
completion claim.* A dying run persists its work, its spend, its inputs and its
cause of death; it never advances an artifact a detector reads as proof it
finished. So the manifest write lives in a ``finally``, and :func:`run_unit`
re-raises **after** the record is durable.

**No third success state.** ``status`` is ``ok`` | ``failed`` |
``not_applicable`` and nothing else. ``not_applicable`` is not a softer ``ok``:
it means the unit did not run because it correctly had nothing to do, it
requires a reason from :data:`NOT_APPLICABLE_REASONS`, and it is *counted* — a
unit that answers ``not_applicable`` every cycle is a unit that has stopped
working, and the board can see that only because the non-run left a record. A
run that produced a partial or defective artifact is ``failed``.

Usage::

    from nousergon_lib.run_manifest import S3ManifestSink, run_unit

    sink = S3ManifestSink(bucket="alpha-engine-research")

    def collect(ctx):
        ctx.record_input("metron/holdings_universe.json", etag=etag)
        df = fetch()
        ctx.rows_in = len(raw)
        ctx.reject("unpriced_symbol", 7)
        key = write_parquet(df)
        ctx.record_output(key, rows_out=len(df), etag=put_etag)

    result = run_unit("D19", collect, sink=sink, trigger="scheduled",
                      trading_day="2026-09-14", log_location=log_group)

A unit with nothing to do raises :class:`NotApplicable` from inside ``fn``; the
manifest is still written, with the declared reason.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

# `resolve_code_sha` / `new_run_id` moved to `run_identity` (I10831 deliverable
# 2) — neither reads or writes anything schema-shaped, so crucible's
# `run_manifest.v2` can import them directly without coupling to this module's
# `data_run_manifest.v1` contract. `_REAL_SHA_RE` stays imported (not
# re-derived) so this module's own sha-shape refusal below cannot drift from
# `run_identity`'s.
from nousergon_lib.run_identity import (
    _REAL_SHA_RE,
    CODE_SHA_ENV,
    CodeShaError,
    new_run_id,
    resolve_code_sha,
)

__all__ = [
    "CODE_SHA_ENV",
    "DEFAULT_MANIFEST_PREFIX",
    "NOT_APPLICABLE_REASONS",
    "SCHEMA_VERSION",
    "TRIGGERS",
    "VERSION_CAPTURES",
    "CodeShaError",
    "LocalDirManifestSink",
    "ManifestSink",
    "NotApplicable",
    "ObjectVersion",
    "S3ManifestSink",
    "UnitRun",
    "UnitRunResult",
    "manifest_key",
    "new_run_id",
    "resolve_code_sha",
    "run_unit",
]

logger = logging.getLogger(__name__)

#: The contract this module writes. Matches the schema's ``schema_version``
#: const, asserted by ``tests/test_run_manifest.py`` so the two cannot drift.
SCHEMA_VERSION = "data_run_manifest.v1"

#: Where the records live, per plan §2 row 7. The full key is
#: ``{prefix}/{unit_id}/{trading_day}/{run_id}.json``.
DEFAULT_MANIFEST_PREFIX = "data_collection/runs"

#: The CLOSED reason list a ``not_applicable`` status draws from. It grows only
#: by PR, with the cycle that needed it named. A free-text reason is how a unit
#: quietly stops being graded (`observability-policy` §3.5's N/A taxonomy is the
#: same rule one level up).
#:
#: One line of semantics per member (`alpha-engine-config-I10831`):
#:
#: * ``not_a_trading_day`` — the trading-calendar axis says this cycle has no
#:   session at all (a weekend, a market holiday).
#: * ``no_new_data_declared`` — an upstream explicitly declared there is
#:   nothing new for THIS run to collect (a vendor feed with no fresh rows, a
#:   target date already published). Prefer the two more specific members
#:   below when the real cause is one of them; this stays the catch-all for
#:   every other "declared nothing new" shape.
#: * ``disabled_by_declaration`` — the unit itself is switched off by a
#:   standing config declaration (a collector's ``enabled: false`` in
#:   ``config.yaml``), independent of what any upstream published this cycle.
#:   The non-run is an operator decision, not a data observation.
#: * ``outside_session_window`` — the unit's own schedule fires more often
#:   than its declared window (a 5-minute intraday timer gated to NYSE market
#:   hours; a phase gated to run only before a cutoff time of day), and this
#:   tick landed outside it. The non-run is a clock fact, not a data
#:   observation or an operator decision.
NOT_APPLICABLE_REASONS: frozenset[str] = frozenset(
    {
        "not_a_trading_day",
        "no_new_data_declared",
        "disabled_by_declaration",
        "outside_session_window",
    }
)

#: How an execution was started. Closed: an unknown trigger is a run nobody can
#: attribute, and ``manual`` exists so a hand-run repair is COUNTED as one
#: rather than disguised as a schedule (plan §4.4).
TRIGGERS: frozenset[str] = frozenset({"scheduled", "on_demand", "manual", "gha", "backfill"})

#: Optional environment declarations for the compute row, exported by the box
#: shell that launched the workload.
_INSTANCE_TYPE_ENV = "NE_DATA_INSTANCE_TYPE"
_LIFECYCLE_ENV = "NE_DATA_LIFECYCLE"
_ESCALATED_ENV = "NE_DATA_ESCALATED_ON_DEMAND"
_REGION_ENVS = ("AWS_REGION", "AWS_DEFAULT_REGION")

_UNIT_ID_RE = re.compile(r"^D[0-9]{2}[A-Z]?$")


class NotApplicable(Exception):
    """Raised by a unit body that correctly had nothing to do this cycle.

    ``reason`` must be a member of :data:`NOT_APPLICABLE_REASONS`. The manifest
    is still written — a non-run that leaves no record is indistinguishable
    from a unit that silently stopped running, which is the failure the whole
    run-record objective exists to end.
    """

    def __init__(self, reason: str, detail: str = ""):
        if reason not in NOT_APPLICABLE_REASONS:
            raise ValueError(
                f"not_applicable reason {reason!r} is not in the closed list "
                f"{sorted(NOT_APPLICABLE_REASONS)}. The list grows only by PR, with "
                "the cycle that needed the new reason named — a free-text reason is "
                "how a unit quietly stops being graded."
            )
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


# ---------------------------------------------------------------------------
# Sinks — where a manifest lands.
# ---------------------------------------------------------------------------


class ManifestSink(Protocol):
    """Where :func:`run_unit` puts the record.

    A protocol rather than a hard-wired ``boto3`` call so a test, a local
    rehearsal and a future non-S3 store are the same code path as production
    (`principles.md` §2.8).
    """

    def write(self, key: str, payload: bytes) -> str | None:
        """Persist ``payload`` at ``key``; return the object's ETag if it has one."""


#: How an output's ``etag`` / ``version_id`` / ``bytes`` came to be on the
#: record. Closed, and written on every output so a ``null`` version is never
#: ambiguous between "nobody looked" and "the object was not there":
#:
#: * ``caller`` — the write site passed every field itself.
#: * ``head_object`` — measured by a HeadObject against the sink's store right
#:   after the write was recorded (alpha-engine-config-I10892).
#: * ``object_absent`` — the HeadObject answered 404: the key is not a single
#:   object in the sink's bucket (an ArcticDB library reference such as
#:   ``arcticdb/universe``, a prefix, or a key in another bucket).
#: * ``unavailable`` — the HeadObject failed for any other reason; the error is
#:   carried in ``version_capture_error``.
#: * ``not_captured`` — the sink has no way to look (a local-directory sink, a
#:   hand-built :class:`UnitRun`).
VERSION_CAPTURES: frozenset[str] = frozenset(
    {"caller", "head_object", "object_absent", "unavailable", "not_captured"}
)


@dataclass(frozen=True)
class ObjectVersion:
    """What a store says about one object as it stands now."""

    etag: str | None
    version_id: str | None
    bytes: int | None


def _clean_etag(value: Any) -> str | None:
    return value.strip('"') if isinstance(value, str) else None


def _split_s3_uri(key: str, default_bucket: str) -> tuple[str, str]:
    """``s3://bucket/key`` addresses its own bucket; a bare key is the sink's."""
    if key.startswith("s3://"):
        bucket, _, rest = key[len("s3://") :].partition("/")
        return bucket, rest
    return default_bucket, key


@dataclass
class S3ManifestSink:
    """The production sink: one JSON object per run under the declared prefix.

    It also answers :meth:`head` — an output object's version as it stands
    right after the unit recorded it — so every ``record_output`` carries the
    ``etag`` / ``version_id`` / ``bytes`` a later reader needs to fetch THAT
    object rather than whatever a later run overwrote it with
    (alpha-engine-config-I10892). ``head_key`` maps a recorded output key to the
    physical key to look at, for a caller whose writes are redirected below the
    S3 client (a shadow run's output root); the default is the key itself.
    HeadObject needs ``s3:GetObject`` on the output key; a role without it
    records ``version_capture: unavailable`` rather than failing the unit.

    Every identity that runs a unit needs ``s3:PutObject`` on
    ``<bucket>/data_collection/runs/*`` — this is NOT implied by a unit's own
    ``writes[]`` grants. The earlier claim here, that collector identities
    already held ``alpha-engine-research/*``, was false for the dashboard box's
    ``alpha-engine-dashboard-role``: D36 daily-news wrote all of its data and
    then died on AccessDenied at this sink (2026-09-15). The grant is asserted
    per writer role in ``nous-ergon-ops``.
    """

    bucket: str
    prefix: str = DEFAULT_MANIFEST_PREFIX
    s3_client: Any = None
    head_key: Callable[[str], str] | None = None

    def _client(self) -> Any:
        if self.s3_client is None:
            import boto3  # local import: the lib core stays importable without boto3

            self.s3_client = boto3.client("s3")
        return self.s3_client

    def write(self, key: str, payload: bytes) -> str | None:
        resp = self._client().put_object(
            Bucket=self.bucket,
            Key=key,
            Body=payload,
            ContentType="application/json",
        )
        etag = resp.get("ETag") if isinstance(resp, Mapping) else None
        return etag.strip('"') if isinstance(etag, str) else None

    def head(self, key: str) -> ObjectVersion | None:
        """The object's current ETag, VersionId and size, or ``None`` on a 404.

        Any other failure RAISES; :meth:`UnitRun.record_output` decides what a
        failed look means for the record. ``version_id`` is ``None`` when the
        bucket is unversioned (S3 omits the header, or returns ``"null"`` for an
        object written before versioning was enabled).
        """
        physical = self.head_key(key) if self.head_key is not None else key
        bucket, object_key = _split_s3_uri(physical, self.bucket)
        try:
            resp = self._client().head_object(Bucket=bucket, Key=object_key)
        except Exception as exc:
            response = getattr(exc, "response", None)
            code = ""
            if isinstance(response, Mapping):
                code = str((response.get("Error") or {}).get("Code") or "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise
        version = resp.get("VersionId")
        size = resp.get("ContentLength")
        return ObjectVersion(
            etag=_clean_etag(resp.get("ETag")),
            version_id=version if isinstance(version, str) and version and version != "null" else None,
            bytes=int(size) if isinstance(size, int) and not isinstance(size, bool) else None,
        )


@dataclass
class LocalDirManifestSink:
    """A local-directory sink for rehearsals and tests. Same key shape as S3."""

    root: str
    prefix: str = DEFAULT_MANIFEST_PREFIX

    def write(self, key: str, payload: bytes) -> str | None:
        import pathlib

        path = pathlib.Path(self.root) / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return None


# ---------------------------------------------------------------------------
# Identity and environment.
#
# `new_run_id` and `resolve_code_sha` live in `run_identity` (I10831
# deliverable 2) and are imported at the top of this module; re-exported here
# via `__all__` so existing importers of `nousergon_lib.run_manifest` keep
# working unchanged.
# ---------------------------------------------------------------------------


def _resolve_compute() -> dict[str, Any]:
    """The compute row, from what the box declared — never guessed.

    An undeclared environment is a laptop or CI run, and says so: ``local``,
    not spot, not escalated. That is a measured statement about where this ran,
    which is exactly what a cost roll-up needs it to be.
    """
    instance_type = os.environ.get(_INSTANCE_TYPE_ENV) or "local"
    lifecycle = (os.environ.get(_LIFECYCLE_ENV) or "").strip().lower()
    region = next((os.environ[v] for v in _REGION_ENVS if os.environ.get(v)), None)
    return {
        "instance_type": instance_type,
        "spot": lifecycle == "spot",
        "escalated_on_demand": (os.environ.get(_ESCALATED_ENV) or "").strip().lower() in ("1", "true", "yes"),
        "interruptions": 0,
        "region": region,
    }


def manifest_key(unit_id: str, trading_day: str, run_id: str, prefix: str = DEFAULT_MANIFEST_PREFIX) -> str:
    """``{prefix}/{unit_id}/{trading_day}/{run_id}.json`` — plan §2 row 7."""
    return f"{prefix.rstrip('/')}/{unit_id}/{trading_day}/{run_id}.json"


# ---------------------------------------------------------------------------
# The run context — what a unit body records about itself.
# ---------------------------------------------------------------------------


@dataclass
class UnitRun:
    """The handle a unit body records its lineage on.

    Every mutator here is additive and cheap, so a body that dies halfway still
    leaves behind everything it had established — which is the point: the
    failure manifest is only useful if it carries the inputs and partial counts
    that explain the failure.
    """

    run_id: str
    unit_id: str
    trading_day: str
    calendar_date: str
    trigger: str
    started: dt.datetime
    code_sha: str
    log_location: str

    rows_in: int = 0
    cost_usd: float = 0.0
    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    rejected: dict[str, int] = field(default_factory=dict)
    guards: list[dict[str, Any]] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)
    denominator: dict[str, Any] | None = None
    compute: dict[str, Any] = field(default_factory=_resolve_compute)
    #: Looks up an output object's current version. Set by :func:`run_unit`
    #: from a sink that has a ``head`` method; never written to the manifest.
    object_head: Callable[[str], ObjectVersion | None] | None = field(default=None, repr=False)

    def record_input(
        self,
        key: str,
        *,
        etag: str | None = None,
        version: str | None = None,
        schema_version: str | None = None,
    ) -> None:
        """One artifact this run READ, at the version it was read at."""
        self.inputs.append({"key": key, "etag": etag, "version": version, "schema_version": schema_version})

    def record_output(
        self,
        key: str,
        *,
        rows_out: int,
        etag: str | None = None,
        schema_version: str | None = None,
        bytes_: int | None = None,
        version_id: str | None = None,
    ) -> None:
        """One artifact this run PUBLISHED, with the row count that landed in it.

        ``rows_out`` is required and has no ``None``: "we did not count" and "we
        counted zero" are different facts with opposite consequences, and the
        empty-but-fresh objective is computed from this number.

        **The version is measured here, not passed in** (alpha-engine-config-I10892).
        Unless the caller supplied ``etag``, ``bytes_`` and ``version_id`` all
        three, the object is looked up through :attr:`object_head` and the
        missing fields are filled from what the store says NOW — immediately
        after the write this call records. That is what lets a later reader
        fetch the exact object this run published after a subsequent run has
        overwritten the key. How the fields were obtained is written as
        ``version_capture`` (:data:`VERSION_CAPTURES`), so a ``null`` is never
        ambiguous.
        """
        if rows_out < 0:
            raise ValueError(f"rows_out must be >= 0, got {rows_out}")
        record: dict[str, Any] = {
            "key": key,
            "etag": etag,
            "schema_version": schema_version,
            "rows_out": int(rows_out),
            "bytes": bytes_,
            "version_id": version_id,
        }
        if etag is not None and bytes_ is not None and version_id is not None:
            record["version_capture"] = "caller"
        elif self.object_head is None:
            record["version_capture"] = "not_captured"
        else:
            try:
                head = self.object_head(key)
            except Exception as exc:  # noqa: BLE001 -- recorded on the manifest, see below
                # Deliberate, narrow deviation from raise-by-default:
                # (a) swallowed: a HeadObject that failed for a reason other than
                #     404 (a writer role holding PutObject but not GetObject, a
                #     throttle). (b) The primary deliverable survives: the object
                #     was ALREADY written, and failing the unit here would mark a
                #     successful publish `failed` over its lineage record, not its
                #     data. (c) Recording surface: this output's own
                #     `version_capture: unavailable` + `version_capture_error`,
                #     which a parity reader treats as "no recorded version" and
                #     never as a match — plus the WARNING below.
                logger.warning("unit %s: HeadObject for output %s failed: %s", self.unit_id, key, exc)
                record["version_capture"] = "unavailable"
                record["version_capture_error"] = f"{type(exc).__name__}: {exc}"[:500]
            else:
                if head is None:
                    record["version_capture"] = "object_absent"
                else:
                    record["version_capture"] = "head_object"
                    if record["etag"] is None:
                        record["etag"] = head.etag
                    if record["bytes"] is None:
                        record["bytes"] = head.bytes
                    if record["version_id"] is None:
                        record["version_id"] = head.version_id
        self.outputs.append(record)

    def reject(self, reason: str, count: int = 1) -> None:
        """Record ``count`` rejected records under ``reason``. Never a bare count."""
        if count <= 0:
            raise ValueError(f"rejected count must be >= 1, got {count}")
        self.rejected[reason] = self.rejected.get(reason, 0) + int(count)

    def spend(self, usd: float) -> None:
        self.cost_usd += float(usd)

    def record_guard(
        self,
        guard: str,
        *,
        mode: str,
        verdict: str,
        detail: str,
        key: str | None = None,
        value: float | None = None,
        baseline: float | None = None,
    ) -> None:
        """One write-time guard's reading — including the ones that PASSED.

        A guard that records only when it fires is indistinguishable from a
        guard that stopped running (`principles.md` §2.7).
        """
        self.guards.append(
            {
                "guard": guard,
                "mode": mode,
                "verdict": verdict,
                "detail": detail,
                "key": key,
                "value": value,
                "baseline": baseline,
            }
        )

    def record_metric(self, record: Any) -> None:
        """Attach a MetricRecord (or a plain dict of one) to this run."""
        if hasattr(record, "model_dump"):
            self.metrics.append(record.model_dump(mode="json", exclude_none=True))
        elif isinstance(record, Mapping):
            self.metrics.append(dict(record))
        else:
            raise TypeError(f"metric must be a MetricRecord or a mapping, got {type(record)!r}")

    def declare_denominator(self, source: str, count: int, floor: float | None = None) -> None:
        self.denominator = {"source": source, "count": int(count), "floor": floor}

    def exclude(self, id_: str, declared_class: str, source: str | None = None) -> None:
        """One record left out of the denominator, by a DECLARED exclusion class."""
        self.excluded.append({"id": id_, "declared_class": declared_class, "source": source})

    @property
    def rows_out(self) -> int:
        return sum(int(o["rows_out"]) for o in self.outputs)


@dataclass(frozen=True)
class UnitRunResult:
    """What :func:`run_unit` returns on a run that did not raise."""

    status: str
    reason: str
    manifest: dict[str, Any]
    key: str | None
    value: Any = None


def _rfc3339(moment: dt.datetime) -> str:
    return moment.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_manifest(ctx: UnitRun, *, status: str, reason: str, finished: dt.datetime) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": ctx.run_id,
        "unit_id": ctx.unit_id,
        "trigger": ctx.trigger,
        "trading_day": ctx.trading_day,
        "calendar_date": ctx.calendar_date,
        "status": status,
        "reason": reason,
        "started": _rfc3339(ctx.started),
        "finished": _rfc3339(finished),
        "code_sha": ctx.code_sha,
        "log_location": ctx.log_location,
        "inputs": list(ctx.inputs),
        "outputs": list(ctx.outputs),
        "rows_in": int(ctx.rows_in),
        "rows_out": ctx.rows_out,
        "rows_rejected": [{"reason": r, "count": int(n)} for r, n in sorted(ctx.rejected.items())],
        "cost_usd": float(ctx.cost_usd),
        "compute": dict(ctx.compute),
        **({"denominator": ctx.denominator} if ctx.denominator is not None else {}),
        **({"excluded": list(ctx.excluded)} if ctx.excluded else {}),
        **({"guards": list(ctx.guards)} if ctx.guards else {}),
        **({"metrics": list(ctx.metrics)} if ctx.metrics else {}),
    }


def run_unit(
    unit_id: str,
    fn: Callable[[UnitRun], Any],
    *,
    sink: ManifestSink | None,
    trigger: str,
    trading_day: str,
    log_location: str,
    calendar_date: str | None = None,
    now: dt.datetime | None = None,
    code_sha: str | None = None,
    prefix: str = DEFAULT_MANIFEST_PREFIX,
    inputs: Iterable[Mapping[str, Any]] = (),
) -> UnitRunResult:
    """Run ``fn`` as unit ``unit_id`` and write its manifest, whatever happens.

    ``fn`` receives a :class:`UnitRun` and records its own lineage on it. The
    manifest is written from a ``finally``: on success, on failure, and on
    :class:`NotApplicable`. On failure the exception is **re-raised after the
    record is durable** — the caller's own error handling is unchanged, and the
    run is never lost because the write came last.

    ``sink=None`` runs ``fn`` exactly as a real invocation would and writes
    NOTHING, for a dry run. One line is logged in the manifest's place, naming
    the unit and trading day, so a dry run is visibly a dry run rather than
    silent. It is the caller's job to ensure ``fn`` itself writes nothing real
    under a dry run; this only governs the one write this function makes.

    Resolution order matters: ``code_sha`` and the argument validation happen
    BEFORE ``fn`` runs, so an invocation that could never have written a
    well-formed record costs nothing and fails where the defect is.
    """
    if not _UNIT_ID_RE.match(unit_id):
        raise ValueError(
            f"unit_id {unit_id!r} is not an audit unit id (D01..D46, optionally suffixed). "
            "The manifest's address and the board's denominator both come from this id, so "
            "a run that cannot name its unit is refused rather than filed where nobody looks."
        )
    if trigger not in TRIGGERS:
        raise ValueError(f"trigger {trigger!r} is not one of {sorted(TRIGGERS)}")
    if not log_location:
        raise ValueError(
            "log_location is required — a manifest that summarises a run without saying "
            "where to read its full logs turns every diagnosis into a search "
            "(observability-policy §6.2)."
        )
    resolved_sha = code_sha if code_sha is not None else resolve_code_sha()
    if not _REAL_SHA_RE.match(resolved_sha):
        raise CodeShaError(f"code_sha {resolved_sha!r} is not a real 40-character git sha")

    started = now or dt.datetime.now(dt.timezone.utc)
    ctx = UnitRun(
        run_id=new_run_id(started),
        unit_id=unit_id,
        trading_day=trading_day,
        calendar_date=calendar_date or started.astimezone(dt.timezone.utc).date().isoformat(),
        trigger=trigger,
        started=started,
        code_sha=resolved_sha,
        log_location=log_location,
        object_head=getattr(sink, "head", None) if sink is not None else None,
    )
    for ref in inputs:
        ctx.record_input(
            str(ref["key"]),
            etag=ref.get("etag"),
            version=ref.get("version"),
            schema_version=ref.get("schema_version"),
        )

    status = "failed"
    reason = "run did not reach a terminal state"
    value: Any = None
    try:
        value = fn(ctx)
        status, reason = "ok", ""
    except NotApplicable as na:
        status, reason = "not_applicable", na.reason
        logger.info(
            "unit %s run %s: not_applicable (%s) %s",
            unit_id,
            ctx.run_id,
            na.reason,
            na.detail,
        )
    except BaseException as exc:  # noqa: BLE001 -- recorded and RE-RAISED below
        status = "failed"
        reason = f"{type(exc).__name__}: {exc}"[:2000] or type(exc).__name__
        raise
    finally:
        manifest = _build_manifest(ctx, status=status, reason=reason, finished=dt.datetime.now(dt.timezone.utc))
        key = manifest_key(unit_id, trading_day, ctx.run_id, prefix)
        if sink is None:
            logger.info(
                "dry run: unit %s trading_day %s status %s — NO manifest written (would be %s)",
                unit_id,
                trading_day,
                status,
                key,
            )
        else:
            # Deliberately NOT wrapped in try/except. A manifest that failed to
            # write is a run that did not happen as far as every downstream
            # reader is concerned, and swallowing the write error would leave
            # the loudest possible defect — an invisible run — reported as
            # nothing at all. On the failure path the raise from here replaces
            # the body's exception, and that is the correct precedence: the
            # body's failure is recoverable evidence in the logs, an
            # unrecordable run is not.
            sink.write(key, json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8"))
            logger.info("unit %s run %s status=%s manifest=%s", unit_id, ctx.run_id, status, key)

    return UnitRunResult(
        status=status,
        reason=reason,
        manifest=manifest,
        key=None if sink is None else key,
        value=value,
    )
