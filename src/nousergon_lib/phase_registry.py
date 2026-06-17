"""
nousergon_lib.phase_registry — reusable phase-execution framework for the
Saturday/weekday spot pipelines (L4528).

Lifted verbatim-in-spirit from the alpha-engine-backtester's
``pipeline_common.py`` (where it was proven across the 2026-04→06 orchestration
hardening: phase markers, per-phase watchdog, the 3-way outcome taxonomy, and
the L4524 artifact-validated checkpoints). The data + predictor weekly spots are
sibling orchestration that run monolithically today; hosting the framework here
lets each consume one battle-tested implementation rather than re-growing its
own (per [[feedback_lift_invariants_to_chokepoint_after_second_recurrence]] —
second adoption is the consolidation signal).

The two backtester couplings are generalized for cross-repo use:

* **Marker key prefix** is injectable — ``PhaseRegistry(marker_prefix=...)``
  writes ``{marker_prefix}/{date}/.phases/{phase}.json`` (backtester passes
  ``"backtest"`` to keep its existing S3 keys; data/predictor pass their own).
* **Hard-caps file** — ``load_phase_hard_caps(path)`` resolves ``path`` as given
  (absolute or CWD-relative); no repo-root ``__file__`` magic.

The structured ``PHASE_START``/``PHASE_END`` log lines are unchanged, so existing
CloudWatch/SSM greps on the ``PHASE_START ``/``PHASE_END `` prefix keep working;
the logger NAME is configurable so a consumer can preserve its own logging
config (the backtester passes ``"backtest.phase"``).

Public surface: ``PhaseStatus``, ``PhaseOutcome``, ``PhaseRegistry``,
``PhaseContext``, ``PhaseTimeoutError``, ``phase``, ``load_phase_hard_caps``,
``MARKER_SCHEMA_VERSION``.
"""

from __future__ import annotations

import enum
import faulthandler
import json
import logging
import sys
import threading
import time
import _thread
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import boto3
import yaml
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

#: Default logger name for the structured PHASE_START/PHASE_END marker lines.
#: A consumer can override per-call / per-registry to preserve its own logging
#: config (e.g. the backtester passes ``"backtest.phase"``).
DEFAULT_PHASE_LOGGER = "nousergon_lib.phase"

MARKER_SCHEMA_VERSION = 1


# ── Phase outcome taxonomy (3-way: SUCCESS | EMPTY | FAILURE) ────────────────
#
# Binary ok/fail is WRONG because it forces the EMPTY case into a harmful
# answer (treat-as-success silently drops a result; treat-as-failure kills the
# whole pipeline when a gate merely did its job).
#
#   SUCCESS — produced its declared admissible result.
#   EMPTY   — ran correctly, produced NO admissible result (e.g. all combos
#             gated out). A first-class FINDING, surfaced LOUDLY so a suspicious
#             degeneracy is never silent, but downstream no-ops gracefully.
#   FAILURE — an infra/contract break (absent input, exception, contract
#             violation). Fatal, fail-loud.


class PhaseStatus(enum.Enum):
    """3-way phase outcome."""

    SUCCESS = "success"
    EMPTY = "empty"
    FAILURE = "failure"


@dataclass
class PhaseOutcome:
    """Structured result of running (or classifying) a pipeline stage.

    ``status`` is the load-bearing field; the rest are observability so a
    degenerate run is legible at a glance. ``reason`` is the human/log message;
    ``degeneracy_reason`` names *why* an EMPTY produced nothing;
    ``n_inputs``/``n_admissible`` quantify the gating; ``artifacts_written``
    records what hit S3.
    """

    status: PhaseStatus
    phase: str
    reason: str = ""
    n_inputs: int | None = None
    n_admissible: int | None = None
    degeneracy_reason: str | None = None
    artifacts_written: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status is PhaseStatus.SUCCESS

    @property
    def is_empty(self) -> bool:
        return self.status is PhaseStatus.EMPTY

    @property
    def is_failure(self) -> bool:
        return self.status is PhaseStatus.FAILURE

    def to_dict(self) -> dict:
        """JSON-serializable record for structured logging / markers."""
        return {
            "status": self.status.value,
            "phase": self.phase,
            "reason": self.reason,
            "n_inputs": self.n_inputs,
            "n_admissible": self.n_admissible,
            "degeneracy_reason": self.degeneracy_reason,
            "artifacts_written": list(self.artifacts_written),
            "detail": dict(self.detail),
        }


# ── Phase markers ────────────────────────────────────────────────────────────
#
# Structured begin/end log lines around each pipeline phase so any timeout
# investigation can attribute wall time to a specific phase without correlating
# log gaps against source. Format is parseable: future tooling can grep the
# ``PHASE_START ``/``PHASE_END `` prefix and pull name + duration from one line.


def _phase_logger(name: str = DEFAULT_PHASE_LOGGER) -> logging.Logger:
    """Dedicated logger for phase markers so callers don't need to pass one."""
    return logging.getLogger(name)


@contextmanager
def phase(name: str, *, logger_name: str = DEFAULT_PHASE_LOGGER, **context):
    """Emit ``PHASE_START name=X ...`` / ``PHASE_END name=X duration_s=Y status=ok|error ...``.

    Duration is measured with monotonic time so clock adjustments don't lie.
    stdout is flushed after each marker so a dying SSM agent doesn't eat
    buffered output.
    """
    plog = _phase_logger(logger_name)
    kv = " ".join(f"{k}={v}" for k, v in context.items())
    plog.info("PHASE_START name=%s %s", name, kv)
    sys.stdout.flush()
    t0 = time.monotonic()
    status = "ok"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        dur = time.monotonic() - t0
        plog.info("PHASE_END name=%s duration_s=%.2f status=%s %s", name, dur, status, kv)
        sys.stdout.flush()


def _marker_key(marker_prefix: str, date: str, phase_name: str) -> str:
    return f"{marker_prefix}/{date}/.phases/{phase_name}.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Phase watchdog (per-phase hard cap) ─────────────────────────────────────
#
# A per-phase hard-cap timer that dumps all-thread stack traces and raises in
# the main thread if a phase exceeds its cap. A threading.Timer (not SIGALRM)
# so it works from any thread; on trip it dumps via faulthandler + interrupts
# main, which the phase context manager catches and maps to PhaseTimeoutError.
# Caps are opt-in per-phase via PhaseRegistry(hard_caps={...}).


class PhaseTimeoutError(RuntimeError):
    """Raised when a phase exceeds its hard cap. All-thread stack traces have
    been written to stderr by faulthandler before the exception is raised."""


def _default_watchdog_trip(name: str, cap_s: float, logger_name: str = DEFAULT_PHASE_LOGGER) -> None:
    """Default trip handler: log PHASE_TIMEOUT, dump all-thread stacks,
    interrupt main. Exposed so tests can swap in a no-op handler."""
    plog = _phase_logger(logger_name)
    plog.warning(
        "PHASE_TIMEOUT name=%s cap_s=%.1f — dumping all-thread stacks to stderr "
        "and raising PhaseTimeoutError in main thread", name, cap_s,
    )
    sys.stderr.write(
        f"\n── PHASE_TIMEOUT name={name} cap_s={cap_s:.1f} ────────────────\n"
    )
    sys.stderr.flush()
    try:
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    except Exception as dump_exc:
        sys.stderr.write(f"(faulthandler.dump_traceback failed: {dump_exc})\n")
    sys.stderr.flush()
    _thread.interrupt_main()


def _start_watchdog(
    name: str,
    cap_s: float,
    on_trip: Callable[[str, float], None] | None = None,
) -> tuple[threading.Timer, dict]:
    """Start a watchdog Timer; return ``(timer, state-dict)``.

    State dict has ``tripped: bool`` so the phase context manager can
    distinguish "KeyboardInterrupt from watchdog" vs "KeyboardInterrupt from
    operator Ctrl+C" and raise PhaseTimeoutError only in the former.
    """
    state = {"tripped": False, "name": name, "cap_s": cap_s}
    handler = on_trip or _default_watchdog_trip

    def _fire():
        state["tripped"] = True
        try:
            handler(name, cap_s)
        except Exception as handler_exc:
            logger.error(
                "phase watchdog handler raised: %s — falling back to interrupt_main",
                handler_exc,
            )
            _thread.interrupt_main()

    timer = threading.Timer(cap_s, _fire)
    timer.daemon = True
    timer.start()
    return timer, state


class PhaseRegistry:
    """Drives per-phase skip/force decisions and writes completion markers.

    Lifecycle:
      1. Caller constructs a registry from CLI flags / config.
      2. For each phase, caller uses ``with registry.phase(name, ...)`` —
         either (a) it's already complete for this date → ``ctx.skipped=True``,
         caller loads the artifact from S3 instead of recomputing; or (b)
         caller runs the compute + registers any artifact keys via
         ``ctx.record_artifact(key)`` before the block exits.
      3. On ``__exit__`` the registry writes an END marker to S3 with
         duration_s + status + artifact_keys.

    A phase is "auto-skippable" only when the caller passes
    ``supports_auto_skip=True`` — otherwise a stale marker can't trick the
    pipeline into skipping a phase whose output isn't actually on S3.

    Markers are written under ``{marker_prefix}/{date}/.phases/{phase}.json`` so
    each consuming repo namespaces its own (backtester ``"backtest"``, data
    ``"data"``, predictor ``"predictor"``, …). Marker reads are cached per phase.
    """

    def __init__(
        self,
        *,
        date: str,
        bucket: str,
        marker_prefix: str = "pipeline",
        skip_phases: Iterable[str] | None = None,
        only_phases: Iterable[str] | None = None,
        force: bool = False,
        force_phases: Iterable[str] | None = None,
        hard_caps: dict[str, float] | None = None,
        s3_client=None,
        phase_logger_name: str = DEFAULT_PHASE_LOGGER,
    ):
        self.date = date
        self.bucket = bucket
        self.marker_prefix = marker_prefix
        self._explicit_skip = set(skip_phases or [])
        self._only = set(only_phases) if only_phases else None
        self._force_all = bool(force)
        self._force_phases = set(force_phases or [])
        self._hard_caps = dict(hard_caps or {})
        self._markers: dict[str, dict | None] = {}
        # Per-phase cache of artifact-validation results (L4524) so a marker
        # whose declared artifacts were head_object'd once isn't re-checked on
        # the second should_run call. Value is the first-missing key, or None.
        self._artifact_checks: dict[str, str | None] = {}
        self._s3 = s3_client  # lazy-init if None
        self._phase_logger_name = phase_logger_name
        # Names of phases that wrote a marker with status=error during THIS
        # invocation — lets a smoke/budget check catch a false-PASS where the
        # outer phase swallowed an inner error but wall-clock looked healthy.
        self.phase_errors: list[str] = []

    # ── S3 helpers ───────────────────────────────────────────────────────

    def _client(self):
        if self._s3 is None:
            self._s3 = boto3.client("s3")
        return self._s3

    @property
    def s3_client(self):
        """Public accessor so artifact save/load helpers can use the same
        client the registry writes markers with."""
        return self._client()

    def _marker_key(self, phase_name: str) -> str:
        return _marker_key(self.marker_prefix, self.date, phase_name)

    def _read_marker(self, phase_name: str) -> dict | None:
        """Return the marker dict for (date, phase), or None if absent/corrupt.

        Result is cached. A corrupt marker (unparseable JSON, missing required
        fields) is treated as absent and logged loud.
        """
        if phase_name in self._markers:
            return self._markers[phase_name]

        key = self._marker_key(phase_name)
        try:
            obj = self._client().get_object(Bucket=self.bucket, Key=key)
            body = obj["Body"].read()
            try:
                marker = json.loads(body)
            except (json.JSONDecodeError, ValueError):
                marker = None
            if not isinstance(marker, dict) or marker.get("status") not in ("ok", "error"):
                logger.warning(
                    "phase_registry: marker at s3://%s/%s malformed — ignoring "
                    "and recomputing phase %s. Body: %s",
                    self.bucket, key, phase_name, body[:200],
                )
                marker = None
            self._markers[phase_name] = marker
            return marker
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404"):
                self._markers[phase_name] = None
                return None
            # Network / permission errors: fail loud rather than silently
            # "marker absent → recompute." A transient S3 blip shouldn't cause
            # a long pipeline to silently redo work it already did.
            raise

    def _write_marker(self, marker: dict) -> None:
        key = self._marker_key(marker["phase"])
        self._client().put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(marker, indent=2).encode(),
            ContentType="application/json",
        )
        self._markers[marker["phase"]] = marker
        if marker.get("status") == "error":
            self.phase_errors.append(marker["phase"])

    def _first_missing_artifact(self, artifact_keys: Iterable[str]) -> str | None:
        """Return the first declared artifact key that is absent on S3, or None
        if every key is present (or none were declared).

        L4524 — artifact-validated checkpoints. A ``status=ok`` marker only
        earns an auto-skip if the outputs it *claims* to have produced are
        actually on S3; a marker whose declared artifact has gone missing is
        lying (a phase marked ok while its critical output was never written /
        was pruned), so it must be treated as INVALID → re-run.

        Existence is probed with ``head_object`` (metadata only). Error posture
        mirrors ``_read_marker``: 404/NotFound → absent (invalidate); any other
        S3 error RAISES rather than silently flipping a skip/re-run decision.
        """
        client = self._client()
        for key in artifact_keys:
            if not key:
                continue
            try:
                client.head_object(Bucket=self.bucket, Key=key)
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in ("NoSuchKey", "NoSuchBucket", "NotFound", "404"):
                    return key
                raise
        return None

    def _marker_artifact_missing(self, phase_name: str, marker: dict) -> str | None:
        """Validate a phase's marker against its declared artifacts (cached)."""
        if phase_name not in self._artifact_checks:
            self._artifact_checks[phase_name] = self._first_missing_artifact(
                marker.get("artifact_keys") or []
            )
        return self._artifact_checks[phase_name]

    # ── Decision logic ───────────────────────────────────────────────────

    def should_run(self, phase_name: str, supports_auto_skip: bool = False) -> tuple[bool, str]:
        """Return ``(run: bool, reason: str)``.

        Precedence: ``--only`` restricts the set; explicit ``--skip`` /
        ``--force`` wins; ``--force`` overrides auto-skip; auto-skip if the
        phase is auto-skippable AND a prior-run marker is present with
        status=ok AND every artifact the marker declares still exists on S3
        (L4524 — a marker whose declared artifact is missing is invalid →
        re-run); else run.

        Reason strings are grep-able: ``only_phases_filter`` | ``explicit_skip``
        | ``auto_skip_marker_ok`` | ``force_rerun`` | ``force_phase_rerun`` |
        ``default_run`` | ``not_auto_skippable`` | ``marker_artifact_missing``.
        """
        if self._only is not None and phase_name not in self._only:
            return False, "only_phases_filter"
        if phase_name in self._explicit_skip:
            return False, "explicit_skip"
        if self._force_all:
            return True, "force_rerun"
        if phase_name in self._force_phases:
            return True, "force_phase_rerun"
        if not supports_auto_skip:
            return True, "not_auto_skippable"
        marker = self._read_marker(phase_name)
        if marker is not None and marker.get("status") == "ok":
            missing = self._marker_artifact_missing(phase_name, marker)
            if missing is not None:
                logger.warning(
                    "phase_registry: phase %s (date %s) has a status=ok marker but "
                    "its declared artifact s3://%s/%s is absent — marker INVALID, "
                    "re-running the phase (L4524 artifact-validated checkpoint).",
                    phase_name, self.date, self.bucket, missing,
                )
                return True, "marker_artifact_missing"
            return False, "auto_skip_marker_ok"
        return True, "default_run"

    def load_marker(self, phase_name: str) -> dict | None:
        """Public accessor for a phase's marker — used by loaders."""
        return self._read_marker(phase_name)

    # ── Phase context manager ────────────────────────────────────────────

    @contextmanager
    def phase(self, name: str, *, supports_auto_skip: bool = False, **log_ctx):
        """Phase context manager — writes a START/END marker to S3 around the block.

        Yields a :class:`PhaseContext`:
          - ``ctx.skipped``: True if the phase should not run (caller loads its
            artifact instead of recomputing).
          - ``ctx.record_artifact(s3_key)``: call before exiting to attach an
            artifact key to the END marker.

        If ``ctx.skipped`` the body still executes — the caller checks
        ``ctx.skipped`` at the top of the block and loads from S3 rather than
        recomputing, so the skip decision lives with the compute code.
        """
        run, reason = self.should_run(name, supports_auto_skip=supports_auto_skip)
        plog = _phase_logger(self._phase_logger_name)
        kv = " ".join(f"{k}={v}" for k, v in log_ctx.items())

        ctx = PhaseContext(name=name, skipped=not run, skip_reason=reason)

        if not run:
            plog.info("PHASE_SKIP name=%s reason=%s %s", name, reason, kv)
            sys.stdout.flush()
            yield ctx
            return

        started_at = _now_iso()
        cap_s = self._hard_caps.get(name)
        if cap_s is not None:
            plog.info("PHASE_START name=%s hard_cap_s=%.1f %s", name, cap_s, kv)
        else:
            plog.info("PHASE_START name=%s %s", name, kv)
        sys.stdout.flush()
        t0 = time.monotonic()
        status = "ok"
        err_msg: str | None = None
        watchdog_timer: threading.Timer | None = None
        watchdog_state: dict | None = None
        if cap_s is not None and cap_s > 0:
            watchdog_timer, watchdog_state = _start_watchdog(name, cap_s)
        try:
            yield ctx
        except BaseException as exc:
            status = "error"
            if (
                watchdog_state is not None
                and watchdog_state.get("tripped")
                and isinstance(exc, KeyboardInterrupt)
            ):
                err_msg = (
                    f"PhaseTimeoutError: phase {name!r} exceeded hard cap "
                    f"{cap_s:.1f}s (see PHASE_TIMEOUT + faulthandler dump on stderr)"
                )
                raise PhaseTimeoutError(err_msg) from exc
            err_msg = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if watchdog_timer is not None:
                watchdog_timer.cancel()
            dur = time.monotonic() - t0
            completed_at = _now_iso()
            plog.info(
                "PHASE_END name=%s duration_s=%.2f status=%s %s",
                name, dur, status, kv,
            )
            sys.stdout.flush()
            # Best-effort marker write. A marker write failure must NOT fail the
            # pipeline — the phase already did its work — but log loud so silent
            # marker-write drift doesn't build up across runs.
            try:
                self._write_marker({
                    "schema_version": MARKER_SCHEMA_VERSION,
                    "phase": name,
                    "date": self.date,
                    "status": status,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "duration_s": round(dur, 2),
                    "artifact_keys": sorted(ctx._artifact_keys),
                    "error": err_msg,
                })
            except Exception as marker_exc:
                logger.warning(
                    "phase_registry: failed to write marker for phase %s: %s. "
                    "Phase compute succeeded; future runs will not see this "
                    "completion and will re-run the phase.",
                    name, marker_exc,
                )


class PhaseContext:
    """Yielded by ``PhaseRegistry.phase()`` so callers can query skip state and
    register artifact keys before the phase ends."""

    def __init__(self, *, name: str, skipped: bool, skip_reason: str):
        self.name = name
        self.skipped = skipped
        self.skip_reason = skip_reason
        self._artifact_keys: set[str] = set()

    def record_artifact(self, s3_key: str) -> None:
        """Attach an S3 key to the phase's END marker (recorded on exit).

        Downstream phases / loaders read ``load_marker(name)["artifact_keys"]``
        to find the outputs; the L4524 validation re-checks they still exist.
        """
        if not isinstance(s3_key, str) or not s3_key:
            raise ValueError(f"record_artifact: expected non-empty str, got {s3_key!r}")
        self._artifact_keys.add(s3_key)


def load_phase_hard_caps(
    path: str | Path,
    *,
    caps_key: str = "full_run_hard_caps_seconds",
) -> dict[str, float]:
    """Load per-phase hard caps (seconds) from the ``caps_key`` block of a YAML
    file. Returns an empty dict if the file or block is absent (watchdog stays
    off — no behavior change).

    ``path`` is resolved as given (absolute, or relative to the current working
    directory) — the caller owns path resolution (no repo-root assumption), so
    the same loader works from any consuming repo. Keyed by phase name; values
    are floats. Non-numeric entries are dropped with a loud log.
    """
    p = Path(path)
    if not p.exists():
        logger.info("phase hard-caps file not found at %s — no phase watchdogs", p)
        return {}
    try:
        with open(p) as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning(
            "phase hard-caps file at %s failed to parse: %s — no phase watchdogs",
            p, exc,
        )
        return {}
    caps = data.get(caps_key) or {}
    if not isinstance(caps, dict):
        logger.warning(
            "phase hard-caps: %s is not a dict (got %s) — no phase watchdogs",
            caps_key, type(caps).__name__,
        )
        return {}
    out: dict[str, float] = {}
    for name, cap in caps.items():
        try:
            out[str(name)] = float(cap)
        except (TypeError, ValueError):
            logger.warning(
                "phase hard-caps: phase %r has non-numeric cap %r — skipping",
                name, cap,
            )
    return out
