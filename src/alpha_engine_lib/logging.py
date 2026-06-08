"""
Shared structured logging + Flow Doctor integration.

Replaces near-identical copies of ``log_config.py`` in alpha-engine-data
and alpha-engine/executor. Consumers call :func:`setup_logging` once at
process startup; subsequent call sites retrieve the Flow Doctor instance
via :func:`get_flow_doctor`.

Modes:

- Text (default): human-readable single-line log format.
- JSON: activated by ``ALPHA_ENGINE_JSON_LOGS=1``. Emits one JSON object
  per log record, including tracebacks for errors.

Flow Doctor activates only when ``FLOW_DOCTOR_ENABLED=1`` and a
``flow_doctor_yaml`` path is provided. ERROR-level records (including
``logger.exception``) fire the FlowDoctorHandler, which dispatches per
the yaml config (email + GitHub issue with dedup + rate limits).

Requires the ``flow_doctor`` optional extra when FLOW_DOCTOR_ENABLED=1
(``alpha-engine-lib[flow_doctor]``).
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

# ``${VAR}`` interpolation tokens in a flow-doctor.yaml. flow-doctor
# resolves these from ``os.environ`` eagerly at ``FlowDoctor.from_config()``
# time — before any lazy ``get_secret()`` consumer-site call runs — so
# the seed below must populate them first.
_FD_VAR_RE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")

# Singleton populated by setup_logging() when FLOW_DOCTOR_ENABLED=1.
# ``Optional[object]`` typing avoids forcing a flow_doctor import here.
_fd_instance: Optional[object] = None


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "func": record.funcName,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exc"] = self.formatException(record.exc_info)
        if hasattr(record, "ctx"):
            entry["ctx"] = record.ctx
        return json.dumps(entry, default=str)


# Default-active secrets redaction patterns. Applied at the logging-handler
# layer by SecretsRedactingFilter so every log record reaching stdout / CW
# Logs / flow-doctor has the matching substrings replaced. Conservative
# by design — false positives (over-redaction) are visible; false negatives
# (leaked keys) reach public CW Logs.
#
# Pattern list is closed-form: every alpha-engine secret class known to leak
# at the data-collector / research / predictor / backtester log sites. Adding
# a new pattern is a one-line addition; opt-out per-record/per-attach is
# possible via ``record.no_redact = True`` (rare — see ``SecretsRedactingFilter``).
#
# Origin: 2026-05-24 audit on alpha-engine-research-runner CW Logs surfaced
# the FMP API key in plaintext inside HTTP-error WARNING lines (the FMP
# /stable 402 paid-tier errors mid-Research). alpha-engine-data #255 had
# shipped a repo-local scrubber for the same defect class earlier; lifting
# to the lib chokepoint per [[feedback_lift_invariants_to_chokepoint_after_second_recurrence]]
# closes the recurrence permanently across every repo that consumes
# ``alpha_engine_lib.logging.setup_logging``.
_SECRET_REDACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # URL-query-string credentials: `?apikey=...&symbol=X` /
    # `?api_key=...` / `?key=...` / `?token=...`. Conservative length
    # floor (16 chars) avoids redacting short ID-like tokens that look
    # alphanumeric but aren't secrets.
    (
        re.compile(
            r"([?&](?:apikey|api_key|key|token|access_token|auth_token)=)"
            r"([A-Za-z0-9_\-\.]{16,})"
        ),
        r"\1<REDACTED>",
    ),
    # AWS Access Key ID — exactly 16 alphanumerics after AKIA prefix.
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"), "<AWS_ACCESS_KEY_REDACTED>"),
    # Anthropic API keys — sk-ant-{api,sid,oat}-...
    (re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}"), "<ANTHROPIC_KEY_REDACTED>"),
    # OpenAI-style secret keys — sk-... 32+ chars (covers project keys).
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{32,}"), "<OPENAI_KEY_REDACTED>"),
    # Authorization: Bearer <token> headers, case-insensitive.
    (
        re.compile(r"(authorization:\s*bearer\s+)([A-Za-z0-9_\-\.]+)", re.IGNORECASE),
        r"\1<REDACTED>",
    ),
    # GitHub personal access tokens — classic (ghp_*) and fine-grained
    # (github_pat_*). Both have well-known prefixes; min-length 30
    # guards against the prefix-alone false-positive.
    (re.compile(r"\b(?:ghp_|gho_|ghu_|ghs_|ghr_)[A-Za-z0-9_]{30,}"), "<GITHUB_TOKEN_REDACTED>"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), "<GITHUB_TOKEN_REDACTED>"),
)


class SecretsRedactingFilter(logging.Filter):
    """logging.Filter that redacts known secret-shaped substrings from
    every log record's formatted message.

    Attached by default to the handler created by :func:`setup_logging`.
    Opt-out via the ``ALPHA_ENGINE_DISABLE_LOG_REDACTION=1`` env var (for
    cases where redaction obscures debugging of a non-secret pattern that
    matches a redaction regex by coincidence).

    Per-record opt-out is also possible via ``record.no_redact = True``
    on a record where the redaction is known-safe to skip. Use rarely —
    the default-active posture is a security property.

    Never raises. A regex compilation or replacement error is caught and
    the original record passes through unmodified (better to log
    something than nothing). The catch is intentionally broad: a logging
    filter that crashes is worse than one that silently no-ops.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "no_redact", False):
            return True
        try:
            # Render the message with its args so substitution sees the
            # actual emitted text (otherwise a `logger.warning("...%s...", key)`
            # would pass `record.msg` containing only the format string).
            rendered = record.getMessage()
            redacted = rendered
            for pattern, replacement in _SECRET_REDACTION_PATTERNS:
                redacted = pattern.sub(replacement, redacted)
            if redacted != rendered:
                # Replace the format string + clear args so downstream
                # formatters re-emit the redacted text rather than
                # re-interpolating.
                record.msg = redacted
                record.args = ()
        except Exception:  # noqa: BLE001 - filter MUST NOT crash logging
            pass
        return True


def get_flow_doctor():
    """Return the shared flow-doctor instance, or None if not initialized."""
    return _fd_instance


def _seed_flow_doctor_secrets(yaml_path: str) -> None:
    """Populate the flow-doctor ``${VAR}`` secrets into ``os.environ``.

    flow-doctor resolves every ``${VAR}`` in its yaml from ``os.environ``
    eagerly inside ``FlowDoctor.from_config()``, before any consumer-site
    :func:`alpha_engine_lib.secrets.get_secret` call has had a chance to
    run. With the legacy ``ssm_secrets.load_secrets()`` bulk-load shim
    retired (PR 9g), systemd/Step-Functions-launched entrypoints have no
    ``.env`` source, so those ``${VAR}`` refs would resolve to nothing
    and flow-doctor's email + GitHub dispatch would silently misfire.

    This is the single chokepoint every repo reaches flow-doctor
    through, so seeding here closes the gap system-wide with no
    per-repo code. The var set is derived from the yaml itself rather
    than hardcoded — each repo's flow-doctor.yaml carries a different
    ``${VAR}`` set, and a yaml-added secret must not silently re-open
    the gap.

    Invariants (mirroring the retired shim):

    - A var already present in ``os.environ`` wins — never overwritten.
    - A genuinely unresolvable secret is left **unset**, so
      flow-doctor's own ``ConfigError`` fires loudly rather than being
      masked with ``""`` (see ``feedback_no_silent_fails``).
    - A secrets-backend hiccup never blocks logging setup; it is logged
      at WARNING and the var is left unset (same loud-failure path).
    """
    try:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            yaml_text = fh.read()
    except OSError:
        # Missing/unreadable yaml is reported by _attach_flow_doctor's
        # own os.path.exists guard with a clearer message.
        return

    from alpha_engine_lib.secrets import get_secret

    for var in sorted(set(_FD_VAR_RE.findall(yaml_text))):
        if os.environ.get(var):
            continue
        try:
            value = get_secret(var, required=False)
        except Exception as exc:  # noqa: BLE001 - backend hiccup is non-fatal
            logging.getLogger(__name__).warning(
                "flow-doctor secret seed: get_secret(%s) raised %r; "
                "leaving unset so flow-doctor fails loudly", var, exc,
            )
            continue
        if value:
            os.environ[var] = value


def _attach_flow_doctor(
    yaml_path: str,
    exclude_patterns: list[str] | None = None,
) -> None:
    """Initialize the shared flow-doctor instance and attach a log handler.

    ``exclude_patterns`` is a list of regex strings forwarded to
    ``FlowDoctorHandler(exclude_patterns=...)``. Log records whose
    rendered message matches any pattern are dropped before entering
    the flow-doctor dispatch pipeline (email / GitHub issue). Use for
    benign ERROR-level noise that would otherwise dedup-spam on-call.
    """
    global _fd_instance
    try:
        import flow_doctor
    except ImportError as exc:
        raise RuntimeError(
            "FLOW_DOCTOR_ENABLED=1 but flow-doctor is not installed. Install "
            "via alpha-engine-lib[flow_doctor] or add flow-doctor[diagnosis] "
            f"to requirements: {exc}"
        ) from exc

    if not os.path.exists(yaml_path):
        raise RuntimeError(
            f"FLOW_DOCTOR_ENABLED=1 but flow-doctor config not found at {yaml_path}"
        )

    _seed_flow_doctor_secrets(yaml_path)
    # flow-doctor 0.6.0 removed the deprecated ``flow_doctor.init()`` free
    # function in favour of ``FlowDoctor.from_config()`` (identical
    # config_path contract). Prefer from_config when present; fall back to
    # init() on flow-doctor < 0.6 so this works across the soak window
    # regardless of which flow-doctor the consumer has pinned. Drop the
    # fallback once the fleet floor is flow-doctor>=0.6.0.
    if hasattr(flow_doctor.FlowDoctor, "from_config"):
        _fd_instance = flow_doctor.FlowDoctor.from_config(config_path=yaml_path)
    else:
        _fd_instance = flow_doctor.init(config_path=yaml_path)
    handler_kwargs: dict = {"level": logging.ERROR}
    if exclude_patterns:
        handler_kwargs["exclude_patterns"] = exclude_patterns
    handler = flow_doctor.FlowDoctorHandler(_fd_instance, **handler_kwargs)
    logging.getLogger().addHandler(handler)


def setup_logging(
    name: str,
    flow_doctor_yaml: str | None = None,
    exclude_patterns: list[str] | None = None,
) -> None:
    """Configure the root logger for an Alpha Engine entrypoint.

    :param name: Logger name shown in the text-mode prefix
        (``"%(asctime)s %(levelname)s [{name}] %(message)s"``). Typically
        the module name (``"data-collector"``, ``"executor"``, etc.).
    :param flow_doctor_yaml: Absolute or CWD-relative path to the
        flow-doctor yaml config. Required if ``FLOW_DOCTOR_ENABLED=1``;
        ignored otherwise.
    :param exclude_patterns: Optional list of regex strings. When
        ``FLOW_DOCTOR_ENABLED=1``, these are forwarded to
        ``FlowDoctorHandler`` so matching ERROR-level records are
        dropped before the flow-doctor dispatch pipeline. Use sparingly
        — this silences *alerts*, not logs. The records still appear in
        stdout / JSON logs; only flow-doctor's email + GitHub issue
        routing is suppressed. Example: the executor passes
        ``[r"Error 10197"]`` to suppress benign IB Gateway noise when
        the iOS app steals the live-data session.

    Env vars consulted:

    - ``ALPHA_ENGINE_JSON_LOGS`` — ``"1"`` enables JSON formatter.
    - ``FLOW_DOCTOR_ENABLED`` — ``"1"`` attaches FlowDoctorHandler.
    """
    json_mode = os.environ.get("ALPHA_ENGINE_JSON_LOGS", "0") == "1"

    handler = logging.StreamHandler()
    if json_mode:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            f"%(asctime)s %(levelname)s [{name}] %(message)s"
        ))

    # Attach secrets-redacting filter by default. Closes the FMP-API-key /
    # AWS-key / Anthropic-key / GitHub-PAT plaintext-log class system-wide
    # (every consumer of alpha_engine_lib.logging.setup_logging inherits
    # the filter on its next lib pin bump). Opt-out via
    # ALPHA_ENGINE_DISABLE_LOG_REDACTION=1 for the rare debugging scenario
    # where a redaction regex matches a non-secret pattern.
    if os.environ.get("ALPHA_ENGINE_DISABLE_LOG_REDACTION", "0") != "1":
        handler.addFilter(SecretsRedactingFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    if os.environ.get("FLOW_DOCTOR_ENABLED", "0") == "1":
        if not flow_doctor_yaml:
            raise RuntimeError(
                "FLOW_DOCTOR_ENABLED=1 but setup_logging() was not given a "
                "flow_doctor_yaml path"
            )
        _attach_flow_doctor(flow_doctor_yaml, exclude_patterns=exclude_patterns)
