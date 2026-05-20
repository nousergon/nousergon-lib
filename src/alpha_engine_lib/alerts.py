"""
Unified failure-surveillance fan-out for Alpha Engine modules.

Consolidation substrate for the **"fire an operator alert from a failure
site"** pattern that has appeared inline across the fleet:

* :file:`alpha-engine/infrastructure/health_checker.sh` — raw ``curl`` to
  Telegram bot API
* :file:`alpha-engine-data/infrastructure/lambdas/changelog-incident-mirror/deploy.sh`
  — raw ``aws sns publish`` to ``alpha-engine-alerts``
* ROADMAP L116/L117 — names 5 more Lambda-deploying repos that need the
  same canary-rollback alert primitive ("Mirror in all 5 Lambda-deploying
  repos … same recurrence class as ``feedback_env_regression_recurs_per_repo_spot_script``
  — fix forward across all repos in one pass, not per-repo at incident time")

Per the ``~/Development/CLAUDE.md`` SOTA / institutional-approach rule
(sub-sub-rule: lift to lib when ≥2 consumers exist), this module is the
canonical Python primitive backing all consumers. Bash callers reach it
via the CLI entry (``python -m alpha_engine_lib.alerts publish ...``) —
mirrors the :mod:`alpha_engine_lib.transparency` ``--cadence daily/weekly``
CLI convention.

**Public API:**

- :func:`publish` — fan-out to both SNS (``alpha-engine-alerts`` topic →
  email) and Telegram (``@nous_ergon_alerts_bot`` channel) by default.
  Each channel is independently best-effort — failure in one does not
  block the other. Returns a :class:`PublishResult` dataclass with the
  per-channel outcome for caller observability.
- CLI: ``python -m alpha_engine_lib.alerts publish --message "..."
  --severity error --source "..."``. Designed for Bash failure-trap
  callers (``cleanup()`` in spot dispatchers, ``deploy.sh`` rollback
  branches). Exit code is ``0`` if *either* channel succeeded, ``1`` if
  *both* failed.

**Severity tiering.** ``severity`` is a free-form string that is
prepended to the message (``[ERROR] ...`` / ``[WARNING] ...``) for both
channels. Telegram pushes (``disable_notification=False``) for
``error``/``critical``; in-channel silent for ``info``/``warning``. SNS
delivery is identical regardless of severity — downstream subscribers
choose how to fan out.

**SNS topic resolution.** Defaults to
``arn:aws:sns:{region}:{account_id}:alpha-engine-alerts``, with
``region`` from ``AWS_REGION``/``AWS_DEFAULT_REGION`` (fallback
``us-east-1``) and ``account_id`` resolved via ``sts:GetCallerIdentity``.
Override with the ``--sns-topic-arn`` CLI flag or ``sns_topic_arn``
kwarg.

**Failure behavior.** Never raises. SNS errors (boto3 ``ClientError``,
network) and Telegram errors both log at WARNING and return a
:class:`PublishResult` with the failed channel marked ``ok=False``. This
is by design — the caller is already in a failure path; secondary
surveillance failure must not mask the primary error.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Final

logger = logging.getLogger(__name__)

DEFAULT_SNS_TOPIC_NAME: Final[str] = "alpha-engine-alerts"
DEFAULT_REGION: Final[str] = "us-east-1"
SEVERITY_PUSH: Final[frozenset[str]] = frozenset({"error", "critical"})


@dataclass
class ChannelResult:
    """Per-channel outcome from a :func:`publish` call."""

    ok: bool
    detail: str = ""


@dataclass
class PublishResult:
    """Aggregated outcome from a :func:`publish` call.

    ``sns`` and ``telegram`` are independent — a publish may succeed in
    one channel and fail in the other. :attr:`any_ok` is the typical
    caller gate (success = at least one channel delivered the alert);
    :attr:`all_ok` is the strict variant for callers that want both.
    """

    sns: ChannelResult = field(default_factory=lambda: ChannelResult(ok=False, detail="not attempted"))
    telegram: ChannelResult = field(default_factory=lambda: ChannelResult(ok=False, detail="not attempted"))

    @property
    def any_ok(self) -> bool:
        return self.sns.ok or self.telegram.ok

    @property
    def all_ok(self) -> bool:
        return self.sns.ok and self.telegram.ok


def _resolve_sns_topic_arn(explicit: str | None) -> str | None:
    """Return the SNS topic ARN, resolving from env + STS if not explicit."""
    if explicit:
        return explicit
    region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or DEFAULT_REGION
    )
    try:
        import boto3

        account_id = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
    except Exception as exc:  # boto3 missing, STS unreachable, creds bad
        logger.warning("alerts.publish: SNS topic ARN resolution failed: %s", exc)
        return None
    return f"arn:aws:sns:{region}:{account_id}:{DEFAULT_SNS_TOPIC_NAME}"


def _format_message(message: str, severity: str, source: str | None) -> str:
    """Prepend severity tag + source prefix to the message body."""
    tag = f"[{severity.upper()}]"
    if source:
        return f"{tag} {source}: {message}"
    return f"{tag} {message}"


def _publish_sns(arn: str, message: str, subject: str | None = None) -> ChannelResult:
    try:
        import boto3

        region = arn.split(":")[3] if ":" in arn else DEFAULT_REGION
        client = boto3.client("sns", region_name=region)
        kwargs: dict = {"TopicArn": arn, "Message": message}
        if subject:
            # SNS subject is limited to 100 chars + ASCII + no newlines.
            cleaned = subject.replace("\n", " ").replace("\r", " ")[:100]
            kwargs["Subject"] = cleaned
        resp = client.publish(**kwargs)
        return ChannelResult(ok=True, detail=resp.get("MessageId", "<no id>"))
    except Exception as exc:
        logger.warning("alerts.publish: SNS publish failed: %s", exc)
        return ChannelResult(ok=False, detail=f"sns error: {exc!r}")


def _publish_telegram(message: str, severity: str) -> ChannelResult:
    try:
        from alpha_engine_lib.telegram import send_message

        # Push for error/critical, silent in-channel for info/warning.
        silent = severity.lower() not in SEVERITY_PUSH
        ok = send_message(message, disable_notification=silent)
        return ChannelResult(ok=bool(ok), detail="sent" if ok else "send_message returned False")
    except Exception as exc:  # send_message itself never raises, but defensive
        logger.warning("alerts.publish: Telegram fan-out failed: %s", exc)
        return ChannelResult(ok=False, detail=f"telegram error: {exc!r}")


def publish(
    message: str,
    *,
    severity: str = "error",
    source: str | None = None,
    sns: bool = True,
    telegram: bool = True,
    sns_topic_arn: str | None = None,
) -> PublishResult:
    """Fan out a failure alert to the operator-surveillance channels.

    Default: publish to both ``alpha-engine-alerts`` SNS (→ email) AND
    Telegram (``@nous_ergon_alerts_bot``). Pass ``sns=False`` /
    ``telegram=False`` to suppress individual channels (useful for
    tests, or for callers that have a narrower target).

    :param message: The alert body. Severity tag + source prefix are
        prepended automatically (e.g. ``"[ERROR] spot_backtest.sh: <body>"``).
    :param severity: Free-form severity string (``error`` / ``critical``
        push on Telegram; everything else is silent in-channel). The tag
        is uppercased in the rendered message.
    :param source: Optional source identifier (script path, repo, Lambda
        name) inserted between the tag and the message body. Helps the
        operator triage at a glance.
    :param sns: When ``False``, skip the SNS publish entirely.
    :param telegram: When ``False``, skip the Telegram fan-out entirely.
    :param sns_topic_arn: Explicit topic ARN. Defaults to
        ``arn:aws:sns:{region}:{account_id}:alpha-engine-alerts`` resolved
        from env + STS.
    :returns: :class:`PublishResult` — caller can inspect per-channel
        outcomes. :attr:`PublishResult.any_ok` is the typical success
        gate; :attr:`PublishResult.all_ok` is the strict variant.
    """
    result = PublishResult()
    formatted = _format_message(message, severity, source)

    if sns:
        arn = _resolve_sns_topic_arn(sns_topic_arn)
        if arn is None:
            result.sns = ChannelResult(ok=False, detail="topic ARN resolution failed")
        else:
            # SNS subject — concise header, falls back to severity tag.
            subject = f"Alpha Engine alert [{severity.upper()}]"
            if source:
                subject += f" — {source}"
            result.sns = _publish_sns(arn, formatted, subject=subject)

    if telegram:
        result.telegram = _publish_telegram(formatted, severity=severity)

    return result


# ─── CLI entry ──────────────────────────────────────────────────────────────
# Designed for Bash callers that need failure surveillance from a script
# (spot dispatcher `cleanup` traps, deploy.sh rollback branches, etc.).
# Mirrors the :mod:`alpha_engine_lib.transparency` ``python -m`` pattern so
# Bash callers reach this primitive without bootstrapping a full Python
# project. Exit code is 0 if *any* channel succeeded, 1 if both failed.


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m alpha_engine_lib.alerts",
        description=(
            "Publish a failure alert to alpha-engine's operator-surveillance "
            "channels (SNS topic alpha-engine-alerts + Telegram). Designed "
            "for Bash callers — exit code 0 if any channel succeeded, 1 if "
            "both failed. Never raises."
        ),
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    pub = subparsers.add_parser("publish", help="Publish an alert message.")
    pub.add_argument("--message", required=True, help="Alert body text.")
    pub.add_argument(
        "--severity",
        default="error",
        help=(
            "Severity tag (default: error). 'error' and 'critical' push on "
            "Telegram; all others are silent in-channel."
        ),
    )
    pub.add_argument(
        "--source",
        default=None,
        help=(
            "Optional source identifier (script path, repo, Lambda name) "
            "rendered between the severity tag and the message body."
        ),
    )
    pub.add_argument("--no-sns", action="store_true", help="Skip SNS publish.")
    pub.add_argument("--no-telegram", action="store_true", help="Skip Telegram fan-out.")
    pub.add_argument(
        "--sns-topic-arn",
        default=None,
        help=(
            "Override the SNS topic ARN. Defaults to "
            "arn:aws:sns:{region}:{account_id}:alpha-engine-alerts."
        ),
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)

    result = publish(
        args.message,
        severity=args.severity,
        source=args.source,
        sns=not args.no_sns,
        telegram=not args.no_telegram,
        sns_topic_arn=args.sns_topic_arn,
    )

    # One-line status to stderr (stdout reserved for structured output if
    # any caller starts parsing it). Bash callers can ignore.
    print(
        f"alerts.publish: sns.ok={result.sns.ok} ({result.sns.detail}); "
        f"telegram.ok={result.telegram.ok} ({result.telegram.detail})",
        file=sys.stderr,
    )

    return 0 if result.any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
