"""Verbatim Python parity for the SF JSON ``States.Format`` message templates.

The Step Function JSON files (touched in Phase 3 of the revamp) inline
the success + failure email message bodies via ``States.Format``. These
Python functions render the SAME bodies — used by:

1. Unit tests that assert the SF JSON template's substituted output equals
   the Python rendering byte-for-byte (parity guard against the two
   drifting).
2. Future non-SF consumers (Slack subscriber, ``ae pipeline status`` CLI)
   that want to render the same body without re-implementing the template.

The functions never raise — bad inputs render best-effort placeholder
strings rather than failing the email path, mirroring the SF JSON's
behavior (``States.Format`` substitutes ``$.field`` even if absent).

**Console URL**: the dashboard host is hardcoded here as the lib-canonical
deep-link base. If the dashboard host changes (e.g., new vanity domain),
edit :data:`CONSOLE_BASE_URL` + the SF JSON templates in lockstep — the
parity tests catch the drift.
"""

from __future__ import annotations

from typing import Final

# Hardcoded console base URL. The dashboard is reachable at three hosts:
#
#   - console.nousergon.ai — private (Cloudflare Access gated)
#   - live.nousergon.ai    — public Streamlit page (subset of pages)
#   - <ec2-host>:8501      — direct (debug only)
#
# Page 25 (Pipeline Status) lives on the PRIVATE console — operator-only
# surface. The success / failure emails are operator-only too, so
# console.nousergon.ai is the right deep-link target.
CONSOLE_BASE_URL: Final[str] = "https://console.nousergon.ai"
PIPELINE_STATUS_PAGE: Final[str] = "Pipeline_Status"

# Cause truncation — kept in lockstep with sf-telegram-notifier
# (alpha-engine-data/infrastructure/lambdas/sf-telegram-notifier/index.py L69).
# The SF JSON's ``States.Format`` doesn't truncate; the truncation here is
# only meaningful when a Python consumer (Slack, CLI) renders. The SF JSON
# templates in Phase 3 will use ``States.StringSplit`` + the first N chars
# to approximate.
_CAUSE_MAX_CHARS = 280


def _console_link(execution_arn: str) -> str:
    """Return the page-25 deep-link for a given execution ARN.

    Streamlit's query-string convention is ``?<key>=<value>``; the page
    consumes ``?run=<arn>`` and filters its rendered tables to that
    execution.
    """
    # Streamlit query-string handling tolerates colons + slashes in the
    # value, so no URL encoding is needed for the ARN. Keep this simple
    # so the SF JSON ``States.Format`` template renders the same string.
    return f"{CONSOLE_BASE_URL}/{PIPELINE_STATUS_PAGE}?run={execution_arn}"


def format_success_message(
    *,
    pretty_label: str,
    execution_arn: str,
) -> str:
    """Render the 2-line success email body.

    Body shape (verbatim — the SF JSON ``States.Format`` template renders
    this same string):

        {pretty_label} SUCCEEDED
        Console: {console_link}

    Parameters
    ----------
    pretty_label:
        Human-readable SF label, e.g. ``"Weekly Freshness SF"``. Sourced from
        :data:`nousergon_lib.pipeline_status.registry.PIPELINE_LABELS`.
    execution_arn:
        Full SF execution ARN. Page 25 filters its tables to this ARN
        via the ``?run=`` query string.
    """
    link = _console_link(execution_arn)
    return f"{pretty_label} SUCCEEDED\nConsole: {link}"


def format_failure_message(
    *,
    pretty_label: str,
    execution_arn: str,
    failing_state: str,
    cause: str,
) -> str:
    """Render the 4-line failure email body.

    Body shape (verbatim):

        {pretty_label} FAILED at state {failing_state}
        Console: {console_link}

        Cause (first 280 chars):
        {truncated_cause}

    The Python rendering truncates ``cause`` at :data:`_CAUSE_MAX_CHARS`
    chars with an ellipsis suffix on overflow. The SF JSON template
    approximates via ``States.StringSplit`` + the first N chars; the
    Phase-3 parity test asserts the two render the same string for
    representative cause values.
    """
    link = _console_link(execution_arn)
    snippet = (cause or "").strip()
    if len(snippet) > _CAUSE_MAX_CHARS:
        snippet = snippet[: _CAUSE_MAX_CHARS - 1] + "…"
    return (
        f"{pretty_label} FAILED at state {failing_state}\n"
        f"Console: {link}\n"
        f"\n"
        f"Cause (first {_CAUSE_MAX_CHARS} chars):\n"
        f"{snippet}"
    )
