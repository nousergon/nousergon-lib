"""Unit tests for ``nousergon_lib.pipeline_status.templates``.

These functions render the same email-body strings that the SF JSON
``States.Format`` templates will produce after Phase 3. They exist so
future non-SF consumers (Slack subscriber, ``ae pipeline status`` CLI)
can render byte-identical message bodies without re-implementing the
template; the parity guard against the SF JSON template drift lives
downstream in the alpha-engine-data Phase 3 PR.
"""

from __future__ import annotations

from nousergon_lib.pipeline_status import templates


# ── format_success_message ────────────────────────────────────────────────


def test_format_success_message_basic_shape():
    out = templates.format_success_message(
        pretty_label="Saturday SF",
        execution_arn="arn:aws:states:us-east-1:711398986525:execution:alpha-engine-saturday-pipeline:run-abc",
    )
    expected = (
        "Saturday SF SUCCEEDED\n"
        "Console: https://console.nousergon.ai/Pipeline_Status"
        "?run=arn:aws:states:us-east-1:711398986525:execution:"
        "alpha-engine-saturday-pipeline:run-abc"
    )
    assert out == expected


def test_format_success_message_for_each_pipeline_label():
    """Render for all 3 pipelines — pin that the body shape is consistent."""
    for label in ("Saturday SF", "Weekday SF", "EOD SF"):
        out = templates.format_success_message(
            pretty_label=label, execution_arn="arn:fake"
        )
        assert out.startswith(f"{label} SUCCEEDED\n")
        assert "Console: https://console.nousergon.ai/Pipeline_Status?run=arn:fake" in out


def test_format_success_message_is_two_lines():
    """Body shape contract — success is exactly 2 lines (per plan doc §3.4)."""
    out = templates.format_success_message(pretty_label="X", execution_arn="y")
    assert out.count("\n") == 1, "Success body MUST be exactly 2 lines"


# ── format_failure_message ────────────────────────────────────────────────


def test_format_failure_message_basic_shape():
    out = templates.format_failure_message(
        pretty_label="Weekday SF",
        execution_arn="arn:aws:states:us-east-1:711398986525:execution:alpha-engine-weekday-pipeline:run-xyz",
        failing_state="MorningEnrich",
        cause="States.TaskFailed: exit 1",
    )
    assert out.startswith("Weekday SF FAILED at state MorningEnrich\n")
    assert (
        "Console: https://console.nousergon.ai/Pipeline_Status"
        "?run=arn:aws:states:us-east-1:711398986525:execution:"
        "alpha-engine-weekday-pipeline:run-xyz" in out
    )
    assert "Cause (first 280 chars):\nStates.TaskFailed: exit 1" in out


def test_format_failure_message_truncates_long_cause():
    """Cause field MUST truncate at 280 chars with ellipsis suffix."""
    long_cause = "A" * 500
    out = templates.format_failure_message(
        pretty_label="EOD SF",
        execution_arn="arn:test",
        failing_state="EODReconcile",
        cause=long_cause,
    )
    # Last line of body is the truncated cause; check it ends with the ellipsis
    cause_line = out.rsplit("\n", 1)[-1]
    assert cause_line.endswith("…")
    # Total length of the cause snippet = 279 chars of A + the ellipsis = 280
    assert len(cause_line) == 280
    assert cause_line[:279] == "A" * 279


def test_format_failure_message_passes_through_short_cause_untruncated():
    short = "boom: exit 1"
    out = templates.format_failure_message(
        pretty_label="Saturday SF",
        execution_arn="arn:test",
        failing_state="DataPhase1",
        cause=short,
    )
    assert out.endswith(short)
    assert "…" not in out  # no truncation marker


def test_format_failure_message_handles_empty_cause():
    """Empty cause renders empty after the 'Cause' header — never raises."""
    out = templates.format_failure_message(
        pretty_label="Weekday SF",
        execution_arn="arn:test",
        failing_state="PredictorInference",
        cause="",
    )
    assert out.endswith("Cause (first 280 chars):\n")


def test_format_failure_message_strips_whitespace_on_cause():
    out = templates.format_failure_message(
        pretty_label="EOD SF",
        execution_arn="arn:test",
        failing_state="EODReconcile",
        cause="   actual error message   \n",
    )
    assert out.endswith("actual error message")


def test_console_link_is_hardcoded_to_private_console():
    """The deep-link MUST point to the private console (Cloudflare Access
    gated); operator-only surface for operator-only emails. Drift would
    leak the link to the public site, which doesn't host page 25."""
    assert templates.CONSOLE_BASE_URL == "https://console.nousergon.ai"
    assert templates.PIPELINE_STATUS_PAGE == "Pipeline_Status"
