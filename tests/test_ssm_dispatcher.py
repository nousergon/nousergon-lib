"""
Unit tests for ``alpha_engine_lib.ssm_dispatcher``.

Pins the institutional-chokepoint contract that the three alpha-engine
dispatcher scripts (alpha-engine-predictor ``spot_train.sh``,
alpha-engine-data ``spot_data_weekly.sh``,
alpha-engine-backtester ``spot_backtest.sh``) will rely on after the
2026-05-26 ``run_ssm`` bash-helper lift to lib:

* Success → exit 0; Failed / TimedOut / Cancelled / TerminalError →
  exit 1; unknown status → exit 1 (no silent skip per
  ``[[feedback_no_silent_fails]]``).
* StdOut streaming: deltas are written to the dispatcher's stdout as
  the inner command produces output; 24KB-cap rotation is detected and
  surfaced (without aborting the poll).
* StdErr is emitted to the dispatcher's stderr only on terminal
  non-Success, prefixed with the description so a multi-step log is
  greppable.
* InvocationDoesNotExist during the registration grace window
  (≤60s post-SendCommand) keeps polling; after the grace window it
  fails loud — closes the 2026-05-23 substrate weakness at the
  chokepoint level rather than per-SF JSON Retry block.
* Throttling exceptions during polling are transient (keep polling).
* The script body is base64-wrapped before transport so embedded
  heredocs, Python, and single quotes survive AWS-RunShellScript.
* The CLI accepts the script body from stdin or --script-file; empty
  body is refused with exit 2.
"""

from __future__ import annotations

import io
import sys
from unittest.mock import MagicMock

import pytest

from alpha_engine_lib import ssm_dispatcher


class _FakeClientError(Exception):
    """Stand-in for botocore.exceptions.ClientError carrying .response."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{code}: {message}")
        self.response = {"Error": {"Code": code, "Message": message}}


def _fake_ssm(*, send_resp=None, send_raises=None, poll_sequence=None,
              poll_raises_sequence=None):
    """Build a MagicMock ssm client driven by ``poll_sequence``.

    Args:
        send_resp: dict to return from send_command. Defaults to a
            CommandId.
        send_raises: exception to raise from send_command instead.
        poll_sequence: list of dicts returned by successive
            get_command_invocation calls.
        poll_raises_sequence: optional list of exceptions/Nones. None
            entries mean "return the next dict from poll_sequence";
            non-None means raise that exception on that call.
    """
    ssm = MagicMock()
    if send_raises is not None:
        ssm.send_command.side_effect = send_raises
    else:
        ssm.send_command.return_value = send_resp or {
            "Command": {"CommandId": "cmd-abc"}
        }
    if poll_raises_sequence:
        # Combine raises + returns so order is preserved
        def _next(*a, **k):
            try:
                exc = poll_raises_sequence.pop(0)
            except IndexError:
                exc = None
            if exc is not None:
                raise exc
            return (poll_sequence or []).pop(0)

        ssm.get_command_invocation.side_effect = _next
    elif poll_sequence is not None:
        ssm.get_command_invocation.side_effect = list(poll_sequence)
    return ssm


# ---------------------------------------------------------------------------
# Encode/transport
# ---------------------------------------------------------------------------


class TestEncodeCommandPayload:
    def test_wraps_with_base64_decode_pipe(self):
        wrapped = ssm_dispatcher._encode_command_payload("echo hello")
        assert wrapped.startswith("echo ")
        assert wrapped.endswith(" | base64 -d | bash")

    def test_round_trips_heredoc_with_quotes_unchanged(self):
        import base64

        script = "python3 - <<'PY'\nprint('hi')\nPY\n"
        wrapped = ssm_dispatcher._encode_command_payload(script)
        token = wrapped.split()[1]
        decoded = base64.b64decode(token).decode("utf-8")
        assert decoded == script

    def test_empty_script_still_encodes(self):
        wrapped = ssm_dispatcher._encode_command_payload("")
        # base64 of empty string is empty; the wrapper still emits the
        # full pipeline shape so any future caller behavior is consistent.
        assert "base64 -d | bash" in wrapped


# ---------------------------------------------------------------------------
# Happy path: Success on first/second poll
# ---------------------------------------------------------------------------


class TestRunSuccess:
    def test_success_first_poll_returns_zero(self):
        ssm = _fake_ssm(
            poll_sequence=[{"Status": "Success", "StandardOutputContent": "ok\n"}]
        )
        out = io.StringIO()
        err = io.StringIO()
        rc = ssm_dispatcher.run(
            "i-abc",
            "bootstrap",
            "echo ok",
            output_bucket="bkt",
            output_key_prefix="pfx",
            stdout_stream=out,
            stderr_stream=err,
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert rc == 0
        assert "ok" in out.getvalue()
        # command-id line lands on stderr (operator-visible)
        assert "cmd-abc" in err.getvalue()

    def test_success_after_inprogress_streams_deltas(self):
        ssm = _fake_ssm(
            poll_sequence=[
                {"Status": "InProgress", "StandardOutputContent": "line1\n"},
                {
                    "Status": "InProgress",
                    "StandardOutputContent": "line1\nline2\n",
                },
                {
                    "Status": "Success",
                    "StandardOutputContent": "line1\nline2\nline3\n",
                },
            ]
        )
        out = io.StringIO()
        rc = ssm_dispatcher.run(
            "i-abc",
            "bootstrap",
            "x",
            stdout_stream=out,
            stderr_stream=io.StringIO(),
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert rc == 0
        full = out.getvalue()
        assert full.count("line1") == 1  # delta-streamed, not repeated
        assert "line2" in full and "line3" in full
        # Streamed in order
        assert full.index("line1") < full.index("line2") < full.index("line3")

    def test_buffer_rotation_at_24kb_emits_cap_note(self):
        # Simulate buffer shrinking (cap hit + rotation)
        ssm = _fake_ssm(
            poll_sequence=[
                {"Status": "InProgress", "StandardOutputContent": "X" * 24000},
                {"Status": "InProgress", "StandardOutputContent": "later"},
                {
                    "Status": "Success",
                    "StandardOutputContent": "later",
                },
            ]
        )
        out = io.StringIO()
        err = io.StringIO()
        rc = ssm_dispatcher.run(
            "i",
            "ragingest",
            "x",
            output_bucket="b",
            output_key_prefix="p",
            stdout_stream=out,
            stderr_stream=err,
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert rc == 0
        assert "24KB cap" in err.getvalue()
        assert "s3://b/p" in err.getvalue()

    def test_buffer_rotation_without_bucket_emits_configure_hint(self):
        ssm = _fake_ssm(
            poll_sequence=[
                {"Status": "InProgress", "StandardOutputContent": "X" * 24000},
                {"Status": "InProgress", "StandardOutputContent": "y"},
                {"Status": "Success", "StandardOutputContent": "y"},
            ]
        )
        err = io.StringIO()
        rc = ssm_dispatcher.run(
            "i",
            "ragingest",
            "x",
            stdout_stream=io.StringIO(),
            stderr_stream=err,
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert rc == 0
        assert "configure --output-bucket" in err.getvalue()


# ---------------------------------------------------------------------------
# Terminal non-Success paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status", ["Failed", "TimedOut", "Cancelled", "Cancelling", "TerminalError"]
)
class TestRunTerminalFailures:
    def test_terminal_status_returns_one(self, status):
        ssm = _fake_ssm(
            poll_sequence=[
                {
                    "Status": status,
                    "StandardOutputContent": "",
                    "StandardErrorContent": "oops",
                }
            ]
        )
        err = io.StringIO()
        rc = ssm_dispatcher.run(
            "i",
            "boom",
            "x",
            stdout_stream=io.StringIO(),
            stderr_stream=err,
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert rc == 1
        text = err.getvalue()
        assert "'boom'" in text
        assert status in text
        assert "oops" in text

    def test_terminal_without_stderr_still_returns_one(self, status):
        ssm = _fake_ssm(
            poll_sequence=[{"Status": status, "StandardOutputContent": ""}]
        )
        rc = ssm_dispatcher.run(
            "i",
            "boom",
            "x",
            stdout_stream=io.StringIO(),
            stderr_stream=io.StringIO(),
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert rc == 1


class TestUnknownStatus:
    def test_unknown_status_fails_loud(self):
        # Per [[feedback_no_silent_fails]] — anything outside the
        # documented {Success | terminal-non-Success | pending} set
        # MUST surface, not get treated as Pending forever.
        ssm = _fake_ssm(
            poll_sequence=[{"Status": "Confused", "StandardOutputContent": ""}]
        )
        err = io.StringIO()
        rc = ssm_dispatcher.run(
            "i",
            "unknown",
            "x",
            stdout_stream=io.StringIO(),
            stderr_stream=err,
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert rc == 1
        assert "unknown status" in err.getvalue()


# ---------------------------------------------------------------------------
# Registration race (InvocationDoesNotExist)
# ---------------------------------------------------------------------------


class TestInvocationDoesNotExistGrace:
    def test_within_grace_window_keeps_polling(self):
        # 2 IDNE in a row, then Success — must NOT fail.
        ssm = MagicMock()
        ssm.send_command.return_value = {"Command": {"CommandId": "cmd-x"}}

        polls = [
            _FakeClientError("InvocationDoesNotExist"),
            _FakeClientError("InvocationDoesNotExist"),
            {"Status": "Success", "StandardOutputContent": "ok"},
        ]

        def _next(*a, **k):
            v = polls.pop(0)
            if isinstance(v, Exception):
                raise v
            return v

        ssm.get_command_invocation.side_effect = _next

        # Fake monotonic clock — never exceeds the grace window
        t = [0.0]

        def _mono():
            return t[0]

        def _sleep(_):
            t[0] += 5.0

        rc = ssm_dispatcher.run(
            "i",
            "race",
            "x",
            stdout_stream=io.StringIO(),
            stderr_stream=io.StringIO(),
            sleep=_sleep,
            monotonic=_mono,
            boto3_client=ssm,
        )
        assert rc == 0
        # All three poll cycles consumed
        assert ssm.get_command_invocation.call_count == 3

    def test_past_grace_window_fails_loud(self):
        # IDNE persisting past 60s grace → exit 1 with named reason.
        ssm = MagicMock()
        ssm.send_command.return_value = {"Command": {"CommandId": "cmd-x"}}
        ssm.get_command_invocation.side_effect = _FakeClientError(
            "InvocationDoesNotExist"
        )

        t = [0.0]

        def _mono():
            return t[0]

        def _sleep(_):
            t[0] += 30.0  # 30s/poll → cross 60s at poll 3

        err = io.StringIO()
        rc = ssm_dispatcher.run(
            "i",
            "race",
            "x",
            stdout_stream=io.StringIO(),
            stderr_stream=err,
            sleep=_sleep,
            monotonic=_mono,
            boto3_client=ssm,
        )
        assert rc == 1
        assert "InvocationDoesNotExist" in err.getvalue()
        assert "never registered" in err.getvalue()


# ---------------------------------------------------------------------------
# Throttling: transient
# ---------------------------------------------------------------------------


class TestPollThrottling:
    def test_throttling_keeps_polling(self):
        ssm = MagicMock()
        ssm.send_command.return_value = {"Command": {"CommandId": "c"}}
        polls = [
            _FakeClientError("ThrottlingException"),
            _FakeClientError("RequestLimitExceeded"),
            {"Status": "Success", "StandardOutputContent": ""},
        ]

        def _next(*a, **k):
            v = polls.pop(0)
            if isinstance(v, Exception):
                raise v
            return v

        ssm.get_command_invocation.side_effect = _next
        rc = ssm_dispatcher.run(
            "i",
            "thr",
            "x",
            stdout_stream=io.StringIO(),
            stderr_stream=io.StringIO(),
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert rc == 0

    def test_unexpected_poll_exception_fails_loud(self):
        ssm = MagicMock()
        ssm.send_command.return_value = {"Command": {"CommandId": "c"}}
        ssm.get_command_invocation.side_effect = RuntimeError("network gone")

        err = io.StringIO()
        rc = ssm_dispatcher.run(
            "i",
            "boom",
            "x",
            stdout_stream=io.StringIO(),
            stderr_stream=err,
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert rc == 1
        assert "RuntimeError" in err.getvalue()
        assert "network gone" in err.getvalue()


# ---------------------------------------------------------------------------
# Send-command failures
# ---------------------------------------------------------------------------


class TestSendCommandFailure:
    def test_send_raises_returns_one_with_reason(self):
        ssm = _fake_ssm(send_raises=_FakeClientError("AccessDeniedException"))
        err = io.StringIO()
        rc = ssm_dispatcher.run(
            "i",
            "denied",
            "x",
            stdout_stream=io.StringIO(),
            stderr_stream=err,
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert rc == 1
        assert "send_command failed" in err.getvalue()
        assert "AccessDeniedException" in err.getvalue() or "_FakeClientError" in err.getvalue()

    def test_send_returns_no_command_id_returns_one(self):
        ssm = _fake_ssm(send_resp={"Command": {}})
        err = io.StringIO()
        rc = ssm_dispatcher.run(
            "i",
            "weird",
            "x",
            stdout_stream=io.StringIO(),
            stderr_stream=err,
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert rc == 1
        assert "no CommandId" in err.getvalue()


# ---------------------------------------------------------------------------
# SendCommand parameter shape
# ---------------------------------------------------------------------------


class TestSendCommandShape:
    def test_passes_base64_wrapped_payload(self):
        ssm = _fake_ssm(
            poll_sequence=[{"Status": "Success", "StandardOutputContent": ""}]
        )
        ssm_dispatcher.run(
            "i-123",
            "boot",
            "echo wow",
            timeout_seconds=900,
            output_bucket="bkt",
            output_key_prefix="pfx",
            region="us-east-1",
            stdout_stream=io.StringIO(),
            stderr_stream=io.StringIO(),
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert ssm.send_command.call_count == 1
        kw = ssm.send_command.call_args.kwargs
        assert kw["InstanceIds"] == ["i-123"]
        assert kw["DocumentName"] == "AWS-RunShellScript"
        assert kw["TimeoutSeconds"] == 900
        assert kw["OutputS3BucketName"] == "bkt"
        assert kw["OutputS3KeyPrefix"] == "pfx"
        # Payload is base64-wrapped (the inner command body)
        cmds = kw["Parameters"]["commands"]
        assert len(cmds) == 1
        assert cmds[0].endswith(" | base64 -d | bash")

    def test_description_truncated_to_100_for_comment(self):
        ssm = _fake_ssm(
            poll_sequence=[{"Status": "Success", "StandardOutputContent": ""}]
        )
        long_desc = "x" * 250
        ssm_dispatcher.run(
            "i",
            long_desc,
            "echo",
            stdout_stream=io.StringIO(),
            stderr_stream=io.StringIO(),
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        assert len(ssm.send_command.call_args.kwargs["Comment"]) == 100

    def test_omits_output_s3_when_not_configured(self):
        ssm = _fake_ssm(
            poll_sequence=[{"Status": "Success", "StandardOutputContent": ""}]
        )
        ssm_dispatcher.run(
            "i",
            "no-s3",
            "echo",
            stdout_stream=io.StringIO(),
            stderr_stream=io.StringIO(),
            sleep=lambda s: None,
            boto3_client=ssm,
        )
        kw = ssm.send_command.call_args.kwargs
        assert "OutputS3BucketName" not in kw
        assert "OutputS3KeyPrefix" not in kw


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_run_subcommand_reads_script_from_stdin(self, monkeypatch):
        captured: dict = {}

        def fake_run(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return 0

        monkeypatch.setattr(ssm_dispatcher, "run", fake_run)
        monkeypatch.setattr(sys, "stdin", io.StringIO("echo cli\n"))
        rc = ssm_dispatcher.main(
            [
                "run",
                "--instance-id",
                "i-1",
                "--description",
                "cli-test",
                "--timeout",
                "120",
                "--region",
                "us-east-1",
                "--script-stdin",
            ]
        )
        assert rc == 0
        assert captured["kwargs"]["instance_id"] == "i-1"
        assert captured["kwargs"]["script"] == "echo cli\n"
        assert captured["kwargs"]["timeout_seconds"] == 120

    def test_run_subcommand_reads_script_from_file(self, tmp_path, monkeypatch):
        script_file = tmp_path / "body.sh"
        script_file.write_text("echo from-file\n")

        captured: dict = {}

        def fake_run(*args, **kwargs):
            captured["script"] = kwargs["script"]
            return 0

        monkeypatch.setattr(ssm_dispatcher, "run", fake_run)
        rc = ssm_dispatcher.main(
            [
                "run",
                "--instance-id",
                "i-1",
                "--description",
                "cli-file",
                "--script-file",
                str(script_file),
            ]
        )
        assert rc == 0
        assert captured["script"] == "echo from-file\n"

    def test_empty_script_refused_with_exit_two(self, monkeypatch):
        monkeypatch.setattr(sys, "stdin", io.StringIO("   \n"))
        rc = ssm_dispatcher.main(
            [
                "run",
                "--instance-id",
                "i-1",
                "--description",
                "empty",
                "--script-stdin",
            ]
        )
        assert rc == 2

    def test_mutually_exclusive_script_source(self, tmp_path):
        script_file = tmp_path / "body.sh"
        script_file.write_text("x")
        with pytest.raises(SystemExit):
            ssm_dispatcher.main(
                [
                    "run",
                    "--instance-id",
                    "i",
                    "--description",
                    "x",
                    "--script-file",
                    str(script_file),
                    "--script-stdin",
                ]
            )

    def test_missing_script_source_errors(self):
        with pytest.raises(SystemExit):
            ssm_dispatcher.main(
                ["run", "--instance-id", "i", "--description", "x"]
            )

    def test_missing_subcommand_errors(self):
        with pytest.raises(SystemExit):
            ssm_dispatcher.main([])

    def test_help_exits_clean(self):
        with pytest.raises(SystemExit) as exc:
            ssm_dispatcher.main(["--help"])
        assert exc.value.code == 0


# ---------------------------------------------------------------------------
# Module entrypoint
# ---------------------------------------------------------------------------


class TestModuleEntrypoint:
    """The module is invokable as ``python -m alpha_engine_lib.ssm_dispatcher``."""

    def test_module_exposes_main(self):
        assert callable(ssm_dispatcher.main)

    def test_module_exposes_run(self):
        assert callable(ssm_dispatcher.run)
