"""
Unit tests for ``alpha_engine_lib.alerts``.

Pins the failure-surveillance fan-out contract: per-channel independence
(SNS failure doesn't block Telegram and vice-versa), severity-to-push
mapping (error/critical push, info/warning silent), CLI exit codes
(0 if any channel succeeded, 1 only if both failed), and message
formatting (``[SEVERITY] source: body``).

Designed so the Bash dispatcher consumers — spot_backtest.sh's cleanup
trap, the L117 Lambda-deploying repos' canary-rollback branches — can
rely on stable contract semantics across lib versions.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from alpha_engine_lib import alerts


@pytest.fixture
def fake_boto3():
    """boto3 stub that returns mocked SNS + STS clients keyed by service."""
    sts_client = MagicMock()
    sts_client.get_caller_identity.return_value = {"Account": "711398986525"}
    sns_client = MagicMock()
    sns_client.publish.return_value = {"MessageId": "test-msg-id-abc123"}

    fake = MagicMock()

    def _client(service: str, **kwargs):
        if service == "sts":
            return sts_client
        if service == "sns":
            return sns_client
        raise AssertionError(f"unexpected boto3 client request: {service}")

    fake.client.side_effect = _client
    return fake, sts_client, sns_client


class TestFormatMessage:
    def test_with_source(self):
        assert alerts._format_message("boom", "error", "spot_backtest.sh") == "[ERROR] spot_backtest.sh: boom"

    def test_without_source(self):
        assert alerts._format_message("boom", "warning", None) == "[WARNING] boom"

    def test_severity_uppercased(self):
        assert alerts._format_message("x", "Info", "src") == "[INFO] src: x"


class TestResolveSnsTopicArn:
    def test_explicit_override(self, monkeypatch):
        arn = "arn:aws:sns:us-west-2:000000000000:custom-topic"
        assert alerts._resolve_sns_topic_arn(arn) == arn

    def test_defaults_from_env_and_sts(self, monkeypatch, fake_boto3):
        fake, sts, _ = fake_boto3
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        with patch.object(alerts, "__name__", alerts.__name__):
            with patch.dict("sys.modules", {"boto3": fake}):
                result = alerts._resolve_sns_topic_arn(None)
        assert result == "arn:aws:sns:us-east-1:711398986525:alpha-engine-alerts"
        sts.get_caller_identity.assert_called_once()

    def test_returns_none_when_sts_fails(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        fake = MagicMock()
        fake.client.side_effect = RuntimeError("no creds")
        with patch.dict("sys.modules", {"boto3": fake}):
            assert alerts._resolve_sns_topic_arn(None) is None


class TestPublish:
    def test_both_channels_succeed(self, fake_boto3):
        fake, _sts, sns = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=True, detail="sent")):
                result = alerts.publish("boom", source="spot_backtest.sh")
        assert result.sns.ok is True
        assert result.telegram.ok is True
        assert result.any_ok is True
        assert result.all_ok is True
        # SNS publish was called with severity-tagged message + readable subject
        kwargs = sns.publish.call_args.kwargs
        assert "[ERROR] spot_backtest.sh: boom" in kwargs["Message"]
        assert kwargs["Subject"].startswith("Alpha Engine alert [ERROR]")

    def test_sns_failure_doesnt_block_telegram(self, fake_boto3):
        fake, _sts, sns = fake_boto3
        sns.publish.side_effect = RuntimeError("topic ARN bad")
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=True, detail="sent")):
                result = alerts.publish("boom", source="x")
        assert result.sns.ok is False
        assert "sns error" in result.sns.detail
        assert result.telegram.ok is True
        assert result.any_ok is True
        assert result.all_ok is False

    def test_telegram_failure_doesnt_block_sns(self, fake_boto3):
        fake, _sts, _sns = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=False, detail="creds missing")):
                result = alerts.publish("boom", source="x")
        assert result.sns.ok is True
        assert result.telegram.ok is False
        assert result.any_ok is True

    def test_both_failures(self, fake_boto3):
        fake, _sts, sns = fake_boto3
        sns.publish.side_effect = RuntimeError("nope")
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=False, detail="creds missing")):
                result = alerts.publish("boom", source="x")
        assert result.any_ok is False
        assert result.all_ok is False

    def test_sns_disabled(self, fake_boto3):
        fake, _sts, sns = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=True, detail="sent")):
                result = alerts.publish("boom", sns=False)
        sns.publish.assert_not_called()
        assert result.sns.ok is False
        assert "not attempted" in result.sns.detail
        assert result.telegram.ok is True

    def test_telegram_disabled(self, fake_boto3):
        fake, _sts, _sns = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram") as tg:
                result = alerts.publish("boom", telegram=False)
        tg.assert_not_called()
        assert result.sns.ok is True
        assert result.telegram.ok is False

    def test_severity_push_mapping(self, fake_boto3):
        """error/critical → disable_notification=False (push);
        info/warning → disable_notification=True (silent)."""
        fake, _sts, _sns = fake_boto3
        from alpha_engine_lib import telegram as tg_mod

        with patch.dict("sys.modules", {"boto3": fake}):
            for sev, expect_silent in [
                ("error", False),
                ("critical", False),
                ("warning", True),
                ("info", True),
            ]:
                with patch.object(tg_mod, "send_message", return_value=True) as send:
                    alerts.publish("x", severity=sev)
                    silent_kwarg = send.call_args.kwargs.get("disable_notification")
                    assert silent_kwarg is expect_silent, f"severity={sev}: expected silent={expect_silent} got {silent_kwarg}"

    def test_sns_subject_truncated_and_sanitized(self, fake_boto3):
        fake, _sts, sns = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=True)):
                alerts.publish("body", source="x" * 150)
        subject = sns.publish.call_args.kwargs["Subject"]
        assert len(subject) <= 100
        assert "\n" not in subject

    def test_never_raises_on_publish_exception(self):
        # Even with no creds + no mocks at all, publish must not raise.
        result = alerts.publish("boom", source="test", sns_topic_arn=None)
        # Either may fail; result is structured.
        assert isinstance(result, alerts.PublishResult)


class TestCli:
    def test_publish_subcommand_calls_publish(self, fake_boto3):
        fake, _sts, sns = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=True, detail="sent")):
                rc = alerts.main([
                    "publish",
                    "--message", "boom",
                    "--severity", "error",
                    "--source", "spot_backtest.sh",
                ])
        assert rc == 0
        assert sns.publish.called

    def test_exit_code_1_when_both_channels_fail(self, fake_boto3):
        fake, _sts, sns = fake_boto3
        sns.publish.side_effect = RuntimeError("nope")
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=False, detail="creds missing")):
                rc = alerts.main([
                    "publish",
                    "--message", "boom",
                ])
        assert rc == 1

    def test_exit_code_0_when_only_one_channel_ok(self, fake_boto3):
        fake, _sts, sns = fake_boto3
        sns.publish.side_effect = RuntimeError("nope")
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=True, detail="sent")):
                rc = alerts.main([
                    "publish",
                    "--message", "boom",
                ])
        assert rc == 0

    def test_no_sns_flag(self, fake_boto3):
        fake, _sts, sns = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=True, detail="sent")):
                rc = alerts.main(["publish", "--message", "x", "--no-sns"])
        sns.publish.assert_not_called()
        assert rc == 0

    def test_no_telegram_flag(self, fake_boto3):
        fake, _sts, _sns = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram") as tg:
                rc = alerts.main(["publish", "--message", "x", "--no-telegram"])
        tg.assert_not_called()
        assert rc == 0

    def test_custom_sns_topic_arn(self, fake_boto3):
        fake, _sts, sns = fake_boto3
        custom = "arn:aws:sns:us-west-2:000000000000:custom"
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=True, detail="sent")):
                alerts.main(["publish", "--message", "x", "--sns-topic-arn", custom])
        assert sns.publish.call_args.kwargs["TopicArn"] == custom

    def test_missing_message_arg_fails(self):
        with pytest.raises(SystemExit):
            alerts.main(["publish"])
