"""Unit tests for ``nousergon_lib.spot_dispatch``.

Pins the contract for the shared dispatch primitives extracted (config#2106)
from `scheduled-groom-dispatcher`/`ci-watch-dispatcher` (nousergon-data) — a
third independent implementation (`sf-watch-spot-dispatcher`) is built
against this module rather than duplicating it again.

These tests pin BEHAVIOR PARITY with the two pre-extraction implementations:
spot-then-on-demand fallback on ``SpotCapacityExhausted``, fail-safe-open
concurrency checks, best-effort (never-raising) terminate-on-failure.
"""

from __future__ import annotations

from unittest import mock

import pytest

from nousergon_lib import spot_dispatch
from nousergon_lib.ec2_spot import SpotCapacityExhausted, SpotLaunchError

REGION = "us-east-1"


def _common_launch_kwargs(**overrides):
    kwargs = dict(
        instance_types=["t3.medium", "t3a.medium"],
        subnets=["subnet-aaa", "subnet-bbb"],
        image_id="ami-0123456789abcdef0",
        key_name="alpha-engine-key",
        security_group_ids=["sg-0123456789abcdef0"],
        iam_instance_profile="alpha-engine-example-executor-profile",
        volume_size_gb=40,
        tag_name="alpha-engine-example-spot",
        region=REGION,
    )
    kwargs.update(overrides)
    return kwargs


class TestLaunchWithFallback:
    @mock.patch("nousergon_lib.spot_dispatch.ec2_spot")
    def test_spot_success_returns_market_spot(self, mock_ec2_spot):
        mock_ec2_spot.launch.return_value = "i-spot123"

        instance_id, market = spot_dispatch.launch_with_fallback(**_common_launch_kwargs())

        assert (instance_id, market) == ("i-spot123", "spot")
        mock_ec2_spot.launch.assert_called_once()
        _, kwargs = mock_ec2_spot.launch.call_args
        assert kwargs["spot"] is True

    @mock.patch("nousergon_lib.spot_dispatch.ec2_spot")
    def test_spot_capacity_exhausted_falls_back_to_on_demand(self, mock_ec2_spot):
        mock_ec2_spot.launch.side_effect = [
            SpotCapacityExhausted("no capacity"),
            "i-ondemand456",
        ]

        instance_id, market = spot_dispatch.launch_with_fallback(**_common_launch_kwargs())

        assert (instance_id, market) == ("i-ondemand456", "on-demand")
        assert mock_ec2_spot.launch.call_count == 2
        first_call_kwargs = mock_ec2_spot.launch.call_args_list[0].kwargs
        second_call_kwargs = mock_ec2_spot.launch.call_args_list[1].kwargs
        assert first_call_kwargs["spot"] is True
        assert second_call_kwargs["spot"] is False

    @mock.patch("nousergon_lib.spot_dispatch.ec2_spot")
    def test_force_on_demand_skips_spot_attempt_entirely(self, mock_ec2_spot):
        mock_ec2_spot.launch.return_value = "i-ondemand789"

        instance_id, market = spot_dispatch.launch_with_fallback(
            **_common_launch_kwargs(force_on_demand=True)
        )

        assert (instance_id, market) == ("i-ondemand789", "on-demand")
        mock_ec2_spot.launch.assert_called_once()
        _, kwargs = mock_ec2_spot.launch.call_args
        assert kwargs["spot"] is False

    @mock.patch("nousergon_lib.spot_dispatch.ec2_spot")
    def test_launch_error_propagates_when_both_attempts_fail(self, mock_ec2_spot):
        mock_ec2_spot.launch.side_effect = SpotCapacityExhausted("exhausted everywhere")

        with pytest.raises(SpotCapacityExhausted):
            spot_dispatch.launch_with_fallback(**_common_launch_kwargs())

    @mock.patch("nousergon_lib.spot_dispatch.ec2_spot")
    def test_non_capacity_launch_error_propagates_without_fallback(self, mock_ec2_spot):
        mock_ec2_spot.launch.side_effect = SpotLaunchError("some other failure")

        with pytest.raises(SpotLaunchError):
            spot_dispatch.launch_with_fallback(**_common_launch_kwargs())
        mock_ec2_spot.launch.assert_called_once()


class TestWaitSsmOnline:
    @mock.patch("nousergon_lib.spot_dispatch.boto3")
    def test_returns_once_ssm_agent_online(self, mock_boto3):
        mock_ec2 = mock.MagicMock()
        mock_ssm = mock.MagicMock()
        mock_boto3.client.side_effect = lambda service, region_name: {
            "ec2": mock_ec2, "ssm": mock_ssm
        }[service]
        mock_ssm.describe_instance_information.return_value = {
            "InstanceInformationList": [{"PingStatus": "Online"}]
        }

        spot_dispatch.wait_ssm_online("i-abc123", region=REGION, ssm_online_budget_sec=5)

        mock_ec2.get_waiter.assert_called_once_with("instance_running")
        mock_ssm.describe_instance_information.assert_called()

    @mock.patch("nousergon_lib.spot_dispatch.time.sleep", return_value=None)
    @mock.patch("nousergon_lib.spot_dispatch.time.time")
    @mock.patch("nousergon_lib.spot_dispatch.boto3")
    def test_raises_runtime_error_on_timeout(self, mock_boto3, mock_time, mock_sleep):
        mock_ec2 = mock.MagicMock()
        mock_ssm = mock.MagicMock()
        mock_boto3.client.side_effect = lambda service, region_name: {
            "ec2": mock_ec2, "ssm": mock_ssm
        }[service]
        mock_ssm.describe_instance_information.return_value = {"InstanceInformationList": []}
        # deadline = start(0) + budget(5); time() called once for deadline calc,
        # then twice per loop iteration checks — force exactly one loop pass then expire.
        mock_time.side_effect = [0, 0, 10]

        with pytest.raises(RuntimeError, match="SSM agent not Online"):
            spot_dispatch.wait_ssm_online("i-abc123", region=REGION, ssm_online_budget_sec=5)


class TestSendAsyncCommand:
    @mock.patch("nousergon_lib.spot_dispatch.boto3")
    def test_sends_command_and_returns_command_id(self, mock_boto3):
        mock_ssm = mock.MagicMock()
        mock_boto3.client.return_value = mock_ssm
        mock_ssm.send_command.return_value = {"Command": {"CommandId": "cmd-123"}}

        command_id = spot_dispatch.send_async_command(
            "i-abc123",
            "echo hello",
            comment="test comment",
            region=REGION,
            cw_log_group="/alpha-engine/example-spot",
            execution_timeout_seconds=3600,
        )

        assert command_id == "cmd-123"
        _, kwargs = mock_ssm.send_command.call_args
        assert kwargs["InstanceIds"] == ["i-abc123"]
        assert kwargs["Parameters"]["commands"] == ["echo hello"]
        assert kwargs["Parameters"]["executionTimeout"] == ["3600"]
        assert kwargs["TimeoutSeconds"] == 600
        assert kwargs["Comment"] == "test comment"


class TestRunningInstanceIds:
    @mock.patch("nousergon_lib.spot_dispatch.boto3")
    def test_builds_filters_from_tag_name_and_discriminators(self, mock_boto3):
        mock_ec2 = mock.MagicMock()
        mock_boto3.client.return_value = mock_ec2
        mock_ec2.describe_instances.return_value = {
            "Reservations": [{"Instances": [{"InstanceId": "i-existing"}]}]
        }

        result = spot_dispatch.running_instance_ids(
            "alpha-engine-example-spot",
            {"example-cadence": "saturday", "example-pipeline": "ne-weekly-freshness-pipeline"},
            region=REGION,
        )

        assert result == ["i-existing"]
        _, kwargs = mock_ec2.describe_instances.call_args
        filters = kwargs["Filters"]
        assert {"Name": "tag:Name", "Values": ["alpha-engine-example-spot"]} in filters
        assert {"Name": "instance-state-name", "Values": ["pending", "running"]} in filters
        assert {"Name": "tag:example-cadence", "Values": ["saturday"]} in filters
        assert {"Name": "tag:example-pipeline", "Values": ["ne-weekly-freshness-pipeline"]} in filters

    @mock.patch("nousergon_lib.spot_dispatch.boto3")
    def test_fail_safe_open_returns_empty_list_on_api_error(self, mock_boto3):
        mock_ec2 = mock.MagicMock()
        mock_boto3.client.return_value = mock_ec2
        mock_ec2.describe_instances.side_effect = RuntimeError("AWS API hiccup")

        result = spot_dispatch.running_instance_ids(
            "alpha-engine-example-spot", {}, region=REGION
        )

        assert result == []

    @mock.patch("nousergon_lib.spot_dispatch.boto3")
    def test_no_discriminator_tags_still_filters_by_name_and_state(self, mock_boto3):
        mock_ec2 = mock.MagicMock()
        mock_boto3.client.return_value = mock_ec2
        mock_ec2.describe_instances.return_value = {"Reservations": []}

        result = spot_dispatch.running_instance_ids(
            "alpha-engine-example-spot", {}, region=REGION
        )

        assert result == []
        _, kwargs = mock_ec2.describe_instances.call_args
        assert len(kwargs["Filters"]) == 2


class TestTerminateOnFailure:
    @mock.patch("nousergon_lib.spot_dispatch.boto3")
    def test_terminates_instance(self, mock_boto3):
        mock_ec2 = mock.MagicMock()
        mock_boto3.client.return_value = mock_ec2

        spot_dispatch.terminate_on_failure("i-abc123", region=REGION, label="example")

        mock_ec2.terminate_instances.assert_called_once_with(InstanceIds=["i-abc123"])

    @mock.patch("nousergon_lib.spot_dispatch.boto3")
    def test_never_raises_when_terminate_call_fails(self, mock_boto3):
        mock_ec2 = mock.MagicMock()
        mock_boto3.client.return_value = mock_ec2
        mock_ec2.terminate_instances.side_effect = RuntimeError("terminate failed")

        # Must not raise — best-effort cleanup, original caller error takes precedence.
        spot_dispatch.terminate_on_failure("i-abc123", region=REGION)
