"""Unit tests for ``nousergon_lib.spot_dispatch``.

Pins the contract for the shared dispatch primitives extracted (config#2106)
from `scheduled-groom-dispatcher`/`ci-watch-dispatcher` (nousergon-data) — a
third independent implementation (`sf-watch-spot-dispatcher`) is built
against this module rather than duplicating it again.

These tests pin BEHAVIOR PARITY with the two pre-extraction implementations
(spot-then-on-demand fallback on ``SpotCapacityExhausted``, best-effort
never-raising terminate-on-failure) plus one deliberate DIVERGENCE
(config#2267 site 1): the concurrency probe is no longer fail-open — a
DescribeInstances failure raises ``SpotProbeError`` instead of returning
``[]``, so a degraded EC2 API can never masquerade as "no duplicate box".
"""

from __future__ import annotations

from unittest import mock

import pytest

from nousergon_lib import spot_dispatch
from nousergon_lib.ec2_spot import SpotCapacityExhausted, SpotLaunchError, SpotQuotaExceededError
from nousergon_lib.spot_dispatch import SpotProbeError

REGION = "us-east-1"


def _common_launch_kwargs(**overrides):
    kwargs = {
        "instance_types": ["t3.medium", "t3a.medium"],
        "subnets": ["subnet-aaa", "subnet-bbb"],
        "image_id": "ami-0123456789abcdef0",
        "key_name": "alpha-engine-key",
        "security_group_ids": ["sg-0123456789abcdef0"],
        "iam_instance_profile": "alpha-engine-example-executor-profile",
        "volume_size_gb": 40,
        "tag_name": "alpha-engine-example-spot",
        "region": REGION,
    }
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
    def test_extra_tags_threaded_through_to_ec2_spot_launch(self, mock_ec2_spot):
        """config#2292 root fix: extra_tags rides straight through to
        krepis.ec2_spot.launch so discriminator tags land atomically at
        RunInstances time — no separate post-launch create_tags call."""
        mock_ec2_spot.launch.return_value = "i-tagged"

        instance_id, market = spot_dispatch.launch_with_fallback(
            **_common_launch_kwargs(
                extra_tags={"ci-watch-repo": "nousergon/krepis", "ci-watch-sha": "abc123"}
            )
        )

        assert (instance_id, market) == ("i-tagged", "spot")
        _, kwargs = mock_ec2_spot.launch.call_args
        # The caller's tags survive verbatim; launch provenance (I5727) is
        # added alongside rather than replacing them.
        assert kwargs["extra_tags"] == {
            "ci-watch-repo": "nousergon/krepis",
            "ci-watch-sha": "abc123",
            spot_dispatch.LAUNCH_MARKET_TAG: "spot",
            spot_dispatch.LAUNCH_REASON_TAG: spot_dispatch.REASON_SPOT_OK,
        }

    @mock.patch("nousergon_lib.spot_dispatch.ec2_spot")
    def test_provenance_is_tagged_even_with_no_caller_tags(self, mock_ec2_spot):
        """I5727: every launch carries provenance, not only the ones that
        escalate. A tag written only on failure cannot distinguish "spot
        succeeded" from "this box predates the tagging", and absence would
        then read as health."""
        mock_ec2_spot.launch.return_value = "i-notags"

        spot_dispatch.launch_with_fallback(**_common_launch_kwargs())

        _, kwargs = mock_ec2_spot.launch.call_args
        assert kwargs["extra_tags"] == {
            spot_dispatch.LAUNCH_MARKET_TAG: "spot",
            spot_dispatch.LAUNCH_REASON_TAG: spot_dispatch.REASON_SPOT_OK,
        }

    @mock.patch("nousergon_lib.spot_dispatch.ec2_spot")
    def test_capacity_fallback_is_tagged_capacity_exhausted(self, mock_ec2_spot):
        mock_ec2_spot.SpotCapacityExhausted = spot_dispatch.SpotCapacityExhausted
        mock_ec2_spot.launch.side_effect = [
            spot_dispatch.SpotCapacityExhausted("no capacity"), "i-fellback"
        ]

        instance_id, market = spot_dispatch.launch_with_fallback(**_common_launch_kwargs())

        assert (instance_id, market) == ("i-fellback", "on-demand")
        _, kwargs = mock_ec2_spot.launch.call_args
        assert kwargs["spot"] is False
        assert kwargs["extra_tags"][spot_dispatch.LAUNCH_MARKET_TAG] == "on-demand"
        assert (kwargs["extra_tags"][spot_dispatch.LAUNCH_REASON_TAG]
                == spot_dispatch.REASON_CAPACITY)

    @mock.patch("nousergon_lib.spot_dispatch.ec2_spot")
    def test_forced_on_demand_is_tagged_distinctly_from_capacity(self, mock_ec2_spot):
        """A deliberate force_on_demand and an involuntary capacity fallback
        both cost the on-demand premium and have DIFFERENT fixes, so the two
        must not collapse to one value."""
        mock_ec2_spot.launch.return_value = "i-forced"

        spot_dispatch.launch_with_fallback(**_common_launch_kwargs(force_on_demand=True))

        _, kwargs = mock_ec2_spot.launch.call_args
        assert (kwargs["extra_tags"][spot_dispatch.LAUNCH_REASON_TAG]
                == spot_dispatch.REASON_FORCED)
        assert spot_dispatch.REASON_FORCED != spot_dispatch.REASON_CAPACITY

    @mock.patch("nousergon_lib.spot_dispatch.ec2_spot")
    def test_library_provenance_wins_over_a_caller_supplied_collision(self, mock_ec2_spot):
        """These are measured facts about the launch, not a caller preference —
        a caller cannot mislabel its own box as spot."""
        mock_ec2_spot.launch.return_value = "i-liar"

        spot_dispatch.launch_with_fallback(
            **_common_launch_kwargs(
                force_on_demand=True,
                extra_tags={spot_dispatch.LAUNCH_MARKET_TAG: "spot"},
            )
        )

        _, kwargs = mock_ec2_spot.launch.call_args
        assert kwargs["extra_tags"][spot_dispatch.LAUNCH_MARKET_TAG] == "on-demand"

    def test_fallback_reasons_is_a_total_partition_of_launch_reasons(self):
        """Consumers classify on these sets. A reason added to the module but
        to neither set would be silently unclassifiable, so assert the
        partition rather than trusting it."""
        assert spot_dispatch.FALLBACK_REASONS < spot_dispatch.LAUNCH_REASONS
        assert (spot_dispatch.LAUNCH_REASONS - spot_dispatch.FALLBACK_REASONS
                == {spot_dispatch.REASON_SPOT_OK})

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

    @mock.patch("nousergon_lib.spot_dispatch.alerts")
    @mock.patch("nousergon_lib.spot_dispatch.ec2_spot")
    def test_spot_quota_exceeded_falls_back_to_on_demand_no_rotation_and_pages(
        self, mock_ec2_spot, mock_alerts
    ):
        """config#2698 acceptance criterion: stubbed RunInstances returning
        MaxSpotInstanceCountExceeded (surfaced as SpotQuotaExceededError by
        krepis.ec2_spot) => the launcher lands an on-demand instance, spot
        was attempted exactly ONCE (no rotation — ec2_spot itself already
        skips rotation on a quota error), and a warning-level page fires."""
        mock_ec2_spot.launch.side_effect = [
            SpotQuotaExceededError("MaxSpotInstanceCountExceeded (c5.large@subnet-aaa)"),
            "i-ondemand-quota",
        ]

        instance_id, market = spot_dispatch.launch_with_fallback(**_common_launch_kwargs())

        assert (instance_id, market) == ("i-ondemand-quota", "on-demand")
        assert mock_ec2_spot.launch.call_count == 2
        first_call_kwargs = mock_ec2_spot.launch.call_args_list[0].kwargs
        second_call_kwargs = mock_ec2_spot.launch.call_args_list[1].kwargs
        assert first_call_kwargs["spot"] is True
        assert second_call_kwargs["spot"] is False
        mock_alerts.publish.assert_called_once()
        _, publish_kwargs = mock_alerts.publish.call_args
        assert publish_kwargs["severity"] == "warning"
        assert "quota" in mock_alerts.publish.call_args.args[0].lower()

    @mock.patch("nousergon_lib.spot_dispatch.alerts")
    @mock.patch("nousergon_lib.spot_dispatch.ec2_spot")
    def test_quota_exceeded_is_not_capacity_exhausted(self, mock_ec2_spot, mock_alerts):
        """Quota and capacity exhaustion are siblings, not each other — both
        must independently trigger the on-demand fallback branch."""
        assert issubclass(SpotQuotaExceededError, SpotLaunchError)
        assert not issubclass(SpotQuotaExceededError, SpotCapacityExhausted)

    @mock.patch("nousergon_lib.spot_dispatch.alerts")
    @mock.patch("nousergon_lib.spot_dispatch.ec2_spot")
    def test_quota_exceeded_propagates_when_on_demand_also_fails(
        self, mock_ec2_spot, mock_alerts
    ):
        mock_ec2_spot.launch.side_effect = [
            SpotQuotaExceededError("quota"),
            SpotLaunchError("on-demand also failed"),
        ]

        with pytest.raises(SpotLaunchError):
            spot_dispatch.launch_with_fallback(**_common_launch_kwargs())
        assert mock_ec2_spot.launch.call_count == 2


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
    def test_api_error_raises_spot_probe_error_not_empty_list(self, mock_boto3):
        """config#2267 site 1: a failed probe must be LOUD — the old fail-open
        `return []` silently vanished the duplicate-box guard on a degraded
        EC2 API. Only a probe that actually RAN may return `[]`."""
        mock_ec2 = mock.MagicMock()
        mock_boto3.client.return_value = mock_ec2
        mock_ec2.describe_instances.side_effect = RuntimeError("AWS API hiccup")

        with pytest.raises(SpotProbeError, match="concurrency probe failed") as excinfo:
            spot_dispatch.running_instance_ids(
                "alpha-engine-example-spot", {}, region=REGION
            )

        # The original API error is chained, never masked.
        assert isinstance(excinfo.value.__cause__, RuntimeError)

    @mock.patch("nousergon_lib.spot_dispatch.boto3")
    def test_empty_result_still_means_no_instances_not_probe_failure(self, mock_boto3):
        """`[]` stays the clean "probe ran, nothing live" verdict."""
        mock_ec2 = mock.MagicMock()
        mock_boto3.client.return_value = mock_ec2
        mock_ec2.describe_instances.return_value = {"Reservations": []}

        assert spot_dispatch.running_instance_ids(
            "alpha-engine-example-spot", {}, region=REGION
        ) == []

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


class TestTerminationImminent:
    """The reclaim-aware leg of the concurrent guard (groom relaunch race,
    2026-08-02). A spot-reclaimed instance is observably `running` to
    describe-instances for ~2 min after the interruption notice — the guard
    must see past that via the spot request's Status.Code."""

    @mock.patch("nousergon_lib.spot_dispatch.boto3")
    def test_terminated_state_is_imminent(self, mock_boto3):
        mock_ec2 = mock.MagicMock()
        mock_boto3.client.return_value = mock_ec2
        mock_ec2.describe_instances.return_value = {"Reservations": [{"Instances": [
            {"InstanceId": "i-dead", "State": {"Name": "terminated"}},
        ]}]}

        assert spot_dispatch.termination_imminent(["i-dead"], region=REGION) == {"i-dead"}

    @mock.patch("nousergon_lib.spot_dispatch.boto3")
    def test_shutting_down_state_is_imminent(self, mock_boto3):
        mock_ec2 = mock.MagicMock()
        mock_boto3.client.return_value = mock_ec2
        mock_ec2.describe_instances.return_value = {"Reservations": [{"Instances": [
            {"InstanceId": "i-dying", "State": {"Name": "shutting-down"}},
        ]}]}

        assert spot_dispatch.termination_imminent(["i-dying"], region=REGION) == {"i-dying"}

    @mock.patch("nousergon_lib.spot_dispatch.boto3")
    def test_healthy_running_with_no_spot_request_is_not_imminent(self, mock_boto3):
        """A genuinely-live on-demand or spot box with no reclaim notice is the
        real duplicate the guard exists to suppress."""
        mock_ec2 = mock.MagicMock()
        mock_boto3.client.return_value = mock_ec2
        mock_ec2.describe_instances.return_value = {"Reservations": [{"Instances": [
            {"InstanceId": "i-healthy", "State": {"Name": "running"}},
        ]}]}
        # No SpotInstanceRequestId on the instance -> no spot-request lookup.
        mock_ec2.get_paginator.return_value.paginate.return_value = []

        assert spot_dispatch.termination_imminent(["i-healthy"], region=REGION) == set()

    @mock.patch("nousergon_lib.spot_dispatch.boto3")
    def test_running_instance_with_marked_spot_request_is_imminent(self, mock_boto3):
        """The race case: instance still `running`, but the spot request already
        carries `marked-for-termination` from the 2-min notice window. This is
        the exact signal State alone cannot see (2026-08-02 low-only reclaim)."""
        mock_ec2 = mock.MagicMock()
        mock_boto3.client.return_value = mock_ec2
        mock_ec2.describe_instances.return_value = {"Reservations": [{"Instances": [
            {"InstanceId": "i-reclaimed", "State": {"Name": "running"},
             "SpotInstanceRequestId": "sir-abc"},
        ]}]}
        mock_ec2.get_paginator.return_value.paginate.return_value = [
            {"SpotInstanceRequests": [
                {"SpotInstanceRequestId": "sir-abc", "State": "active",
                 "Status": {"Code": "marked-for-termination"}},
            ]}]

        assert spot_dispatch.termination_imminent(["i-reclaimed"], region=REGION) == {"i-reclaimed"}

    @mock.patch("nousergon_lib.spot_dispatch.boto3")
    def test_running_instance_with_closed_spot_request_is_imminent(self, mock_boto3):
        """A closed/cancelled spot request means the instance is gone or going
        regardless of the specific Status.Code (the live 2026-08-02 reclaim read
        State=closed, Status=instance-terminated-no-capacity)."""
        mock_ec2 = mock.MagicMock()
        mock_boto3.client.return_value = mock_ec2
        mock_ec2.describe_instances.return_value = {"Reservations": [{"Instances": [
            {"InstanceId": "i-closed", "State": {"Name": "running"},
             "SpotInstanceRequestId": "sir-closed"},
        ]}]}
        mock_ec2.get_paginator.return_value.paginate.return_value = [
            {"SpotInstanceRequests": [
                {"SpotInstanceRequestId": "sir-closed", "State": "closed",
                 "Status": {"Code": "instance-terminated-no-capacity"}},
            ]}]

        assert spot_dispatch.termination_imminent(["i-closed"], region=REGION) == {"i-closed"}

    @mock.patch("nousergon_lib.spot_dispatch.boto3")
    def test_mixed_set_returns_only_the_dying_subset(self, mock_boto3):
        """One healthy, one reclaimed, one already terminated — only the latter
        two are imminent, so the caller relaunches past only those."""
        mock_ec2 = mock.MagicMock()
        mock_boto3.client.return_value = mock_ec2
        mock_ec2.describe_instances.return_value = {"Reservations": [{"Instances": [
            {"InstanceId": "i-healthy", "State": {"Name": "running"},
             "SpotInstanceRequestId": "sir-healthy"},
            {"InstanceId": "i-reclaimed", "State": {"Name": "running"},
             "SpotInstanceRequestId": "sir-reclaimed"},
            {"InstanceId": "i-dead", "State": {"Name": "terminated"}},
        ]}]}
        mock_ec2.get_paginator.return_value.paginate.return_value = [
            {"SpotInstanceRequests": [
                {"SpotInstanceRequestId": "sir-healthy", "State": "active",
                 "Status": {"Code": "fulfilled"}},
                {"SpotInstanceRequestId": "sir-reclaimed", "State": "active",
                 "Status": {"Code": "marked-for-stop"}},
            ]}]

        result = spot_dispatch.termination_imminent(
            ["i-healthy", "i-reclaimed", "i-dead"], region=REGION)
        assert result == {"i-reclaimed", "i-dead"}

    @mock.patch("nousergon_lib.spot_dispatch.boto3")
    def test_probe_failure_fails_safe_to_empty_set(self, mock_boto3):
        """A broken EC2 API must never risk a duplicate launch — degrade to the
        original guard suppression (empty set = nothing imminent). The failure
        is logged, not swallowed silently."""
        mock_ec2 = mock.MagicMock()
        mock_boto3.client.return_value = mock_ec2
        mock_ec2.describe_instances.side_effect = RuntimeError("EC2 API hiccup")

        assert spot_dispatch.termination_imminent(["i-maybe"], region=REGION) == set()

    def test_empty_input_returns_empty_set_without_api_call(self):
        assert spot_dispatch.termination_imminent([], region=REGION) == set()


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

    @mock.patch("nousergon_lib.spot_dispatch.boto3")
    def test_tags_the_termination_reason_before_terminating(self, mock_boto3):
        """alpha-engine-config-I6199 — the reason must survive the terminate.

        Order matters: create_tags on an already-terminating instance is not
        reliable, so the tag has to be written first.
        """
        mock_ec2 = mock.MagicMock()
        mock_boto3.client.return_value = mock_ec2

        spot_dispatch.terminate_on_failure(
            "i-abc123", region=REGION, label="groom",
            reason="RuntimeError: SSM agent not Online after 180s for i-abc123",
        )

        tags = {t["Key"]: t["Value"]
                for t in mock_ec2.create_tags.call_args.kwargs["Tags"]}
        assert tags[spot_dispatch.TERMINATION_REASON_TAG] == (
            "RuntimeError: SSM agent not Online after 180s for i-abc123"
        )
        assert tags[spot_dispatch.TERMINATION_SOURCE_TAG] == "dispatcher"
        assert mock_ec2.create_tags.call_args.kwargs["Resources"] == ["i-abc123"]
        assert [c[0] for c in mock_ec2.method_calls] == ["create_tags", "terminate_instances"]

    @mock.patch("nousergon_lib.spot_dispatch.boto3")
    def test_reason_is_truncated_to_the_ec2_tag_value_limit(self, mock_boto3):
        """A long traceback string must not make create_tags reject the call."""
        mock_ec2 = mock.MagicMock()
        mock_boto3.client.return_value = mock_ec2

        spot_dispatch.terminate_on_failure(
            "i-abc123", region=REGION, reason="x" * 900
        )

        tags = {t["Key"]: t["Value"]
                for t in mock_ec2.create_tags.call_args.kwargs["Tags"]}
        assert len(tags[spot_dispatch.TERMINATION_REASON_TAG]) == 256

    @mock.patch("nousergon_lib.spot_dispatch.boto3")
    def test_tagging_failure_never_prevents_the_terminate(self, mock_boto3):
        """Tagging is strictly subordinate to not orphaning the box."""
        mock_ec2 = mock.MagicMock()
        mock_boto3.client.return_value = mock_ec2
        mock_ec2.create_tags.side_effect = RuntimeError("tag denied")

        spot_dispatch.terminate_on_failure(
            "i-abc123", region=REGION, reason="RuntimeError: boom"
        )

        mock_ec2.terminate_instances.assert_called_once_with(InstanceIds=["i-abc123"])

    @mock.patch("nousergon_lib.spot_dispatch.boto3")
    def test_no_tag_call_when_no_reason_given(self, mock_boto3):
        """Existing callers that pass no reason must behave exactly as before."""
        mock_ec2 = mock.MagicMock()
        mock_boto3.client.return_value = mock_ec2

        spot_dispatch.terminate_on_failure("i-abc123", region=REGION)

        mock_ec2.create_tags.assert_not_called()
        mock_ec2.terminate_instances.assert_called_once_with(InstanceIds=["i-abc123"])
