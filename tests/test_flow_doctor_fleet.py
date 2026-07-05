"""Tests for canonical fleet flow-doctor Telegram config."""

from __future__ import annotations

import os
import tempfile

import pytest

from nousergon_lib.flow_doctor_fleet import (
    EXECUTOR_FLOW_DOCTOR_TELEGRAM_TOPICS,
    FleetTelegramTopic,
    fleet_telegram_notifier_dicts,
    fleet_telegram_ssm_params,
    trade_alert_dedup_key,
)


def test_fleet_topic_specs_cover_all_enum_members():
    assert len(fleet_telegram_ssm_params()) == len(FleetTelegramTopic)
    for topic in FleetTelegramTopic:
        env = f"FLOW_DOCTOR_TELEGRAM_THREAD_{topic.value.upper()}"
        assert env in fleet_telegram_ssm_params()


def test_executor_profile_has_critical_ops_health_and_trades():
    assert EXECUTOR_FLOW_DOCTOR_TELEGRAM_TOPICS == (
        FleetTelegramTopic.CRITICAL,
        FleetTelegramTopic.OPS_HEALTH,
        FleetTelegramTopic.TRADES,
    )


def test_notifier_dicts_use_env_thread_ids_and_severity_gates():
    critical, ops, trades = fleet_telegram_notifier_dicts(
        EXECUTOR_FLOW_DOCTOR_TELEGRAM_TOPICS
    )
    assert critical["message_thread_id"] == "${FLOW_DOCTOR_TELEGRAM_THREAD_CRITICAL}"
    assert critical["notify_on"] == ["critical"]
    assert ops["notify_on"] == ["error", "warning"]
    assert ops["notify_on_category"] == ["TRANSIENT", "EXTERNAL", "INFRA"]
    assert trades["notify_on"] == ["info"]
    assert trades["message_thread_id"] == "${FLOW_DOCTOR_TELEGRAM_THREAD_TRADES}"


def test_flow_doctor_from_config_accepts_fleet_telegram_notifiers():
    flow_doctor = pytest.importorskip("flow_doctor")
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        os.environ["TELEGRAM_BOT_TOKEN"] = "123:abc"
        os.environ["TELEGRAM_CHAT_ID"] = "-10099"
        os.environ["FLOW_DOCTOR_TELEGRAM_THREAD_CRITICAL"] = "1"
        os.environ["FLOW_DOCTOR_TELEGRAM_THREAD_OPS_HEALTH"] = "2"
        os.environ["FLOW_DOCTOR_TELEGRAM_THREAD_TRADES"] = "3"
        os.environ["FLOW_DOCTOR_SKIP_PREFLIGHT"] = "1"
        try:
            fd = flow_doctor.FlowDoctor.from_config(
                flow_name="fleet-test",
                store={"type": "sqlite", "path": f.name},
                notify=fleet_telegram_notifier_dicts(
                    EXECUTOR_FLOW_DOCTOR_TELEGRAM_TOPICS
                ),
            )
        finally:
            for key in (
                "TELEGRAM_BOT_TOKEN",
                "TELEGRAM_CHAT_ID",
                "FLOW_DOCTOR_TELEGRAM_THREAD_CRITICAL",
                "FLOW_DOCTOR_TELEGRAM_THREAD_OPS_HEALTH",
                "FLOW_DOCTOR_TELEGRAM_THREAD_TRADES",
                "FLOW_DOCTOR_SKIP_PREFLIGHT",
            ):
                os.environ.pop(key, None)

    assert len(fd._notifiers) == 3
    thread_ids = sorted(
        n.message_thread_id for n in fd._notifiers if n.message_thread_id is not None
    )
    assert thread_ids == [1, 2, 3]


def test_trade_alert_dedup_key_is_stable():
    assert trade_alert_dedup_key("BUY", "AAPL", 10, 150.0) == (
        "executor:trade:BUY:AAPL:10:150.0000"
    )
