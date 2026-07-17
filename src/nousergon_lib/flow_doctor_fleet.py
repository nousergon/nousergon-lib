"""Canonical fleet flow-doctor Telegram routing (forum topics + severity gates).

Single source of truth for the N-topic Telegram layout described in
``alpha-engine-config/private-docs/fleet_notification_consolidation_arc_260704.md`` §3.
Each topic is a separate ``TelegramNotifier`` with its own ``message_thread_id`` and
``notify_on`` severity scope.

Thread IDs are **never** hardcoded — they resolve from env (seeded from SSM in prod).
See ``private-docs/fleet_telegram_forum_topics_ops.md`` in alpha-engine-config.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


class FleetTelegramTopic(str, Enum):
    """Fixed forum topics on the operator alerts supergroup."""

    CRITICAL = "critical"
    TRADES = "trades"
    PIPELINE = "pipeline"
    OPS_HEALTH = "ops_health"
    GROOM = "groom"
    RESEARCH = "research"


@dataclass(frozen=True)
class FleetTelegramTopicSpec:
    """Routing metadata for one forum topic."""

    topic: FleetTelegramTopic
    thread_id_env: str
    ssm_param: str
    notify_on: tuple[str, ...]
    notify_on_category: tuple[str, ...] = ()
    disable_notification: bool = False
    parse_mode: str = "Markdown"


# Severity scope per topic — mirrors fleet_notification_consolidation_arc_260704.md §3.
_FLEET_TELEGRAM_TOPIC_SPECS: Mapping[FleetTelegramTopic, FleetTelegramTopicSpec] = {
    FleetTelegramTopic.CRITICAL: FleetTelegramTopicSpec(
        topic=FleetTelegramTopic.CRITICAL,
        thread_id_env="FLOW_DOCTOR_TELEGRAM_THREAD_CRITICAL",
        ssm_param="/alpha-engine/FLOW_DOCTOR_TELEGRAM_THREAD_CRITICAL",
        notify_on=("critical",),
        disable_notification=False,
    ),
    FleetTelegramTopic.TRADES: FleetTelegramTopicSpec(
        topic=FleetTelegramTopic.TRADES,
        thread_id_env="FLOW_DOCTOR_TELEGRAM_THREAD_TRADES",
        ssm_param="/alpha-engine/FLOW_DOCTOR_TELEGRAM_THREAD_TRADES",
        notify_on=("info",),
        disable_notification=False,
    ),
    FleetTelegramTopic.PIPELINE: FleetTelegramTopicSpec(
        topic=FleetTelegramTopic.PIPELINE,
        thread_id_env="FLOW_DOCTOR_TELEGRAM_THREAD_PIPELINE",
        ssm_param="/alpha-engine/FLOW_DOCTOR_TELEGRAM_THREAD_PIPELINE",
        notify_on=("info", "warning"),
        disable_notification=False,
    ),
    FleetTelegramTopic.OPS_HEALTH: FleetTelegramTopicSpec(
        topic=FleetTelegramTopic.OPS_HEALTH,
        thread_id_env="FLOW_DOCTOR_TELEGRAM_THREAD_OPS_HEALTH",
        ssm_param="/alpha-engine/FLOW_DOCTOR_TELEGRAM_THREAD_OPS_HEALTH",
        notify_on=("error", "warning"),
        notify_on_category=("TRANSIENT", "EXTERNAL", "INFRA"),
        disable_notification=False,
    ),
    FleetTelegramTopic.GROOM: FleetTelegramTopicSpec(
        topic=FleetTelegramTopic.GROOM,
        thread_id_env="FLOW_DOCTOR_TELEGRAM_THREAD_GROOM",
        ssm_param="/alpha-engine/FLOW_DOCTOR_TELEGRAM_THREAD_GROOM",
        notify_on=("info",),
        disable_notification=True,
    ),
    FleetTelegramTopic.RESEARCH: FleetTelegramTopicSpec(
        topic=FleetTelegramTopic.RESEARCH,
        thread_id_env="FLOW_DOCTOR_TELEGRAM_THREAD_RESEARCH",
        ssm_param="/alpha-engine/FLOW_DOCTOR_TELEGRAM_THREAD_RESEARCH",
        notify_on=("info", "warning"),
        disable_notification=False,
    ),
}

# Executor daemon: error mirror + trade fills + ops degradation.
EXECUTOR_FLOW_DOCTOR_TELEGRAM_TOPICS: tuple[FleetTelegramTopic, ...] = (
    FleetTelegramTopic.CRITICAL,
    FleetTelegramTopic.OPS_HEALTH,
    FleetTelegramTopic.TRADES,
)

# External observers / SF notifiers (T2): pipeline milestones + critical mirror.
PIPELINE_OBSERVER_TELEGRAM_TOPICS: tuple[FleetTelegramTopic, ...] = (
    FleetTelegramTopic.CRITICAL,
    FleetTelegramTopic.PIPELINE,
    FleetTelegramTopic.OPS_HEALTH,
)


def fleet_telegram_topic_spec(topic: FleetTelegramTopic) -> FleetTelegramTopicSpec:
    """Return the canonical spec for ``topic``."""
    return _FLEET_TELEGRAM_TOPIC_SPECS[topic]


def fleet_telegram_thread_id_env(topic: FleetTelegramTopic) -> str:
    """Env var name holding the forum ``message_thread_id`` for ``topic``."""
    return fleet_telegram_topic_spec(topic).thread_id_env


def fleet_telegram_notifier_dict(
    topic: FleetTelegramTopic,
    *,
    # S107-suppressed defaults: these are literal `${ENV_VAR}` YAML-interpolation
    # placeholders emitted into the flow-doctor config template, not real
    # credentials — the actual secret is resolved by flow-doctor from env at
    # its own runtime, never held here.
    bot_token: str = "${TELEGRAM_BOT_TOKEN}",  # noqa: S107
    chat_id: str = "${TELEGRAM_CHAT_ID}",  # noqa: S107
) -> dict[str, Any]:
    """Build one flow-doctor yaml-compatible telegram notifier dict."""
    spec = fleet_telegram_topic_spec(topic)
    out: dict[str, Any] = {
        "type": "telegram",
        "bot_token": bot_token,
        "chat_id": chat_id,
        "message_thread_id": f"${{{spec.thread_id_env}}}",
        "parse_mode": spec.parse_mode,
        "disable_notification": spec.disable_notification,
        "notify_on": list(spec.notify_on),
    }
    if spec.notify_on_category:
        out["notify_on_category"] = list(spec.notify_on_category)
    return out


def fleet_telegram_notifier_dicts(
    topics: Sequence[FleetTelegramTopic],
    *,
    # S107-suppressed defaults: these are literal `${ENV_VAR}` YAML-interpolation
    # placeholders emitted into the flow-doctor config template, not real
    # credentials — the actual secret is resolved by flow-doctor from env at
    # its own runtime, never held here.
    bot_token: str = "${TELEGRAM_BOT_TOKEN}",  # noqa: S107
    chat_id: str = "${TELEGRAM_CHAT_ID}",  # noqa: S107
) -> list[dict[str, Any]]:
    """Build ordered notifier dicts for ``FlowDoctor.from_config(notify=...)``."""
    return [
        fleet_telegram_notifier_dict(
            topic, bot_token=bot_token, chat_id=chat_id
        )
        for topic in topics
    ]


def fleet_telegram_ssm_params() -> dict[str, str]:
    """Map env var → SSM parameter path for ops seeding (values NOT in git)."""
    return {
        spec.thread_id_env: spec.ssm_param
        for spec in _FLEET_TELEGRAM_TOPIC_SPECS.values()
    }


def trade_alert_dedup_key(action: str, ticker: str, shares: int, price: float) -> str:
    """Stable dedup key for a single fill notification (same fill, one ping)."""
    return f"executor:trade:{action}:{ticker}:{shares}:{price:.4f}"


__all__ = [
    "EXECUTOR_FLOW_DOCTOR_TELEGRAM_TOPICS",
    "PIPELINE_OBSERVER_TELEGRAM_TOPICS",
    "FleetTelegramTopic",
    "FleetTelegramTopicSpec",
    "fleet_telegram_notifier_dict",
    "fleet_telegram_notifier_dicts",
    "fleet_telegram_ssm_params",
    "fleet_telegram_thread_id_env",
    "fleet_telegram_topic_spec",
    "trade_alert_dedup_key",
]
