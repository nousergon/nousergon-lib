"""A rehearsal's substrate check stays off the production surfaces (alpha-engine-config-I12582).

``rehearsal-2026-10-21-1`` ran WeeklySubstrateHealthCheck through the same
``nousergon_lib.transparency --cadence weekly --alert`` call as production. It
put ``SubstrateChecksFailed=2`` into ``AlphaEngine/Substrate``, which tripped
``alpha-engine-substrate-aggregate-failures``, and its ``--alert`` published to
the production ``alpha-engine-alerts`` topic.

These tests drive ``main()`` with every AWS client stubbed:

  * under ``RUN_TOKEN=rehearsal-*`` nothing is put to CloudWatch or SNS, and the
    exit code still reports the failing row;
  * a production token, and no token, emit and page exactly as before;
  * the decision holds on a krepis without ``krepis.rehearsal`` (pre-0.59.78).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from nousergon_lib import transparency


@pytest.fixture
def clients(monkeypatch):
    made: dict[str, MagicMock] = {}

    def fake_client(service, *args, **kwargs):
        return made.setdefault(service, MagicMock(name=service))

    import boto3

    monkeypatch.setattr(boto3, "client", fake_client)
    monkeypatch.setattr(
        transparency,
        "check_inventory",
        lambda *a, **k: [
            transparency.CheckResult("data_quality", "weekly", "fail", "boom", "2026-10-22"),
            transparency.CheckResult("pnl_attribution", "weekly", "ok", "ok", "2026-10-22"),
        ],
    )
    for var in ("RUN_TOKEN", "AE_PIPELINE_ROLE", "SNS_TOPIC_ARN"):
        monkeypatch.delenv(var, raising=False)
    return made


def _calls(made, service, method):
    client = made.get(service)
    return [] if client is None else getattr(client, method).call_args_list


def test_rehearsal_emits_no_metric_and_sends_no_page(clients, monkeypatch, capsys):
    monkeypatch.setenv("RUN_TOKEN", "rehearsal-2026-10-21-1")
    rc = transparency.main(["--cadence", "weekly", "--alert"])
    assert rc == 1, "the failing row must still fail the step on the rehearsal's own execution"
    assert _calls(clients, "cloudwatch", "put_metric_data") == []
    assert _calls(clients, "sns", "publish") == []
    assert "REHEARSAL (rehearsal-2026-10-21-1)" in capsys.readouterr().out


def test_explicit_rehearsal_role_is_a_rehearsal(clients, monkeypatch):
    monkeypatch.setenv("AE_PIPELINE_ROLE", "rehearsal")
    transparency.main(["--cadence", "weekly", "--alert"])
    assert _calls(clients, "cloudwatch", "put_metric_data") == []
    assert _calls(clients, "sns", "publish") == []


@pytest.mark.parametrize("token", [None, "a1b2c3d4-weekly", "watch-rerun-2026-10-24"])
def test_production_still_emits_and_pages(clients, monkeypatch, token):
    if token is not None:
        monkeypatch.setenv("RUN_TOKEN", token)
    rc = transparency.main(["--cadence", "weekly", "--alert"])
    assert rc == 1
    published = [
        m for c in _calls(clients, "cloudwatch", "put_metric_data") for m in c.kwargs["MetricData"]
    ]
    assert {c.kwargs["Namespace"] for c in _calls(clients, "cloudwatch", "put_metric_data")} == {
        "AlphaEngine/Substrate"
    }
    failed = next(m for m in published if m["MetricName"] == "SubstrateChecksFailed")
    assert failed["Value"] == 1.0
    pages = _calls(clients, "sns", "publish")
    assert len(pages) == 1
    assert pages[0].kwargs["TopicArn"] == transparency.DEFAULT_SNS_TOPIC


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"RUN_TOKEN": "rehearsal-2026-10-22-1"}, "rehearsal-2026-10-22-1"),
        (
            {"RUN_TOKEN": "arn:aws:states:us-east-1:1:execution:sm:rehearsal-2026-10-22-1"},
            "rehearsal-2026-10-22-1",
        ),
        ({"AE_PIPELINE_ROLE": "rehearsal"}, "rehearsal-unnamed"),
        ({"RUN_TOKEN": "prod-run"}, None),
        ({}, None),
    ],
)
def test_fallback_without_krepis_rehearsal_agrees(monkeypatch, env, expected):
    monkeypatch.setitem(sys.modules, "krepis.rehearsal", None)  # import raises ImportError
    assert transparency._rehearsal_token(env) == expected


def test_krepis_rehearsal_is_the_definition_when_present(monkeypatch):
    import types

    seen = {}

    def rehearsal_token(execution_name=None, *, env=None):
        seen["env"] = env
        return "rehearsal-from-krepis"

    fake = types.ModuleType("krepis.rehearsal")
    fake.rehearsal_token = rehearsal_token  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "krepis.rehearsal", fake)
    env = {"RUN_TOKEN": "whatever"}
    assert transparency._rehearsal_token(env) == "rehearsal-from-krepis"
    assert seen["env"] is env
