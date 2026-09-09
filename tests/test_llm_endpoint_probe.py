"""Tests for the shared LLM endpoint serve-probe (alpha-engine-config-I10230).

Lifted from `nous-ergon-ops/alpha-engine-dashboard/live/infrastructure/bin/
router_funded_depth_probe.py` (copy 1, authenticated router-edge probe) and
`router_degraded_mode_drill.py` (copy 2, unauthenticated loopback D5 probe) on
the third-adoption trigger. The real-body cases below are reproduced from each
copy's own test suite (`tests/test_router_funded_depth_probe.py::
TestClassification` / `TestVerdictVocabulary`, `tests/
test_router_degraded_mode_drill.py::test_serve_verdicts_are_classified_by_
the_documented_order`) to prove this module's `classify()` produces the SAME
verdict each copy already asserted for the same real response, on at least one
case per vocabulary member.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nousergon_lib import llm_endpoint_probe as probe

# ── vocabulary is closed ──────────────────────────────────────────────────


def test_vocabulary_is_closed():
    expected = {
        "servable", "timeout", "unreachable", "dlp_blocked", "unfunded",
        "unauthorized", "route_unconfigured", "cooled_down", "wrong_path",
        "not_served", "error",
    }
    assert set(probe.VERDICTS) == expected


def test_classify_never_returns_outside_the_closed_vocabulary():
    cases = [
        (200, "{}"), (0, "TimeoutError"), (0, "URLError"), (402, ""),
        (401, "x"), (403, "x"), (400, "unknown upstream host 'x'"),
        (429, "x"), (404, "<html>"), (404, "{}"), (500, "x"), (418, "x"),
        (400, "blocked by DLP"),
    ]
    for status, body in cases:
        assert probe.classify(status, body) in probe.VERDICTS


# ── real-case parity with copy 1 (router_funded_depth_probe.classify) ─────


class TestParityWithFundedDepthProbe:
    @pytest.mark.parametrize("status,body", [
        # anthropic, 2026-08-16, req_011Ce6fDKQDUBQx6fbfwSGVu
        (400, ("{'type':'invalid_request_error','message':'Your credit balance is too low "
               "to access the Anthropic API. Please go to Plans & Billing'}")),
        # xai via the router edge, 2026-08-16
        (400, ("Error doing the fallback: OpenAIException - Error code: 403 - "
               "{'code':'permission-denied','error':'Your team 3634e351 has either used all "
               "available credits or ...'}")),
        # deepseek, groom/sf-watch, 2026-08-07 (config-I6613)
        (402, "litellm.AuthenticationError: Insufficient Balance"),
        (402, ""),
    ])
    def test_exhaustion_is_unfunded(self, status, body):
        assert probe.classify(status, body) == "unfunded"

    def test_a_credential_refusal_is_not_a_money_problem(self):
        assert probe.classify(401, "<html><title>401 Authorization Required</title>") == "unauthorized"
        assert probe.classify(403, "forbidden") == "unauthorized"

    def test_ok_is_servable(self):
        assert probe.classify(200, '{"choices":[]}') == "servable"

    def test_a_transport_failure_says_nothing_about_funding(self):
        assert probe.classify(0, "URLError: connection refused") == "unreachable"

    def test_a_model_the_proxy_just_listed_but_will_not_serve(self):
        assert probe.classify(404, "model not found") == "not_served"

    #: Shape reproduced from the dashboard box's
    #: s3://alpha-engine-research/ops/router-funded-depth/latest.json,
    #: ran_at 2026-08-25T11:31:47Z (a live "unknown upstream host" refusal),
    #: with the provider hostnames replaced by placeholders — this test
    #: exercises `classify()`'s marker match, not any specific provider, and
    #: a literal upstream host here would trip this repo's own
    #: provider-linkage guards.
    LIVE_400 = (
        '{"error":{"message":"litellm.BadRequestError: OpenAIException - '
        "unknown upstream host 'upstream-example.test' — configured: "
        "['configured-a.example', 'configured-b.example']No fallback model "
        'group found for original model_group=example-model."}}'
    )

    def test_an_upstream_the_proxy_will_not_route_is_its_own_verdict(self):
        assert probe.classify(400, self.LIVE_400) == "route_unconfigured"

    def test_an_html_body_is_a_wrong_path_not_a_missing_model(self):
        """Measured 2026-08-25: prefix `/api` + the router's `/chat/completions`
        reached OpenRouter's WEB edge — 404 text/html with a dpl_ id."""
        assert probe.classify(404, "<!DOCTYPE html><html>...dpl_9x") == "wrong_path"
        assert probe.classify(404, '{"error":{"message":"model not found"}}') == "not_served"

    def test_a_401_challenge_page_is_still_a_credential_problem(self):
        assert probe.classify(401, "<html><title>401 Unauthorized</title></html>") == "unauthorized"

    def test_a_cooldown_is_transient_and_not_folded_into_error(self):
        body = ("Error code: 429 - No deployments available for selected model, "
                "Try again in 120 seconds. cooldown_list=['c21bc880','46f70d22']")
        assert probe.classify(429, body) == "cooled_down"

    def test_a_hung_upstream_is_not_a_dead_listener(self):
        assert probe.classify(0, "TimeoutError: timed out") == "timeout"
        assert probe.classify(0, "URLError: connection refused") == "unreachable"


# ── real-case parity with copy 2 (router_degraded_mode_drill.classify_serve) ──


@pytest.mark.parametrize("status,body,expected", [
    (200, '{"content":[]}', "servable"),
    (429, "rate limited", "cooled_down"),
    (0, "TimeoutError: timed out", "timeout"),
    (0, "URLError: Connection refused", "unreachable"),
    (402, "payment required", "unfunded"),
    (400, "insufficient balance", "unfunded"),
    (401, "invalid api key", "unauthorized"),
    (400, "unknown upstream host 'upstream-example.test' — configured: []", "route_unconfigured"),
    (404, "not found", "not_served"),
    (400, "blocked by DLP: secret detected", "dlp_blocked"),
    (500, "upstream exploded", "error"),
])
def test_serve_verdicts_match_the_drills_documented_order(status, body, expected):
    assert probe.classify(status, body) == expected


def test_an_exhausted_account_is_unfunded_not_unauthorized():
    assert probe.classify(401, "Your credit balance is too low") == "unfunded"


def test_a_dlp_block_is_not_graded_as_a_broken_endpoint():
    assert probe.classify(400, "request blocked by gitleaks-egress") == "dlp_blocked"


# ── the two hard requirements the dispatch called out explicitly ──────────


def test_429_passes_is_expressible_by_a_caller():
    """`cooled_down` is never itself a failure verdict — a caller's own
    SERVE_OK_VERDICTS-shaped set decides whether it counts as a pass."""
    assert probe.classify(429, "cooldown_list=['x']") == "cooled_down"
    assert probe.classify(429, "cooldown_list=['x']") != "error"


def test_unfunded_is_always_a_violation_never_folded_into_ok():
    verdict = probe.classify(402, "insufficient balance")
    assert verdict == "unfunded"
    assert verdict not in ("servable", "cooled_down")


# ── ordering contract: dlp_blocked ahead of the whole 4xx family ──────────


def test_dlp_blocked_outranks_unfunded_and_unauthorized():
    """A DLP-blocked body that also happens to carry money/credential
    language must still classify dlp_blocked — the control's own refusal
    outranks any provider-shaped text a scanner quoted back."""
    assert probe.classify(400, "blocked by DLP: quota exceeded, insufficient balance") == "dlp_blocked"


def test_wrong_path_outranks_not_served():
    assert probe.classify(404, "<html><body>404</body></html>") == "wrong_path"


# ── request(): wire-aware path selection and header contract ──────────────


def _urlopen_ok(status=200, body=b"{}"):
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = body
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def test_request_defaults_to_the_wires_declared_path():
    with patch("nousergon_lib.llm_endpoint_probe.urllib.request.urlopen") as mocked, \
         patch("nousergon_lib.llm_endpoint_probe.urllib.request.Request") as req_cls:
        mocked.return_value = _urlopen_ok()
        probe.request("http://127.0.0.1:8971", wire="anthropic", model="m", timeout=1)
        url = req_cls.call_args[0][0]
        assert url == "http://127.0.0.1:8971/v1/messages"


def test_request_openai_wire_path():
    with patch("nousergon_lib.llm_endpoint_probe.urllib.request.urlopen") as mocked, \
         patch("nousergon_lib.llm_endpoint_probe.urllib.request.Request") as req_cls:
        mocked.return_value = _urlopen_ok()
        probe.request("http://127.0.0.1:8990", wire="openai", model="m", timeout=1)
        url = req_cls.call_args[0][0]
        assert url == "http://127.0.0.1:8990/v1/chat/completions"


def test_request_explicit_path_overrides_the_wire_default():
    """`router_funded_depth_probe.py` posts to `/chat/completions` (no `/v1`
    prefix) against the router's TLS edge — the wire default must not force
    that call onto a path it never used."""
    with patch("nousergon_lib.llm_endpoint_probe.urllib.request.urlopen") as mocked, \
         patch("nousergon_lib.llm_endpoint_probe.urllib.request.Request") as req_cls:
        mocked.return_value = _urlopen_ok()
        probe.request("https://router.example:8443", wire="openai", model="m",
                      path="/chat/completions", timeout=1)
        url = req_cls.call_args[0][0]
        assert url == "https://router.example:8443/chat/completions"


def test_anthropic_wire_carries_the_version_header():
    with patch("nousergon_lib.llm_endpoint_probe.urllib.request.urlopen") as mocked, \
         patch("nousergon_lib.llm_endpoint_probe.urllib.request.Request") as req_cls:
        mocked.return_value = _urlopen_ok()
        probe.request("http://127.0.0.1:8971", wire="anthropic", model="m", timeout=1)
        headers = req_cls.call_args.kwargs["headers"]
        assert headers["anthropic-version"] == probe.ANTHROPIC_VERSION_HEADER


def test_caller_headers_are_merged_never_silently_dropped():
    with patch("nousergon_lib.llm_endpoint_probe.urllib.request.urlopen") as mocked, \
         patch("nousergon_lib.llm_endpoint_probe.urllib.request.Request") as req_cls:
        mocked.return_value = _urlopen_ok()
        probe.request("http://127.0.0.1:8990", wire="openai", model="m",
                      headers={"X-Upstream-Host": "upstream-example.test"}, timeout=1)
        headers = req_cls.call_args.kwargs["headers"]
        assert headers["X-Upstream-Host"] == "upstream-example.test"
        assert headers["Content-Type"] == "application/json"


def test_unknown_wire_with_no_explicit_path_raises():
    with pytest.raises(ValueError):
        probe.request("http://x", wire="carrier-pigeon", model="m", timeout=1)


def test_only_one_token_is_ever_requested():
    with patch("nousergon_lib.llm_endpoint_probe.urllib.request.urlopen") as mocked, \
         patch("nousergon_lib.llm_endpoint_probe.urllib.request.Request") as req_cls:
        mocked.return_value = _urlopen_ok()
        probe.request("http://x", wire="openai", model="m", timeout=1)
        import json
        sent = json.loads(req_cls.call_args.kwargs["data"])
        assert sent["max_tokens"] == 1


def test_transport_failure_yields_status_zero_with_exception_detail():
    import urllib.error
    with patch("nousergon_lib.llm_endpoint_probe.urllib.request.urlopen") as mocked:
        mocked.side_effect = urllib.error.URLError("connection refused")
        status, body = probe.request("http://x", wire="openai", model="m", timeout=1)
        assert status == 0
        assert "connection refused" in body


def test_http_error_returns_its_status_and_body():
    import urllib.error
    with patch("nousergon_lib.llm_endpoint_probe.urllib.request.urlopen") as mocked:
        exc = urllib.error.HTTPError("http://x", 402, "payment required",
                                     hdrs=None, fp=MagicMock(read=lambda: b"insufficient balance"))
        mocked.side_effect = exc
        status, body = probe.request("http://x", wire="openai", model="m", timeout=1)
        assert status == 402
        assert body == "insufficient balance"


# ── probe(): the convenience wrapper ───────────────────────────────────────


def test_probe_wraps_request_and_classify():
    with patch("nousergon_lib.llm_endpoint_probe.request") as mocked_request:
        mocked_request.return_value = (200, "{}")
        result = probe.probe("http://x", wire="openai", model="m", timeout=1)
        assert result == {"http_status": 200, "body": "{}", "verdict": "servable"}
