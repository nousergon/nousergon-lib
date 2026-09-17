"""The rolling-issue adapter's invariants.

Two of these are authority properties rather than behaviour: the adapter
carries NO close method, and its body PATCH cannot be made to carry a
``state`` key regardless of what a caller passes. Both are asserted against
the module's own syntax tree, because a convention about authority is exactly
the thing that has failed before.

The ordering invariant — comment BEFORE headline, and a failed post means no
message — is tested here as well as on the consumer side, because a lift that
carried the mechanism and dropped the ordering would pass every other test in
this file.
"""

from __future__ import annotations

import ast
import inspect
import json
import pathlib

import pytest

from nousergon_lib.gates import tracker as tracker_module
from nousergon_lib.gates.report import UndeliveredError
from nousergon_lib.gates.tracker import (
    Tracker,
    TrackerConfig,
    TrackerCredentialError,
    TrackerError,
)

CONFIG = TrackerConfig(
    repo="nousergon/alpha-engine-config",
    token_var="DATA_TRACKER_TOKEN",
    app_ssm_prefix_var="DATA_TRACKER_APP_SSM_PREFIX",
)


class FakeGitHub:
    """Records every request and answers from a scripted queue."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        status, payload = self.responses.pop(0)
        return status, json.dumps(payload).encode("utf-8")


def _tracker(monkeypatch, responses):
    monkeypatch.setenv("DATA_TRACKER_TOKEN", "t0ken")
    monkeypatch.delenv("DATA_TRACKER_APP_SSM_PREFIX", raising=False)
    opener = FakeGitHub(responses)
    return Tracker(CONFIG, opener=opener), opener


# --------------------------------------------------------------------------
# authority: it may never close


def test_the_adapter_carries_no_close_method():
    forbidden = {"close", "close_issue", "reopen", "set_state", "update_issue"}
    assert not forbidden & set(dir(Tracker))


def test_the_body_patch_payload_is_a_literal_and_can_never_carry_state():
    source = pathlib.Path(inspect.getsourcefile(tracker_module)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    patches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "method" and getattr(kw.value, "value", None) == "PATCH"
    ]
    assert patches, "no PATCH call found — this guard must be re-anchored, not deleted"
    for call in patches:
        payload = next(kw.value for kw in call.keywords if kw.arg == "payload")
        assert isinstance(payload, ast.Dict), "a PATCH payload must be a literal dict"
        keys = [k.value for k in payload.keys]
        assert keys == ["body"], f"a PATCH payload may carry only `body`, not {keys}"


# --------------------------------------------------------------------------
# credential


def test_an_absent_credential_raises_rather_than_skipping_the_post(monkeypatch):
    monkeypatch.delenv("DATA_TRACKER_TOKEN", raising=False)
    monkeypatch.delenv("DATA_TRACKER_APP_SSM_PREFIX", raising=False)
    tracker = Tracker(CONFIG, opener=FakeGitHub([]))
    assert tracker.credential() is None
    with pytest.raises(TrackerError, match="DATA_TRACKER_TOKEN"):
        tracker.post_comment(1, "body")


def test_an_explicit_token_wins_over_the_app_prefix(monkeypatch):
    monkeypatch.setenv("DATA_TRACKER_TOKEN", "t0ken")
    monkeypatch.setenv("DATA_TRACKER_APP_SSM_PREFIX", "/ne/app/")
    assert Tracker(CONFIG).credential() == "t0ken"


def test_a_configured_but_broken_mint_is_not_reported_as_absent(monkeypatch):
    monkeypatch.delenv("DATA_TRACKER_TOKEN", raising=False)
    monkeypatch.setenv("DATA_TRACKER_APP_SSM_PREFIX", "/ne/app/")

    def boom(**_kwargs):
        raise RuntimeError("SSM said no")

    monkeypatch.setattr("nousergon_lib.github_app.installation_token", boom)
    with pytest.raises(TrackerCredentialError, match="SSM said no"):
        Tracker(CONFIG).credential()


def test_a_mint_is_narrowed_to_issues_write(monkeypatch):
    monkeypatch.delenv("DATA_TRACKER_TOKEN", raising=False)
    monkeypatch.setenv("DATA_TRACKER_APP_SSM_PREFIX", "/ne/app/")
    seen = {}

    def fake(*, ssm_prefix, permissions, **_kwargs):
        seen["prefix"] = ssm_prefix
        seen["permissions"] = permissions
        return "minted"

    monkeypatch.setattr("nousergon_lib.github_app.installation_token", fake)
    assert Tracker(CONFIG).credential() == "minted"
    assert seen == {"prefix": "/ne/app/", "permissions": {"issues": "write"}}


def test_a_typed_mint_failure_names_the_grant(monkeypatch):
    monkeypatch.delenv("DATA_TRACKER_TOKEN", raising=False)
    monkeypatch.setenv("DATA_TRACKER_APP_SSM_PREFIX", "/ne/app/")
    from nousergon_lib.github_app import GitHubAppTokenError

    def boom(**_kwargs):
        raise GitHubAppTokenError("no private key")

    monkeypatch.setattr("nousergon_lib.github_app.installation_token", boom)
    with pytest.raises(TrackerCredentialError, match="no private key"):
        Tracker(CONFIG).credential()


# --------------------------------------------------------------------------
# find / create


def test_an_exact_title_match_returns_its_number(monkeypatch):
    tracker, _ = _tracker(monkeypatch, [(200, {"items": [{"number": 7, "title": "daily", "state": "open"}]})])
    assert tracker.find_issue_by_title("daily") == 7


def test_a_phrase_match_that_is_not_the_exact_title_does_not_count(monkeypatch):
    tracker, _ = _tracker(monkeypatch, [(200, {"items": [{"number": 7, "title": "daily update", "state": "open"}]})])
    assert tracker.find_issue_by_title("daily") is None


def test_two_open_issues_with_that_title_raise_and_never_pick_one(monkeypatch):
    tracker, _ = _tracker(
        monkeypatch,
        [
            (
                200,
                {
                    "items": [
                        {"number": 7, "title": "daily", "state": "open"},
                        {"number": 9, "title": "daily", "state": "open"},
                    ]
                },
            )
        ],
    )
    with pytest.raises(TrackerError, match="refusing to pick one"):
        tracker.find_issue_by_title("daily")


def test_a_non_200_search_raises_rather_than_reading_as_absent(monkeypatch):
    tracker, _ = _tracker(monkeypatch, [(403, {"message": "denied"})])
    with pytest.raises(TrackerError, match="403"):
        tracker.find_issue_by_title("daily")


def test_a_search_answering_without_items_raises(monkeypatch):
    tracker, _ = _tracker(monkeypatch, [(200, {"total_count": 0})])
    with pytest.raises(TrackerError, match="items"):
        tracker.find_issue_by_title("daily")


def test_find_or_create_creates_only_when_nothing_is_there(monkeypatch):
    tracker, opener = _tracker(monkeypatch, [(200, {"items": []}), (201, {"number": 11, "html_url": "u"})])
    assert tracker.find_or_create_issue(title="daily", body="b") == 11
    assert opener.requests[1].method == "POST"
    assert json.loads(opener.requests[1].data) == {"title": "daily", "body": "b"}


def test_find_or_create_does_not_create_when_the_issue_exists(monkeypatch):
    tracker, opener = _tracker(monkeypatch, [(200, {"items": [{"number": 7, "title": "daily", "state": "open"}]})])
    assert tracker.find_or_create_issue(title="daily", body="b") == 7
    assert len(opener.requests) == 1


def test_an_empty_title_is_refused(monkeypatch):
    tracker, _ = _tracker(monkeypatch, [])
    with pytest.raises(TrackerError, match="empty title"):
        tracker.create_issue(title="  ", body="b")


def test_a_create_that_answers_without_a_number_raises(monkeypatch):
    tracker, _ = _tracker(monkeypatch, [(201, {"html_url": "u"})])
    with pytest.raises(TrackerError, match="no usable"):
        tracker.create_issue(title="daily", body="b")


# --------------------------------------------------------------------------
# comment / body


def test_a_comment_returns_its_permalink(monkeypatch):
    tracker, opener = _tracker(monkeypatch, [(201, {"html_url": "https://gh/c/1"})])
    assert tracker.post_comment(7, "full update") == "https://gh/c/1"
    request = opener.requests[0]
    assert request.method == "POST"
    assert request.get_header("Authorization") == "Bearer t0ken"
    assert request.get_header("X-github-api-version") == "2022-11-28"


def test_an_empty_comment_is_refused(monkeypatch):
    tracker, _ = _tracker(monkeypatch, [])
    with pytest.raises(TrackerError, match="empty comment"):
        tracker.post_comment(7, "   ")


def test_a_comment_accepted_without_a_url_raises(monkeypatch):
    tracker, _ = _tracker(monkeypatch, [(201, {})])
    with pytest.raises(TrackerError, match="html_url"):
        tracker.post_comment(7, "body")


def test_a_refused_comment_raises_with_githubs_own_answer(monkeypatch):
    tracker, _ = _tracker(monkeypatch, [(422, {"message": "unprocessable"})])
    with pytest.raises(TrackerError, match="422"):
        tracker.post_comment(7, "body")


def test_a_transport_failure_on_a_comment_raises(monkeypatch):
    monkeypatch.setenv("DATA_TRACKER_TOKEN", "t0ken")

    def boom(_request):
        raise OSError("connection reset")

    with pytest.raises(TrackerError, match="connection reset"):
        Tracker(CONFIG, opener=boom).post_comment(7, "body")


def test_a_body_rewrite_patches_only_the_body(monkeypatch):
    tracker, opener = _tracker(monkeypatch, [(200, {})])
    tracker.update_issue_body(7, "# index")
    request = opener.requests[0]
    assert request.method == "PATCH"
    assert json.loads(request.data) == {"body": "# index"}


def test_a_refused_body_rewrite_raises(monkeypatch):
    tracker, _ = _tracker(monkeypatch, [(404, {"message": "not found"})])
    with pytest.raises(TrackerError, match="404"):
        tracker.update_issue_body(7, "# index")


def test_a_non_json_answer_raises_rather_than_reading_as_success(monkeypatch):
    monkeypatch.setenv("DATA_TRACKER_TOKEN", "t0ken")

    def html(_request):
        return 201, b"<html>rate limited</html>"

    with pytest.raises(TrackerError, match="did not answer with JSON"):
        Tracker(CONFIG, opener=html).post_comment(7, "body")


# --------------------------------------------------------------------------
# the ordering invariant


def _publish_never(*_args, **_kwargs):
    raise AssertionError("a message was sent although the tracker post failed")


def test_a_failed_post_means_no_message_is_ever_rendered_or_sent(monkeypatch):
    """The ordering invariant, from the delivery side.

    The headline's indispensable content is the comment's permalink, so the
    comment is posted FIRST. Because `post_comment` raises, a caller that
    renders the headline from its return value cannot reach the send at all —
    the manifest reads `failed` and nothing goes out.
    """
    monkeypatch.setattr("nousergon_lib.gates.report._krepis_publish", _publish_never)
    tracker, _ = _tracker(monkeypatch, [(403, {"message": "denied"})])

    sent = []

    def job():
        permalink = tracker.post_comment(7, "full update")  # raises
        headline = f"full update: {permalink}"
        sent.append(headline)
        from nousergon_lib.gates.report import deliver

        deliver(headline, severity="info", source="s", console_artifact="a")

    with pytest.raises(TrackerError):
        job()
    assert sent == []


def test_a_successful_post_supplies_the_permalink_the_headline_needs(monkeypatch):
    calls = {}

    def fake(message, **kwargs):
        calls["message"] = message
        return type("R", (), {"any_ok": True, "dedup_skipped": False, "muted": False})()

    monkeypatch.setattr("nousergon_lib.gates.report._krepis_publish", fake)
    tracker, _ = _tracker(monkeypatch, [(201, {"html_url": "https://gh/c/9"})])

    from nousergon_lib.gates.report import deliver

    permalink = tracker.post_comment(7, "full update")
    deliver(f"full update: {permalink}", severity="info", source="s", console_artifact="a")
    assert calls["message"] == "full update: https://gh/c/9"


def test_an_undelivered_headline_is_still_a_failure_after_a_good_post(monkeypatch):
    """A posted comment does not make a report delivered.

    The tracker carries the record; the message is the pointer at it, and a
    pointer nobody received is the accountability gap either way.
    """

    def muted(_message, **_kwargs):
        return type("R", (), {"any_ok": True, "dedup_skipped": False, "muted": True})()

    monkeypatch.setattr("nousergon_lib.gates.report._krepis_publish", muted)
    tracker, _ = _tracker(monkeypatch, [(201, {"html_url": "https://gh/c/9"})])

    from nousergon_lib.gates.report import deliver

    permalink = tracker.post_comment(7, "full update")
    with pytest.raises(UndeliveredError):
        deliver(f"see {permalink}", severity="info", source="s", console_artifact="a")
