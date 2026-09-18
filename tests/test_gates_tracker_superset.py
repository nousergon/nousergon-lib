"""The tracker reads `crucible/tracker.py` carries that this adapter did not.

`shared-code-policy` §3.1. The lift (`alpha-engine-config-I10951`) moved the
WRITES — comment, create, rewrite a body. The fork also READS, and its two
readers are opposites on purpose:

* :meth:`Tracker.read_issue` never raises, because its caller renders one
  board row per issue and a board that died on one optional read tells nobody
  anything. Every fault becomes ``access_problem=True``, which a consumer
  renders UNMEASURABLE rather than as a verdict.
* :meth:`Tracker.comment_bodies` raises on everything, because its caller is a
  WRITER checking whether a record is already posted. A listing that silently
  came back short posts a duplicate.

Closing the fork without both would downgrade an access fault into a false
verdict on one side and into a duplicate post on the other.
"""

from __future__ import annotations

import json

import pytest

from nousergon_lib.gates.tracker import (
    COMMENT_PAGE_CEILING,
    COMMENT_PAGE_SIZE,
    ISSUE_STATES,
    Tracker,
    TrackerConfig,
    TrackerError,
)

CONFIG = TrackerConfig(
    repo="nousergon/alpha-engine-config",
    token_var="X_TRACKER_TOKEN",
    app_ssm_prefix_var="X_TRACKER_APP_SSM_PREFIX",
)


class FakeGitHub:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        status, payload = self.responses.pop(0)
        return status, json.dumps(payload).encode("utf-8")


def _tracker(monkeypatch, responses, *, config=CONFIG):
    monkeypatch.setenv(config.token_var, "t0ken")
    monkeypatch.delenv(config.app_ssm_prefix_var, raising=False)
    opener = FakeGitHub(responses)
    return Tracker(config, opener=opener), opener


def _unconfigured(monkeypatch, *, config=CONFIG):
    monkeypatch.delenv(config.token_var, raising=False)
    monkeypatch.delenv(config.app_ssm_prefix_var, raising=False)
    return Tracker(config, opener=FakeGitHub([]))


# ---------------------------------------------------------------------------
# read_issue: lenient, and an access fault is its own third fact


def test_a_read_issue_returns_the_state_it_was_told(monkeypatch):
    tracker, opener = _tracker(monkeypatch, [(200, {"state": "closed"})])
    read = tracker.read_issue(10123)
    assert read.state == "closed"
    assert read.closed is True
    assert read.problem is None
    assert read.access_problem is False
    assert opener.requests[0].full_url.endswith("/repos/nousergon/alpha-engine-config/issues/10123")


def test_an_absent_credential_is_an_access_problem_and_never_raises(monkeypatch):
    read = _unconfigured(monkeypatch).read_issue(10123)
    assert read.state is None
    assert read.access_problem is True
    assert "no tracker credential" in read.problem
    assert read.closed is False


def test_a_denied_read_is_an_access_problem_not_an_open_issue(monkeypatch):
    """The whole point: a 403 must not render as "not closed"."""
    tracker, _ = _tracker(monkeypatch, [(403, {"message": "Resource not accessible"})])
    read = tracker.read_issue(10123)
    assert read.state is None
    assert read.access_problem is True
    assert "403" in read.problem


def test_a_transport_failure_is_an_access_problem(monkeypatch):
    monkeypatch.setenv(CONFIG.token_var, "t0ken")
    monkeypatch.delenv(CONFIG.app_ssm_prefix_var, raising=False)

    def explode(request):
        raise OSError("connection reset")

    read = Tracker(CONFIG, opener=explode).read_issue(1)
    assert read.access_problem is True
    assert "connection reset" in read.problem


def test_a_non_json_answer_is_an_access_problem(monkeypatch):
    monkeypatch.setenv(CONFIG.token_var, "t0ken")
    monkeypatch.delenv(CONFIG.app_ssm_prefix_var, raising=False)
    read = Tracker(CONFIG, opener=lambda request: (200, b"<html>")).read_issue(1)
    assert read.access_problem is True
    assert read.state is None


def test_a_state_outside_the_closed_vocabulary_is_refused_not_passed_through(monkeypatch):
    tracker, _ = _tracker(monkeypatch, [(200, {"state": "draft"})])
    read = tracker.read_issue(1)
    assert read.state is None
    assert read.access_problem is True
    assert str(ISSUE_STATES) in read.problem


# ---------------------------------------------------------------------------
# comment_bodies: strict, paginated, refuses to truncate


def _page(n):
    return [{"body": f"comment {i}"} for i in range(n)]


def test_comment_bodies_returns_every_body_oldest_first(monkeypatch):
    tracker, opener = _tracker(monkeypatch, [(200, [{"body": "a"}, {"body": "b"}])])
    assert tracker.comment_bodies(7) == ["a", "b"]
    assert f"per_page={COMMENT_PAGE_SIZE}&page=1" in opener.requests[0].full_url


def test_a_full_page_is_followed_by_the_next_one(monkeypatch):
    tracker, opener = _tracker(
        monkeypatch,
        [(200, _page(COMMENT_PAGE_SIZE)), (200, [{"body": "last"}])],
    )
    bodies = tracker.comment_bodies(7)
    assert len(bodies) == COMMENT_PAGE_SIZE + 1
    assert bodies[-1] == "last"
    assert "page=2" in opener.requests[1].full_url


def test_a_listing_past_the_ceiling_raises_rather_than_truncating(monkeypatch):
    """A short listing makes the writer post a duplicate. Guessing past it is
    the one direction that silently produces a wrong record."""
    tracker, _ = _tracker(monkeypatch, [(200, _page(COMMENT_PAGE_SIZE))] * COMMENT_PAGE_CEILING)
    with pytest.raises(TrackerError, match="more than"):
        tracker.comment_bodies(7)


def test_an_absent_credential_raises_and_names_the_operator_grant(monkeypatch):
    config = TrackerConfig(
        repo=CONFIG.repo,
        token_var=CONFIG.token_var,
        app_ssm_prefix_var=CONFIG.app_ssm_prefix_var,
        grant_hint="apply the stack; laptop: export X_TRACKER_TOKEN",
    )
    with pytest.raises(TrackerError, match="apply the stack"):
        _unconfigured(monkeypatch, config=config).comment_bodies(7)


def test_a_non_list_answer_raises(monkeypatch):
    tracker, _ = _tracker(monkeypatch, [(200, {"message": "nope"})])
    with pytest.raises(TrackerError, match="not a list"):
        tracker.comment_bodies(7)


def test_a_transport_failure_raises(monkeypatch):
    monkeypatch.setenv(CONFIG.token_var, "t0ken")
    monkeypatch.delenv(CONFIG.app_ssm_prefix_var, raising=False)

    def explode(request):
        raise OSError("connection reset")

    with pytest.raises(TrackerError, match="connection reset"):
        Tracker(CONFIG, opener=explode).comment_bodies(7)


# ---------------------------------------------------------------------------
# the grant hint reaches every credential-absence message


def test_a_write_names_the_operator_grant_when_one_is_declared(monkeypatch):
    config = TrackerConfig(
        repo=CONFIG.repo,
        token_var=CONFIG.token_var,
        app_ssm_prefix_var=CONFIG.app_ssm_prefix_var,
        grant_hint="set the repo variable and apply the stack",
    )
    with pytest.raises(TrackerError, match="set the repo variable"):
        _unconfigured(monkeypatch, config=config).post_comment(7, "body")


def test_a_config_with_no_grant_hint_still_names_both_variables(monkeypatch):
    with pytest.raises(TrackerError) as caught:
        _unconfigured(monkeypatch).post_comment(7, "body")
    assert CONFIG.token_var in str(caught.value)
    assert CONFIG.app_ssm_prefix_var in str(caught.value)


# ---------------------------------------------------------------------------
# the reads are reads


def test_neither_reader_can_construct_a_mutating_request(monkeypatch):
    tracker, opener = _tracker(monkeypatch, [(200, {"state": "open"}), (200, [])])
    tracker.read_issue(1)
    tracker.comment_bodies(1)
    assert [request.get_method() for request in opener.requests] == ["GET", "GET"]
