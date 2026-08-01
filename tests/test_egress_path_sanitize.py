"""Unit tests for egress proxy request-path sanitizer (CodeQL py/partial-ssrf)."""

import pytest

from nousergon_lib.egress.proxy import UpstreamError, _sanitize_request_path


@pytest.mark.parametrize(
    "path",
    [
        "/v1/messages",
        "/anthropic/v1/messages",
        "/v1/chat/completions?foo=bar&x=1",
        "/a-b_c.d~e%2F",
        "/v1/foo:bar@baz",
    ],
)
def test_sanitize_accepts_allowlisted_paths(path):
    assert _sanitize_request_path(path) == path


@pytest.mark.parametrize(
    "path",
    [
        "",
        "v1/messages",  # not absolute
        "/../etc/passwd",
        "/v1/../secret",
        "/v1/messages\r\nHost: evil",
        "/v1/messages with space",
        "/v1/messages<script>",
        "\\\\evil",
    ],
)
def test_sanitize_rejects_unsafe_paths(path):
    with pytest.raises(UpstreamError):
        _sanitize_request_path(path)
