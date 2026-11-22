"""The shared action-pin classifier: parsing, ref shape, and the three outcomes."""

from __future__ import annotations

import pytest

from nousergon_lib.action_pins import (
    ObjectKind,
    RefShape,
    ResolutionError,
    UsesKind,
    classify_object,
    parse_uses,
    ref_shape,
)

SHA_A = "a" * 40
SHA_B = "b" * 40


class _Table:
    """A ``get`` answering from a path -> (status, body) table; 404 otherwise."""

    def __init__(self, table: dict[str, tuple[int, object]]) -> None:
        self.table = table
        self.calls: list[str] = []

    def __call__(self, path: str) -> tuple[int, object]:
        self.calls.append(path)
        return self.table.get(path, (404, None))


# ── parsing ──────────────────────────────────────────────────────────────────


def test_parses_owner_repo_and_ref() -> None:
    p = parse_uses(f"actions/checkout@{SHA_A}")
    assert (p.kind, p.owner, p.repo, p.path, p.ref) == (UsesKind.PARSED, "actions", "checkout", "", SHA_A)
    assert p.owner_repo == "actions/checkout"


def test_parses_the_path_form_and_queries_only_owner_repo() -> None:
    p = parse_uses(f"nousergon/nousergon-lib/.github/workflows/x.yml@{SHA_A}")
    assert p.kind is UsesKind.PARSED
    assert p.path == "/.github/workflows/x.yml"
    assert p.owner_repo == "nousergon/nousergon-lib"


@pytest.mark.parametrize("value", ["./.github/actions/x", ".\\x", "docker://alpine:3.20"])
def test_local_and_docker_are_skipped(value: str) -> None:
    assert parse_uses(value).kind is UsesKind.SKIPPED


def test_unpinned_and_unparseable_are_distinguished() -> None:
    assert parse_uses("actions/checkout").kind is UsesKind.UNPINNED
    assert parse_uses("checkout@v4").kind is UsesKind.UNPARSEABLE


# ── ref shape ────────────────────────────────────────────────────────────────


def test_only_40_lowercase_hex_is_a_sha() -> None:
    assert ref_shape(SHA_A) is RefShape.SHA
    assert ref_shape("0123456789abcdef" * 2 + "01234567") is RefShape.SHA


def test_short_and_long_hex_are_their_own_shapes() -> None:
    assert ref_shape("abc1234") is RefShape.SHORT_HEX
    assert ref_shape("a" * 39) is RefShape.SHORT_HEX
    assert ref_shape("a" * 41) is RefShape.LONG_HEX
    assert ref_shape("A" * 39) is RefShape.SHORT_HEX


def test_uppercase_40_hex_and_names_are_other() -> None:
    assert ref_shape("A" * 40) is RefShape.OTHER
    for ref in ("main", "v4", "release-1.2", "g" * 40):
        assert ref_shape(ref) is RefShape.OTHER


# ── classification ───────────────────────────────────────────────────────────


def test_commit() -> None:
    v = classify_object(_Table({f"repos/o/r/commits/{SHA_A}": (200, {"sha": SHA_A})}), "o/r", SHA_A)
    assert v.kind is ObjectKind.COMMIT


def test_annotated_tag_object_names_tag_and_target() -> None:
    get = _Table({f"repos/o/r/git/tags/{SHA_A}": (200, {"tag": "v1", "object": {"sha": SHA_B}})})
    v = classify_object(get, "o/r", SHA_A)
    assert (v.kind, v.detail, v.tag_name, v.target_sha) == (ObjectKind.INVALID, "tag_object", "v1", SHA_B)


def test_blob() -> None:
    v = classify_object(_Table({f"repos/o/r/git/blobs/{SHA_A}": (200, {})}), "o/r", SHA_A)
    assert (v.kind, v.detail) == (ObjectKind.INVALID, "blob")


def test_missing_object_in_a_visible_repo_is_invalid() -> None:
    v = classify_object(_Table({"repos/o/r": (200, {})}), "o/r", SHA_A)
    assert (v.kind, v.detail) == (ObjectKind.INVALID, "missing")


def test_invisible_repo_is_unreadable_not_invalid() -> None:
    v = classify_object(_Table({}), "o/private", SHA_A)
    assert (v.kind, v.detail) == (ObjectKind.UNREADABLE, "repo_not_visible")


def test_transport_faults_propagate() -> None:
    def boom(path: str) -> tuple[int, object]:
        raise ResolutionError(f"GET /{path} -> HTTP 500")

    with pytest.raises(ResolutionError):
        classify_object(boom, "o/r", SHA_A)


def test_a_commit_costs_one_request() -> None:
    get = _Table({f"repos/o/r/commits/{SHA_A}": (200, {})})
    classify_object(get, "o/r", SHA_A)
    assert get.calls == [f"repos/o/r/commits/{SHA_A}"]
