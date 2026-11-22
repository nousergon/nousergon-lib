"""Shared classification of GitHub Actions ``uses:`` pins (alpha-engine-config-I13935).

Two questions, answered once for every guard that has to ask them:

1. **What is this ``uses:`` value?** :func:`parse_uses` splits
   ``owner/repo[/path]@ref`` and skips local (``./``) and ``docker://``
   references. :func:`ref_shape` separates a 40-character *lowercase*
   hexadecimal SHA from a short or a long hexadecimal ref, which are different
   mistakes (an abbreviated SHA pasted from ``git log``, a SHA with a stray
   suffix) and deserve different words than a branch or tag name does.
2. **Is that SHA a commit?** :func:`classify_object` asks the GitHub REST API
   through an *injected* ``get`` callable, so this module owns no transport,
   token or retry policy and a test can answer from a table. It separates three
   outcomes that must never be merged:

   * ``COMMIT`` -- the SHA resolves to a commit;
   * ``INVALID`` -- the SHA is definitively not a commit: a blob (the ``.sha``
     that ``GET /contents/{path}`` returns), an annotated tag object (with the
     commit it dereferences to), or no reachable object at all;
   * ``UNREADABLE`` -- the repository itself is invisible to the caller's
     token, so nothing can be said about the SHA. A cross-repo private ``uses:``
     404s at every object endpoint, byte-identical to a deleted SHA; calling it
     invalid sends someone hunting a SHA that is fine. The fix is a token.

A transport fault is not an outcome. ``get`` raises :class:`ResolutionError`
and it propagates: a classifier that turns an unreachable API into "no
findings" reports green precisely when it has checked nothing.

The wording of a finding is the caller's. This module returns data
(:class:`ObjectVerdict`), because the same verdict is an annotation in one
guard and a table row in another.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "ActionRef",
    "GetFn",
    "ObjectKind",
    "ObjectVerdict",
    "RefShape",
    "ResolutionError",
    "UsesKind",
    "classify_object",
    "parse_uses",
    "ref_shape",
]

SHA_LENGTH = 40

# Lowercase only: git object IDs render lowercase everywhere GitHub emits them,
# and an uppercase ref is far more likely to be hand-mangled than deliberate.
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")

# owner/repo, optionally followed by a path inside that repo (reusable
# workflows and repo-hosted composite actions both use the longer form).
_USES_RE = re.compile(r"^(?P<owner>[A-Za-z0-9._-]+)/(?P<repo>[A-Za-z0-9._-]+)(?P<path>/[^@]*)?@(?P<ref>.+)$")


class ResolutionError(RuntimeError):
    """The check could not be completed -- not a finding, an infrastructure fault."""


#: ``get(path) -> (http_status, decoded_json_or_None)``. ``path`` has no leading
#: slash (``repos/o/r/commits/<sha>``). 404 and 422 are *answers* (no such
#: object of that type) and are returned; any other failure raises
#: :class:`ResolutionError`.
GetFn = Callable[[str], "tuple[int, object]"]


# ── parsing ──────────────────────────────────────────────────────────────────


class UsesKind(str, Enum):
    """What a ``uses:`` value is, before any ref is judged."""

    SKIPPED = "skipped"  # local action or container image: no cross-repo ref
    UNPINNED = "unpinned"  # no ``@ref`` at all
    UNPARSEABLE = "unparseable"  # an ``@`` but not ``owner/repo[/path]@ref``
    PARSED = "parsed"


@dataclass(frozen=True)
class ActionRef:
    """One parsed ``uses:`` value. ``owner``/``repo``/``ref`` are set only when PARSED."""

    kind: UsesKind
    owner: str = ""
    repo: str = ""
    path: str = ""
    ref: str = ""

    @property
    def owner_repo(self) -> str:
        """The repository to query -- never ``owner/repo/.github/workflows/x.yml``."""
        return f"{self.owner}/{self.repo}"


def parse_uses(value: str) -> ActionRef:
    """Parse one ``uses:`` value. Local ``./`` and ``docker://`` values are SKIPPED."""
    if value.startswith("./") or value.startswith(".\\") or value.startswith("docker://"):
        return ActionRef(UsesKind.SKIPPED)
    match = _USES_RE.match(value)
    if not match:
        return ActionRef(UsesKind.UNPINNED if "@" not in value else UsesKind.UNPARSEABLE)
    return ActionRef(
        UsesKind.PARSED,
        owner=match.group("owner"),
        repo=match.group("repo"),
        path=match.group("path") or "",
        ref=match.group("ref"),
    )


class RefShape(str, Enum):
    """The lexical shape of a ref, decided without touching the network."""

    SHA = "sha"  # exactly 40 lowercase hex characters
    SHORT_HEX = "short_hex"  # hexadecimal, fewer than 40 characters
    LONG_HEX = "long_hex"  # hexadecimal, more than 40 characters
    OTHER = "other"  # a branch or tag name, or 40 hex with an uppercase digit


def ref_shape(ref: str) -> RefShape:
    """Classify *ref* by shape. Only :attr:`RefShape.SHA` is an acceptable pin."""
    if _SHA_RE.match(ref):
        return RefShape.SHA
    if _HEX_RE.match(ref):
        if len(ref) < SHA_LENGTH:
            return RefShape.SHORT_HEX
        if len(ref) > SHA_LENGTH:
            return RefShape.LONG_HEX
    return RefShape.OTHER


# ── resolution ───────────────────────────────────────────────────────────────


class ObjectKind(str, Enum):
    COMMIT = "commit"
    INVALID = "invalid"
    UNREADABLE = "unreadable"


@dataclass(frozen=True)
class ObjectVerdict:
    """The outcome for one SHA.

    ``detail`` names *why* for the non-commit outcomes: ``tag_object``,
    ``blob``, ``missing`` (all INVALID) or ``repo_not_visible`` (UNREADABLE).
    ``tag_name`` and ``target_sha`` are set for ``tag_object`` only;
    ``target_sha`` is the commit the tag dereferences to, i.e. what to use.
    """

    kind: ObjectKind
    detail: str = "commit"
    tag_name: str = ""
    target_sha: str = ""


def _as_dict(body: object) -> dict[str, object] | None:
    if not isinstance(body, dict):
        return None
    return {str(k): v for k, v in body.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]


def classify_object(get: GetFn, owner_repo: str, sha: str) -> ObjectVerdict:
    """Decide what *sha* is in *owner_repo*, using the injected *get*.

    The message a person ends up reading is the deliverable. "Not found" sends
    them back to the same endpoint that produced the bad value; "this is the
    blob SHA of the file" ends the investigation. So the wrong-type probes run
    before the verdict is allowed to be a bare "missing".
    """
    status, _ = get(f"repos/{owner_repo}/commits/{sha}")
    if status == 200:
        return ObjectVerdict(ObjectKind.COMMIT)

    # Wrong-type probes, cheapest first. Both are real objects in the repo, so
    # the only thing that distinguishes them from a commit is the endpoint that
    # accepts them.
    tag_status, tag_body = get(f"repos/{owner_repo}/git/tags/{sha}")
    tag = _as_dict(tag_body)
    if tag_status == 200 and tag is not None:
        target = _as_dict(tag.get("object")) or {}
        return ObjectVerdict(
            ObjectKind.INVALID,
            "tag_object",
            tag_name=str(tag.get("tag", "?")),
            target_sha=str(target.get("sha", "?")),
        )

    blob_status, _ = get(f"repos/{owner_repo}/git/blobs/{sha}")
    if blob_status == 200:
        return ObjectVerdict(ObjectKind.INVALID, "blob")

    # Everything above 404s identically when the repository itself is invisible
    # to this token. Separate the two by asking whether the repo is visible.
    repo_status, _ = get(f"repos/{owner_repo}")
    if repo_status != 200:
        return ObjectVerdict(ObjectKind.UNREADABLE, "repo_not_visible")

    return ObjectVerdict(ObjectKind.INVALID, "missing")
