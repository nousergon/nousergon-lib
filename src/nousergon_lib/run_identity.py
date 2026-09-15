"""Run identity primitives: a run id and the code sha it ran at.

`alpha-engine-config-I10831` deliverable 2. Both functions started life inside
:mod:`nousergon_lib.run_manifest`, bound in name (if never in behavior) to that
module's ``data_run_manifest.v1`` contract. Crucible's ``run_manifest.v2``
needs the same two primitives — a lexically sortable run id and a refused,
never-guessed commit sha — and re-importing them from ``run_manifest`` would
couple crucible's writer to a schema module that is not its own
(`alpha-engine-config-I10784`, `alpha-engine-config-I10810`; crucible-PR305).

Neither function reads or writes anything schema-shaped: no ``schema_version``,
no manifest field name, nothing from either contract. That is what makes this
module safe for both writers to import directly. ``run_manifest`` re-exports
both names unchanged, so every existing importer keeps working.
"""

from __future__ import annotations

import datetime as dt
import os
import random
import re
import subprocess

__all__ = [
    "CODE_SHA_ENV",
    "CodeShaError",
    "new_run_id",
    "resolve_code_sha",
]

#: The box's dispatcher exports the sha it deployed; off the box a real git
#: checkout answers for itself. Named here so a caller can export it.
CODE_SHA_ENV = "NE_DATA_CODE_SHA"

#: A real, non-placeholder git sha: forty lowercase hex characters, explicitly
#: NOT the all-zero placeholder. Mirrors crucible's ``_REAL_SHA_RE``.
_REAL_SHA_RE = re.compile(r"^(?!0{40}$)[0-9a-f]{40}$")

_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


class CodeShaError(RuntimeError):
    """The running tree's commit sha could not be measured.

    Raised BEFORE a run body runs, never at write time: a manifest cannot
    record a value nobody measured, and discovering that later would pit the
    refusal against "manifest or it did not happen".
    """


def new_run_id(now: dt.datetime | None = None) -> str:
    """A ULID: 48 bits of millisecond timestamp then 80 bits of randomness.

    Lexically sortable by creation time, which is what makes a listing of one
    unit's (or job's) day readable in execution order without parsing the ids.
    """
    moment = now or dt.datetime.now(dt.timezone.utc)
    value = (int(moment.timestamp() * 1000) << 80) | random.getrandbits(80)  # noqa: S311 -- an id, not a secret
    out = []
    for _ in range(26):
        out.append(_ULID_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


def resolve_code_sha(env_var: str = CODE_SHA_ENV, cwd: str | None = None) -> str:
    """The commit sha of the tree that is running, or raise :class:`CodeShaError`.

    ``$NE_DATA_CODE_SHA`` wins when set — the box's own answer, carried in the
    release rather than read from a working tree a wheel install does not have.
    Off the box (a laptop or CI run inside a real checkout) the variable is
    normally unset and ``git rev-parse HEAD`` is the real answer.

    Either source producing something other than a real 40-character lowercase
    sha is REFUSED rather than written as the all-zero placeholder that used to
    validate and answer nothing.
    """
    declared = os.environ.get(env_var)
    if declared is not None:
        if not _REAL_SHA_RE.match(declared):
            raise CodeShaError(
                f"${env_var}={declared!r} is not a real 40-character lowercase git sha "
                "(or is the all-zero placeholder). The box exports this from the sha it "
                "deployed; a malformed value there is a deploy-time defect, and code_sha "
                "cannot be written as a value nobody measured."
            )
        return declared
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S603,S607 -- fixed argv, no shell
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            cwd=cwd,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CodeShaError(
            f"${env_var} is unset and `git rev-parse HEAD` could not run ({exc}). "
            f"Export ${env_var} on a box with no git checkout, or run from inside one."
        ) from exc
    sha = out.stdout.strip()
    if out.returncode != 0 or not _REAL_SHA_RE.match(sha):
        raise CodeShaError(
            f"${env_var} is unset and `git rev-parse HEAD` did not return a real sha "
            f"(exit {out.returncode}, stdout {sha!r}). code_sha cannot be written as a "
            "value nobody measured."
        )
    return sha
