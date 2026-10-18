"""Is this process part of a weekly-SF rehearsal? (alpha-engine-config-I12351)

Brian's rule (2026-10-16): a rehearsal must be runnable at any time and stay
isolated from live files. ``rehearsal-2026-10-16-2`` still wrote live
bookkeeping through three writers in this library: the ``health/<module>.json``
heartbeat (:func:`nousergon_lib.health.write_health`), the console row
``ops/checks/<id>/latest.json`` (:func:`nousergon_lib.fleet_check_result.emit`)
and the multiple-testing counter ``backtest/cumulative_trial_count.json``
(:func:`nousergon_lib.quant.stats.trial_accumulator.increment_trial_count`),
which its seven sweep increments moved by 137 trials.

**The signal** is the one every other rehearsal guard in the fleet keys on, so
a process is never a rehearsal to one layer and live to another:

* ``RUN_TOKEN`` — the Step Functions execution name, exported by
  ``krepis.ssm_log_capture`` and forwarded onto every spot by the launchers —
  starting with ``rehearsal-`` (the same prefix
  :data:`nousergon_lib.phase_registry.REHEARSAL_EXECUTION_PREFIX` keys on);
* ``AE_PIPELINE_ROLE=rehearsal``, the explicit override for a hand run.

**Where a rehearsal's records go.** :func:`namespaced_key` puts them under a
``.rehearsal/<token>/`` segment directly below the record's own top-level
prefix, the layout :mod:`nousergon_lib.phase_registry` already uses for
completion markers. Readers address live records by exact key or by a date
partition, so they never see the rehearsal's copy, and a role allowed to write
the live record under that top-level prefix may write the copy too.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

#: Same value as :data:`nousergon_lib.phase_registry.REHEARSAL_EXECUTION_PREFIX`
#: (asserted by tests). Not imported from there: that module pulls boto3 and
#: PyYAML at import time, and the writers below must stay importable without them.
REHEARSAL_EXECUTION_PREFIX = "rehearsal-"

RUN_TOKEN_ENV = "RUN_TOKEN"  # noqa: S105 — an env var NAME, not a secret
PIPELINE_ROLE_ENV = "AE_PIPELINE_ROLE"
#: Token used when ``AE_PIPELINE_ROLE=rehearsal`` is set without an execution name.
UNNAMED_TOKEN = "rehearsal-unnamed"  # noqa: S105 — a path segment, not a secret
NAMESPACE_SEGMENT = ".rehearsal"

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def rehearsal_token(env: Mapping[str, str] | None = None) -> str | None:
    """This process's rehearsal execution name, or ``None`` outside a rehearsal.

    The token is safe to use as one S3 path segment.
    """
    env = os.environ if env is None else env
    token = (env.get(RUN_TOKEN_ENV) or "").strip()
    if token.startswith(REHEARSAL_EXECUTION_PREFIX):
        return _UNSAFE.sub("", token) or UNNAMED_TOKEN
    if (env.get(PIPELINE_ROLE_ENV) or "").strip().lower() == "rehearsal":
        return UNNAMED_TOKEN
    return None


def namespaced_key(key: str, token: str | None) -> str:
    """``key`` as a rehearsal's copy of it; ``key`` itself when ``token`` is None.

    ``health/backtester.json`` → ``health/.rehearsal/<token>/backtester.json``.
    """
    if not token:
        return key
    head, sep, rest = key.partition("/")
    if not sep:
        return f"{NAMESPACE_SEGMENT}/{token}/{key}"
    return f"{head}/{NAMESPACE_SEGMENT}/{token}/{rest}"


__all__ = [
    "NAMESPACE_SEGMENT",
    "PIPELINE_ROLE_ENV",
    "REHEARSAL_EXECUTION_PREFIX",
    "RUN_TOKEN_ENV",
    "UNNAMED_TOKEN",
    "namespaced_key",
    "rehearsal_token",
]
