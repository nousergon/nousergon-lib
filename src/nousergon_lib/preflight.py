"""
Preflight: fast fail-fast connectivity + freshness checks.

``BasePreflight`` provides the shared primitives; consumer modules
subclass it and override ``run()`` to compose a module-specific check
sequence. The base raises ``RuntimeError`` on any failure — consumers
catch nothing, so the raise propagates up through ``main()`` → non-zero
exit → the orchestration layer's failure handler.

Design context (2026-04-14): the alpha-engine-data DailyData step
silently ran against a stale ArcticDB universe library for two
weekdays because an ``ImportError`` on ``arcticdb`` was caught at debug
level. A freshness check on SPY would have flagged the outage in ~1s.
Preflight exists to catch that class of failure *before* spending 30
minutes on real work.

Scope is deliberately narrow: **external-world handshakes only** (env
vars, S3 reachability, ArcticDB symbol freshness). Data-correctness
hard-fails still live in the hardened collectors themselves.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

log = logging.getLogger(__name__)

# Default location for the deploy-time GIT_SHA stamp inside a Lambda
# image. Stamped by deploy.sh via ``--build-arg GIT_SHA=…`` then COPYed
# to /var/task/GIT_SHA.txt; consumers running outside Lambda can pass an
# alternate path.
_DEFAULT_GIT_SHA_FILE = Path("/var/task/GIT_SHA.txt")


class BasePreflight:
    """Shared preflight primitives.

    Subclass and override :meth:`run` to compose a module-specific
    check sequence. Each primitive raises :class:`RuntimeError` on
    failure with an explanatory message that includes what was checked
    and what went wrong.
    """

    def __init__(self, bucket: str, region: str | None = None):
        if not bucket:
            raise ValueError("BasePreflight: bucket is required")
        self.bucket = bucket
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")

    # ── Composition entry point ──────────────────────────────────────────

    def run(self) -> None:
        """Execute the preflight check sequence.

        Subclasses override this to compose primitives. The default
        raises to prevent a misuse where a subclass forgets to override
        and silently passes.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override run() to compose preflight checks"
        )

    # ── Primitives ───────────────────────────────────────────────────────

    def check_env_vars(self, *names: str) -> None:
        """Raise if any of the given env vars are unset or empty."""
        missing = [n for n in names if not os.environ.get(n)]
        if missing:
            raise RuntimeError(f"Pre-flight: required env vars missing: {missing}")

    def check_s3_bucket(self) -> None:
        """Raise if the configured bucket is not reachable (auth, network, or missing)."""
        import boto3
        try:
            boto3.client("s3").head_bucket(Bucket=self.bucket)
        except Exception as exc:
            raise RuntimeError(
                f"Pre-flight: S3 bucket {self.bucket!r} unreachable: {exc}"
            ) from exc

    def check_s3_key(self, key: str, max_age_days: int | None = None) -> None:
        """Raise if ``s3://{bucket}/{key}`` is missing or older than ``max_age_days``.

        ``max_age_days=None`` disables the freshness check — existence only.
        """
        import boto3
        from botocore.exceptions import ClientError
        try:
            head = boto3.client("s3").head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            err_code = exc.response.get("Error", {}).get("Code")
            if err_code in ("404", "NoSuchKey"):
                raise RuntimeError(
                    f"Pre-flight: S3 key s3://{self.bucket}/{key} does not exist"
                ) from exc
            raise RuntimeError(
                f"Pre-flight: S3 key s3://{self.bucket}/{key} unreachable: {exc}"
            ) from exc
        if max_age_days is not None:
            last_modified = head["LastModified"]
            age_days = (datetime.now(timezone.utc) - last_modified).days
            if age_days > max_age_days:
                raise RuntimeError(
                    f"Pre-flight: S3 key s3://{self.bucket}/{key} is "
                    f"{age_days} days stale (threshold {max_age_days})"
                )

    def check_arcticdb_fresh(
        self,
        library: str,
        symbol: str,
        max_stale_days: int,
    ) -> None:
        """Raise if ``arcticdb`` is unavailable, the library/symbol is
        unreadable, or the last date in ``symbol`` is older than
        ``max_stale_days`` calendar days from today (UTC).

        Requires the ``arcticdb`` optional extra
        (``nousergon-lib[arcticdb]``).
        """
        try:
            import arcticdb as adb
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Pre-flight: arcticdb not importable — install "
                "nousergon-lib[arcticdb] or add arcticdb to the deploy image: "
                f"{exc}"
            ) from exc

        uri = (
            f"s3s://s3.{self.region}.amazonaws.com:{self.bucket}"
            "?path_prefix=arcticdb&aws_auth=true"
        )
        try:
            lib = adb.Arctic(uri).get_library(library)
        except Exception as exc:
            raise RuntimeError(
                f"Pre-flight: ArcticDB library {library!r} unreachable "
                f"at {uri}: {exc}"
            ) from exc

        try:
            # ArcticDB's VersionedItem.data is typed as a broad
            # NormalizableType union (DataFrame/Series/ndarray/ExpressionNode
            # /LazyDataFrame/etc, since ArcticDB symbols can hold any
            # normalizable type, and lib.read() without a lazy query builder
            # never returns the Lazy* variants); this method reads a price
            # DataFrame by contract, so the cast trusts that over the
            # unstubbed union.
            df = cast("pd.DataFrame", lib.read(symbol).data)
        except Exception as exc:
            raise RuntimeError(
                f"Pre-flight: ArcticDB {library}/{symbol} read failed: {exc}"
            ) from exc

        if df.empty:
            raise RuntimeError(
                f"Pre-flight: ArcticDB {library}/{symbol} is empty"
            )

        # df.index[-1]'s static type is a broad Index.__getitem__ overload
        # union (pyright can't narrow the scalar element type from an
        # unstubbed index); pd.Timestamp(...) does the actual runtime
        # coercion/validation regardless of the input's concrete type. The
        # df.empty check above guarantees a real index label here, so
        # pd.Timestamp's NaT branch (its ctor stub's fallback for
        # None/unparseable input) can never actually fire — cast it away.
        last_ts = cast("pd.Timestamp", pd.Timestamp(cast(Any, df.index[-1])))
        # Normalize to tz-naive date for comparison against today's UTC date.
        if last_ts.tzinfo is not None:
            last_ts = last_ts.tz_convert("UTC").tz_localize(None)
        today = pd.Timestamp(datetime.now(timezone.utc).date())
        age_days = (today - last_ts.normalize()).days
        if age_days > max_stale_days:
            raise RuntimeError(
                f"Pre-flight: ArcticDB {library}/{symbol} last date "
                f"{last_ts.date()} is {age_days} days stale "
                f"(threshold {max_stale_days})"
            )

    def check_arcticdb_universe_fresh(
        self,
        library: str,
        max_stale_days: int,
        *,
        max_workers: int = 20,
    ) -> None:
        """[DEPRECATED 2026-05-05] Per-symbol freshness scan over an
        ArcticDB library.

        Deprecated because data-freshness now lives upstream in
        ``alpha-engine-data``'s preflight, which runs before any
        consumer in every Step Function. Consumers (executor,
        backtester, predictor) dropped their calls in 2026-05-05's
        consolidation arc. Scheduled for removal after 6-month soak;
        current callers should migrate to trusting SF ordering.

        Original docstring follows.

        Scan every symbol in ``library`` and raise if any symbol's
        last_date is older than ``max_stale_days`` calendar days from
        today (UTC).

        Where :meth:`check_arcticdb_fresh` covers a single canonical
        liveness probe (e.g. macro/SPY), this primitive catches the
        partial-write class — individual tickers stop receiving writes
        while the canonical SPY symbol stays fresh, so the single-symbol
        check reports healthy but downstream consumers fail two hours
        deep on stale per-ticker reads.

        Motivation (2026-04-21 backtester incident): macro.SPY was fresh,
        ASGN + MOH had stalled at 2026-04-01 because daily_append silently
        skipped them, executor's load_atr_14_pct guard aborted the
        backtester ~2 hours into its predictor-backtest mode. This scan
        catches the same class at preflight in ~5-10 seconds (20 threads
        × ~900 tickers × tail(1) read each).

        Implementation notes:
        - Reads ``tail(1)`` rather than the full series — ~20ms/symbol.
        - Read errors on any symbol are themselves fatal: a silent read
          error here would mask exactly the kind of write-skip this
          primitive exists to catch.
        - Stale list is sorted by stalest-first so the operator sees
          the worst offenders without scrolling.

        Requires the ``arcticdb`` optional extra
        (``nousergon-lib[arcticdb]``).

        Args:
            library: ArcticDB library name to scan (e.g. ``"universe"``).
            max_stale_days: Symbols with ``last_date`` older than today
                minus this many calendar days are flagged as stale.
            max_workers: Thread pool size for the per-symbol scan.
                Default 20 matches backtester precedent. Tune lower for
                rate-limited backends; higher for fan-out-bound cases.

        Raises:
            RuntimeError: If arcticdb is unimportable, the library is
                unreachable, the library is empty, any symbol's
                ``tail(1)`` read raises, or ANY symbol is stale beyond
                the threshold.
        """
        warnings.warn(
            "BasePreflight.check_arcticdb_universe_fresh is deprecated; "
            "data-freshness now lives upstream in alpha-engine-data's "
            "preflight (runs before consumers in every Step Function). "
            "Scheduled for removal after 6-month soak.",
            DeprecationWarning,
            stacklevel=2,
        )

        from concurrent.futures import ThreadPoolExecutor
        from datetime import date, timedelta

        try:
            import arcticdb as adb
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Pre-flight: arcticdb not importable — install "
                "nousergon-lib[arcticdb] or add arcticdb to the deploy image: "
                f"{exc}"
            ) from exc

        uri = (
            f"s3s://s3.{self.region}.amazonaws.com:{self.bucket}"
            "?path_prefix=arcticdb&aws_auth=true"
        )
        try:
            lib = adb.Arctic(uri).get_library(library)
        except Exception as exc:
            raise RuntimeError(
                f"Pre-flight: ArcticDB library {library!r} unreachable "
                f"at {uri}: {exc}"
            ) from exc

        symbols = list(lib.list_symbols())
        if not symbols:
            raise RuntimeError(
                f"Pre-flight: ArcticDB library {library!r} on bucket "
                f"{self.bucket!r} has zero symbols — upstream pipeline "
                "has not written anything."
            )

        today = date.today()
        cutoff = today - timedelta(days=max_stale_days)

        def _last_date_for(sym: str) -> tuple[str, date | None, str | None]:
            try:
                # See the analogous cast on check_arcticdb_universe_fresh
                # above: VersionedItem.data's declared type is a broad
                # NormalizableType union; this reads a price DataFrame by
                # contract.
                df = cast("pd.DataFrame", lib.tail(sym, n=1).data)
                if df.empty:
                    return sym, None, "empty frame"
                # See the analogous cast + comment above: the df.empty
                # check guarantees a real index label, so NaT can't
                # actually fire.
                last_ts = cast("pd.Timestamp", pd.Timestamp(cast(Any, df.index[-1])))
                if last_ts.tzinfo is not None:
                    last_ts = last_ts.tz_convert("UTC").tz_localize(None)
                return sym, last_ts.date(), None
            except Exception as exc:  # pragma: no cover — covered via mock
                return sym, None, str(exc)

        stale: list[tuple[str, date]] = []
        errored: list[tuple[str, str]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for sym, last_date, err in pool.map(_last_date_for, symbols):
                if err is not None:
                    errored.append((sym, err))
                elif last_date is None:
                    errored.append((sym, "no last_date"))
                elif last_date < cutoff:
                    stale.append((sym, last_date))

        if errored:
            sample = [f"{s}({e[:40]})" for s, e in errored[:5]]
            raise RuntimeError(
                f"Pre-flight: {len(errored)} symbol(s) in ArcticDB "
                f"library {library!r} could not be read for freshness check. "
                f"Sample: {sample}. Treated as fatal because a silent read "
                "error here would mask exactly the kind of per-symbol write "
                "skip this scan exists to catch."
            )

        if stale:
            stale.sort(key=lambda x: x[1])
            summary = [f"{sym} (last={d.isoformat()})" for sym, d in stale[:10]]
            more = f" (+{len(stale) - 10} more)" if len(stale) > 10 else ""
            raise RuntimeError(
                f"Pre-flight: {len(stale)}/{len(symbols)} symbol(s) in "
                f"ArcticDB library {library!r} have stale data (older "
                f"than {max_stale_days} calendar days, "
                f"cutoff={cutoff.isoformat()}). Top offenders: "
                f"{summary}{more}. Backfill upstream or investigate "
                "the per-symbol write path before re-running."
            )

    def check_ib_paper_account(self, account_id: str) -> None:
        """Raise if ``account_id`` doesn't start with 'D' (IBKR paper prefix).

        Defensive check for the executor — prevents live credentials
        leaking into a paper-trading run (or vice versa).
        """
        if not account_id:
            raise RuntimeError("Pre-flight: IB account_id is empty")
        if not account_id.startswith("D"):
            raise RuntimeError(
                f"Pre-flight: IB account_id {account_id!r} is not a paper "
                "account (paper accounts start with 'D')"
            )

    def check_deploy_drift(
        self,
        repo: str,
        branch: str = "main",
        *,
        sha_file: Path | None = None,
        timeout: float = 5.0,
    ) -> None:
        """Hard-fail if the deploy-baked SHA lags ``repo@branch`` HEAD.

        The deployed image is stamped with ``GIT_SHA`` at build time
        (via Docker ``--build-arg GIT_SHA=…``); this check compares
        that stamp against the current ``branch`` HEAD SHA on GitHub.
        A mismatch means a merge landed on main but the CI deploy
        workflow either failed, was skipped by a paths filter, or
        hasn't run yet — i.e. the deployed code is a prior commit,
        which is exactly the deploy-drift mode that motivated this
        check (2026-04-20 coverage-gap session).

        A mismatch is not automatically drift: rapid-fire back-to-back
        merges mean the image can finish building from an older `main`
        HEAD after a newer commit has already landed. If ``baked`` is
        still an ancestor of ``upstream`` (i.e. reachable by walking
        upstream's history backwards), the deployed code is a strict
        prefix of current main — a benign race, not drift. Only a SHA
        that is *not* an ancestor (force-push, history rewrite, or a
        deploy built from an unrelated branch) hard-fails.

        Degraded modes (warn, don't fail) — chosen so a GitHub outage
        or an unstamped legacy image doesn't block a trading-hours
        Lambda:
        - Stamp file missing or "unknown"  → image predates drift
          checking; log warn and continue.
        - GitHub API unreachable           → log warn and continue.

        Hard-fail mode — when both stamps are present, differ, and
        ``baked`` is not an ancestor of ``upstream`` (including when
        the ancestry check itself can't be resolved — an unmet
        mismatch defaults to hard-fail, never to a silent pass).

        Args:
            repo: ``"owner/name"`` — e.g. ``"nousergon/crucible-predictor"``.
            branch: Branch HEAD to compare against. Default ``"main"``.
            sha_file: Path to the GIT_SHA stamp. Defaults to
                ``/var/task/GIT_SHA.txt`` (Lambda image convention).
            timeout: GitHub API timeout in seconds.
        """
        baked = _read_baked_git_sha(sha_file or _DEFAULT_GIT_SHA_FILE)
        if baked is None:
            log.warning(
                "Deploy-drift: no baked GIT_SHA in image at %s (legacy build "
                "or build-arg omitted). Rebuild via deploy.sh to enable this check.",
                sha_file or _DEFAULT_GIT_SHA_FILE,
            )
            return

        upstream = _fetch_origin_main_sha(repo, branch=branch, timeout=timeout)
        if upstream is None:
            # _fetch_origin_main_sha already logged the reason
            return

        if baked == upstream:
            log.info("Deploy-drift: image at %s matches %s@%s ✓", baked[:12], repo, branch)
            return

        ancestry = _is_ancestor(repo, base=baked, head=upstream, timeout=timeout)
        if ancestry:
            log.info(
                "Deploy-drift: image at %s is behind %s@%s (%s) but is a valid "
                "ancestor — benign build/merge race, not drift.",
                baked[:12], repo, branch, upstream[:12],
            )
            return

        if ancestry is None:
            # Fail closed, but say what actually happened. Asserting "is not an
            # ancestor" here would be a claim GitHub never made — on 2026-08-20
            # a single unretried HTTP 500 produced exactly that sentence about
            # a SHA that WAS an ancestor, and it failed a deploy canary.
            raise RuntimeError(
                f"Deploy drift UNRESOLVED: image was built from {baked[:12]} and "
                f"{repo}@{branch} is now at {upstream[:12]}, but the GitHub compare "
                f"API could not be reached to establish whether {baked[:12]} is an "
                f"ancestor (see the preceding warning for the transport error). "
                f"This may be a benign build/merge race or real drift — refusing to "
                f"guess. Re-run once GitHub is reachable; if it persists, re-run "
                f"`.github/workflows/deploy.yml` on main (or the local deploy.sh). "
                f"Refusing to proceed — running stale code on new signals is how "
                f"2026-04-20 happened."
            )

        raise RuntimeError(
            f"Deploy drift: image was built from {baked[:12]} but "
            f"{repo}@{branch} is now at {upstream[:12]}, and {baked[:12]} is "
            f"not an ancestor of it. The CI deploy workflow did not promote "
            f"the latest commit (or history was rewritten/force-pushed). "
            f"Re-run `.github/workflows/deploy.yml` on main (or the local "
            f"deploy.sh) before resuming. Refusing to proceed — "
            f"running stale code on new signals is how 2026-04-20 happened."
        )


def _read_baked_git_sha(sha_file: Path) -> str | None:
    """Return the SHA baked into the image by ``deploy.sh --build-arg GIT_SHA=…``.

    Returns ``None`` if the stamp file is missing (legacy image) or holds
    ``"unknown"`` (build-arg omitted). Callers decide whether ``None`` is
    warn-and-continue or hard-fail.
    """
    try:
        sha = sha_file.read_text().strip()
    except FileNotFoundError:
        return None
    if not sha or sha == "unknown":
        return None
    return sha


def _safe_urlopen(req, **kwargs):
    """urlopen wrapper that fails loudly on any non-https scheme (S310: bandit
    cannot statically prove the URL's scheme, but every call site here builds
    it from a hardcoded https:// base -- this makes that guarantee explicit
    and enforced at runtime rather than just asserted by code review)."""
    url = req.full_url if isinstance(req, urllib.request.Request) else req
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-https URL: {url!r}")
    return urllib.request.urlopen(req, **kwargs)  # noqa: S310 -- scheme validated above


def _fetch_origin_main_sha(
    repo: str,
    branch: str = "main",
    timeout: float = 5.0,
    *,
    stats: dict[str, Any] | None = None,
    allow_ambient_auth: bool = False,
) -> str | None:
    """Fetch HEAD SHA of ``branch`` for ``repo`` via GitHub REST API.

    Returns ``None`` on any network/parse error — the drift check treats a
    GitHub outage as "unknown, proceed with warning" rather than blocking
    the consumer. ``repo`` is ``"owner/name"`` (e.g.
    ``"nousergon/crucible-predictor"``).
    """
    payload, definitive = github_get_json(
        f"https://api.github.com/repos/{repo}/branches/{branch}",
        timeout=timeout,
        stats=stats,
        allow_ambient_auth=allow_ambient_auth,
    )
    if payload is None:
        # Both the transient-outage and the definitive-404 case land here as
        # "cannot compare". Unlike the ancestry check below, a missing upstream
        # SHA leaves nothing to compare against at all, so this stays
        # warn-and-continue rather than blocking a trading-hours Lambda.
        log.warning(
            "Deploy-drift: no %s@%s HEAD from GitHub (definitive=%s) — cannot compare",
            repo, branch, definitive,
        )
        return None
    return payload.get("commit", {}).get("sha")


#: HTTP statuses that mean "GitHub could not answer *right now*" as opposed to
#: "GitHub answered, and the answer is no". A 5xx or a 429 says nothing about
#: ancestry; collapsing it into a definitive verdict is what turned a transient
#: 500 into a false "not an ancestor" deploy-drift halt on 2026-08-20.
_TRANSIENT_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})

#: HTTP statuses that are a verdict about the CALLER'S CREDENTIAL, never about
#: the resource. GitHub returns 401 for a token it rejects and 403 for one that
#: is valid but not permitted (or for an exhausted rate limit). Neither says
#: anything about whether the repo, branch or commit exists, so neither may
#: ever be returned as ``definitive`` — the whole point of the definitive flag
#: is "GitHub answered, and the answer is no about the THING YOU ASKED FOR".
#:
#: Measured 2026-08-21 (alpha-engine-config-I7924): predictor Lambda v516,
#: deployed 01:50 UTC, was the first production execution of the auth header
#: added the previous day. It picked up the long-expired ``GITHUB_TOKEN`` that
#: the SSM-to-env builder injects, GitHub answered 401, this helper classified
#: it ``definitive``, ``check_deploy_drift`` omitted ``sf_drift`` as unmeasured,
#: and the preopen ``DeployDriftGate`` fail-closed branch halted the 12:15 UTC
#: trading pipeline three hours later. An expired credential must never be able
#: to halt trading: every repo this probe reads is PUBLIC and answers the same
#: question with no credential at all.
_GITHUB_CREDENTIAL_HTTP_STATUSES = frozenset({401, 403})

#: Retry schedule (seconds to sleep *before* attempts 2 and 3) for transient
#: GitHub failures. Bounded deliberately: this runs inside a Lambda preflight,
#: so the whole retry budget stays under ~2s of sleep on top of the per-attempt
#: timeout rather than eating the invocation.
_GITHUB_RETRY_BACKOFF = (0.5, 1.5)


def _github_auth_headers(*, allow_ambient: bool = False) -> dict[str, str]:
    """Bearer header from ``GITHUB_TOKEN``/``GH_TOKEN`` when one is present.

    Absence is normal, never an error: every GitHub read this module performs
    is against a PUBLIC repo, so the unauthenticated 60 req/hr ceiling is a
    complete fallback rather than a degraded one. A token, when present, buys
    only rate-limit headroom (5000 req/hr).

    It is therefore NEVER correct for a token to make a call fail that would
    have succeeded without it. ``github_get_json`` enforces that by retrying
    unauthenticated on 401/403 — do not add a call site that assumes a token
    must be present, and do not "fix" a credential rejection by making the
    caller fail closed.

    The pre-2026-08-21 docstring asserted "Lambda images carry no token". That
    was false: the alpha-engine SSM-to-env builder injects ``GITHUB_TOKEN``
    into the predictor Lambda, whose own CI test forbids reading it in-repo —
    the invariant was enforced in the repo and bypassed through this library.

    ``GITHUB_TOKEN`` is a pinned secret (alpha-engine-config-I7925, #345):
    resolved through :func:`krepis.secrets.get_secret` — the same
    SSM-with-env-fallback path every other pinned secret in the fleet uses —
    rather than a direct environ read of that name, which is exactly the
    pattern consumer repos' CI forbids in their own tree and could not see
    inside this installed library. ``GH_TOKEN`` (the GitHub CLI's own env var,
    not a pinned secret) is read directly — a convenience alias for local dev,
    never provisioned by the fleet.

    **And the pickup itself is OPT-IN, off by default (I7925, #346).**
    These two are complements, not alternatives, and both are needed:

    - Routing through ``get_secret`` fixes *how* the credential is looked up,
      so the read is visible to the fleet's secret-scan and can come from SSM.
    - ``allow_ambient`` fixes *whether* it is looked up at all — and that is
      the half that answers the 2026-08-21 halt. ``get_secret(required=False)``
      still falls back to the environment, so on its own it would still have
      **found** the dead value frozen in the predictor Lambda's environment.
      The token was not passed to anything; it was found, by code that had no
      idea it was there, in an environment nothing writes and therefore nothing
      can rotate. Eleven Lambdas are in that state.

    A credential must be *offered* to be used. Measured 2026-08-21, **no caller
    of this module needs one** — every consumer (``deploy_drift``,
    ``lib_pin_drift``, ``crucible-executor``'s preflight, ``BasePreflight``)
    runs in a Lambda or on an EC2 box against PUBLIC repos, where the anonymous
    ceiling is ample and cannot expire. Pass ``allow_ambient=True`` only from a
    CI job or laptop script that genuinely needs the 5000 req/hr ceiling, and
    only where a stale ambient value could not be silently present.

    The opt-in check comes FIRST, before any lookup: a default call must not
    reach SSM, both because the credential is not wanted and because a
    per-invocation SSM round-trip on a preflight path is a cost nobody asked
    for.
    """
    if not allow_ambient:
        return {}

    from krepis.secrets import get_secret

    token = get_secret("GITHUB_TOKEN", required=False) or os.environ.get("GH_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def github_get_json(
    url: str,
    *,
    timeout: float = 5.0,
    attempts: int = 3,
    sleep: Any = None,
    stats: dict[str, Any] | None = None,
    allow_ambient_auth: bool = False,
) -> tuple[dict[str, Any] | None, bool]:
    """GET *url* from the GitHub REST API, retrying transient failures.

    Returns ``(payload, definitive)``:

    - ``(payload, True)``  — GitHub answered; the payload is the answer.
    - ``(None, True)``     — GitHub answered *definitively negative* (404 for
      an unknown repo/commit, 422 for an unprocessable comparison). The caller
      may treat this as a real finding, not as an outage.
    - ``(None, False)``    — could not be resolved after *attempts* tries
      (network error, timeout, 5xx, rate limit, unparseable body). The caller
      must NOT report this as a substantive answer.

    The (payload, definitive) split exists because the single-attempt
    predecessor of this helper returned a bare ``False`` for both "GitHub says
    no" and "GitHub did not answer", so a transient 500 rendered as a
    confidently-worded drift halt naming an ancestry that was in fact fine.

    A 401/403 is neither of those: it is a verdict about the CALLER, and it is
    never returned as ``definitive``. When a token was sent, the whole attempt
    budget is retried once WITHOUT it, because every repo read through this
    helper is public and answers identically unauthenticated. See
    ``_GITHUB_CREDENTIAL_HTTP_STATUSES`` for the incident that made this
    mandatory rather than merely nice.

    ``allow_ambient_auth`` (default ``False``) governs whether a
    ``GITHUB_TOKEN``/``GH_TOKEN`` found in the surrounding environment is used.
    Off by default because a credential that is *found* rather than *offered* is
    what halted trading on 2026-08-21 — see ``_github_auth_headers``. Every repo
    read through this helper is public, so the default costs nothing but the
    5000 req/hr ceiling.

    ``stats``, when supplied, is populated in place with observability the
    tuple has no room for — currently ``github_credential_rejected`` (bool)
    and ``github_credential_status`` (int). A caller that surfaces a verdict to
    an operator should pass one and render it: an expired token that no longer
    breaks the call is still a defect that must be fixed before it breaks a
    call that has no anonymous fallback.
    """
    if sleep is None:  # pragma: no cover - trivial default indirection
        sleep = time.sleep
    auth = _github_auth_headers(allow_ambient=allow_ambient_auth)
    headers = {"Accept": "application/vnd.github+json", **auth}
    last_exc: Exception | None = None
    # One unauthenticated re-attempt is allowed after a credential rejection.
    # Consumed rather than looped so a persistently-403 anonymous caller (rate
    # limit) still terminates on the normal attempt budget.
    auth_fallback_available = bool(auth)
    attempt = 0
    while attempt < attempts:
        try:
            # S310: ruff cannot statically prove the scheme through a variable
            # URL. _safe_urlopen enforces https:// at runtime and raises
            # otherwise, which is the guarantee the rule is asking for; every
            # caller here also builds the URL from a hardcoded https:// base.
            req = urllib.request.Request(url, headers=headers)  # noqa: S310
            with _safe_urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read()), True
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in _GITHUB_CREDENTIAL_HTTP_STATUSES:
                if auth_fallback_available:
                    # The credential is the problem, not the request. Drop it
                    # and start over anonymously — every repo read here is
                    # public. Loud at ERROR because a rejected token is a real
                    # operational defect that must be fixed, even though it no
                    # longer breaks this call (sf-pipeline-policy 2.3: the
                    # degradation is visible, it is just not fatal).
                    log.error(
                        "GitHub API %s -> HTTP %s with GITHUB_TOKEN/GH_TOKEN "
                        "present: the credential was REJECTED. Retrying "
                        "unauthenticated (public repo). Rotate or remove the "
                        "token — alpha-engine-config-I7924.",
                        url, exc.code,
                    )
                    if stats is not None:
                        stats["github_credential_rejected"] = True
                        stats["github_credential_status"] = exc.code
                    auth_fallback_available = False
                    headers = {"Accept": "application/vnd.github+json"}
                    attempt = 0
                    continue
                # No credential in play (or already stripped). 401/403 without
                # a token is a rate limit or an org policy — still not a
                # verdict about the resource, so never definitive.
                log.warning(
                    "GitHub API %s -> HTTP %s unauthenticated — no answer, "
                    "not a negative answer", url, exc.code,
                )
            elif exc.code not in _TRANSIENT_HTTP_STATUSES:
                # A definitive negative: the resource genuinely is not there /
                # is not comparable. Retrying cannot change this answer.
                log.warning("GitHub API %s -> HTTP %s (definitive)", url, exc.code)
                return None, True
        except (OSError, json.JSONDecodeError) as exc:
            # OSError covers URLError plus the bare TimeoutError urlopen raises
            # on a read-phase timeout (2026-05-07 weekday SF DeployDriftCheck).
            last_exc = exc
        if attempt < len(_GITHUB_RETRY_BACKOFF):
            sleep(_GITHUB_RETRY_BACKOFF[attempt])
        attempt += 1
    log.warning(
        "GitHub API %s unresolved after %d attempts (%s) — no answer, not a negative answer",
        url, attempts, last_exc,
    )
    return None, False


def _is_ancestor(
    repo: str, base: str, head: str, timeout: float = 5.0, *, sleep: Any = None,
    allow_ambient_auth: bool = False,
) -> bool | None:
    """Return whether ``base`` is an ancestor of (or equal to) ``head``.

    Uses GitHub's compare-commits API rather than local ``git
    merge-base`` — Lambda images never bake in a ``.git`` object
    database (only source directories are ``COPY``'d), so there is no
    local history to walk. ``status`` is one of ``identical`` (same
    commit), ``ahead`` (``head`` has commits ``base`` doesn't, ``base``
    reachable from ``head``), ``behind`` (``base`` has commits ``head``
    doesn't — ``base`` is *not* an ancestor of ``head``), or
    ``diverged`` (neither is an ancestor of the other). Only
    ``identical``/``ahead`` count as a benign race; ``behind``/
    ``diverged`` are real drift.

    Tri-state, deliberately:

    - ``True``  — ancestor (benign build/merge race).
    - ``False`` — GitHub answered and the answer is "not an ancestor",
      including a definitive 404/422 (a SHA main has never heard of is
      exactly the force-push/rewrite case this guard exists for).
    - ``None``  — ancestry could not be resolved after retries. The
      caller still fails closed, but must say *that*, not assert a
      relationship GitHub never confirmed.
    """
    payload, definitive = github_get_json(
        f"https://api.github.com/repos/{repo}/compare/{base}...{head}",
        timeout=timeout,
        sleep=sleep,
        allow_ambient_auth=allow_ambient_auth,
    )
    if not definitive:
        return None
    if payload is None:
        return False
    return payload.get("status") in ("identical", "ahead")
