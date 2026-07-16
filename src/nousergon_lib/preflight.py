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

        Degraded modes (warn, don't fail) — chosen so a GitHub outage
        or an unstamped legacy image doesn't block a trading-hours
        Lambda:
        - Stamp file missing or "unknown"  → image predates drift
          checking; log warn and continue.
        - GitHub API unreachable           → log warn and continue.

        Hard-fail mode — when both stamps are present and differ.

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

        if baked != upstream:
            raise RuntimeError(
                f"Deploy drift: image was built from {baked[:12]} but "
                f"{repo}@{branch} is now at {upstream[:12]}. The CI deploy "
                f"workflow did not promote the latest commit. Re-run "
                f"`.github/workflows/deploy.yml` on main (or the local "
                f"deploy.sh) before resuming. Refusing to proceed — "
                f"running stale code on new signals is how 2026-04-20 happened."
            )

        log.info("Deploy-drift: image at %s matches %s@%s ✓", baked[:12], repo, branch)


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


def _fetch_origin_main_sha(repo: str, branch: str = "main", timeout: float = 5.0) -> str | None:
    """Fetch HEAD SHA of ``branch`` for ``repo`` via GitHub REST API.

    Returns ``None`` on any network/parse error — the drift check treats a
    GitHub outage as "unknown, proceed with warning" rather than blocking
    the consumer. ``repo`` is ``"owner/name"`` (e.g.
    ``"nousergon/crucible-predictor"``).
    """
    # S310 fires on urllib.request.Request(...) whenever the URL argument is
    # a variable rather than an inline literal (it cannot statically prove
    # the scheme through the indirection) — inlining the f-string directly,
    # same as the alpha-engine-config precedent (config#2532), keeps the
    # scheme provably-hardcoded-https at the call site instead of needing a
    # noqa here on top of the _safe_urlopen runtime guard below.
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/branches/{branch}",
        headers={"Accept": "application/vnd.github+json"},
    )
    try:
        with _safe_urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        return payload.get("commit", {}).get("sha")
    except (OSError, json.JSONDecodeError) as exc:
        # OSError covers urllib.error.URLError/HTTPError plus the bare
        # TimeoutError that urlopen raises on a read-phase timeout (the
        # 2026-05-07 weekday SF DeployDriftCheck failure: read timed out
        # inside http.client.getresponse, which is past urllib's
        # OSError → URLError wrap point in do_open).
        log.warning("Deploy-drift: GitHub API unreachable (%s) — cannot compare", exc)
        return None
