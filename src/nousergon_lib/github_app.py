"""GitHub App installation-token minting — the fleet's bot write identity.

Consolidation point (config-I2785, lifted on second adoption from
``alpha-engine-config/scripts/github_app_token.py``): the console Decision
Queue loader and the scheduled-groom dispatcher both authenticate GitHub
REST calls as the org's GitHub App (``ne-groomer[bot]``) instead of an
operator PAT. The 2026-07-16 GitHub partial outage (config-I2784) 503'd
every user-token REST call for ~an hour while App installation tokens rode
through untouched — the App identity is the resilient default; operator
PATs are the fallback, decided by each consumer.

Requires the ``[github_app]`` extra (``PyJWT[crypto]``)::

    from nousergon_lib.github_app import installation_token

    token = installation_token()  # SSM /alpha-engine/groom/github_app_*

Installation tokens live one hour; :func:`installation_token` caches the
minted token in-process and re-mints once it is within
:data:`REFRESH_MARGIN_SECONDS` of expiry. Callers should re-call per
request burst rather than store the string long-term.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
JWT_TTL_SECONDS = 9 * 60  # GitHub caps App JWTs at 10 minutes
REFRESH_MARGIN_SECONDS = 5 * 60  # re-mint when < this remains on a cached token
# Fleet App credentials. Full SSM paths — krepis.secrets.get_secret is NOT
# usable here (it enforces a flat /alpha-engine/<NAME> namespace and rejects
# nested names), so this module reads SSM directly via boto3.
DEFAULT_SSM_PREFIX = "/alpha-engine/groom/"
# Env-var override (checked before SSM) — same contract the groom harness's
# bootstrap already exports for scripts/github_app_token.py.
_ENV_OVERRIDES = {
    "github_app_id": "GROOM_GH_APP_ID",
    "github_app_installation_id": "GROOM_GH_APP_INSTALLATION_ID",
    "github_app_private_key": "GROOM_GH_APP_PRIVATE_KEY",
}
_USER_AGENT = "nousergon-lib-github-app"

_cache: dict[str, InstallationToken] = {}
_cache_lock = threading.Lock()


def _safe_urlopen(req: urllib.request.Request, **kwargs):
    """urlopen wrapper that fails loudly on any non-https scheme (S310:
    bandit cannot statically prove the URL's scheme — mirrors
    ``preflight._safe_urlopen``; the only call site builds the URL from the
    https ``GITHUB_API`` default or a caller-supplied ``api`` base, and this
    makes the https guarantee enforced at runtime)."""
    if not req.full_url.startswith("https://"):
        raise GitHubAppTokenError(f"refusing non-https URL: {req.full_url!r}")
    return urllib.request.urlopen(req, **kwargs)  # noqa: S310 -- scheme validated above


class GitHubAppTokenError(RuntimeError):
    """An installation token could not be minted (fail loud, no fallback here)."""


def normalize_pem(pem: str) -> str:
    """SSM/env-held private keys often arrive with literal ``\\n`` escapes."""
    return pem.replace("\\n", "\n").strip() + "\n"


def build_app_jwt(app_id: str, private_key_pem: str) -> str:
    """RS256 App JWT (the credential that authenticates *as the App*)."""
    try:
        import jwt
    except ImportError as exc:  # pragma: no cover — packaging error, not logic
        raise GitHubAppTokenError(
            "PyJWT not installed — pip install 'nousergon-lib[github_app]'"
        ) from exc
    now = int(time.time())
    payload = {"iat": now - 60, "exp": now + JWT_TTL_SECONDS, "iss": str(app_id)}
    try:
        return jwt.encode(payload, private_key_pem, algorithm="RS256")
    except Exception as exc:  # corrupt/wrong-format key material in SSM —
        # normalize to the module's error so consumers' PAT fallback engages.
        raise GitHubAppTokenError(f"App JWT signing failed: {exc}") from exc


@dataclass(frozen=True)
class InstallationToken:
    token: str
    expires_at: datetime  # aware UTC

    def expiring_within(self, seconds: float, *, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        return (self.expires_at - now).total_seconds() <= seconds


def mint_installation_token(
    *,
    app_id: str,
    installation_id: str,
    private_key_pem: str,
    api: str = GITHUB_API,
) -> InstallationToken:
    """POST /app/installations/{id}/access_tokens → short-lived ``ghs_`` token.

    Raises :class:`GitHubAppTokenError` on any transport or contract failure —
    consumers own the decision to fall back to a PAT, this module never does.
    """
    app_jwt = build_app_jwt(app_id, private_key_pem)
    url = f"{api}/app/installations/{installation_id}/access_tokens"
    # S310 can't see through the ``api`` parameter indirection (kept for
    # testability); constructing a Request does no I/O — the https scheme is
    # enforced at runtime by _safe_urlopen below, mirroring preflight.py.
    req = urllib.request.Request(  # noqa: S310 -- scheme validated in _safe_urlopen
        url,
        data=b"{}",
        method="POST",
        headers={
            "Authorization": f"Bearer {app_jwt}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": _USER_AGENT,
        },
    )
    try:
        with _safe_urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise GitHubAppTokenError(
            f"GitHub App token mint failed: HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise GitHubAppTokenError(f"GitHub App token mint failed: {exc}") from exc
    token = body.get("token")
    if not token:
        raise GitHubAppTokenError(f"token-mint response missing 'token': {body!r}")
    expires_at = _parse_expiry(body.get("expires_at"))
    return InstallationToken(token=str(token), expires_at=expires_at)


def _parse_expiry(raw: str | None) -> datetime:
    """GitHub returns ISO-8601 Zulu; a missing/odd value degrades to +55 min
    (safe: earlier-than-real expiry only causes a premature re-mint)."""
    if raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            logger.warning("github_app: unparseable expires_at %r — assuming 55min", raw)
    return datetime.now(timezone.utc) + timedelta(minutes=55)


def _read_credential(name: str, ssm_prefix: str, region: str | None) -> str:
    """Env override first (local dev / groom harness), then SSM full path."""
    import os

    env_name = _ENV_OVERRIDES.get(name)
    if env_name and os.environ.get(env_name, "").strip():
        return os.environ[env_name].strip()
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError as exc:  # pragma: no cover — boto3 is a core lib dep
        raise GitHubAppTokenError(f"boto3 unavailable for SSM read: {exc}") from exc
    try:
        ssm = boto3.client("ssm", region_name=region) if region else boto3.client("ssm")
        resp = ssm.get_parameter(Name=f"{ssm_prefix}{name}", WithDecryption=True)
        value = resp["Parameter"]["Value"].strip()
    except (BotoCoreError, ClientError) as exc:
        raise GitHubAppTokenError(
            f"App credential unreadable at SSM {ssm_prefix}{name} "
            f"(and no ${env_name} env override): {exc}"
        ) from exc
    if not value:
        raise GitHubAppTokenError(f"App credential empty at SSM {ssm_prefix}{name}")
    return value


def installation_token(
    *,
    ssm_prefix: str = DEFAULT_SSM_PREFIX,
    region: str | None = None,
    api: str = GITHUB_API,
) -> str:
    """Cached installation token from SSM-held App credentials.

    Reads ``{ssm_prefix}github_app_id`` / ``_installation_id`` /
    ``_private_key`` (env overrides ``GROOM_GH_APP_*`` win — the groom
    harness's existing contract). The lock is held across the mint so
    concurrent callers (Streamlit threads, Lambda warm starts) can't
    stampede the GitHub endpoint.
    """
    with _cache_lock:
        cached = _cache.get(ssm_prefix)
        if cached and not cached.expiring_within(REFRESH_MARGIN_SECONDS):
            return cached.token
        app_id = _read_credential("github_app_id", ssm_prefix, region)
        inst_id = _read_credential("github_app_installation_id", ssm_prefix, region)
        pem = normalize_pem(_read_credential("github_app_private_key", ssm_prefix, region))
        minted = mint_installation_token(
            app_id=app_id, installation_id=inst_id, private_key_pem=pem, api=api
        )
        _cache[ssm_prefix] = minted
        return minted.token


def clear_cache() -> None:
    """Drop cached tokens (tests / credential rotation)."""
    with _cache_lock:
        _cache.clear()
