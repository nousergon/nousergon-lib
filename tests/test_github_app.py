"""Unit tests for nousergon_lib.github_app (config-I2785).

No network: the mint path is exercised against a mocked ``urlopen``; the
JWT path is verified end-to-end with a real RSA keypair (PyJWT[crypto] is
installed via the ``github_app`` extra in CI).
"""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import datetime, timedelta, timezone
from unittest import mock

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from nousergon_lib import github_app


@pytest.fixture(autouse=True)
def _fresh_cache():
    github_app.clear_cache()
    yield
    github_app.clear_cache()


@pytest.fixture(scope="module")
def rsa_keys():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    return pem, key.public_key()


def _mint_response(token="ghs_testtoken", expires_at="2099-01-01T00:00:00Z"):
    body = json.dumps({"token": token, "expires_at": expires_at}).encode()
    resp = mock.MagicMock()
    resp.read.return_value = body
    resp.__enter__ = mock.Mock(return_value=resp)
    resp.__exit__ = mock.Mock(return_value=False)
    return resp


# ── JWT ──────────────────────────────────────────────────────────────────────


def test_build_app_jwt_claims_roundtrip(rsa_keys):
    pem, public_key = rsa_keys
    encoded = github_app.build_app_jwt("12345", pem)
    claims = pyjwt.decode(encoded, public_key, algorithms=["RS256"])
    assert claims["iss"] == "12345"
    # iat is backdated 60s against clock skew; ttl measured from "now" ≈ iat+60
    assert claims["exp"] - claims["iat"] == github_app.JWT_TTL_SECONDS + 60


def test_normalize_pem_unescapes_and_terminates():
    assert github_app.normalize_pem("a\\nb") == "a\nb\n"
    assert github_app.normalize_pem("a\nb\n\n") == "a\nb\n"


# ── mint ─────────────────────────────────────────────────────────────────────


def test_mint_installation_token_success(rsa_keys):
    pem, _ = rsa_keys
    with mock.patch.object(github_app.urllib.request, "urlopen", return_value=_mint_response()) as m:
        minted = github_app.mint_installation_token(
            app_id="1", installation_id="99", private_key_pem=pem
        )
    assert minted.token == "ghs_testtoken"
    assert minted.expires_at == datetime(2099, 1, 1, tzinfo=timezone.utc)
    req = m.call_args[0][0]
    assert req.full_url.endswith("/app/installations/99/access_tokens")
    assert req.get_method() == "POST"
    assert req.get_header("Authorization", "").startswith("Bearer ")


def test_mint_http_error_raises(rsa_keys):
    pem, _ = rsa_keys
    err = urllib.error.HTTPError(
        "https://api.github.com/x", 401, "Unauthorized", None, io.BytesIO(b'{"message":"bad"}')
    )
    with mock.patch.object(github_app.urllib.request, "urlopen", side_effect=err):
        with pytest.raises(github_app.GitHubAppTokenError, match="HTTP 401"):
            github_app.mint_installation_token(
                app_id="1", installation_id="99", private_key_pem=pem
            )


def test_mint_missing_token_field_raises(rsa_keys):
    pem, _ = rsa_keys
    with mock.patch.object(
        github_app.urllib.request, "urlopen", return_value=_mint_response(token="")
    ):
        with pytest.raises(github_app.GitHubAppTokenError, match="missing 'token'"):
            github_app.mint_installation_token(
                app_id="1", installation_id="99", private_key_pem=pem
            )


def test_parse_expiry_degrades_to_55min():
    before = datetime.now(timezone.utc)
    parsed = github_app._parse_expiry("not-a-date")
    assert parsed - before >= timedelta(minutes=54)
    assert github_app._parse_expiry("2099-01-01T00:00:00Z") == datetime(
        2099, 1, 1, tzinfo=timezone.utc
    )


# ── cached SSM-backed convenience ───────────────────────────────────────────


def _patch_secrets(monkeypatch):
    """Populate the GROOM_GH_APP_* env overrides (checked before SSM)."""
    monkeypatch.setenv("GROOM_GH_APP_ID", "1")
    monkeypatch.setenv("GROOM_GH_APP_INSTALLATION_ID", "99")
    monkeypatch.setenv("GROOM_GH_APP_PRIVATE_KEY", "PEM")


def test_installation_token_mints_once_and_caches(monkeypatch):
    _patch_secrets(monkeypatch)
    fresh = github_app.InstallationToken(
        token="ghs_cached", expires_at=datetime.now(timezone.utc) + timedelta(minutes=55)
    )
    with mock.patch.object(github_app, "mint_installation_token", return_value=fresh) as m:
        assert github_app.installation_token() == "ghs_cached"
        assert github_app.installation_token() == "ghs_cached"
    assert m.call_count == 1
    assert m.call_args.kwargs["private_key_pem"] == "PEM\n"  # normalized


def test_installation_token_remints_near_expiry(monkeypatch):
    _patch_secrets(monkeypatch)
    nearly_dead = github_app.InstallationToken(
        token="ghs_old", expires_at=datetime.now(timezone.utc) + timedelta(minutes=2)
    )
    fresh = github_app.InstallationToken(
        token="ghs_new", expires_at=datetime.now(timezone.utc) + timedelta(minutes=55)
    )
    with mock.patch.object(
        github_app, "mint_installation_token", side_effect=[nearly_dead, fresh]
    ) as m:
        assert github_app.installation_token() == "ghs_old"
        assert github_app.installation_token() == "ghs_new"  # margin forces re-mint
    assert m.call_count == 2


def test_installation_token_missing_secrets_raises(monkeypatch):
    for env in ("GROOM_GH_APP_ID", "GROOM_GH_APP_INSTALLATION_ID", "GROOM_GH_APP_PRIVATE_KEY"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    from moto import mock_aws

    with mock_aws():  # empty SSM — parameter absent
        with pytest.raises(github_app.GitHubAppTokenError, match="github_app_id"):
            github_app.installation_token()


def test_installation_token_reads_ssm_when_no_env(monkeypatch):
    for env in ("GROOM_GH_APP_ID", "GROOM_GH_APP_INSTALLATION_ID", "GROOM_GH_APP_PRIVATE_KEY"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    from moto import mock_aws

    fresh = github_app.InstallationToken(
        token="ghs_ssm", expires_at=datetime.now(timezone.utc) + timedelta(minutes=55)
    )
    with mock_aws():
        import boto3

        ssm = boto3.client("ssm", region_name="us-east-1")
        for name, value in [
            ("github_app_id", "42"),
            ("github_app_installation_id", "4242"),
            ("github_app_private_key", "KEY\\nMATERIAL"),
        ]:
            ssm.put_parameter(
                Name=f"/alpha-engine/groom/{name}", Value=value, Type="SecureString"
            )
        with mock.patch.object(github_app, "mint_installation_token", return_value=fresh) as m:
            assert github_app.installation_token() == "ghs_ssm"
    assert m.call_args.kwargs["app_id"] == "42"
    assert m.call_args.kwargs["installation_id"] == "4242"
    assert m.call_args.kwargs["private_key_pem"] == "KEY\nMATERIAL\n"  # unescaped + terminated


def test_expiring_within():
    tok = github_app.InstallationToken(
        token="t", expires_at=datetime(2030, 1, 1, tzinfo=timezone.utc)
    )
    at = datetime(2029, 12, 31, 23, 56, tzinfo=timezone.utc)  # 4 min left
    assert tok.expiring_within(300, now=at)
    assert not tok.expiring_within(120, now=at)
