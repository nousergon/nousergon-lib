"""The packaged egress DLP ruleset must be krepis's ruleset, rule for rule.

alpha-engine-config-I9712. ``gitleaks-custom.toml`` -- the fleet's own
secret-shape rules -- existed in four copies (krepis package data,
alpha-engine-config's groom routing dir, claude-code-config's laptop egress
proxy, and this package's ``egress/``) and diverged in BOTH directions. This
copy predated alpha-engine-config-I4946's two false-positive fixes (the
Anthropic rule fired on a bare prefix in prose; the IBKR rule matched its own
description) and lacked claude-code-config-I112's two detection fixes (a
Telegram token ending in ``-`` was unmatched; the stock AWS key-id rule fired
on ~5-25% of key ids). Nothing compared the copies, so each fix lived on one
side only.

krepis is PUBLIC and its packaged ``dlp_config/`` is the canonical copy, so this
guard needs no credential: it fetches both TOML files from krepis ``main``
unauthenticated, parses both sides, and fails on ANY difference in the parsed
documents -- rule ids, regexes, allowlists, entropy, keywords, tags,
descriptions, ``[extend]``, ``[allowlist]``. Comments are not part of the
parsed document, so this copy may keep its own provenance header; it may not
keep its own rules.

A fetch failure FAILS, never skips. A parity guard that goes green when it
cannot see the other side is the silent pass it exists to remove; the fetch is
retried across two independent endpoints before it gives up.
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import pytest

try:  # 3.11+
    import tomllib
except ImportError:  # 3.9/3.10: tomli is a declared dev dependency there
    import tomli as tomllib  # type: ignore[no-redef]

EGRESS_DIR = Path(__file__).resolve().parent.parent / "src" / "nousergon_lib" / "egress"
FILES = ("gitleaks-custom.toml", "gitleaks-egress.toml")
KREPIS_PATH = "src/krepis/dlp_config"
SOURCES = (
    "https://raw.githubusercontent.com/nousergon/krepis/main/{path}",
    # Second, not first: unauthenticated contents-API calls are rate limited
    # per IP (60/h), and hosted runners share IPs.
    "https://api.github.com/repos/nousergon/krepis/contents/{path}?ref=main",
)
ATTEMPTS = 3


def _fetch_krepis(name: str) -> str:
    errors = []
    for template in SOURCES:
        url = template.format(path=f"{KREPIS_PATH}/{name}")
        for attempt in range(1, ATTEMPTS + 1):
            try:
                # Both URLs are https literals from SOURCES above; no
                # caller-supplied scheme can reach urlopen.
                req = urllib.request.Request(  # noqa: S310
                    url,
                    headers={
                        "Accept": "application/vnd.github.raw",
                        "User-Agent": "nousergon-lib-dlp-ruleset-parity",
                    },
                )
                with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
                    return resp.read().decode("utf-8")
            except Exception as exc:  # noqa: BLE001 -- every failure is reported
                errors.append(f"{url} attempt {attempt}: {exc}")
                time.sleep(2 * attempt)
    pytest.fail(
        "krepis main unreachable -- the DLP ruleset parity guard cannot run, "
        "and a guard that cannot see the canonical copy must not pass "
        "(alpha-engine-config-I9712):\n  " + "\n  ".join(errors)
    )


def _differences(local: dict, canonical: dict) -> list:
    out = []
    for key in sorted(set(local) | set(canonical)):
        if key != "rules" and local.get(key) != canonical.get(key):
            out.append(f"[{key}] local={local.get(key)!r} krepis={canonical.get(key)!r}")
    lr = {r.get("id"): r for r in local.get("rules", [])}
    cr = {r.get("id"): r for r in canonical.get("rules", [])}
    out += [f"rule {rid!r} only in this copy" for rid in sorted(set(lr) - set(cr))]
    out += [f"rule {rid!r} only in krepis" for rid in sorted(set(cr) - set(lr))]
    for rid in sorted(set(lr) & set(cr)):
        for field in sorted(set(lr[rid]) | set(cr[rid])):
            if lr[rid].get(field) != cr[rid].get(field):
                out.append(f"rule {rid!r} field {field!r}: local={lr[rid].get(field)!r} krepis={cr[rid].get(field)!r}")
    if not out and local != canonical:
        out.append("parsed documents differ (rule order or an unclassified field)")
    return out


@pytest.mark.parametrize("name", FILES)
def test_packaged_ruleset_matches_krepis_main(name):
    local = tomllib.loads((EGRESS_DIR / name).read_text(encoding="utf-8"))
    canonical = tomllib.loads(_fetch_krepis(name))
    diffs = _differences(local, canonical)
    assert not diffs, (
        f"src/nousergon_lib/egress/{name} diverges from krepis main "
        "(src/krepis/dlp_config/). krepis is the canonical DLP ruleset: land the "
        "rule change there first, then copy it here byte-for-byte "
        "(alpha-engine-config-I9712).\n  " + "\n  ".join(diffs)
    )


def test_the_comparison_sees_a_one_character_regex_change():
    """The guard's own teeth: the exact delta I9712 found (a dropped trailing
    ``\\b``) must register as a difference, not compare equal."""
    base = tomllib.loads((EGRESS_DIR / "gitleaks-custom.toml").read_text(encoding="utf-8"))
    mutated = tomllib.loads((EGRESS_DIR / "gitleaks-custom.toml").read_text(encoding="utf-8"))
    mutated["rules"][0]["regex"] += r"\b"
    assert _differences(mutated, base)
    assert not _differences(base, base)
