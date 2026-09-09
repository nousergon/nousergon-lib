#!/usr/bin/env python3
"""llm_endpoint_probe — one `max_tokens=1` completion, one bounded verdict.

WHY THIS EXISTS (alpha-engine-config-I10230, second adoption of I10037)
-------------------------------------------------------------------------
The same shape — POST one `max_tokens=1` completion at an LLM endpoint, then
classify the response into a bounded verdict vocabulary — existed in THREE
places with three drifting copies of the marker lists and vocabularies:

  1. `nous-ergon-ops/alpha-engine-dashboard/live/infrastructure/bin/
     router_funded_depth_probe.py` (`_request`, `classify`, `probe_entry`) —
     an AUTHENTICATED call against the router's TLS edge, one `(provider,
     model, route, credential)` tuple at a time, to prove an entry actually
     serves rather than merely being funded.
  2. `nous-ergon-ops/alpha-engine-dashboard/live/infrastructure/bin/
     router_degraded_mode_drill.py` (`_serve_probe_request`, `classify_serve`)
     — added by `nous-ergon-ops-PR1130`'s D5 assertion, an UNAUTHENTICATED
     call at a resolved LOOPBACK endpoint while the router is down, to prove
     the wire the drill just resolved to actually carries a byte.
     `shared-code-policy.md`'s second-adoption trigger fired here and was
     deliberately NOT taken, with the rationale recorded inline in that
     file: lifting would have gated PR1130 on a `nousergon-lib` release, and
     `router_funded_depth_probe.py` was (and remains) a sibling script on the
     box with no importable package.

The third adoption is this module. `alpha-engine-config/scripts/
probe_llm_capabilities.py` was named as a candidate third copy in
I10230's filing but, on inspection at lift time, answers a DIFFERENT
question — whether a DECLARED capability (structured outputs, tool_choice,
web_search, streaming) is honoured, with its own {ok, fail, advisory,
skipped} vocabulary over a full completion — not "does this endpoint serve
at all" over a `max_tokens=1` probe. It has no `UNFUNDED_MARKERS`-shaped
constant, no `classify(status, body)` function, and nothing in this fleet's
tree does. It is out of scope for this lift; see the migration PR for the
measured finding.

THE VOCABULARY THAT WON, AND WHAT WAS LOSSY
---------------------------------------------
`classify()` below keeps copy 1's (`router_funded_depth_probe.classify`)
verdict set and documented ORDERING CONTRACT verbatim — money before
credentials, the proxy's own refusal before the generic 400 bucket — and
inserts copy 2's `dlp_blocked` verdict ahead of the whole 4xx family, exactly
where copy 2 already placed it. The two marker lists (`UNFUNDED_MARKERS`,
copy 2's DLP markers) are unioned rather than picked: nothing a either copy's
list already caught stops being caught.

Nothing is lost migrating copy 1 (`router_funded_depth_probe.py`): its
verdict set was already the superset minus `dlp_blocked`, which it cannot
reach in practice (it calls the router's TLS edge directly with its own
credential, never the box's DLP-scanning loopback proxy) but which costs it
nothing to carry.

Migrating copy 2 (`router_degraded_mode_drill.py`) is a **measured, benign
behaviour change**: `classify_serve` never carried the HTML-body
(`wrong_path`) test, so a body opening `<!doctype ...>` fell through to
`error` there. After this lift the same body classifies `wrong_path`. Copy
2's own `SERVE_OK_VERDICTS = {"servable", "cooled_down"}` treats both
`error` and `wrong_path` as failures, so the drill's D5 PASS/FAIL outcome is
UNCHANGED — only the label attached to an HTML-body failure becomes more
specific (a route-prefix defect rather than an unclassified one).

CONTRACT PRESERVED VERBATIM (do not weaken any of these in this module)
---------------------------------------------------------------------
* One `max_tokens=1` completion per call — `request()` never sends more.
* No credential is injected by this module. A caller that wants an
  authenticated call passes its own `Authorization` header; a caller that
  wants the egress proxy to inject its own resolved upstream key (loopback,
  no inbound `Authorization`/`x-api-key`) passes none.
* `429` classifies `cooled_down`, never a failure verdict on its own —
  callers decide whether `cooled_down` counts as OK for their use.
* `unfunded` and `unauthorized` are never folded into each other or into
  `error` — different fix, different owner.
* The verdict vocabulary is CLOSED (`VERDICTS`) — a bounded CloudWatch
  metric dimension per `observability-policy` §4. `classify()` cannot
  return anything outside it.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

_ALLOWED_SCHEMES = ("http", "https")


def _validate_url(url: str) -> str:
    """Refuse a `file:`/custom scheme before it reaches `urlopen` (S310)."""
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"refusing to open {url!r}: scheme {scheme!r} is not one of {_ALLOWED_SCHEMES}"
        )
    return url


# ── wire contract ──────────────────────────────────────────────────────────

#: The request path each wire posts to, when the caller does not override it
#: with an explicit ``path=``. The wire IS the path — one upstream host can
#: serve both wires under different prefixes, so the endpoint alone never
#: says which wire is being spoken.
WIRE_PATHS: dict[str, str] = {
    "anthropic": "/v1/messages",
    "openai": "/v1/chat/completions",
}

#: Header the Anthropic Messages wire requires; the proxy forwards it
#: untouched. Absent it, an anthropic-wire endpoint 400s and the verdict
#: would say "the leg is broken" about a missing header rather than the
#: leg itself.
ANTHROPIC_VERSION_HEADER = "2023-06-01"


def request(
    base_url: str,
    *,
    wire: str,
    model: str,
    path: str | None = None,
    headers: dict[str, str] | None = None,
    timeout: float,
) -> tuple[int, str]:
    """One `max_tokens=1` completion at ``base_url``. Returns ``(status, body)``.

    Never raises on an HTTP error or a transport failure — the status (and,
    for a transport failure, a synthetic ``0``) IS the measurement, read by
    :func:`classify`. ``status == 0`` carries
    ``"{ExceptionName}: {message}"`` as the body so a timeout and a refused
    connection are told apart by :func:`classify`, not by an exception type
    escaping here.

    ``path`` defaults to ``WIRE_PATHS[wire]``; pass it explicitly for a
    caller (e.g. an authenticated router-edge probe) whose path is not the
    wire's own prefix. ``headers`` are merged over the wire's own required
    headers (``anthropic-version`` for ``wire="anthropic"``) — a caller may
    not use this to unset ``Content-Type`` or the version header.
    """
    if path is None:
        if wire not in WIRE_PATHS:
            raise ValueError(
                f"unknown wire {wire!r} has no declared path — pass path= "
                f"explicitly or add it to WIRE_PATHS"
            )
        path = WIRE_PATHS[wire]

    body = {
        "model": model,
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "ok"}],
    }
    hdrs: dict[str, str] = {"Content-Type": "application/json"}
    if wire == "anthropic":
        hdrs["anthropic-version"] = ANTHROPIC_VERSION_HEADER
    if headers:
        hdrs.update(headers)

    req = urllib.request.Request(  # noqa: S310 -- scheme validated by _validate_url
        _validate_url(base_url.rstrip("/") + path),
        data=json.dumps(body).encode(),
        headers=hdrs,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 -- see above
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, f"{type(exc).__name__}: {exc}"


# ── verdict vocabulary ───────────────────────────────────────────────────────

#: The closed set `classify()` may return. A bounded CloudWatch metric
#: dimension (`observability-policy` §4) — adding a verdict is a deliberate
#: edit here, never an implicit new string at a call site.
VERDICTS: tuple[str, ...] = (
    "servable",
    "timeout",
    "unreachable",
    "dlp_blocked",
    "unfunded",
    "unauthorized",
    "route_unconfigured",
    "cooled_down",
    "wrong_path",
    "not_served",
    "error",
)

#: Substrings that turn a refusal into an UNFUNDED verdict, matched
#: case-insensitively against the response body. Union of both prior
#: copies' lists — every phrase either fleet component actually received
#: from a provider. Extending this list is how a new provider's exhaustion
#: language is taught. An unmatched refusal falls through to
#: `unauthorized`/`error` rather than being guessed at: calling a credential
#: problem "unfunded" sends someone to a billing page over a rotated key.
UNFUNDED_MARKERS: tuple[str, ...] = (
    "credit balance is too low",       # anthropic
    "insufficient balance",            # deepseek
    "insufficient_quota",
    "quota",
    "used all available credits",      # xai
    "out of credits",
    "quota exceeded",
    "payment required",
    "billing",
)

#: The egress proxy's OWN refusal when a row names an upstream absent from
#: the host's `--routes` table. Matched on the proxy's message rather than a
#: status alone because 400 is also what a malformed request returns, and
#: the two need different people. The literal is the proxy's, not a
#: provider's, so it is stable across every provider behind it.
ROUTE_UNCONFIGURED_MARKER = "unknown upstream host"

#: The proxy's DLP refusal. A BLOCK is not a broken leg — it is the control
#: working on a body the caller sent, and it must never be graded as a
#: provider/endpoint failure. Copy 2's addition; copy 1 cannot reach this
#: verdict in practice (it calls the router's TLS edge directly, never the
#: box's DLP-scanning loopback proxy) but carries it at zero cost.
DLP_BLOCK_MARKERS: tuple[str, ...] = ("dlp", "secret detected", "blocked by", "gitleaks")

#: A body that is an HTML DOCUMENT rather than an API error. Provider APIs
#: answer JSON; HTML means the request left the API surface for a web edge,
#: which is a route-PREFIX defect rather than a missing model. Matched on
#: shape, with no provider name in it.
_HTML_PREFIXES = ("<!doctype", "<html", "<?xml", "<head", "<body")


def _is_html(body: str) -> bool:
    """Does this body open as an HTML document?

    Substring over the head rather than `startswith`, because a gateway can
    prepend a status line, a BOM or blank lines and the body is still HTML.
    The window is bounded so a JSON error that merely QUOTES a tag deep in a
    provider message cannot be mistaken for a page.
    """
    head = body.lstrip()[:200].lower()
    return any(prefix in head for prefix in _HTML_PREFIXES)


def classify(status: int, body: str) -> str:
    """One verdict per response, always a member of :data:`VERDICTS`.

    ORDER IS THE CONTRACT — carried verbatim from `router_funded_depth_
    probe.classify`, with `dlp_blocked` inserted ahead of the whole 4xx
    family exactly where `router_degraded_mode_drill.classify_serve` already
    placed it:

    * `dlp_blocked` first among refusals, because it is the only 4xx
      produced by THIS fleet's own control rather than by the provider, and
      calling it `error` (or `unauthorized`) would send a reader to a
      provider's status page for something that never left the box;
    * money before credentials, because an exhausted account answers
      401/403 at several providers and calling that `unauthorized` sends
      someone to rotate a working key;
    * the proxy's own refusal (`route_unconfigured`) before the generic 400
      bucket, because it is the only verdict here that no provider can fix;
    * `unauthorized` before the HTML test, because a 401 challenge page is
      an HTML body and it is still a credential problem — the path was
      right enough to be refused by the API;
    * `wrong_path` (HTML body) before `not_served` (404), because a 404
      that is HTML is a route-prefix defect, not a missing model;
    * `not_served` (404) last among the classified verdicts, because it is
      the answer when nothing above applies: the listener is there, the
      path is not.
    """
    lowered = body.lower()
    if status == 200:
        return "servable"
    if status == 0:
        # `request()` returns status 0 with `"{ExceptionName}: {message}"`
        # for every transport failure. A timeout is a listener that took
        # the connection and did not answer; a refusal is one that is not
        # there. Same red, different next step.
        if "timeout" in lowered or "timed out" in lowered:
            return "timeout"
        return "unreachable"
    if any(marker in lowered for marker in DLP_BLOCK_MARKERS):
        return "dlp_blocked"
    if status == 402 or any(marker in lowered for marker in UNFUNDED_MARKERS):
        return "unfunded"
    if status in (401, 403):
        return "unauthorized"
    if ROUTE_UNCONFIGURED_MARKER in lowered:
        return "route_unconfigured"
    if status == 429:
        return "cooled_down"
    if _is_html(body):
        return "wrong_path"
    if status == 404:
        return "not_served"
    return "error"


def probe(
    base_url: str,
    *,
    wire: str,
    model: str,
    path: str | None = None,
    headers: dict[str, str] | None = None,
    timeout: float,
) -> dict[str, Any]:
    """Convenience wrapper: one `request()` + `classify()`, as a result dict.

    Neither prior copy called anything named `probe` — each built its own
    result envelope with fields the other did not carry (`model` vs.
    `registry_id`, `duration_s`, `skip_reason`, ...), so this module does
    not impose one. It provides `request`/`classify` as the two functions
    every call site actually shares, and this wrapper for a caller that
    wants the common case (status, body, verdict) with no envelope of its
    own opinion.
    """
    status, body = request(
        base_url, wire=wire, model=model, path=path, headers=headers, timeout=timeout
    )
    return {"http_status": status, "body": body, "verdict": classify(status, body)}
