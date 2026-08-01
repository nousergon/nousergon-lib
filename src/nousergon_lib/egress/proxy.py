#!/usr/bin/env python3
"""Generic LLM egress proxy (content-scanning outbound gateway), v2.1.

Canonical source-of-truth copy, consolidated from Brian's laptop-local
``claude-code-config/llm-routing/`` and ``alpha-engine-config/infrastructure/
groom-llm-routing/`` into ``nousergon-lib`` (config-I3482).

Usage
    As a standalone script::

        python3 -m nousergon_lib.egress.proxy --port 8972 \\
            --upstream-host api.deepseek.com --api-key-env DEEPSEEK_API_KEY

    Or via the installed entry point::

        llm-egress-proxy --port 8972 --upstream-host api.deepseek.com \\
            --api-key-env DEEPSEEK_API_KEY

Gitleaks configs are resolved from:
    1. ``$LLM_ROUTING_DIR`` env var (if set — contains gitleaks-egress.toml,
       gitleaks-custom.toml)
    2. The ``nousergon_lib.egress`` package directory (when pip-installed)
    3. The directory containing this file (when run from source)

Fail-closed throughout: missing gitleaks/config/API key, a scan error, or a
gitleaks finding all result in the request being blocked, never silently
forwarded.

v2 (2026-07-22) - incremental scanning, error taxonomy, response salvage,
    block forensics, observability.  See the ARCHITECTURE.md entry at
    alpha-engine-config#3482 for the full v2 changelog.
v2.1 (2026-07-25) - consolidated into nousergon-lib as a proper installable
    package; CONFIG_DIR now supports importlib.resources when pip-installed.
    The ``shutil.which("gitleaks")`` improvement from the vendored copy is
    canonicalised.
---
symposion's spawn-if-not-running logic remains as a harmless fallback:
their health check sees the launchd-owned instance and reuses it.

Usage: started by launchd (see above); the LLM-routed wrapper scripts /
symposion's own process manager retain fallback spawn logic. Not invoked
directly.
  python3 llm_egress_proxy.py --port 8971 --upstream-host api.deepseek.com --api-key-env DEEPSEEK_API_KEY --upstream-prefix /anthropic
  python3 llm_egress_proxy.py --port 8972 --upstream-host api.deepseek.com --api-key-env DEEPSEEK_API_KEY --upstream-prefix ""
  python3 llm_egress_proxy.py --port 8973 --upstream-host api.x.ai --api-key-env XAI_API_KEY --upstream-prefix ""
"""

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar, Optional

__version__ = "2.1.0"

# config-I3007 dev-security 1/5: when run as a PyInstaller-frozen binary (the
# distinct-identity build that lets LuLu firewall THIS process specifically vs
# the shared python interpreter — see bin/build-egress-proxy.sh), `__file__`
# points into the onefile temp-extract dir, not the real .llm-routing dir where
# gitleaks-egress.toml and the logs live.
# v2.1 (config-I3482): consolidated under nousergon-lib.
#   CONFIG_DIR: where gitleaks-egress.toml lives (read-only is fine)
#   DATA_DIR:   where blocked-events/ and per-instance logs go (must be writable)
# Resolution order (configs): $LLM_ROUTING_DIR > importlib.resources > __file__
# Resolution order (data):    $LLM_ROUTING_DIR > $LLM_EGRESS_DATA_DIR > ~/.llm-routing
def _resolve_dirs() -> tuple[str, str]:
    env_override = os.environ.get("LLM_ROUTING_DIR")
    if env_override:
        # Legacy: one directory holds everything (laptop, vendored copy)
        return env_override, env_override

    if getattr(sys, "frozen", False):
        d = os.path.expanduser("~/Development/.llm-routing")
        return d, d

    # Configs: find the package_data gitleaks-egress.toml
    config_dir = None
    try:
        from importlib.resources import files as _rf
        _pkg_cfg = _rf("nousergon_lib.egress") / "gitleaks-egress.toml"
        if _pkg_cfg.is_file():
            config_dir = str(_rf("nousergon_lib.egress"))
    except (ModuleNotFoundError, TypeError, OSError, AttributeError):
        # (a) package resources unavailable (editable install / missing
        #     package_data / Traversable without is_file) — not a hard
        #     failure: CONFIG_DIR falls back to __file__ dirname below.
        # (c) recording surface: this fallback path; no separate log —
        #     proxy start still fails loud later if gitleaks config missing.
        config_dir = None
    if config_dir is None:
        config_dir = os.path.dirname(os.path.abspath(__file__))

    # Data: writable directory for logs + blocked-events
    data_dir = os.environ.get(
        "LLM_EGRESS_DATA_DIR",
        os.path.expanduser("~/.llm-routing"),
    )
    os.makedirs(data_dir, exist_ok=True)
    return config_dir, data_dir

CONFIG_DIR, DATA_DIR = _resolve_dirs()
GITLEAKS_CONFIG = os.path.join(CONFIG_DIR, "gitleaks-egress.toml")
BLOCKED_EVENTS_DIR = os.path.join(DATA_DIR, "blocked-events")
GITLEAKS_TIMEOUT_SECONDS = 8
# Resolved to an absolute path once at import time via shutil.which() rather
# than a bare literal "gitleaks" — ensures S607 compliance and fails loud
# when the binary is missing (the original laptop source used a bare string
# literal; this is the canonical improvement from the vendored copy).
GITLEAKS_BIN = shutil.which("gitleaks") or "gitleaks"

# Large base64 blobs (embedded images) dominate scan time and are not
# meaningfully text-scannable for secrets -- replace them with a short
# placeholder in the SCAN COPY only. The bytes actually forwarded upstream
# are never modified. Applied per string leaf (v1 applied a quote-anchored
# variant to the flattened text, which missed blobs whose surrounding quotes
# were stripped by JSON decoding - this unanchored form catches both).
_BASE64_BLOB_RE = re.compile(r"[A-Za-z0-9+/_-]{2000,}={0,2}")

# Send-phase / connect-phase exceptions that mean "no usable upstream
# response yet" - candidates for response salvage and then one retry.
_TRANSIENT_UPSTREAM_ERRORS = (
    BrokenPipeError,
    ConnectionResetError,
    ConnectionRefusedError,
    ConnectionAbortedError,
    TimeoutError,
    socket.timeout,
    socket.gaierror,
    ssl.SSLEOFError,
    ssl.SSLZeroReturnError,
    http.client.NotConnected,
    http.client.CannotSendRequest,
    http.client.RemoteDisconnected,
)

# Hop-by-hop headers never forwarded in either direction, plus auth headers
# we always replace with the proxy's own key (key-isolation, config#3007).
_STRIP_REQUEST_HEADERS = frozenset({
    "host", "content-length", "authorization", "x-api-key", "connection",
    "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
    "trailer", "transfer-encoding", "upgrade", "expect",
})
_STRIP_RESPONSE_HEADERS = frozenset({
    "connection", "transfer-encoding", "content-length", "keep-alive",
    "trailer", "upgrade",
})


class ClientDisconnected(Exception):
    """The CLIENT hung up while we were relaying - benign, not an upstream fault."""


class UpstreamStreamAbort(Exception):
    """Upstream died mid-stream AFTER headers were already relayed - unrecoverable."""


class UpstreamError(Exception):
    """No usable upstream response after salvage + retry - answer 502."""


def log(message):
    # Mask credential-length tokens so secrets never reach the log file or
    # stderr in clear text — same self-masking used for forensic excerpts.
    # Timestamp is added AFTER masking so it is never itself masked.
    sanitized = mask_secrets(message)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {sanitized}\n"
    log_path = ProxyHandler.log_path
    if log_path is not None:
        try:
            with open(log_path, "a") as f:
                f.write(line)
        except OSError:
            pass
    sys.stderr.write(line)


def die(message):
    sanitized = mask_secrets(message)
    sys.stderr.write(
        f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} FATAL: {sanitized}\n")
    sys.exit(1)


def _collect_strings(value, out: list) -> None:
    """Recursively collect every string leaf in a decoded JSON value.

    Provider request bodies (Anthropic Messages, OpenAI Chat Completions/
    Responses - every wire format this proxy has seen) nest actual content
    (message text, tool inputs/outputs, system prompt) many levels deep in
    JSON. Scanning the RAW JSON bytes misses secrets because JSON
    string-escaping (a literal `"` becomes `\\"`) breaks gitleaks's
    quote-anchored rules (verified empirically 2026-07-13 -- a secret
    gitleaks catches in plain text was missed once JSON-escaped). Scanning
    the decoded string values instead means gitleaks sees exactly the
    characters a human -- or the PreToolUse file-content hooks, which scan
    raw file bytes, not JSON-wrapped ones -- would see.
    """
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            _collect_strings(v, out)
    elif isinstance(value, list):
        for v in value:
            _collect_strings(v, out)
    # numbers/bools/None carry no secret-shaped content


def _mask_token(match):
    tok = match.group(0)
    return f"***[len={len(tok)}]"


def mask_secrets(text: str) -> str:
    """Redact every credential-length token so no secret content reaches output.

    Every run of 8+ non-space chars is replaced with an opaque placeholder
    that preserves only the length (for forensic sizing), never the token
    content.  Over-redaction is fine here — these channels exist purely for
    operational forensics, not to reconstruct original inputs.
    """
    return re.sub(r"\S{8,}", _mask_token, text)


class LeafScanCache:
    """LRU set of string-leaf digests that have already gitleaks-scanned clean.

    Keyed on SHA-256 of the leaf content. Invalidated wholesale whenever the
    gitleaks config chain (gitleaks-egress.toml + everything it [extend]s)
    changes on disk, so a rule tightening always re-scans from scratch.
    Thread-safe: the proxy serves concurrent sessions from a threading server.
    """

    def __init__(self, capacity: int = 100_000):
        self._capacity = capacity
        self._clean = OrderedDict()
        self._lock = threading.Lock()
        self._config_sig = None
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _config_signature():
        sig = []
        paths = [GITLEAKS_CONFIG]
        try:
            with open(GITLEAKS_CONFIG, encoding="utf-8", errors="replace") as f:
                paths.extend(re.findall(r'^\s*path\s*=\s*"([^"]+)"', f.read(), re.M))
        except OSError:
            pass
        for p in paths:
            try:
                st = os.stat(p)
                sig.append((p, st.st_mtime_ns, st.st_size))
            except OSError:
                sig.append((p, None, None))
        return tuple(sig)

    @staticmethod
    def digest(leaf: str) -> bytes:
        return hashlib.sha256(leaf.encode("utf-8", errors="replace")).digest()

    def refresh_config(self):
        sig = self._config_signature()
        with self._lock:
            if sig != self._config_sig:
                if self._config_sig is not None:
                    log("scan cache cleared: gitleaks config chain changed on disk")
                self._clean.clear()
                self._config_sig = sig

    def is_clean(self, digest: bytes) -> bool:
        with self._lock:
            if digest in self._clean:
                self._clean.move_to_end(digest)
                self.hits += 1
                return True
            self.misses += 1
            return False

    def mark_clean(self, digests):
        with self._lock:
            for d in digests:
                self._clean[d] = True
                self._clean.move_to_end(d)
            while len(self._clean) > self._capacity:
                self._clean.popitem(last=False)

    def stats(self):
        with self._lock:
            return {"size": len(self._clean), "capacity": self._capacity,
                    "hits": self.hits, "misses": self.misses}


_scan_cache = LeafScanCache()


def _write_blocked_event(port: int, path: str, findings: list, scan_lines: list,
                         body_len: int) -> str:
    """Persist a self-masked forensic record of a block; return its path.

    v1 logged only the rule id, which made false-positive triage impossible
    (two generic-api-key blocks on 2026-07-20 remain undiagnosable). The raw
    matched secret is never written - excerpts pass through mask_secrets().
    """
    os.makedirs(BLOCKED_EVENTS_DIR, exist_ok=True)
    event = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "proxy_port": port,
        "request_path": path,
        "body_bytes": body_len,
        "findings": [],
    }
    for f in findings:
        start = max(0, int(f.get("StartLine", 1)) - 2)
        end = min(len(scan_lines), int(f.get("EndLine", f.get("StartLine", 1))) + 1)
        excerpt = "\n".join(scan_lines[start:end])[:400]
        event["findings"].append({
            "rule": f.get("RuleID", "unknown"),
            "description": f.get("Description", ""),
            "start_line": f.get("StartLine"),
            "entropy": f.get("Entropy"),
            "masked_excerpt": mask_secrets(excerpt),
        })
    fname = os.path.join(
        BLOCKED_EVENTS_DIR,
        f"{time.strftime('%Y%m%d-%H%M%S')}-port{port}-{os.getpid()}.json",
    )
    try:
        with open(fname, "w", encoding="utf-8") as fh:
            json.dump(event, fh, indent=2)
    except OSError as exc:
        log(f"WARN: could not write blocked-event file: {exc!r}")
        return "(blocked-event write failed)"
    return fname


def scan_for_secrets(body: bytes, port: int = 0, path: str = "") -> tuple:
    """Returns (verdict, reason, scan_ms, cache_ratio).

    verdict: "ok" | "dlp_block" (a real finding - non-retryable 400) |
    "scan_error" (scan infrastructure failed - fail-closed 503, a retry may
    succeed once the infrastructure recovers).

    Incremental: only string leaves never previously scanned clean under the
    current gitleaks config are scanned. Leaves overlapping a finding are
    never cached; clean leaves from a blocked request still are, so a
    client's automatic retry of a near-identical body stays cheap.
    """
    t0 = time.monotonic()
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Fails closed rather than falling back to raw-byte scanning - every
        # real request to this endpoint is JSON; anything else is anomalous
        # enough to treat with suspicion.
        return "dlp_block", "request body is not valid JSON -- failing closed", 0.0, 0.0

    leaves: list = []
    _collect_strings(parsed, leaves)
    _scan_cache.refresh_config()

    new_leaves = []
    new_digests = []
    seen_this_request = set()
    for leaf in leaves:
        d = LeafScanCache.digest(leaf)
        if d in seen_this_request:
            continue
        if _scan_cache.is_clean(d):
            continue
        seen_this_request.add(d)
        new_leaves.append(leaf)
        new_digests.append(d)

    total = len(leaves)
    cache_ratio = 1.0 if not total else 1.0 - (len(new_leaves) / total)
    if not new_leaves:
        return "ok", "", (time.monotonic() - t0) * 1000.0, cache_ratio

    # Flatten only the NEW leaves, tracking each leaf's line range so a
    # finding's line numbers map back to the leaf that must not be cached.
    scan_lines: list = []
    leaf_line_ranges = []  # (first_line_1based, last_line_1based) per new leaf
    for leaf in new_leaves:
        substituted = _BASE64_BLOB_RE.sub("[[large-blob-excluded-from-scan]]", leaf)
        lines = substituted.split("\n")
        first = len(scan_lines) + 1
        scan_lines.extend(lines)
        leaf_line_ranges.append((first, len(scan_lines)))
    scan_text = "\n".join(scan_lines).encode("utf-8", errors="replace")

    report_path = None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(scan_text)
            tmp_path = tmp.name
        report_path = tmp_path + ".report.json"
        result = subprocess.run(
            [
                GITLEAKS_BIN, "detect", "--no-git",
                "--source", tmp_path,
                "--config", GITLEAKS_CONFIG,
                "--no-banner", "--redact", "--exit-code", "1",
                "--report-format", "json", "--report-path", report_path,
            ],
            capture_output=True, text=True, timeout=GITLEAKS_TIMEOUT_SECONDS,
                cwd=CONFIG_DIR,  # resolve [extend].path relative to config dir
        )
    except subprocess.TimeoutExpired:
        return "scan_error", "gitleaks scan timed out -- failing closed", \
            (time.monotonic() - t0) * 1000.0, cache_ratio
    except FileNotFoundError:
        return "scan_error", "gitleaks binary not found on PATH -- failing closed", \
            (time.monotonic() - t0) * 1000.0, cache_ratio
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    scan_ms = (time.monotonic() - t0) * 1000.0

    if result.returncode == 0:
        _scan_cache.mark_clean(new_digests)
        try:
            os.unlink(report_path)
        except OSError:
            pass
        return "ok", "", scan_ms, cache_ratio

    if result.returncode != 1:
        try:
            os.unlink(report_path)
        except OSError:
            pass
        return "scan_error", \
            f"gitleaks scan errored (exit {result.returncode}) -- failing closed", \
            scan_ms, cache_ratio

    # Findings. Parse the JSON report for rule ids + line numbers.
    findings = []
    try:
        with open(report_path, encoding="utf-8") as fh:
            findings = json.load(fh)
    except (OSError, json.JSONDecodeError):
        pass
    finally:
        try:
            os.unlink(report_path)
        except OSError:
            pass

    # Cache the clean leaves even on a block: only leaves overlapping a
    # finding stay uncached and will be re-scanned next time. Gitleaks
    # report line numbers are 1-based (verified empirically 2026-07-22
    # against gitleaks 8.30.1: a secret on file line 3 reports StartLine=3).
    dirty_leaf_idx = set()
    for f in findings:
        f_start = int(f.get("StartLine", 1))
        f_end = int(f.get("EndLine", f.get("StartLine", 1)))
        for i, (first, last) in enumerate(leaf_line_ranges):
            if first <= f_end and last >= f_start:
                dirty_leaf_idx.add(i)
    if not findings:
        dirty_leaf_idx = set(range(len(new_leaves)))  # can't localize: cache nothing
    _scan_cache.mark_clean(
        d for i, d in enumerate(new_digests) if i not in dirty_leaf_idx
    )

    rules = sorted({f.get("RuleID", "unknown") for f in findings}) or ["unknown"]
    event_path = _write_blocked_event(port, path, findings, scan_lines, len(body))
    reason = (
        f"gitleaks flagged outbound request body -- rules {rules} "
        f"-- forensics: {event_path}"
    )
    return "dlp_block", reason, scan_ms, cache_ratio


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # ClassVars set on the class in main() before serve_forever().
    api_key: ClassVar[Optional[str]] = None
    upstream_host: ClassVar[Optional[str]] = None
    upstream_prefix: ClassVar[Optional[str]] = None
    upstream_timeout: ClassVar[int] = 120
    log_path: ClassVar[Optional[str]] = None
    port: ClassVar[int] = 0
    started_at: ClassVar[float] = 0.0

    counters = {
        "requests": 0,
        "forwarded_ok": 0,
        "blocked_dlp": 0,
        "blocked_scan_error": 0,
        "upstream_errors": 0,
        "upstream_retries": 0,
        "salvaged_responses": 0,
        "client_disconnects": 0,
        "upstream_stream_aborts": 0,
    }
    _counters_lock = threading.Lock()

    @classmethod
    def _count(cls, key, n=1):
        with cls._counters_lock:
            cls.counters[key] += n

    def log_message(self, fmt, *args):
        pass  # suppress default stderr access logging; we log decisions ourselves

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def _send_error_response(self, status: int, err_type: str, message: str):
        """Hybrid error body readable by both Anthropic-wire and OpenAI-wire
        clients: Anthropic parses top-level type/error, OpenAI clients read
        .error.message. Status codes are the contract: 400 = do not retry
        (real DLP finding), 502/503 = transient, a client retry is sane.
        """
        payload = json.dumps({
            "type": "error",
            "error": {"type": err_type, "code": err_type, "message": message},
        }).encode()
        self.close_connection = True
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
        except OSError:
            pass  # client already gone; nothing left to tell it

    def _forward(self, method: str, body: bytes) -> int:
        """Forward to upstream and relay the response. Returns upstream status.

        Raises ClientDisconnected / UpstreamStreamAbort / UpstreamError.
        A fresh connection per request is deliberate (no pooling): reusing
        kept-alive upstream sockets is what CAUSES stale-socket resets, and
        a TLS handshake per request is noise next to LLM inference time.
        """
        headers = {}
        for k, v in self.headers.items():
            if k.lower() in _STRIP_REQUEST_HEADERS:
                continue
            headers[k] = v
        # Sent unconditionally regardless of upstream wire format - Bearer
        # covers OpenAI-compatible/native providers (DeepSeek's OpenAI
        # endpoint, xAI, OpenRouter, ...), x-api-key covers Anthropic-wire
        # ones (DeepSeek's Anthropic endpoint). The upstream ignores
        # whichever header its own wire format doesn't use.
        headers["Authorization"] = f"Bearer {self.api_key}"
        headers["x-api-key"] = self.api_key
        headers["Content-Length"] = str(len(body))

        # Reject CRLF injection, null bytes, and path traversal in the
        # client-supplied path before it reaches conn.request().
        if "\r" in self.path or "\n" in self.path or "\0" in self.path:
            raise UpstreamError(
                f"rejected request path with control characters: "
                f"{self.path!r}")
        if "/../" in self.path or self.path.endswith("/..") or \
                self.path.startswith("../") or "\\" in self.path:
            raise UpstreamError(
                f"rejected request path with traversal: {self.path!r}")

        if self.upstream_prefix is None or self.upstream_host is None:
            raise UpstreamError("proxy misconfigured: upstream_host/prefix unset")
        upstream_path = self.upstream_prefix + self.path
        last_exc = None
        for attempt in (1, 2):
            conn = http.client.HTTPSConnection(
                self.upstream_host, 443, timeout=self.upstream_timeout,
                context=ssl.create_default_context(),
            )
            try:
                resp = None
                try:
                    conn.request(method, upstream_path, body=body, headers=headers)
                    resp = conn.getresponse()
                except _TRANSIENT_UPSTREAM_ERRORS as exc:
                    last_exc = exc
                    # Salvage: a send-phase BrokenPipe usually means the
                    # upstream rejected the request and closed early - its
                    # real error response (413/400/...) may already be
                    # readable. Relay THAT instead of inventing a failure.
                    try:
                        resp = conn.getresponse()
                        self._count("salvaged_responses")
                        log(f"salvaged upstream response {resp.status} after "
                            f"send-phase {type(exc).__name__} for {method} {self.path}")
                    except Exception:
                        resp = None
                if resp is None:
                    if attempt == 1:
                        self._count("upstream_retries")
                        log(f"retrying {method} {self.path} after "
                            f"{type(last_exc).__name__}: {last_exc}")
                        time.sleep(0.25)
                        continue
                    raise UpstreamError(
                        f"{type(last_exc).__name__}: {last_exc}") from last_exc
                self._relay(resp)
                return resp.status
            finally:
                conn.close()
        raise UpstreamError(f"{type(last_exc).__name__}: {last_exc}")  # unreachable

    def _relay(self, resp) -> None:
        """Stream an upstream response to the client, distinguishing whose
        socket failed: upstream read errors raise UpstreamStreamAbort, client
        write errors raise ClientDisconnected. Once this method has sent
        anything, the request is past the point of no retry."""
        content_length = resp.getheader("Content-Length")
        try:
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() in _STRIP_RESPONSE_HEADERS:
                    continue
                self.send_header(k, v)
            self.send_header("Connection", "close")
            self.close_connection = True
        except OSError as exc:
            raise ClientDisconnected(str(exc)) from exc

        if content_length is not None:
            # A bounded, non-streamed response - forward its real
            # Content-Length unchanged, nothing to re-frame.
            try:
                data = resp.read()
            except Exception as exc:
                raise UpstreamStreamAbort(f"{type(exc).__name__}: {exc}") from exc
            try:
                self.send_header("Content-Length", content_length)
                self.end_headers()
                self.wfile.write(data)
            except OSError as exc:
                raise ClientDisconnected(str(exc)) from exc
        else:
            # No Content-Length means the upstream response is streamed
            # (SSE/chunked, e.g. a streaming chat completion). Python's
            # http.client already transparently de-chunks it before we
            # can see the raw wire framing, so stripping
            # Transfer-Encoding with nothing to replace it left the
            # response with NO valid HTTP/1.1 framing at all - verified
            # live (symposion-I24) that this made undici/fetch-based
            # clients (OpenCode's, stricter than Claude Code's own
            # client) fail with "socket connection was closed
            # unexpectedly" instead of tolerating a close-delimited
            # body. Re-chunking (not buffer-then-send) keeps this a true
            # pass-through - the client still sees tokens as they
            # arrive, not the whole reply at once.
            try:
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
            except OSError as exc:
                raise ClientDisconnected(str(exc)) from exc
            while True:
                try:
                    chunk = resp.read(8192)
                except Exception as exc:
                    raise UpstreamStreamAbort(
                        f"{type(exc).__name__}: {exc}") from exc
                if not chunk:
                    break
                try:
                    self.wfile.write(f"{len(chunk):x}\r\n".encode())
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except OSError as exc:
                    raise ClientDisconnected(str(exc)) from exc
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except OSError as exc:
                raise ClientDisconnected(str(exc)) from exc

    def _send_health(self):
        payload = json.dumps({
            "status": "ok",
            "version": __version__,
            "pid": os.getpid(),
            "port": self.port,
            "upstream_host": self.upstream_host,
            "uptime_seconds": round(time.monotonic() - self.started_at, 1),
            "counters": dict(self.counters),
            "scan_cache": _scan_cache.stats(),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _handle(self, method: str):
        if self.path == "/__proxy_health__":
            self._send_health()
            return

        self._count("requests")
        t0 = time.monotonic()
        body = self._read_body()

        scan_ms = 0.0
        cache_ratio = 1.0
        if body:
            verdict, reason, scan_ms, cache_ratio = scan_for_secrets(
                body, port=self.port, path=self.path)
            if verdict == "dlp_block":
                self._count("blocked_dlp")
                log(f"BLOCKED {method} {self.path} -- {reason}")
                self._send_error_response(
                    400, "invalid_request_error",
                    f"BLOCKED by LLM egress proxy (secret-content DLP): {reason}")
                return
            if verdict == "scan_error":
                self._count("blocked_scan_error")
                log(f"SCAN-ERROR {method} {self.path} -- {reason}")
                self._send_error_response(
                    503, "api_error",
                    f"LLM egress proxy could not scan the request "
                    f"(fail-closed): {reason}")
                return

        log(f"> {method} {self.path} ({len(body)} bytes) "
            f"scan={scan_ms:.0f}ms cache={cache_ratio:.0%}")
        try:
            status = self._forward(method, body)
            self._count("forwarded_ok")
            log(f"< {method} {self.path} {status} "
                f"total={(time.monotonic() - t0):.1f}s")
        except ClientDisconnected as exc:
            self._count("client_disconnects")
            log(f"client disconnected mid-response for {method} {self.path} "
                f"(benign - request cancelled client-side): {exc}")
        except UpstreamStreamAbort as exc:
            # Headers already relayed; nothing valid we can still send.
            self._count("upstream_stream_aborts")
            log(f"upstream aborted mid-stream for {method} {self.path}: {exc}")
            self.close_connection = True
        except UpstreamError as exc:
            self._count("upstream_errors")
            log(f"upstream error for {method} {self.path} "
                f"(after salvage + 1 retry): {exc}")
            self._send_error_response(
                502, "api_error",
                f"LLM egress proxy could not reach upstream "
                f"{self.upstream_host}: {exc}")

    def do_POST(self):
        self._handle("POST")

    def do_GET(self):
        self._handle("GET")

    def do_PUT(self):
        self._handle("PUT")

    def do_DELETE(self):
        self._handle("DELETE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--upstream-host", required=True,
        help='Upstream provider host to forward to, e.g. "api.deepseek.com" or "api.x.ai".',
    )
    parser.add_argument(
        "--api-key-env", required=True,
        help='Name of the environment variable holding the real upstream API key, '
             'e.g. "DEEPSEEK_API_KEY" or "XAI_API_KEY". Read once at startup.',
    )
    parser.add_argument(
        "--upstream-prefix", default="",
        help='Path prefix prepended to the incoming request path before forwarding upstream. '
             'Empty by default (client baseURL already includes any version segment, e.g. '
             '"/v1"); DeepSeek\'s Anthropic-wire endpoint needs "/anthropic" explicitly.',
    )
    parser.add_argument(
        "--upstream-timeout", type=int, default=120,
        help="Per-socket-operation timeout (seconds) for upstream connections.",
    )
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        die(f"{args.api_key_env} not set in environment -- refusing to start")
    if not os.path.isfile(GITLEAKS_CONFIG):
        die(f"gitleaks config missing at {GITLEAKS_CONFIG} -- refusing to start")
    if shutil.which("gitleaks") is None:
        die("gitleaks binary not found on PATH -- refusing to start")

    ProxyHandler.api_key = api_key
    ProxyHandler.upstream_host = args.upstream_host
    ProxyHandler.upstream_prefix = args.upstream_prefix
    ProxyHandler.upstream_timeout = args.upstream_timeout
    ProxyHandler.port = args.port
    ProxyHandler.started_at = time.monotonic()
    # Per-instance log file (one per port) so concurrently-running proxies
    # for different providers don't interleave into one ambiguous file.
    ProxyHandler.log_path = os.path.join(DATA_DIR, f"llm-egress-proxy-{args.port}.log")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), ProxyHandler)
    server.daemon_threads = True  # a hung stream must never block shutdown
    log(f"LLM egress proxy v{__version__} listening on 127.0.0.1:{args.port} "
        f"(upstream {args.upstream_host!r}, prefix {args.upstream_prefix!r}, "
        f"key env {args.api_key_env!r})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
