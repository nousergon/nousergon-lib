"""LLM egress proxy — content-scanning outbound gateway (config-I3482).

A gitleaks-scanning HTTP proxy that sits between an LLM client and ONE upstream
provider.  Every outbound request body is scanned for secrets before forwarding;
responses stream back untouched and unbuffered.  Fail-closed throughout.

Consolidated canonical source of truth (v2.1, 2026-07-25) — previously vendored
at ``alpha-engine-config/infrastructure/groom-llm-routing/`` and Brian's laptop
``claude-code-config/llm-routing/``.

Usage::

    python3 -m nousergon_lib.egress.proxy --port 8972 \\
        --upstream-host api.deepseek.com --api-key-env DEEPSEEK_API_KEY

Or via entry point::

    llm-egress-proxy --port 8972 --upstream-host api.deepseek.com \\
        --api-key-env DEEPSEEK_API_KEY
"""
# No public API — the proxy is a standalone CLI entry point.
# The module exists for importlib.resources (gitleaks configs) and
# for the pyproject.toml [project.scripts] entry point.
