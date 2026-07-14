"""Shared L1/L2/L3 gate-failure alert helper (market-value-integrity
alerting backbone, config#2459 scope item 4).

The framework's L1 (cross-source agreement, nousergon-data), L2 (per-series
data-contract validation, config#2456, nousergon-lib + nousergon-data), and
L3 (T+1 reconcile / NAV three-way hard gate, config#2457, crucible-executor)
gates each live in a different repo/process. Rather than each layer
reinventing "how do I get this failure to flow-doctor", this module offers
one shared, uniform-shape helper the three layers CAN call into instead of
each hand-rolling their own ``logger.error(...)`` message format. As of
this PR it is a proposed shared home, not yet an enforced one: config#2456
and config#2457 are independently in-flight in parallel and may land their
own ad hoc gate-failure log calls before adopting this helper (see the
"Adoption" note below) — a caller auditing what actually reaches
flow-doctor today should still grep each layer's own gate-check code, not
assume every L1/L2/L3 failure already routes through here.

Deliberately NOT a new alerting integration — it is a thin wrapper around
the existing flow-doctor singleton (``krepis.logging`` /
``nousergon_lib.logging``, re-exported from ``krepis``) that every
alpha-engine entrypoint already initializes via
``setup_logging(name, flow_doctor_yaml=...)``. Once that singleton is
attached (level=ERROR ``FlowDoctorHandler`` on the root logger), a
``logger.error(...)``-or-louder call is captured and dispatched through
flow-doctor's configured notifiers (email / GitHub issue per that
process's ``flow-doctor.yaml``; some repos additionally route through
Telegram via a repo-local helper such as ``nousergon-data``'s
``infrastructure/lambdas/flow_doctor_telegram.notify_via_flow_doctor``
for forum-topic-scoped dispatch — this module does not attempt to
replicate that direct-call path, only the ``logger.error()``-under-the-
singleton convention every entrypoint already has) — see
``nousergon-data/weekly_collector.py``, ``rag/preflight.py``,
``lambda/handler.py``, ``builders/daily_append.py`` for the existing
call-site convention this module matches. NOTE: because
``FlowDoctorHandler`` is attached at level=ERROR, only
``severity="error"``/``"critical"`` actually reach flow-doctor's
dispatch pipeline today — ``"info"``/``"warning"`` are accepted (for
callers that want a uniform call shape across severities) but land as
plain sub-threshold log lines, not flow-doctor notifications. See
``severity`` below.

Usage (from any of L1/L2/L3's gate-check code, in a process that has
already called ``setup_logging(..., flow_doctor_yaml=...)``)::

    from nousergon_lib.gate_alerts import alert_gate_failure

    alert_gate_failure(
        layer="L2",
        series="AAPL",
        detail="continuity gap: 2026-07-11 missing, expected trading day",
        severity="error",
    )

Adoption: config#2456 (L2) and config#2457 (L3) are independently in-flight
as this lands, in different repos, without a cross-PR dependency forcing
convergence on this helper's signature. This module ships ahead of a
confirmed L2/L3 call site as a proposed shared home; the PR that adds it
does NOT retrofit L1/L2/L3's existing gate-check code to call it (see PR
body). A follow-up should either (a) migrate L1/L2/L3 to call
``alert_gate_failure`` directly, reconciling any shape mismatch against
whatever those layers already shipped, or (b) if the layers' needs
diverge enough that a single shared signature doesn't fit, retire this
module rather than let it linger unused.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Valid severities, mirroring flow-doctor's own ``notify_on`` vocabulary
# (see nousergon_lib.flow_doctor_fleet._FLEET_TELEGRAM_TOPIC_SPECS).
_VALID_SEVERITIES = frozenset({"info", "warning", "error", "critical"})


def _log_level_for(severity: str) -> int:
    """Map ``severity`` to a stdlib ``logging`` level.

    ``"critical"`` intentionally maps to ``logging.ERROR``, NOT
    ``logging.CRITICAL`` — ``FlowDoctorHandler`` is attached at
    level=ERROR (see ``krepis.logging._attach_flow_doctor``), so ERROR is
    the threshold that actually reaches flow-doctor's dispatch pipeline;
    ``logging.CRITICAL`` would still cross that threshold too, but using
    it here would incorrectly imply flow-doctor treats CRITICAL as a
    distinct, higher-priority log level than ERROR — it doesn't. The
    ``severity`` string itself (not the stdlib log level) is what carries
    "critical" through to the ``extra`` dict for any downstream
    notifier/context enrichment that wants to distinguish them.
    """
    return logging.ERROR if severity in ("error", "critical") else getattr(
        logging, severity.upper()
    )


def alert_gate_failure(
    layer: str,
    series: str,
    detail: str,
    *,
    severity: str = "error",
) -> None:
    """Route a market-value-integrity gate failure through flow-doctor.

    ``layer`` — which gate raised this (e.g. ``"L1"``, ``"L2"``, ``"L3"``).
    ``series`` — the symbol/series the failure concerns (e.g. ``"AAPL"``,
        or a batch identifier like ``"universe:2026-07-14"`` for a
        whole-run failure).
    ``detail`` — human-readable description of what failed (gate-specific;
        e.g. "cross-source disagreement: polygon=101.2 yfinance=98.7").
    ``severity`` — one of ``"info"``, ``"warning"``, ``"error"``,
        ``"critical"`` (default ``"error"``, matching a gate FAILURE, not
        a soft warning). Invalid values raise ``ValueError`` — a typo'd
        severity silently routing to the wrong Telegram topic (or
        dropping the alert) is worse than a loud failure here.

    Dispatch mechanism: emits a single ``logger.error()`` (or the
    level matching ``severity``) call carrying a uniform, greppable
    message shape (``"[gate-failure] layer=... series=... detail=..."``)
    plus a structured ``extra`` dict (``layer``, ``series``, ``severity``)
    that a JSON-mode formatter or a flow-doctor notifier's
    ``notify_on_category``/context enrichment can key off. This module
    does NOT itself attach a flow-doctor handler or open a Telegram
    connection — it relies entirely on the calling process having already
    run ``setup_logging(name, flow_doctor_yaml=...)`` (the standing
    convention; see module docstring). In a process with no flow-doctor
    handler attached (local dev, a script that never called
    ``setup_logging``), this degrades to a plain log line — the same
    graceful-inactive behavior every other flow-doctor call site has.

    Never raises for a dispatch-layer problem — a failure to ALERT about
    a gate failure must not itself crash the gate-checking pipeline (the
    same fail-loud-but-never-crash posture ``nousergon-data``'s
    ``notify_via_flow_doctor`` uses for its own dispatch fallback, though
    that helper is a separate, repo-local direct-call path — see module
    docstring). ``severity`` validation is the one exception (a
    programming error, not a runtime alerting failure) and DOES raise.
    """
    if severity not in _VALID_SEVERITIES:
        raise ValueError(
            f"alert_gate_failure: severity={severity!r} not in "
            f"{sorted(_VALID_SEVERITIES)}"
        )

    level = _log_level_for(severity)
    message = f"[gate-failure] layer={layer} series={series} detail={detail}"
    try:
        log.log(
            level,
            message,
            extra={"layer": layer, "series": series, "severity": severity},
        )
    except Exception:  # noqa: BLE001 — alerting must never crash the gate.
        # Fall back to a bare print so the failure isn't fully silent even
        # if the logging stack itself is broken. print() is a deliberately
        # weak last resort (no guaranteed capture in every deployment) —
        # it only fires if the logging call itself raised, which the
        # standard library's own logging module is designed not to do
        # except in pathological misconfiguration.
        print(f"alert_gate_failure: logging dispatch failed for: {message}")


__all__ = ["alert_gate_failure"]
