"""Surface collector-style error dicts to Flow Doctor.

Many Alpha Engine orchestrators (alpha-engine-data weekly_collector, the
predictor training pipeline stages, research per-team collectors) catch
exceptions and convert them into a return-dict of the form::

    {"status": "error", "error": "<message>"}

The orchestrator aggregates per-collector dicts into a final results
structure and continues running the remaining collectors. Without an
explicit ``logger.error()`` call, the underlying error never reaches
Flow Doctor's logging-handler-based capture path — the alert pipeline
only sees the orchestrator's generic "non-ok status" summary line, which
dedups all partial runs together and contains none of the real error
text needed for LLM diagnosis or actionable GitHub issues.

:func:`report_collector_errors` walks the collectors dict and emits one
``logger.error()`` per error-status entry. Each emitted record carries
the collector name + original error message, producing distinct dedup
signatures and rich diagnose context.

Typical usage in an orchestrator's finalize step::

    from nousergon_lib.collector_results import report_collector_errors

    # ... run collectors, populate results["collectors"] ...
    report_collector_errors(results["collectors"])
    # write manifest, return results, etc.

Idempotent — safe to call multiple times in the same process.
Flow Doctor's per-yaml dedup window suppresses repeat alerts.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any


def report_collector_errors(
    collectors: Mapping[str, Mapping[str, Any]],
    logger: logging.Logger | None = None,
) -> int:
    """Log one ERROR per collector with ``status == "error"``.

    :param collectors: Mapping of collector name → result dict. Each
        result dict is expected to have a ``"status"`` key; entries
        with status ``"error"`` also typically carry an ``"error"``
        key with the message string.
    :param logger: Logger to emit through. Defaults to
        ``logging.getLogger(__name__)`` (which routes to the root
        logger's handlers — including FlowDoctorHandler when
        ``setup_logging(flow_doctor_yaml=...)`` has been called).
    :return: Number of error entries logged.

    Non-mapping values, missing ``status`` keys, and any non-error
    status are ignored silently. The function never raises.
    """
    log = logger or logging.getLogger(__name__)
    count = 0
    for name, info in collectors.items():
        if not isinstance(info, Mapping):
            continue
        if info.get("status") != "error":
            continue
        err = info.get("error", "<no error message>")
        log.error("collector %s failed: %s", name, err)
        count += 1
    return count
