"""Does an install or deploy path start a timer-driven workload? (config#9099)

THE MECHANISM, measured, not assumed
------------------------------------
``Requires=`` in a **timer's** ``[Unit]`` section is a START dependency *of the
timer*, not a declaration of what the timer triggers. ``systemctl enable --now
<x>.timer`` in an installer therefore enqueues a start job for ``<x>.service``
as well — and does so even when the timer is already active and no calendar
point has elapsed, because the transaction still carries the dependency jobs.
``Wants=`` does the same weakly.

On 2026-08-28 03:00:42 UTC that fired three timer-driven services off-schedule
on the shared dashboard box within four seconds of a deploy, one of them a
weekly OSS bakeoff that spends real LLM tokens. All three failed and box-health
paged (``alpha-engine-config-I9000``). ``Persistent=`` catch-up was the standing
hypothesis and it is WRONG: a ``Persistent=`` replay writes the timer's stamp
under ``/var/lib/systemd/timers/``, and the stamps recorded no elapse.

WHY THIS IS A LIBRARY
---------------------
Three repos install units onto that box and each grew its own copy of this
parser on 2026-08-28 (``crucible-dashboard-PR792``, ``nousergon-data-PR1566``,
``nous-ergon-ops-PR919``). Two of the copies had already fixed a defect the
first still carried — see ``_NOT_A_UNIT_NAME`` below — which is the fork that
``shared-code-policy`` §3 exists to close. This module is the union of all
three — plus one more defect of the same class that none of them had, found by
this module's own tests during the lift: a trailing ``# comment`` after a
``systemctl`` invocation had its WORDS harvested as unit names. All of these
are verdict-neutral (the tokens can never match a real unit) and all of them
put garbage in the output of a guard whose only asset is being believed.

The parser lives here once, and each repo's test supplies its own paths, its
own presence assertions and its own incident narrative.

WHAT STAYS IN THE REPO, DELIBERATELY
------------------------------------
* The ``X-InstallMayStart=yes`` escape hatch's **value**: it is a real
  assignment in the service's own ``[Unit]``, with the reason written beside
  it, never an allowlist in this module or in a test. :func:`may_start` reads
  it; nothing here decides it.
* The failure message. Each caller owns its assertion so the text names that
  repo's units and that repo's fix.
* The live box-side reader. ``crucible-dashboard``'s
  ``infrastructure/box_health.sh::install_start_dependency_scan`` walks the
  MERGED systemd graph via ``systemctl cat`` (``systemctl show`` does not
  surface ``X-`` keys). It is the backstop for what source-text analysis
  structurally cannot see, and it is not replaced by this.

SCOPE, stated because it bounds the guarantee
---------------------------------------------
Pure text analysis — no subprocess, no filesystem beyond the paths the caller
hands in. Two gaps every caller must keep naming in its own docstring:

* a ``systemctl <verb> "$unit"`` whose unit name is a shell variable is
  skipped (tokens containing ``$`` are dropped), so a loop that arms timers
  from a rendered list is invisible;
* dependency edges declared by units the calling repo does not own are
  invisible, so a chain that leaves and re-enters that repo's unit set is not
  seen here.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "DEP_KEYS",
    "INSTALL_MAY_START_KEY",
    "directive",
    "load_units",
    "may_start",
    "scheduled_workloads",
    "sections",
    "start_closure",
    "started_units",
    "triggered_service",
    "violations",
]

#: ``[Unit]`` keys whose values are pulled in when the unit is started. These
#: are the edges that turn "arm a timer" into "run its service now".
DEP_KEYS = ("Requires", "Requisite", "BindsTo", "Wants")

#: The unit-side escape hatch. Its VALUE lives in the unit file, never here.
INSTALL_MAY_START_KEY = "X-InstallMayStart"

# systemctl verbs that cause a unit to be ACTIVATED. `enable` alone and
# `daemon-reload` do not, which is why they are absent. `reenable` IS matched
# and then dropped (see _INERT_VERBS) so that its argument cannot be mistaken
# for the argument of a neighbouring verb.
_START_RE = re.compile(
    r"systemctl\s+(?P<flags>(?:--\S+\s+)*)"
    r"(?P<verb>start|restart|enable|reenable|reload-or-restart|try-restart)\s+"
    r"(?P<rest>[^\n;|&)]*)"
)

# `reenable` rewrites symlinks and does NOT activate.
_INERT_VERBS = frozenset({"reenable"})

# Shell noise that is not a unit name. `>/dev/null` and the `2>` left behind
# when `_START_RE` stops at the `&` of `2>&1` both survive a naive split, and
# `crucible-dashboard-PR792`'s copy of this parser reported them as units named
# `>/dev/null.service` and `2>.service`. They can never match a real unit, so
# they changed no verdict — but a guard whose output contains obvious garbage
# is a guard that gets argued with the day it fires for real, and this is
# precisely the class of guard that gets deleted rather than trusted.
_NOT_A_UNIT_NAME = frozenset("<>&(){}[]!\\`")

# A `systemctl` inside an `echo`/`printf`/`log`/`say`/`cat` is operator guidance
# an installer PRINTS ("Run now: sudo systemctl start x.service"), not something
# it does. Reading those as executions is how a guard cries wolf until it is
# deleted — the same class as the 2026-08-27 scan that matched `ne-admin` inside
# a YAML comment.
_ECHOED_RE = re.compile(r"\b(echo|printf|log|say|cat)\b")


def sections(text: str) -> dict:
    """Split unit-file text into ``{section: [lines]}``, dropping comments.

    Comment lines (``#`` or ``;``) are dropped whole. Note this is the unit-file
    grammar, where a comment is only ever a full line — unlike the shell, where
    :func:`started_units` must be more careful.
    """
    out: dict = {}
    current = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            out.setdefault(current, [])
            continue
        out.setdefault(current, []).append(line)
    return out


def directive(lines, key: str) -> list:
    """Every whitespace-separated value assigned to ``key`` in ``lines``.

    systemd allows a directive to repeat and to carry several values per line;
    both are accumulated, matching how systemd merges list-valued keys.
    """
    vals: list = []
    for line in lines:
        k, _, v = line.partition("=")
        if k.strip() == key:
            vals.extend(v.split())
    return vals


def load_units(systemd_dir: Path) -> dict:
    """Every ``*.service`` / ``*.timer`` in ``systemd_dir``, drop-ins merged.

    ``<name>.service.d/*.conf`` fragments are appended to the base unit's
    sections, which is how systemd itself composes them. A drop-in for a unit
    whose base file this repo does not carry still yields an entry, so its
    edges are visible.
    """
    units: dict = {}
    if not systemd_dir.is_dir():
        return units
    for path in sorted(systemd_dir.glob("*")):
        if path.is_dir() or path.suffix not in (".service", ".timer"):
            continue
        units[path.name] = sections(path.read_text(encoding="utf-8"))
    for dropin_dir in sorted(systemd_dir.glob("*.service.d")):
        target = units.setdefault(dropin_dir.name[: -len(".d")], {})
        for conf in sorted(dropin_dir.glob("*.conf")):
            for section, lines in sections(conf.read_text(encoding="utf-8")).items():
                target.setdefault(section, []).extend(lines)
    return units


def triggered_service(timer: str, timer_sections) -> str:
    """The service a timer activates: explicit ``Unit=``, else same basename."""
    explicit = directive(timer_sections.get("Timer", []), "Unit")
    if explicit:
        return explicit[-1]
    return timer[: -len(".timer")] + ".service"


def scheduled_workloads(units) -> set:
    """``Type=oneshot`` services that a timer in ``units`` exists to trigger.

    DERIVED, never a literal list — that property is the whole point. A new
    timer-driven job joins the protected class by existing, not by being added
    to an allowlist somebody has to remember.
    """
    out: set = set()
    for name, unit_sections in units.items():
        if not name.endswith(".timer"):
            continue
        target = triggered_service(name, unit_sections)
        target_sections = units.get(target)
        if target_sections is None:
            continue
        if "oneshot" in directive(target_sections.get("Service", []), "Type"):
            out.add(target)
    return out


def _normalise(token: str):
    """A bare systemctl argument as a unit name, or ``None`` if it is not one."""
    token = token.strip().strip("\"'")
    if not token or token.startswith("-") or "$" in token or "*" in token:
        return None
    if _NOT_A_UNIT_NAME & set(token):
        return None
    if "." not in token.rsplit("/", 1)[-1]:
        return token + ".service"
    return token


def started_units(script: Path) -> set:
    """Units ``script`` activates, by source text.

    Anti-false-positive rules, each of which exists because a real script in the
    fleet tripped it:

    * **full-line** comments are stripped — and only full-line, because
      truncating at any ``#`` turns a false positive into a false negative
      (``systemctl start x  # why`` would stop being seen);
    * a ``systemctl`` preceded on its line by ``echo``/``printf``/``log``/
      ``say``/``cat`` is printed guidance, not an execution;
    * tokens holding ``$``, ``*`` or shell metacharacters are not unit names,
      and a token starting a trailing comment ends the argument list;
    * ``enable`` counts only with ``--now``; ``reenable`` never counts.
    """
    started: set = set()
    for line in script.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        for match in _START_RE.finditer(line):
            if _ECHOED_RE.search(line[: match.start()]):
                continue
            verb = match.group("verb")
            if verb in _INERT_VERBS:
                continue
            flags = match.group("flags") or ""
            rest = match.group("rest")
            if verb == "enable" and "--now" not in flags and "--now" not in rest:
                continue
            for token in rest.split():
                # A trailing comment's WORDS are not unit names. All three
                # pre-lift copies harvested `seed`, `the` and `cache` from
                # `systemctl start x.service  # seed the cache`; found by this
                # module's own test, not by any of them. Stopping at the `#`
                # TOKEN keeps the full-line-only rule above intact — the
                # execution is still seen, only the prose after it is dropped.
                if token.startswith("#"):
                    break
                unit = _normalise(token)
                if unit:
                    started.add(unit)
    return started


def start_closure(unit: str, units) -> set:
    """``unit`` plus everything starting it pulls in, per ``units``.

    Transitive over :data:`DEP_KEYS` in each unit's ``[Unit]`` section. Names of
    units absent from ``units`` are included (they were named by a real edge)
    but not expanded — the calling repo cannot see edges it does not own.
    """
    seen: set = set()
    stack = [unit]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        unit_sections = units.get(current)
        if unit_sections is None:
            continue
        for key in DEP_KEYS:
            for dep in directive(unit_sections.get("Unit", []), key):
                if dep.endswith((".service", ".timer")) and dep not in seen:
                    stack.append(dep)
    return seen


def may_start(unit: str, units) -> bool:
    """Does ``unit`` declare ``X-InstallMayStart=yes`` in its own ``[Unit]``?

    The reason lives beside the declaration in the unit file. ``X-`` keys are
    ignored by systemd, so this costs nothing on the box — and are invisible to
    ``systemctl show``, which is why the live box-side reader uses
    ``systemctl cat``.
    """
    unit_sections = units.get(unit, {})
    return "yes" in directive(unit_sections.get("Unit", []), INSTALL_MAY_START_KEY)


def violations(script: Path, units) -> dict:
    """``{started unit: scheduled workloads it pulls in}`` for ``script``.

    Empty when the script is clean. The caller owns the assertion and its
    message, so each repo's failure text names that repo's units and that
    repo's fix.
    """
    workloads = scheduled_workloads(units)
    offenders: dict = {}
    for unit in started_units(script):
        pulled = {u for u in start_closure(unit, units) & workloads if not may_start(u, units)}
        if pulled:
            offenders[unit] = pulled
    return offenders
