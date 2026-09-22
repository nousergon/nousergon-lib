"""Price the DIFF: refuse a change that adds spend with no budget line.

Most cost controls are detective-after-spend — a daily monitor grades
yesterday, a budget action fires on a breach that already happened, a
capability gate grades entitlements as they currently stand. Nothing looks at
a *change*. So "a cost above the budget needs human approval" is enforced at
runtime and merely *remembered* at authoring time: a pull request that adds a
call site on a per-request-billed API, or grants an entitlement on a service
with no line in the SSoT, lands with nothing objecting, and the first thing
that notices is a bill.

WHAT COUNTS AS APPROVED

The approved set is read from the SSoT document and from nowhere else::

    approved = every `iam_prefixes` entry on a budgeted service line
             + `capability_gate.free_prefixes`

This adds **no second registry to maintain**: widening the gate means editing
the SSoT in a PR, which is the approval, which is the whole design.

By default the SSoT is the prefix-only document packaged with this library —
derived by allowlist from a private source, carrying service names and billing
prefixes and no amounts at all (see ``data/budget_prefixes.yaml``). A caller
may substitute its own with ``--ssot`` / the ``doc`` argument.

WHY IT READS ONLY ADDED LINES

A gate that grades the whole file reds every PR that touches a file which
already contained a call site, which makes it noise, which gets it removed. It
grades what the diff ADDS. A line that was already on the base branch was
already approved, or is already a finding for a different control with a
different remedy.

THE TWO CLASSES THAT ARE COUNTED *AND* GRADED

  * **new always-on resources.** Every ``Type: AWS::Service::Thing`` the diff
    adds is resolved to a billing prefix through the resource map, and that
    prefix must name a budgeted or declared-free service. A type absent from
    the map is a FINDING, not a pass: "unmapped" and "free" are the two
    answers this gate may never confuse, and a resource nobody has priced is
    exactly the shape that produces a surprise bill.

  * **new schedules.** Two independent questions, both asked:

      1. *Does the target bill to a budgeted service?* The target is resolved
         structurally — ``!GetAtt Dispatcher.Arn`` and ``!Ref EodStateMachine``
         are the shapes fleet templates use, so a line-level regex cannot
         answer it. When it cannot be resolved the gate says so and counts it;
         it never reports an unresolved target as a pass.

      2. *Is the retry bounded?* A schedule target with no ``RetryPolicy``
         inherits the AWS default of **185 attempts**. A per-call price cap
         that says nothing about call count is not a control. Absent, or above
         ``schedule_rules.max_retry_attempts``, is a finding.

    GitHub Actions ``schedule:`` crons are graded on PRIVATE repos only, on
    FREQUENCY. Public-repo Actions minutes are free and unlimited, so grading
    them would be noise with no spend behind it; private minutes bill against
    an allowance. And ``timeout-minutes`` bounds what ONE run costs while
    saying nothing about how many runs there are — for a cron the interval IS
    that bound.

CloudFormation is read STRUCTURALLY for the schedule class, from the head
tree, and only for the schedule resources the diff actually touched.

ZERO metered API calls. This reads two YAML documents and a diff. The
instrument is not exempt from the budget it polices.

Refs alpha-engine-config-I11228.
"""

from __future__ import annotations

import ast
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from nousergon_lib.cost_gate import resource_map as resource_map_mod

#: Where the packaged, derived SSoT lives.
BUDGET_PREFIXES_PATH = resource_map_mod.DATA_DIR / "budget_prefixes.yaml"

#: A cloud SDK client/resource construction, in any of the spellings the fleet
#: uses. The captured group is the service name, which is also the billing
#: prefix for every service in the SSoT.
_CLIENT = re.compile(r"""boto3\s*\.\s*(?:client|resource)\s*\(\s*["']([a-z0-9-]+)["']""")

#: An entitlement string in a policy document or a policy literal. Deliberately
#: anchored on the quote so that prose mentioning ``ce:GetCostAndUsage`` in a
#: comment is not a finding — a comment is not a grant.
#:
#: LINE-LEVEL FALLBACK ONLY. For a JSON/YAML file this is used only when the
#: head file could not be parsed structurally (see ``parse_action_values``);
#: otherwise a Condition key shaped like aws colon SourceAccount, or a
#: component id shaped like pipeline colon unit, reads as a grant it never
#: was. For every other file (Python call sites, shell, etc.) it is still the
#: only mechanism, since there is no structure to parse. DELIBERATELY
#: UNQUOTED in this sentence — a quoted example here would be exactly the
#: false positive this comment describes, on this file's own next diff.
#:
#: LINE-LEVEL DISAMBIGUATION RULE (no document structure to consult, so this is
#: necessarily narrower than the structural pass): a candidate match is graded
#: as an entitlement UNLESS either holds —
#:
#:   1. its prefix is not a real AWS IAM service prefix shape at all — today
#:      that is exactly the ``aws`` global-condition-key namespace, tracked in
#:      ``_NOT_A_SERVICE`` alongside the GitHub-permissions prefixes it already
#:      held; or
#:   2. it is the VALUE of a quoted dict/JSON key, on the SAME line, and that
#:      key is not Action-shaped (``Action``/``Actions``/``NotAction``) — the
#:      boto3 EC2 filter-dict shape ``{"Name": "tag:Name", "Values": [...]}``
#:      and a component-id field (``"id": "pipeline:unit"``) both read this
#:      way: a real grant is written as a bare list element or as the value of
#:      an Action-shaped key, never as the value of ``Name``/``Key``/``id``/
#:      any other field name. See ``_is_keyed_by_non_action``.
_IAM_ACTION = re.compile(r"""["']([a-z0-9-]+):[A-Z*][A-Za-z0-9*]*["']""")

#: The same shape, unquoted — matched against an already-isolated Action
#: VALUE pulled out of a parsed document, where quoting has already been
#: stripped by the YAML/JSON loader.
_ACTION_VALUE = re.compile(r"^([a-z0-9-]+):[A-Z*][A-Za-z0-9*]*$")

#: A line that plausibly touches a schedule. CHEAP PRE-FILTER ONLY — a hit
#: makes the gate read and parse the head file, where the real grading happens.
#: A miss must therefore be conclusive, which is why it is deliberately wide.
_SCHEDULE = re.compile(
    r"(cron\(|rate\(|ScheduleExpression|RetryPolicy|MaximumRetryAttempts"
    r"|AWS::Scheduler::Schedule\b|AWS::Events::Rule\b|^\s*-?\s*cron:\s*['\"])",
    re.MULTILINE,
)

#: A GitHub Actions cron entry: ``- cron: '30 13 * * *'``.
_GHA_CRON = re.compile(r"^\s*-\s*cron:\s*['\"]?([-0-9*/, ?A-Za-z]+?)['\"]?\s*(?:#.*)?$")

#: A new always-on resource, as a FULL CloudFormation type name.
#:
#: The negative lookbehind is load-bearing: ``"ResourceType": "AWS::..."`` in a
#: ``resources-to-import`` manifest names a resource that ALREADY EXISTS and is
#: being adopted into a stack. That is not new spend, and grading it would red
#: the PR that brings an orphaned resource under IaC — the opposite of the
#: incentive this gate should create.
_RESOURCE_TYPE = re.compile(
    r"""(?<![A-Za-z])["']?Type["']?\s*:\s*["']?(AWS::[A-Za-z0-9]+::[A-Za-z0-9]+)"""
)

#: The two CloudFormation types that can carry a schedule.
_SCHEDULE_TYPES = ("AWS::Scheduler::Schedule", "AWS::Events::Rule")

#: Prefixes that name no cloud service at all and appear in these patterns only
#: as false positives. Kept SHORT and justified: every entry here is a hole.
_NOT_A_SERVICE = frozenset({
    # `boto3.client("...")` never takes these; they appear in the entitlement
    # pattern from non-AWS grant strings (GitHub workflow `permissions:`
    # blocks) that this gate does not govern.
    "actions", "contents", "id-token", "pull-requests", "issues", "checks",
    "packages", "statuses", "deployments", "security-events", "metadata",
    "models", "pages", "discussions", "attestations", "repository-projects",
    # `aws` is IAM's GLOBAL CONDITION KEY namespace (`aws:SourceAccount`,
    # `aws:sourceVpc`, `aws:CalledVia`, ...), never a service prefix — there is
    # no `boto3.client("aws")` and no service literally named `aws`. In a
    # structurally-parsed policy document this is already excluded because a
    # Condition key is not an Action value; this entry is what gives the
    # LINE-LEVEL fallback (source files, and IaC files that failed to parse)
    # the same answer with no document structure to consult.
    "aws",
})

#: A quoted JSON/dict KEY immediately preceding a colon, on the same line as a
#: candidate `word:Word` match — used to tell whether that value sits under an
#: Action-shaped key or under something else entirely. LINE-LEVEL ONLY: there
#: is no document structure to walk here, so only same-line context counts.
_KEYED_BY = re.compile(r"""["'](?P<key>[A-Za-z_][A-Za-z0-9_-]*)["']\s*:\s*$""")

#: Key names that plausibly introduce an IAM action/entitlement value. A
#: `word:Word` token keyed by anything ELSE on the same line is someone else's
#: value, not a grant, no matter how it is shaped.
_ACTION_KEY_NAMES = frozenset({"action", "actions", "notaction"})

#: A TEST. Its call sites and grant strings are both fixtures. This gate's own
#: test file constructs a `textract` client and names `textract:*` actions to
#: prove the gate fires. **A gate that reds the test proving it works cannot be
#: merged, let alone required.**
#:
#: Suppresses BOTH classes, because a test has a legitimate reason to write
#: either shape.
_TEST = re.compile(r"(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$)", re.IGNORECASE)

#: A CAPTURED-STATE artifact — a ``roles_backup_*.json`` records entitlements as
#: they ALREADY were, so a retirement is reversible. Grading it would red the
#: PR that makes a destructive change reversible.
#:
#: Suppresses the ENTITLEMENT class ONLY. A capture file has no reason to
#: construct a client, so that half stays graded.
_CAPTURE = re.compile(r"(^|/)(backups?/|[^/]*_?backups?[._])", re.IGNORECASE)

#: PROSE. A ``.md`` provisions nothing, and a changelog entry quoting
#: ``Type: AWS::CloudFront::Distribution`` is a record of what happened, not a
#: request for it. Suppresses the RESOURCE and SCHEDULE classes only — the
#: client and entitlement classes keep their own, narrower rules above.
_PROSE = re.compile(r"\.(md|rst|txt)$", re.IGNORECASE)

#: A GitHub Actions workflow file, where a ``schedule:`` cron buys runner
#: minutes rather than a cloud resource.
_WORKFLOW = re.compile(r"(^|/)\.github/workflows/[^/]+\.ya?ml$")

#: A file that could hold a CloudFormation template.
_IAC_FILE = re.compile(r"\.(ya?ml|json|template)$", re.IGNORECASE)

#: Creating a schedule through the SDK rather than a template.
#:
#: CHEAP PRE-FILTER ONLY, exactly like ``_SCHEDULE``. A hit makes the gate
#: parse the head file and ask whether any of these tokens is really a CALL
#: (see :func:`parse_sdk_schedule_calls`); a miss is conclusive. It is NOT a
#: verdict on its own, because the same four tokens occur as ordinary text in
#: any module that *searches for* them — including this one, whose own
#: pattern list and "not graded" message both name all four. Grading the
#: token rather than the syntax made this module's own source report a
#: permanently unresolvable schedule (alpha-engine-config-I11289): the
#: scanner matched its own pattern list. Same class as the `Condition`-key
#: and component-id false positives above — a token graded outside the
#: context that gives it meaning.
_SDK_SCHEDULE = re.compile(
    r"(create_schedule|update_schedule|put_rule|put_targets|ScheduleExpression\s*=)"
)

#: The boto3 methods that create or retarget a schedule.
_SDK_SCHEDULE_METHODS = frozenset({
    "create_schedule", "update_schedule", "put_rule", "put_targets",
})

#: The keyword argument that carries a cadence into any of them.
_SDK_SCHEDULE_KWARG = "ScheduleExpression"


class SdkScheduleParseError(RuntimeError):
    """``text`` is not parseable Python, so an SDK schedule call site in it
    cannot be told apart from the same token in a string literal, a comment
    or a pattern list by position. The caller falls back to the line-level
    pre-filter and reports the file as NOT GRADED — never as "no schedule
    here". The twin of :class:`resource_map.ActionParseError`, and for the
    same reason."""


def parse_sdk_schedule_calls(text: str) -> list:
    """Every ``(first_line, last_line)`` span in ``text`` occupied by a REAL
    SDK schedule call site, resolved from Python SYNTAX rather than from text.

    Three shapes count, and nothing else does:

      * a call whose callee is named ``create_schedule`` / ``update_schedule``
        / ``put_rule`` / ``put_targets`` — ``client.put_rule(...)`` or a bare
        ``put_rule(...)``;
      * any call passing ``ScheduleExpression=`` as a keyword argument;
      * an assignment to a name or attribute called ``ScheduleExpression``,
        which is the remaining shape the ``ScheduleExpression\\s*=`` branch of
        the pre-filter was written for.

    A string literal, a comment, a docstring or a regex alternation
    *containing* any of those tokens is none of those three, so it is not a
    call site. That is the whole fix for the self-match: it needs no
    per-file exclusion, so it also covers the next module that names these
    strings — a test fixture, a doc example, a second grader.

    Raises :class:`SdkScheduleParseError` when ``text`` does not parse.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError) as exc:  # ValueError: NUL bytes
        raise SdkScheduleParseError(f"could not parse as Python: {exc}") from exc

    spans: list = []

    def span(node: ast.AST) -> tuple:
        first = getattr(node, "lineno", 0)
        return (first, getattr(node, "end_lineno", None) or first)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in _SDK_SCHEDULE_METHODS or any(
                kw.arg == _SDK_SCHEDULE_KWARG for kw in node.keywords
            ):
                spans.append(span(node))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                tname = (
                    target.attr if isinstance(target, ast.Attribute)
                    else getattr(target, "id", None)
                )
                if tname == _SDK_SCHEDULE_KWARG:
                    spans.append(span(node))
                    break
    return spans

#: The one sentence every finding ends with. It names the remedy rather than
#: the suppression, because the alternative is that the next reader disables
#: the check instead of adding the line.
REMEDY = (
    "The remedy is a budget line, not a suppression. Add the service to the "
    "cost SSoT in the repo that owns it — that line IS the approval — and "
    "regenerate the published prefix map. If the service genuinely cannot "
    "bill, add it to `capability_gate.free_prefixes` WITH A REASON."
)

_NO_LINE = (
    "names no budgeted service in the cost SSoT and is not declared free"
)


class SsotError(ValueError):
    """The SSoT document is unusable. Raised, never degraded to an empty set:
    an empty approved set would approve nothing and red every caller, which
    reads as the gate being broken rather than as the document being wrong."""


def load_ssot(path: Path | str | None = None) -> dict:
    """The budget-prefix document. Defaults to the one packaged here.

    Raises :class:`SsotError` rather than falling back to the packaged copy
    when an explicit path is unreadable: a caller that asked to grade against
    its own document and was silently graded against another one is worse than
    a caller that failed.
    """
    p = Path(path) if path is not None else BUDGET_PREFIXES_PATH
    if not p.is_file():
        raise SsotError(f"{p} does not exist")
    doc = yaml.safe_load(p.read_text())
    if not isinstance(doc, dict):
        raise SsotError(f"{p} is not a mapping")
    budgeted, free = approved_prefixes(doc)
    if not budgeted or not free:
        raise SsotError(
            f"{p}: derived {len(budgeted)} budgeted prefix(es) and {len(free)} "
            "free prefix(es); an empty set would approve nothing and red every "
            "caller"
        )
    return doc


def approved_prefixes(doc: dict) -> tuple:
    """``(prefix -> the budgeted service that covers it, free prefixes)``.

    Both come from the SSoT. A prefix in neither map is unbudgeted capability.
    """
    budgeted: dict = {}
    for provider in (doc.get("providers") or {}).values():
        for svc, row in (provider.get("services") or {}).items():
            for prefix in (row.get("iam_prefixes") or []):
                budgeted[prefix] = svc
    free = set((doc.get("capability_gate") or {}).get("free_prefixes") or {})
    return budgeted, free


#: ``@@ -12,0 +13,4 @@`` — the new-file start line and count. ``--unified=0``
#: omits the count when it is 1, so the group is optional.
_HUNK = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@")


def added_with_lines(diff: str) -> list:
    """``[(path, 1-based new-file line number, added line)]`` — additions only.

    ``+++ b/path`` is itself a line starting with ``+``; dropping it by prefix
    would also drop a genuine addition that happens to start with ``++``, so
    the header is matched explicitly instead.

    Line numbers are what let the schedule class ask "did this diff touch THIS
    resource?" instead of "does this file contain a schedule anywhere?". They
    are tracked across context lines too, so the caller may pass a diff taken
    at any ``--unified`` width.
    """
    out: list = []
    path = "?"
    lineno = 0
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            path = raw[4:].removeprefix("b/").strip()
            continue
        hunk = _HUNK.match(raw)
        if hunk:
            lineno = int(hunk.group(1))
            continue
        if raw.startswith(("--- ", "diff ", "index ", "@@", "-", "\\")):
            continue
        if raw.startswith("+"):
            out.append((path, lineno, raw[1:]))
            lineno += 1
        else:  # a context line: it occupies a line in the new file too
            lineno += 1
    return out


def added_lines(diff: str) -> list:
    """:func:`added_with_lines` without the line numbers."""
    return [(path, text) for path, _, text in added_with_lines(diff)]


def cron_interval_minutes(expr: str) -> int | None:
    """The SHORTEST gap between two firings of a 5-field Actions cron.

    ``None`` for an expression this cannot read, which the caller reports as
    ungraded rather than as a pass.

    The estimate is deliberately the *shortest* gap rather than the average:
    the question is how many runs a month may bill, and ``0,1,2,30 * * * *``
    bills like a one-minute cadence for three minutes of every hour.
    """
    fields = expr.split()
    if len(fields) != 5:
        return None

    def gap(field: str) -> int | None:
        """Smallest step within one period of this field, or None if it names a
        single fixed value (in which case the next-coarser field decides)."""
        if field in ("*", "?"):
            return 1
        if field.startswith("*/"):
            step = field[2:]
            return int(step) if step.isdigit() and int(step) > 0 else None
        if "-" in field or "/" in field:
            return 1  # a range fires on consecutive units
        values = sorted({int(v) for v in field.split(",") if v.isdigit()})
        if len(values) != len(field.split(",")):
            return None  # a name (`MON`) or something unparsed — do not guess
        if len(values) >= 2:
            return min(b - a for a, b in zip(values, values[1:]))
        return None

    minutes = gap(fields[0])
    if minutes is not None:
        return minutes
    hours = gap(fields[1])
    if hours is not None:
        return hours * 60
    return 1440  # a fixed minute AND a fixed hour: at most once a day


def _note_ungraded(counts: dict, key: str, detail: str) -> None:
    """Increment ``counts[key]`` (``ungraded_schedules``/``ungraded_targets``)
    and record WHAT could not be read, so the "NOT GRADED" line in the CLI
    output can name it instead of only counting it. A could-not-grade that
    names no file is indistinguishable from one that never happened —
    alpha-engine-config-I11289 was exactly a caller unable to identify which
    of ~150 commits, let alone which file within it, tripped this counter.
    """
    counts[key] += 1
    counts["ungraded"].append(detail)


def _note_fallback(counts: dict, path: str, reason: str) -> None:
    """Increment ``actions_context_fallback`` and record WHICH file fell back
    and why. The count alone told a caller that some file in the diff was
    graded by the weaker line-level scan without saying which, so the one
    action it asks for — fix that file's syntax — could not be taken
    (alpha-engine-config-I11289). Same shape as :func:`_note_ungraded`."""
    counts["actions_context_fallback"] += 1
    counts["fallback"].append(f"{path}: {reason}")


def _sdk_schedule_is_real(
    path: str,
    touched: set,
    root_path: Path | None,
    read_file: Callable[[str], str | None] | None,
) -> bool:
    """Whether the SDK-schedule pre-filter hit on ``path`` names a real call
    site the diff touched.

    FAIL-CLOSED. ``True`` (report it ungraded) whenever the question cannot
    be answered structurally: a non-Python source file, a head file that
    cannot be read, or one that will not parse. ``False`` only when the head
    file parsed and holds no schedule call site on any line the diff added —
    the token was text, not syntax.
    """
    if not path.endswith(".py"):
        return True
    text = _read_head(path, root_path, read_file)
    if text is None:
        return True
    try:
        spans = parse_sdk_schedule_calls(text)
    except SdkScheduleParseError:
        return True
    return any(first <= n <= last for first, last in spans for n in touched)


def _grade_gha_crons(
    path: str,
    lines: list,
    min_interval: int,
    found: dict,
    counts: dict,
) -> None:
    for line in lines:
        m = _GHA_CRON.match(line)
        if not m:
            continue
        counts["schedules"] += 1
        expr = m.group(1).strip()
        interval = cron_interval_minutes(expr)
        if interval is None:
            _note_ungraded(
                counts, "ungraded_schedules",
                f"{path}: workflow schedule `{expr}` — cron expression could not be parsed",
            )
            continue
        if interval < min_interval:
            found[("cron", expr, path)] = (
                f"{path}: adds a workflow schedule `{expr}` firing as often as every "
                f"{interval} minute(s), under the declared floor of {min_interval}. "
                f"Actions minutes on a PRIVATE repo bill against the included "
                f"allowance; the cron interval is the only thing bounding how many "
                f"runs there will be (`timeout-minutes` bounds one run, not the "
                f"count). Widen the cadence, or raise "
                f"`schedule_rules.min_gha_cron_interval_minutes` in the resource "
                f"map with a reason."
            )


def _is_keyed_by_non_action(line: str, match_start: int) -> bool:
    """True when the quoted ``word:Word`` token starting at ``match_start`` on
    ``line`` is immediately preceded by ``"<key>":`` for a key that is not
    Action-shaped — see the ``_IAM_ACTION`` docstring for the rule and its
    rationale. Only same-line context is available at the line level, so the
    check is deliberately narrow: no preceding key at all (a bare list
    element) is NOT suppressed — that is the shape a real grant takes when
    each action sits on its own line.
    """
    m = _KEYED_BY.search(line[:match_start].rstrip())
    return m is not None and m.group("key").lower() not in _ACTION_KEY_NAMES


def _grade_actions_by_line(
    path: str, lines: list, known: set, found: dict, counts: dict
) -> None:
    """The pre-context-aware behaviour: every quoted ``prefix:Word`` token on
    an added line is graded, with no regard for whether it sits in an
    ``Action`` value, a ``Condition`` key, or a ``Principal`` — except for the
    same-line keyed-value disambiguation documented on ``_IAM_ACTION``. Used
    ONLY as the fallback when a file that should be gradeable structurally
    could not be parsed, or is not a structured file at all (Python, shell,
    ...) — never silently, always counted (see ``actions_context_fallback``
    in :func:`findings`)."""
    for line in lines:
        for m in _IAM_ACTION.finditer(line):
            counts["actions"] += 1
            if _is_keyed_by_non_action(line, m.start()):
                continue
            prefix = m.group(1)
            if prefix in known or prefix in _NOT_A_SERVICE:
                continue
            found[("action", prefix, path)] = (
                f"{path}: grants `{prefix}:*` actions, and `{prefix}` {_NO_LINE}."
            )


def _grade_actions_structurally(
    path: str, touched: set, values: list, known: set, found: dict, counts: dict
) -> None:
    """Only an ``Action``/``NotAction`` value the diff actually added is
    graded — resolved from the head file's real structure, not from any
    quoted string that happens to look like one."""
    for av in values:
        if not av.touches(touched):
            continue
        counts["actions"] += 1
        m = _ACTION_VALUE.match(av.value)
        if not m:
            continue
        prefix = m.group(1)
        if prefix in known or prefix in _NOT_A_SERVICE:
            continue
        found[("action", prefix, path)] = (
            f"{path}: grants `{prefix}:*` actions, and `{prefix}` {_NO_LINE}."
        )


def _grade_cfn_schedules(
    path: str,
    touched: set,
    text: str,
    known: set,
    resource_types: dict,
    max_retries: int,
    found: dict,
    counts: dict,
) -> None:
    resources = resource_map_mod.parse_cfn_resources(text)
    if not resources:
        # A document with no `Resources:` mapping declares no cloud resource,
        # so this is the TRUE answer, not an unexamined one — and counting it
        # as ungraded would bury the cases that really are.
        return
    by_lid = {r.logical_id: r.type for r in resources}

    for res in resources:
        if res.type not in _SCHEDULE_TYPES or not res.touches(touched):
            continue
        props = res.properties
        # An `AWS::Events::Rule` with an event pattern and no schedule fires on
        # an event, not a clock. It is a resource (graded as one above), not a
        # schedule, and has no cadence to bound.
        if not props.get("ScheduleExpression"):
            continue

        targets = props.get("Targets")
        if not isinstance(targets, list):
            single = props.get("Target")
            targets = [single] if isinstance(single, dict) else []
        if not targets:
            _note_ungraded(
                counts, "ungraded_schedules",
                f"{path}: schedule `{res.logical_id}` declares no `Target(s)` to resolve",
            )
            continue

        for target in targets:
            counts["schedules"] += 1
            if not isinstance(target, dict):
                _note_ungraded(
                    counts, "ungraded_schedules",
                    f"{path}: schedule `{res.logical_id}` has a `Target` entry that is "
                    f"not a mapping and could not be read",
                )
                continue

            retry = target.get("RetryPolicy")
            attempts = retry.get("MaximumRetryAttempts") if isinstance(retry, dict) else None
            if not isinstance(attempts, int) or isinstance(attempts, bool):
                found[("retry", res.logical_id, path)] = (
                    f"{path}: schedule `{res.logical_id}` declares no "
                    f"`RetryPolicy.MaximumRetryAttempts`, so it inherits the AWS "
                    f"default of 185 attempts. On a per-request-billed target that "
                    f"default is a spend multiplier nobody approved. Declare a "
                    f"bound at or below {max_retries}."
                )
            elif attempts > max_retries:
                found[("retry", res.logical_id, path)] = (
                    f"{path}: schedule `{res.logical_id}` retries up to {attempts} "
                    f"times, above the declared maximum of {max_retries}. A retry "
                    f"count is a spend multiplier on the target's per-request price."
                )

            prefix = resource_map_mod.resolve_target_service(
                target.get("Arn"), by_lid, resource_types
            )
            if prefix is None:
                _note_ungraded(
                    counts, "ungraded_targets",
                    f"{path}: schedule `{res.logical_id}`'s target `Arn` could not be "
                    f"resolved to a billing service",
                )
                continue
            if prefix not in known:
                found[("target", prefix, path)] = (
                    f"{path}: schedule `{res.logical_id}` targets a `{prefix}` "
                    f"resource, and `{prefix}` {_NO_LINE}."
                )


def _read_head(
    path: str,
    root_path: Path | None,
    read_file: Callable[[str], str | None] | None,
) -> str | None:
    """The head-tree text for ``path``, or ``None`` if it cannot be read —
    shared by the schedule class and the action class, both of which grade a
    file's STRUCTURE rather than its added lines alone."""
    if read_file is not None:
        return read_file(path)
    if root_path is not None:
        candidate = root_path / path
        if candidate.is_file():
            return candidate.read_text(errors="replace")
    return None


def findings(
    diff: str,
    doc: dict,
    *,
    resource_map: dict | None = None,
    root: Path | str | None = None,
    read_file: Callable[[str], str | None] | None = None,
    repo_visibility: str = "private",
) -> tuple:
    """Unbudgeted capability the diff ADDS, plus the counts it only observed.

    ``root`` is the head checkout. The schedule class reads files from it, and
    says so when it cannot: a file absent from ``root`` is reported as an
    UNGRADED schedule, never as a schedule that passed.

    ``read_file`` replaces that filesystem read with an arbitrary
    ``path -> text or None`` reader — used by a history replay that hands over
    ``git show <sha>:<path>`` without checking out every commit.
    """
    budgeted, free = approved_prefixes(doc)
    known = set(budgeted) | free
    rmap = resource_map if resource_map is not None else resource_map_mod.load()
    resource_types = rmap["resource_types"]
    rules = rmap.get("schedule_rules") or {}
    max_retries = rules.get(
        "max_retry_attempts", resource_map_mod.DEFAULT_MAX_RETRY_ATTEMPTS
    )
    min_interval = rules.get(
        "min_gha_cron_interval_minutes",
        resource_map_mod.DEFAULT_MIN_GHA_CRON_INTERVAL_MINUTES,
    )
    root_path = Path(root) if root is not None else None

    found: dict = {}
    counts = {
        "lines": 0, "schedules": 0, "resources": 0, "clients": 0, "actions": 0,
        "ungraded_schedules": 0, "ungraded_targets": 0, "actions_context_fallback": 0,
        # Named detail for every ungraded_schedules/ungraded_targets increment
        # above — see _note_ungraded. A count with no name attached is exactly
        # the diagnostics gap alpha-engine-config-I11289 hit: "1 schedule not
        # graded" over 150 commits, with no way to say which one.
        "ungraded": [],
        # Named detail for every actions_context_fallback increment — see
        # _note_fallback. The remedy the fallback line names ("fix the file's
        # syntax") is unactionable without the file's name.
        "fallback": [],
    }
    # Per-file, because a schedule is a BLOCK and the resources it names live
    # elsewhere in the same template.
    by_file: dict = {}

    for path, lineno, line in added_with_lines(diff):
        counts["lines"] += 1
        touched, texts = by_file.setdefault(path, (set(), []))
        touched.add(lineno)
        texts.append(line)

        # A CloudFormation `Type:` declares a resource only inside a TEMPLATE.
        # The same string in a `.py` comment or docstring is prose — this very
        # module's docstring names `Type: AWS::Service::Thing`. A script that
        # really creates a resource does it through an SDK client, and the
        # CLIENT class grades that.
        if _IAC_FILE.search(path) and not (_TEST.search(path) or _PROSE.search(path)):
            for type_name in _RESOURCE_TYPE.findall(line):
                counts["resources"] += 1
                prefix = resource_types.get(type_name)
                if prefix is None:
                    found[("resource", type_name, path)] = (
                        f"{path}: adds a `{type_name}` resource, and that type is "
                        f"not in the published resource map. UNMAPPED IS NOT FREE "
                        f"— map the type to the billing prefix of the service it "
                        f"bills to, and give that service a budget line if it has "
                        f"none."
                    )
                elif prefix not in known:
                    found[("resource", type_name, path)] = (
                        f"{path}: adds a `{type_name}` resource, which bills to "
                        f"`{prefix}`, and `{prefix}` {_NO_LINE}."
                    )

        if _TEST.search(path):
            continue
        for prefix in _CLIENT.findall(line):
            counts["clients"] += 1
            if prefix in known or prefix in _NOT_A_SERVICE:
                continue
            found[("client", prefix, path)] = (
                f"{path}: constructs a `{prefix}` client, and `{prefix}` {_NO_LINE}."
            )

        if _CAPTURE.search(path):
            continue
        if _IAC_FILE.search(path):
            # A JSON/YAML file is graded STRUCTURALLY, per file, below — an
            # `Action` value needs the file's structure to tell it apart from
            # a `Condition` key or a component id shaped `word:word`, which a
            # line-level regex cannot do. See the action-class pass.
            continue
        for m in _IAM_ACTION.finditer(line):
            counts["actions"] += 1
            if _is_keyed_by_non_action(line, m.start()):
                continue
            prefix = m.group(1)
            if prefix in known or prefix in _NOT_A_SERVICE:
                continue
            found[("action", prefix, path)] = (
                f"{path}: grants `{prefix}:*` actions, and `{prefix}` {_NO_LINE}."
            )

    # -- the action class, per JSON/YAML file ----------------------------------
    #
    # Graded structurally: an `Action`/`NotAction` value the diff added is a
    # finding on an unbudgeted prefix; nothing else in the document is, no
    # matter what it looks like quoted. A file this cannot parse falls back to
    # the line-level scan and says so — never silently, and never as a pass.
    for path, (touched, texts) in by_file.items():
        if _TEST.search(path) or _CAPTURE.search(path) or not _IAC_FILE.search(path):
            continue
        text = _read_head(path, root_path, read_file)
        if text is None:
            _note_fallback(
                counts, path,
                "head file could not be read (absent from `--root`, or no "
                "`read_file` supplied)",
            )
            _grade_actions_by_line(path, texts, known, found, counts)
            continue
        try:
            values = resource_map_mod.parse_action_values(text)
        except resource_map_mod.ActionParseError as exc:
            _note_fallback(counts, path, str(exc))
            _grade_actions_by_line(path, texts, known, found, counts)
            continue
        _grade_actions_structurally(path, touched, values, known, found, counts)

    # -- the schedule class, per file -----------------------------------------
    for path, (touched, texts) in by_file.items():
        if _TEST.search(path) or _PROSE.search(path):
            continue
        if not any(_SCHEDULE.search(line) for line in texts):
            continue

        if _WORKFLOW.search(path):
            # Public-repo Actions minutes are free and unlimited. Grading them
            # would be noise with no spend behind it, and noise is what gets a
            # gate muted.
            if repo_visibility == "public":
                counts["schedules"] += sum(1 for line in texts if _GHA_CRON.match(line))
                continue
            _grade_gha_crons(path, texts, min_interval, found, counts)
            continue

        if not _IAC_FILE.search(path):
            # A `.py` that CREATES a schedule does it through an SDK call. Its
            # client is graded above; its cadence and retry bound are not
            # readable from here, so the gate says so rather than passing it.
            #
            # A `.py` that merely NAMES `MaximumRetryAttempts` — this module's
            # own source does, in a docstring — is prose. Counting that as an
            # ungraded schedule would put a NOT GRADED line on most Python PRs,
            # and a signal that fires on everything is read as firing on
            # nothing.
            #
            # The pre-filter hit is confirmed against the head file's SYNTAX
            # before it is reported: the same four tokens are ordinary text
            # in any module that searches for them, this one included.
            if any(_SDK_SCHEDULE.search(line) for line in texts) and _sdk_schedule_is_real(
                path, touched, root_path, read_file
            ):
                _note_ungraded(
                    counts, "ungraded_schedules",
                    f"{path}: names an SDK schedule call site "
                    f"(`create_schedule`/`put_rule`/`put_targets`/`ScheduleExpression=`) "
                    f"whose cadence and retry bound cannot be read from Python source",
                )
            continue

        text = _read_head(path, root_path, read_file)
        if text is None:
            # NOT a pass. The head file is how a schedule becomes gradeable at
            # all, and a schedule this gate did not read must not print the
            # same green zero as one it read and approved.
            _note_ungraded(
                counts, "ungraded_schedules",
                f"{path}: head file could not be read (absent from `--root`, or no "
                f"`read_file` supplied) — its schedule(s) could not be graded",
            )
            continue

        _grade_cfn_schedules(
            path, touched, text, known, resource_types, max_retries, found, counts
        )

    return [found[k] for k in sorted(found)], counts


class MergeBaseUnreachable(RuntimeError):
    """``base`` and ``head`` share no reachable ancestor in this checkout.

    MEASURED: ``git diff origin/main...HEAD`` exits **128** the moment the base
    branch moves ahead of a shallow PR checkout, because the merge base is
    simply not in the object store. A ``CalledProcessError`` traceback out of a
    REQUIRED check is a hard block on every PR that is behind its base — which,
    on a repo with a merge queue, is most of them.

    Raised rather than degraded to a two-dot diff: two-dot would report
    everything that landed on the base since the branch was cut, so the gate
    would red on someone else's merged change, which is how a check gets muted.
    An unreachable merge base is a thing to FIX (deepen the fetch), never a
    thing to grade around.
    """


class RootNotAGitWorkTree(RuntimeError):
    """``--root`` does not name a path inside a git work tree.

    Every git invocation in this module is anchored on ``--root`` via
    ``cwd=`` — NEVER on the process's own working directory (see
    ``resolve_root``). That means a ``--root`` this gate cannot resolve to a
    work tree cannot be graded at all: a named, exit-2 cause here, rather
    than a bare subprocess traceback, or — the actual defect this class
    exists to close — a silent grade of whatever repo the process happened to
    be launched from.
    """


def resolve_root(root: Path | str) -> Path:
    """Validate ``root`` against git ITSELF, anchored on ``root``, and return
    its resolved absolute path — the value every subsequent git call in this
    module passes as ``cwd``.

    Raises :class:`RootNotAGitWorkTree` when ``root`` is not a directory, or
    is a directory git does not recognise as (inside) a work tree when asked
    with ``cwd=root``. This is the fix for the CWD/``--root`` disagreement:
    the process's OWN working directory is never consulted here, so it is
    irrelevant which repo the caller happened to launch the CLI from.
    """
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise RootNotAGitWorkTree(f"--root {root} does not exist or is not a directory")
    probe = _git(["rev-parse", "--is-inside-work-tree"], cwd=root_path, check=False)
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        cause = probe.stderr.strip() or probe.stdout.strip() or f"exit {probe.returncode}"
        raise RootNotAGitWorkTree(
            f"--root {root} is not inside a git work tree ({cause}). Every git "
            "ref this gate resolves (--base/--head) is resolved AGAINST "
            "--root, never against the process's own working directory — "
            "pass the checkout that actually holds the history to grade."
        )
    return root_path


def _git(
    args: list, *, cwd: Path | str | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    """Run git. ``cwd`` — always ``--root``, resolved by :func:`resolve_root` —
    is the ONLY thing that decides which repo's history this reads.
    ``subprocess.run(cwd=...)`` is used rather than a bare process launch plus
    ``git -C``, because it also fixes the historical failure mode: a caller
    invoking this CLI from a shell whose CWD is a DIFFERENT repo's checkout
    used to grade THAT repo's history instead of the one named by ``--root``,
    silently and without error (alpha-engine-config-I11289)."""
    return subprocess.run(
        ["git", *args],  # noqa: S607 — git from PATH
        cwd=cwd,
        capture_output=True, text=True, check=check,
    )


def is_shallow(root: Path | str | None = None) -> bool:
    return (
        _git(["rev-parse", "--is-shallow-repository"], cwd=root, check=False)
        .stdout.strip() == "true"
    )


def merge_base(base: str, head: str, root: Path | str | None = None) -> str | None:
    """The merge base, deepening a shallow clone once if that is what is missing.

    The repair is here and not only in a workflow because this runs from repos
    whose checkout step it does not own. Detecting a shallow clone and
    reporting it would leave the caller to fix a condition this can fix itself.

    ``root`` is passed to every git call as ``cwd`` — see ``resolve_root``.
    """
    probe = _git(["merge-base", base, head], cwd=root, check=False)
    if probe.returncode == 0 and probe.stdout.strip():
        return probe.stdout.strip()
    if is_shallow(root):
        deepen = _git(
            ["fetch", "--quiet", "--unshallow", "--no-tags", "origin"], cwd=root, check=False
        )
        if deepen.returncode != 0:
            return None
        probe = _git(["merge-base", base, head], cwd=root, check=False)
        if probe.returncode == 0 and probe.stdout.strip():
            return probe.stdout.strip()
    return None


def git_diff(base: str, head: str, root: Path | str | None = None) -> str:
    """What HEAD adds relative to its merge base with ``base``, resolved
    entirely against ``root`` (see ``resolve_root``) — never against the
    process's own working directory.

    Equivalent to ``git diff base...head``, but the merge base is resolved
    explicitly so an unreachable one produces a named error instead of a bare
    exit-128 traceback.
    """
    mb = merge_base(base, head, root)
    if mb is None:
        raise MergeBaseUnreachable(
            f"no merge base between {base!r} and {head!r} in this checkout"
            + (" (the clone is SHALLOW and could not be deepened)" if is_shallow(root) else "")
            + ". The pre-merge cost gate grades what a branch ADDS, which is "
            "undefined without a merge base. Remedy: check out with "
            "`fetch-depth: 0`, or `git fetch --unshallow --no-tags origin` "
            "before running this. Do NOT substitute a two-dot diff — it would "
            "grade every change that landed on the base since the branch was cut."
        )
    return _git(["diff", "--unified=0", f"{mb}..{head}"], cwd=root).stdout


def load_yaml_doc(path: Path | str, what: str) -> dict:
    """Read an alternate document. Raises rather than falling back to a packaged
    copy: a caller silently graded against a document it did not name would be
    both wrong and impossible to debug from the output."""
    doc: Any = yaml.safe_load(Path(path).read_text())
    if not isinstance(doc, dict):
        raise SsotError(f"--{what} {path} is not a mapping")
    return doc
