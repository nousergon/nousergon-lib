"""Can this PUBLIC repo carry the cost map without carrying a single amount?

This is the tiering control for the packaged documents under
``src/nousergon_lib/cost_gate/data/``. They are derived from a PRIVATE cost
single-source-of-truth that holds monthly ceilings, what each line was sized
against, reduction markers, owners, and prose about live infrastructure. None
of that may reach an AGPL repo on a public remote.

Two independent assertions, because each catches what the other cannot:

* **over the serialised bytes** — what leaks is what is written to a file, not
  what a dict happens to hold. Regexes for a digit-dollar amount, a 12-digit
  account id and an ARN, plus a blocklist of the private document's field
  names.
* **over the key shape** — a blocklist only catches the fields someone thought
  of. Every key in both documents must be one of a CLOSED set, so a field
  added to the private source tomorrow cannot appear here at all: it is absent
  until someone names it in the producing script's allowlist, in a reviewed PR
  on the private repo.

Service names such as "Amazon EC2 Container Registry (ECR)" are the cloud
vendor's own public product names and carry digits, which is why the byte
check hunts dollar-shaped amounts and identifiers rather than digits.

Refs alpha-engine-config-I11228.
"""

from __future__ import annotations

import re

import yaml

from nousergon_lib.cost_gate import grade
from nousergon_lib.cost_gate import resource_map as crm

BUDGETS_PATH = grade.BUDGET_PREFIXES_PATH
RMAP_PATH = crm.RESOURCE_MAP_PATH
PACKAGED = [BUDGETS_PATH, RMAP_PATH]

#: A currency amount in any of the spellings the private document uses:
#: `$100`, `100 USD`, `monthly_budget_usd: 12.5`.
_DOLLAR = re.compile(r"\$\s*\d|\d\s*(?:usd|USD)\b")

#: An AWS account id.
_ACCOUNT_ID = re.compile(r"(?<!\d)\d{12}(?!\d)")

#: Any ARN — it names a live resource, a region and an account at once.
_ARN = re.compile(r"arn:[a-z0-9-]*:", re.IGNORECASE)

#: An EC2 instance id.
_INSTANCE_ID = re.compile(r"\bi-[0-9a-f]{8,17}\b")

#: Field names from the private document. Every one of these is a tier-3 fact.
_BANNED_SUBSTRINGS = (
    "monthly_budget_usd", "limit_usd", "sized_against", "flagged",
    "revisit", "owner:", "notification_ladder", "subscriber",
)

#: The CLOSED key set of the derived budget document.
_BUDGET_KEYS = {"schema_version", "_doc", "providers", "capability_gate"}

#: The CLOSED key set of the published resource map. `schedule_rules` carries
#: two integers — a retry ceiling and a cron-interval floor. They are grading
#: bounds, not spend: the same two numbers are this package's own
#: ``DEFAULT_MAX_RETRY_ATTEMPTS`` / ``DEFAULT_MIN_GHA_CRON_INTERVAL_MINUTES``.
#: They ship so a bound tightened in the private source reaches every caller
#: instead of silently diverging from the library defaults.
_RMAP_KEYS = {
    "schema_version", "resource_types", "schedule_rules", "updated", "updated_by",
}
_SCHEDULE_RULE_KEYS = {"max_retry_attempts", "min_gha_cron_interval_minutes"}


def test_every_packaged_document_exists_and_says_it_is_generated():
    for path in PACKAGED:
        assert path.is_file(), path
        head = path.read_text()[:1200]
        assert "GENERATED" in head, path
        assert "DO NOT EDIT" in head.upper(), path
        assert "PRIVATE" in head.upper(), path


def test_no_amount_or_identifier_survives_into_the_packaged_bytes():
    """The load-bearing assertion. Over the BYTES, comments included — a
    rationale comment is as public as a data line once the file is pushed."""
    for path in PACKAGED:
        text = path.read_text()
        assert not _DOLLAR.search(text), f"{path}: currency amount"
        assert not _ACCOUNT_ID.search(text), f"{path}: 12-digit account id"
        assert not _ARN.search(text), f"{path}: ARN"
        assert not _INSTANCE_ID.search(text), f"{path}: instance id"
        lowered = text.lower()
        for banned in _BANNED_SUBSTRINGS:
            assert banned.lower() not in lowered, f"{path}: {banned}"


def test_the_budget_document_keys_are_a_closed_set():
    doc = yaml.safe_load(BUDGETS_PATH.read_text())
    assert set(doc) == _BUDGET_KEYS
    for provider in doc["providers"].values():
        assert set(provider) == {"services"}
        for name, row in provider["services"].items():
            assert set(row) == {"iam_prefixes"}, name
            assert all(isinstance(p, str) for p in row["iam_prefixes"]), name
    assert set(doc["capability_gate"]) == {"free_prefixes"}
    for prefix, reason in doc["capability_gate"]["free_prefixes"].items():
        assert re.fullmatch(r"[a-z0-9-]+", prefix), prefix
        # The private document carries a written REASON per entry, and a
        # reason is prose someone wrote about our infrastructure. The gate
        # reads the KEYS; the values are replaced with one constant.
        assert isinstance(reason, str)
    assert len(set(doc["capability_gate"]["free_prefixes"].values())) == 1


def test_the_resource_map_keys_are_a_closed_set():
    doc = yaml.safe_load(RMAP_PATH.read_text())
    assert set(doc) <= _RMAP_KEYS, set(doc) - _RMAP_KEYS
    for type_name, prefix in doc["resource_types"].items():
        assert re.fullmatch(r"AWS::[A-Za-z0-9]+::[A-Za-z0-9]+", type_name), type_name
        assert re.fullmatch(r"[a-z0-9-]+", prefix), (type_name, prefix)
    assert set(doc["schedule_rules"]) <= _SCHEDULE_RULE_KEYS
    for value in doc["schedule_rules"].values():
        assert isinstance(value, int) and not isinstance(value, bool)


def test_the_shipped_bounds_match_this_packages_defaults():
    """If they ever diverge, a caller grading with the packaged map and a
    caller grading with the library defaults are running two different gates."""
    rules = crm.load()["schedule_rules"]
    assert rules["max_retry_attempts"] == crm.DEFAULT_MAX_RETRY_ATTEMPTS
    assert rules["min_gha_cron_interval_minutes"] == (
        crm.DEFAULT_MIN_GHA_CRON_INTERVAL_MINUTES
    )


def test_the_leak_patterns_actually_fire():
    """A detector nobody has seen fire is unverified. Each pattern is proven
    against the exact shape it exists to catch."""
    assert _DOLLAR.search("monthly ceiling $100")
    assert _DOLLAR.search("limit: 100 USD")
    assert _ACCOUNT_ID.search("account 123456789012 owns it")
    assert not _ACCOUNT_ID.search("updated: 2026-09-20")
    assert _ARN.search("arn:aws:sns:us-east-1:1:alerts")
    assert _INSTANCE_ID.search("box i-0abc1234def567890")
    assert not _INSTANCE_ID.search("i-am-not-an-instance")
