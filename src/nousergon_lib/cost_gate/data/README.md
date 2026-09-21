# `cost_gate/data` — generated, never hand-edited

Two documents, both **derived by allowlist** from a private cost
single-source-of-truth and published here by pull request:

| File | What it holds |
|---|---|
| `budget_prefixes.yaml` | Cloud service names and the billing prefixes each covers, plus the set of prefixes declared free. **No amounts.** |
| `resource_map.yaml` | `AWS::Service::Type` &rarr; billing prefix, and the two schedule bounds (`max_retry_attempts`, `min_gha_cron_interval_minutes`). |

## Why they exist here and not behind a fetch

The gate has to run in every repo where spend is *created*, not only in the
one where spend is *declared*. A private reusable workflow cannot be called by
a public repo, and a cross-repo fetch would need a credential in every caller.
Shipping the derived documents as package data means every repo runs the same
grader from the library it already pins — identical behaviour for a public and
a private caller, no credential, and a caller whose pin is behind simply
grades against an older map rather than failing open.

## Why allowlist, not redaction

Every field that crosses into these documents is named explicitly in the
producing script. A redaction pass — load the private document, delete the
sensitive keys, publish the rest — fails **open** the moment someone adds a
field, and that failure is silent and public. This one fails closed: a new
private field is simply absent until someone adds it to the allowlist, in a
reviewed PR, on the private repo.

`tests/test_cost_gate_no_leak.py` asserts the guarantee over the serialised
bytes (what leaks is what is written to a file, not what a dict happens to
hold) *and* over the key shape (a substring blocklist only catches the fields
someone thought of).

## Regenerating

Run the derivation in the private repo and open a PR here with the result —
`alpha-engine-config`'s `scripts/check_public_cost_map_in_sync.py` re-derives
from the live source, compares it to the copy in the pinned `nousergon-lib`,
and prints the exact command when they differ.
