"""Does the pre-merge cost gate actually refuse — and does it stay quiet?

Ported from the single-repo original on its second adoption
(``policy-shared-code``). The load-bearing test is
``test_a_new_call_site_on_an_unbudgeted_service_is_refused``, and it asserts a
non-zero exit rather than merely that some text was printed.

The gate has to be shown to fire and shown to STAY QUIET, in equal measure. A
check that cannot reach zero on ordinary work becomes ``--warn-only`` forever,
which is indistinguishable from not having it.

Tests that needed the PRIVATE source document were dropped rather than copied:
the fleet-tree coverage sweep and the "unbudgeted mapped prefix is deliberate"
pin both grade a file this repo does not and must not hold. They stay in
``alpha-engine-config``, where the source lives.

Refs alpha-engine-config-I11228.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from nousergon_lib.cost_gate import cli
from nousergon_lib.cost_gate import grade as gate
from nousergon_lib.cost_gate import resource_map as crm

DOC = gate.load_ssot()
RMAP = crm.load()


def _diff(path: str, *added: str) -> str:
    body = "\n".join(f"+{ln}" for ln in added)
    return (
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
        f"@@ -0,0 +1,{len(added)} @@\n{body}\n"
    )


def _whole_file_diff(path: str, text: str) -> str:
    """A diff that adds ``text`` in full, with line numbers that match it.

    The schedule class asks "did this diff touch THIS resource?", so a fixture
    whose line numbers do not correspond to the file would prove nothing.
    """
    return _diff(path, *text.splitlines())


# -- the packaged documents load at all --------------------------------------


def test_the_packaged_ssot_and_resource_map_load():
    budgeted, free = gate.approved_prefixes(DOC)
    assert "ce" in budgeted
    assert "iam" in free
    assert RMAP["resource_types"]["AWS::Lambda::Function"] == "lambda"


def test_an_ssot_that_is_not_there_raises_rather_than_falling_back(tmp_path):
    """A caller that named a document and was silently graded against another
    one is worse than a caller that failed."""
    with pytest.raises(gate.SsotError):
        gate.load_ssot(tmp_path / "nope.yaml")


def test_an_empty_ssot_raises_rather_than_approving_nothing(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("providers: {}\ncapability_gate: {}\n")
    with pytest.raises(gate.SsotError):
        gate.load_ssot(p)


def test_an_ssot_that_is_not_a_mapping_raises(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(gate.SsotError):
        gate.load_ssot(p)


def test_load_yaml_doc_refuses_a_non_mapping(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("- a\n")
    with pytest.raises(gate.SsotError):
        gate.load_yaml_doc(p, "ssot")


# -- it fires ----------------------------------------------------------------


def test_a_new_call_site_on_an_unbudgeted_service_is_refused():
    """``textract`` names no budgeted service and is not declared free."""
    issues, _ = gate.findings(_diff("scripts/x.py", '    c = boto3.client("textract")'), DOC)
    assert len(issues) == 1
    assert "textract" in issues[0]
    assert "no budgeted service" in issues[0]


def test_a_new_entitlement_on_an_unbudgeted_service_is_refused():
    issues, _ = gate.findings(
        _diff("infrastructure/iam/thing.json", '        "textract:DetectDocumentText",'), DOC
    )
    assert len(issues) == 1
    assert "textract" in issues[0]


def test_the_remedy_is_a_budget_line_not_a_suppression(tmp_path, capsys):
    """The failure message has to state what to do, or the next person
    suppresses the check instead of adding the line."""
    d = tmp_path / "x.diff"
    d.write_text(_diff("a.py", 'boto3.client("textract")'))
    rc = cli.main(["--diff-file", str(d)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "budget line, not a suppression" in out
    assert "free_prefixes" in out


# -- it stays quiet ----------------------------------------------------------


def test_a_budgeted_service_passes():
    issues, _ = gate.findings(_diff("scripts/x.py", '    ce = boto3.client("ce")'), DOC)
    assert issues == []


def test_a_declared_free_service_passes():
    issues, _ = gate.findings(_diff("scripts/x.py", '    i = boto3.client("iam")'), DOC)
    assert issues == []


def test_an_alias_of_a_budgeted_service_passes():
    """``stepfunctions`` and ``states`` are two names for one service. An alias
    absent from ``iam_prefixes`` would read as unbudgeted capability."""
    issues, _ = gate.findings(
        _diff("scripts/x.py", '    sf = boto3.client("stepfunctions")'), DOC
    )
    assert issues == []


def test_a_removed_line_is_not_a_finding():
    """It grades what the diff ADDS."""
    d = (
        'diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +0,0 @@\n'
        '-c = boto3.client("textract")\n'
    )
    issues, _ = gate.findings(d, DOC)
    assert issues == []


def test_a_mention_in_prose_is_not_a_grant():
    issues, _ = gate.findings(
        _diff("README.md", "we should think about textract:DetectDocumentText one day"), DOC
    )
    assert issues == []


def test_a_test_fixture_naming_an_unbudgeted_service_is_not_a_grant():
    """THIS FILE names ``textract:*`` to prove the gate fires. A gate that reds
    the test proving it works cannot be merged, let alone required."""
    issues, _ = gate.findings(
        _diff("tests/test_cost_gate.py", '    "textract:DetectDocumentText"'), DOC
    )
    assert issues == []


def test_a_captured_entitlement_backup_is_not_a_new_grant():
    issues, _ = gate.findings(
        _diff("private-docs/retirement/roles_backup_260920.json",
              '            "textract:DetectDocumentText",'), DOC
    )
    assert issues == []


def test_a_client_inside_a_capture_file_is_still_graded():
    """The capture exclusion suppresses the entitlement class only: a backup of
    entitlement state has no reason to construct a client."""
    issues, _ = gate.findings(
        _diff("private-docs/backups/thing.py", 'c = boto3.client("textract")'), DOC
    )
    assert len(issues) == 1


def test_a_client_construction_inside_a_test_file_is_not_graded():
    issues, _ = gate.findings(
        _diff("tests/test_cost_gate.py", '    c = boto3.client("textract")'), DOC
    )
    assert issues == []


def test_a_workflow_permissions_block_is_not_a_cloud_grant():
    """``contents: read`` is a GitHub token scope, not a service this gate
    governs. Every entry in that exclusion list is a hole, so it is pinned."""
    issues, _ = gate.findings(
        _diff("scripts/x.py", '    perms = "contents:Read"'), DOC
    )
    assert issues == []


# -- always-on resources -----------------------------------------------------

_TEMPLATE = """\
Resources:
  Dispatcher:
    Type: AWS::Lambda::Function
    Properties:
      Handler: index.handler
{extra}
"""


def _template(extra: str) -> str:
    return _TEMPLATE.format(extra=extra.strip("\n"))


def test_a_budgeted_resource_type_passes():
    issues, counts = gate.findings(
        _diff("infra/x.yaml", "    Type: AWS::Lambda::Function"), DOC, resource_map=RMAP
    )
    assert issues == []
    assert counts["resources"] == 1


def test_an_unmapped_resource_type_is_a_finding():
    """UNMAPPED IS NOT FREE. A resource nobody has priced is exactly the shape
    that produces a surprise bill."""
    issues, _ = gate.findings(
        _diff("infra/x.yaml", "    Type: AWS::Kinesis::Stream"), DOC, resource_map=RMAP
    )
    assert len(issues) == 1
    assert "not in the published resource map" in issues[0]
    assert "UNMAPPED IS NOT FREE" in issues[0]


def test_a_mapped_type_on_an_unbudgeted_service_is_a_finding():
    """``cloudfront`` is mapped deliberately even without a budget line — so
    the message names the real remedy rather than reading as a map defect."""
    issues, _ = gate.findings(
        _diff("infra/x.yaml", "    Type: AWS::CloudFront::Distribution"),
        DOC, resource_map=RMAP,
    )
    assert len(issues) == 1
    assert "bills to `cloudfront`" in issues[0]


def test_a_free_resource_type_passes():
    issues, _ = gate.findings(
        _diff("infra/x.yaml", "    Type: AWS::IAM::Role"), DOC, resource_map=RMAP
    )
    assert issues == []


def test_an_imported_resource_is_not_a_new_resource():
    """``"ResourceType": "AWS::..."`` in a resources-to-import manifest names
    something that ALREADY EXISTS."""
    issues, counts = gate.findings(
        _diff("infra/import.json", '    "ResourceType": "AWS::CloudFront::Distribution",'),
        DOC, resource_map=RMAP,
    )
    assert issues == []
    assert counts["resources"] == 0


def test_a_resource_type_named_in_python_prose_is_not_a_resource():
    issues, counts = gate.findings(
        _diff("src/nousergon_lib/cost_gate/grade.py",
              "#: `Type: AWS::CloudFront::Distribution` is a record, not a request"),
        DOC, resource_map=RMAP,
    )
    assert issues == []
    assert counts["resources"] == 0


def test_a_changelog_quoting_a_resource_type_is_not_a_resource():
    issues, _ = gate.findings(
        _diff("docs/changelog.d/x.md", "    Type: AWS::CloudFront::Distribution"),
        DOC, resource_map=RMAP,
    )
    assert issues == []


# -- schedules ---------------------------------------------------------------


def _graded(tmp_path: Path, path: str, text: str):
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return gate.findings(
        _whole_file_diff(path, text), DOC, resource_map=RMAP, root=tmp_path
    )


def test_a_bounded_schedule_on_a_budgeted_target_passes(tmp_path):
    issues, counts = _graded(tmp_path, "infra/s.yaml", _template("""
  Sched:
    Type: AWS::Scheduler::Schedule
    Properties:
      ScheduleExpression: cron(0 3 * * ? *)
      Target:
        Arn: !GetAtt Dispatcher.Arn
        RetryPolicy:
          MaximumRetryAttempts: 0
"""))
    assert issues == []
    assert counts["schedules"] == 1
    assert counts["ungraded_schedules"] == 0
    assert counts["ungraded_targets"] == 0


def test_a_schedule_with_no_retry_policy_is_a_finding(tmp_path):
    """Absent means the AWS default of 185 attempts. A per-call price cap that
    says nothing about call count is not a control."""
    issues, _ = _graded(tmp_path, "infra/s.yaml", _template("""
  Sched:
    Type: AWS::Scheduler::Schedule
    Properties:
      ScheduleExpression: cron(0 3 * * ? *)
      Target:
        Arn: !GetAtt Dispatcher.Arn
"""))
    assert len(issues) == 1
    assert "declares no `RetryPolicy.MaximumRetryAttempts`" in issues[0]
    assert "185" in issues[0]


def test_a_schedule_retrying_above_the_declared_maximum_is_a_finding(tmp_path):
    issues, _ = _graded(tmp_path, "infra/s.yaml", _template("""
  Sched:
    Type: AWS::Scheduler::Schedule
    Properties:
      ScheduleExpression: cron(0 3 * * ? *)
      Target:
        Arn: !GetAtt Dispatcher.Arn
        RetryPolicy:
          MaximumRetryAttempts: 185
"""))
    assert len(issues) == 1
    assert "retries up to 185 times" in issues[0]


def test_a_schedule_targeting_an_unbudgeted_service_is_a_finding(tmp_path):
    """The target is a ``!GetAtt`` at a logical id declared elsewhere in the
    template — which is why this class is graded structurally."""
    issues, _ = _graded(tmp_path, "infra/s.yaml", _template("""
  Edge:
    Type: AWS::CloudFront::Distribution
    Properties: {}
  Sched:
    Type: AWS::Scheduler::Schedule
    Properties:
      ScheduleExpression: cron(0 3 * * ? *)
      Target:
        Arn: !GetAtt Edge.Arn
        RetryPolicy:
          MaximumRetryAttempts: 1
"""))
    assert any("targets a `cloudfront` resource" in i for i in issues)


def test_an_events_rule_target_is_graded_through_a_ref(tmp_path):
    issues, _ = _graded(tmp_path, "infra/s.yaml", _template("""
  Rule:
    Type: AWS::Events::Rule
    Properties:
      ScheduleExpression: rate(1 day)
      Targets:
        - Arn: !Ref Dispatcher
          Id: t
"""))
    assert len(issues) == 1
    assert "schedule `Rule` declares no" in issues[0]


def test_an_event_pattern_rule_has_no_cadence_to_bound(tmp_path):
    """A rule with no ``ScheduleExpression`` fires on an event, not a clock."""
    issues, counts = _graded(tmp_path, "infra/s.yaml", _template("""
  Rule:
    Type: AWS::Events::Rule
    Properties:
      EventPattern:
        source: [aws.ec2]
      Targets:
        - Arn: !Ref Dispatcher
          Id: t
"""))
    assert issues == []
    assert counts["schedules"] == 0


def test_a_schedule_with_no_target_at_all_is_ungraded_not_passed(tmp_path):
    _, counts = _graded(tmp_path, "infra/s.yaml", _template("""
  Sched:
    Type: AWS::Scheduler::Schedule
    Properties:
      ScheduleExpression: cron(0 3 * * ? *)
"""))
    assert counts["ungraded_schedules"] == 1


def test_a_schedule_the_diff_did_not_touch_is_not_regraded(tmp_path):
    """Editing one line of a template must not demand a retry policy be
    re-added to every other schedule in it."""
    text = _template("""
  Sched:
    Type: AWS::Scheduler::Schedule
    Properties:
      ScheduleExpression: cron(0 3 * * ? *)
      Target:
        Arn: !GetAtt Dispatcher.Arn
""")
    target = tmp_path / "infra/s.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    diff = (
        "diff --git a/infra/s.yaml b/infra/s.yaml\n--- a/infra/s.yaml\n"
        "+++ b/infra/s.yaml\n@@ -5,0 +5 @@\n+      # ScheduleExpression note\n"
    )
    issues, _ = gate.findings(diff, DOC, resource_map=RMAP, root=tmp_path)
    assert issues == []


def test_a_schedule_whose_definition_cannot_be_read_is_not_a_pass(tmp_path):
    """The gate said NOTHING about this schedule; *no data* is never green."""
    _, counts = gate.findings(
        _diff("infra/absent.yaml", "      ScheduleExpression: cron(0 3 * * ? *)"),
        DOC, resource_map=RMAP, root=tmp_path,
    )
    assert counts["ungraded_schedules"] == 1


def test_a_read_file_hook_replaces_the_filesystem(tmp_path):
    """The history replay hands over ``git show <sha>:<path>`` rather than
    checking every commit out."""
    text = _template("""
  Sched:
    Type: AWS::Scheduler::Schedule
    Properties:
      ScheduleExpression: cron(0 3 * * ? *)
      Target:
        Arn: !GetAtt Dispatcher.Arn
        RetryPolicy:
          MaximumRetryAttempts: 1
""")
    issues, counts = gate.findings(
        _whole_file_diff("infra/s.yaml", text), DOC, resource_map=RMAP,
        read_file=lambda path: text,
    )
    assert issues == []
    assert counts["ungraded_schedules"] == 0


def test_a_schedule_created_by_an_sdk_call_is_reported_ungraded():
    _, counts = gate.findings(
        _diff("scripts/x.py",
              "        c.create_schedule(ScheduleExpression='rate(1 minute)')"),
        DOC, resource_map=RMAP,
    )
    assert counts["ungraded_schedules"] == 1


def test_a_python_file_that_only_names_a_retry_field_is_not_a_schedule():
    """This package's own source names ``MaximumRetryAttempts`` in a docstring.
    A signal that fires on everything is read as firing on nothing."""
    _, counts = gate.findings(
        _diff("src/nousergon_lib/cost_gate/grade.py",
              "    # absent MaximumRetryAttempts means the AWS default of 185"),
        DOC, resource_map=RMAP,
    )
    assert counts["ungraded_schedules"] == 0


def test_a_template_that_will_not_parse_is_not_a_template_with_no_schedules(tmp_path):
    bad = tmp_path / "infra/broken.yaml"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("Resources:\n  A:\n   - x\n  : :\n")
    with pytest.raises(crm.ResourceMapError):
        gate.findings(
            _diff("infra/broken.yaml", "      ScheduleExpression: cron(0 3 * * ? *)"),
            DOC, resource_map=RMAP, root=tmp_path,
        )


# -- GitHub Actions crons ----------------------------------------------------


def test_a_frequent_cron_on_a_private_repo_is_a_finding():
    issues, _ = gate.findings(
        _diff(".github/workflows/w.yml", "    - cron: '*/5 * * * *'"),
        DOC, resource_map=RMAP, repo_visibility="private",
    )
    assert len(issues) == 1
    assert "every 5 minute(s)" in issues[0]


def test_the_same_cron_on_a_public_repo_is_not_graded():
    """Public-repo Actions minutes are free and unlimited. Grading them would
    be noise with no spend behind it, and noise is what gets a gate muted."""
    issues, counts = gate.findings(
        _diff(".github/workflows/w.yml", "    - cron: '*/5 * * * *'"),
        DOC, resource_map=RMAP, repo_visibility="public",
    )
    assert issues == []
    assert counts["schedules"] == 1


def test_an_ordinary_daily_cron_passes():
    issues, _ = gate.findings(
        _diff(".github/workflows/w.yml", "    - cron: '30 13 * * *'"),
        DOC, resource_map=RMAP,
    )
    assert issues == []


def test_an_unreadable_cron_is_ungraded_not_passed():
    issues, counts = gate.findings(
        _diff(".github/workflows/w.yml", "    - cron: 'MON TUE WED'"),
        DOC, resource_map=RMAP,
    )
    assert issues == []
    assert counts["ungraded_schedules"] == 1


def test_cron_interval_is_the_shortest_gap_not_the_average():
    """``0,1,2,30 * * * *`` bills like a one-minute cadence for three minutes
    of every hour. The question is how many runs may bill, not how evenly."""
    assert gate.cron_interval_minutes("0,1,2,30 * * * *") == 1
    assert gate.cron_interval_minutes("*/15 * * * *") == 15
    assert gate.cron_interval_minutes("0 * * * *") == 60
    assert gate.cron_interval_minutes("30 13 * * *") == 1440
    assert gate.cron_interval_minutes("* * * * *") == 1
    assert gate.cron_interval_minutes("0-5 * * * *") == 1
    assert gate.cron_interval_minutes("*/0 0 * * *") == 1440
    assert gate.cron_interval_minutes("nonsense") is None
    # A fixed minute with an unreadable HOUR field falls back to once-a-day.
    # Ported as-is rather than tightened: the minute field alone already bounds
    # this at one firing per hour at worst, so the fallback cannot understate
    # the cadence by more than a factor of 24 on an expression no fleet repo
    # writes — and diverging from the private copy's behaviour would mean two
    # gates that look alike and grade differently.
    assert gate.cron_interval_minutes("0 MON * * *") == 1440


# -- added-line bookkeeping --------------------------------------------------


def test_it_reports_how_much_it_graded():
    _, counts = gate.findings(
        _diff("scripts/x.py", 'a = boto3.client("s3")', 'b = boto3.client("ec2")'), DOC
    )
    assert counts["clients"] == 2
    assert counts["lines"] == 2


def test_added_lines_drops_the_numbers():
    d = _diff("x.py", "a", "b")
    assert gate.added_lines(d) == [("x.py", "a"), ("x.py", "b")]
    assert [n for _, n, _ in gate.added_with_lines(d)] == [1, 2]


def test_context_lines_advance_the_line_counter():
    d = (
        "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
        "@@ -1,2 +1,3 @@\n ctx\n+added\n"
    )
    assert gate.added_with_lines(d) == [("x.py", 2, "added")]


# -- the merge base ----------------------------------------------------------


def test_an_unreachable_merge_base_is_named_not_a_traceback(tmp_path, monkeypatch):
    """MEASURED: ``git diff base...HEAD`` exits 128 the moment the base moves
    ahead of a shallow PR checkout. A traceback out of a REQUIRED check blocks
    every PR that is behind its base."""
    repo = tmp_path / "r"
    repo.mkdir()

    def run(*a):
        subprocess.run(a, cwd=repo, check=True, capture_output=True)

    run("git", "init", "-q", "-b", "main")
    run("git", "config", "user.email", "t@example.com")
    run("git", "config", "user.name", "t")
    (repo / "a").write_text("a")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "a")
    # A second root commit: a real history with NO merge base against `main`.
    run("git", "checkout", "-q", "--orphan", "other")
    run("git", "rm", "-rqf", ".")
    (repo / "b").write_text("b")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "b")

    monkeypatch.chdir(repo)
    assert gate.is_shallow() is False
    assert gate.merge_base("main", "HEAD") is None
    with pytest.raises(gate.MergeBaseUnreachable) as excinfo:
        gate.git_diff("main", "HEAD")
    assert "no merge base" in str(excinfo.value)
    assert "fetch-depth: 0" in str(excinfo.value)
    assert "two-dot" in str(excinfo.value)


def test_a_reachable_merge_base_produces_a_diff(tmp_path, monkeypatch):
    repo = tmp_path / "r"
    repo.mkdir()

    def run(*a):
        subprocess.run(a, cwd=repo, check=True, capture_output=True)

    run("git", "init", "-q", "-b", "main")
    run("git", "config", "user.email", "t@example.com")
    run("git", "config", "user.name", "t")
    (repo / "a.py").write_text("x = 1\n")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "a")
    run("git", "checkout", "-qb", "feat")
    (repo / "a.py").write_text('x = 1\nc = boto3.client("textract")\n')
    run("git", "add", "-A")
    run("git", "commit", "-qm", "b")

    monkeypatch.chdir(repo)
    assert gate.merge_base("main", "HEAD")
    diff = gate.git_diff("main", "HEAD")
    issues, _ = gate.findings(diff, DOC, resource_map=RMAP)
    assert len(issues) == 1


# -- the exit-code contract --------------------------------------------------


def test_a_clean_diff_exits_0(tmp_path, capsys):
    d = tmp_path / "x.diff"
    d.write_text(_diff("a.py", 'boto3.client("s3")'))
    assert cli.main(["--diff-file", str(d)]) == 0
    assert "0 findings" in capsys.readouterr().out


def test_findings_exit_1(tmp_path):
    d = tmp_path / "x.diff"
    d.write_text(_diff("a.py", 'boto3.client("textract")'))
    assert cli.main(["--diff-file", str(d)]) == 1


def test_warn_only_downgrades_findings_to_0(tmp_path, capsys):
    d = tmp_path / "x.diff"
    d.write_text(_diff("a.py", 'boto3.client("textract")'))
    assert cli.main(["--diff-file", str(d), "--warn-only"]) == 0
    out = capsys.readouterr().out
    assert "1 finding(s)" in out
    assert "--warn-only" in out


def test_warn_only_does_not_downgrade_could_not_grade(tmp_path, monkeypatch, capsys):
    """A gate that could not run at all is a broken control, not a soft
    finding. Exit 2 survives ``--warn-only``."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)  # noqa: S607 — a real work tree
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        gate, "git_diff",
        lambda *a, **k: (_ for _ in ()).throw(gate.MergeBaseUnreachable("boom")),
    )
    assert cli.main(["--base", "main", "--head", "HEAD", "--warn-only"]) == 2
    assert "could not grade" in capsys.readouterr().out


def test_a_no_merge_base_run_exits_2_not_1(tmp_path, monkeypatch, capsys):
    """"I could not grade this" and "I graded it and it is unbudgeted" are
    different states with different remedies."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)  # noqa: S607 — a real work tree
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        gate, "git_diff",
        lambda *a, **k: (_ for _ in ()).throw(gate.MergeBaseUnreachable("boom")),
    )
    assert cli.main(["--base", "main", "--head", "HEAD"]) == 2
    assert "could not grade" in capsys.readouterr().out


def test_repo_visibility_public_reaches_the_grader(tmp_path, capsys):
    d = tmp_path / "x.diff"
    d.write_text(_diff(".github/workflows/w.yml", "    - cron: '*/5 * * * *'"))
    assert cli.main(["--diff-file", str(d), "--repo-visibility", "public"]) == 0
    assert cli.main(["--diff-file", str(d), "--repo-visibility", "private"]) == 1
    capsys.readouterr()


def test_the_ungraded_count_is_printed_not_swallowed(tmp_path, capsys):
    d = tmp_path / "x.diff"
    d.write_text(_diff("scripts/x.py", "  c.create_schedule(ScheduleExpression='x')"))
    cli.main(["--diff-file", str(d), "--root", str(tmp_path)])
    assert "NOT GRADED" in capsys.readouterr().out


def test_the_ungraded_count_names_the_file_not_just_a_number(tmp_path, capsys):
    """alpha-engine-config-I11289: a could-not-grade in this repo's own
    150-commit history could not be identified from the CLI's output — it
    printed a count with no file attached. The NOT GRADED line must now name
    what it could not read."""
    d = tmp_path / "x.diff"
    d.write_text(_diff("scripts/x.py", "  c.create_schedule(ScheduleExpression='x')"))
    cli.main(["--diff-file", str(d), "--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "scripts/x.py" in out
    assert "SDK schedule call site" in out


def test_findings_reports_the_ungraded_detail_in_its_counts():
    issues, counts = gate.findings(
        _diff("scripts/x.py", "  c.create_schedule(ScheduleExpression='x')"), DOC
    )
    assert counts["ungraded_schedules"] == 1
    assert len(counts["ungraded"]) == 1
    assert "scripts/x.py" in counts["ungraded"][0]


# -- `--root` is authoritative for git ref resolution, never the process's
# -- own CWD -------------------------------------------------------------
#
# Refs alpha-engine-config-I11289. Measured independently by five dispatched
# workers plus the orchestrating session: `--base`/`--head` used to be
# resolved by git against the process's CURRENT WORKING DIRECTORY, so running
# the CLI from a shell `cd`'d into a DIFFERENT repo silently graded that
# repo's history instead of the one named by `--root` — no error, a
# plausible-looking wrong answer. Every git call in `grade.py` now takes
# `cwd=root` explicitly (see `resolve_root`/`_git`), and this is the
# regression test that proves it end to end with two real, unrelated repos.


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)  # noqa: S607


def _init_repo_with_two_commits(root: Path, second_commit_adds: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", cwd=root)
    _git("config", "user.email", "test@example.com", cwd=root)
    _git("config", "user.name", "Test", cwd=root)
    (root / "a.py").write_text("x = 1\n")
    _git("add", "a.py", cwd=root)
    _git("commit", "-q", "-m", "base", cwd=root)
    (root / "a.py").write_text("x = 1\n" + second_commit_adds)
    _git("add", "a.py", cwd=root)
    _git("commit", "-q", "-m", "head", cwd=root)


def test_root_is_graded_even_when_cwd_is_a_different_repo(tmp_path, monkeypatch):
    """Repo A (the process's CWD) and Repo B (`--root`) are two unrelated
    repos with different histories. `--base HEAD~1 --head HEAD` resolves in
    BOTH — the bug this fix closes is exactly that ambiguity. The CLI must
    grade B, not A, regardless of where the process was launched from."""
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    _init_repo_with_two_commits(repo_a, "y = 2\n")  # no finding: not a grant
    _init_repo_with_two_commits(repo_b, 'c = boto3.client("textract")\n')  # a real finding

    monkeypatch.chdir(repo_a)
    rc_graded_as_b = cli.main(["--base", "HEAD~1", "--head", "HEAD", "--root", str(repo_b)])
    assert rc_graded_as_b == 1  # repo B's diff has an unbudgeted finding

    monkeypatch.chdir(repo_a)
    rc_graded_as_a = cli.main(["--base", "HEAD~1", "--head", "HEAD", "--root", str(repo_a)])
    assert rc_graded_as_a == 0  # repo A's own diff has none


def test_a_root_outside_any_git_work_tree_exits_2_with_a_named_cause(tmp_path):
    not_a_repo = tmp_path / "plain-dir"
    not_a_repo.mkdir()
    rc = cli.main(["--base", "HEAD~1", "--head", "HEAD", "--root", str(not_a_repo)])
    assert rc == 2


def test_a_root_outside_any_git_work_tree_names_the_cause(tmp_path, capsys):
    not_a_repo = tmp_path / "plain-dir"
    not_a_repo.mkdir()
    cli.main(["--base", "HEAD~1", "--head", "HEAD", "--root", str(not_a_repo)])
    out = capsys.readouterr().out
    assert "is not inside a git work tree" in out
    assert str(not_a_repo) in out


def test_diff_file_mode_does_not_require_root_to_be_a_git_work_tree(tmp_path):
    """`--diff-file` bypasses git entirely, so an arbitrary directory handed
    as `--root` (used only for reading head-tree files) is fine — this is the
    pre-existing shape most unit tests in this file use."""
    d = tmp_path / "x.diff"
    d.write_text(_diff("scripts/x.py", 'c = boto3.client("textract")'))
    rc = cli.main(["--diff-file", str(d), "--root", str(tmp_path)])
    assert rc == 1


# -- the injected SSoT is really used ----------------------------------------


def test_an_injected_ssot_copy_grades_identically_to_the_packaged_one(tmp_path, capsys):
    """The whole point of shipping the map as package data: a caller that
    supplies its own copy of the SAME document must get the SAME verdict, or
    the two paths are two different gates that merely look alike."""
    copy = tmp_path / "ssot.yaml"
    copy.write_text(yaml.safe_dump(DOC, sort_keys=True))

    fixtures = [
        _diff("x.py", 'c = boto3.client("textract")'),
        _diff("x.py", 'c = boto3.client("ce")'),
        _diff("i.yaml", "    Type: AWS::Kinesis::Stream"),
        _diff("i.yaml", "    Type: AWS::Lambda::Function"),
    ]
    codes_packaged, codes_injected = [], []
    for i, text in enumerate(fixtures):
        d = tmp_path / f"{i}.diff"
        d.write_text(text)
        codes_packaged.append(cli.main(["--diff-file", str(d)]))
        codes_injected.append(cli.main(["--diff-file", str(d), "--ssot", str(copy)]))
    capsys.readouterr()

    assert codes_packaged == codes_injected
    # ...and at least one of them actually found something, or the comparison
    # above proves only that two gates agree about nothing.
    assert 1 in codes_packaged and 0 in codes_packaged


def test_an_injected_ssot_that_omits_a_prefix_changes_the_verdict(tmp_path, capsys):
    """Proves the override is READ rather than ignored: the same diff that
    passes against the packaged map must fail against one without ``s3``."""
    narrowed = {
        "providers": {"aws": {"services": {"Some Service": {"iam_prefixes": ["ec2"]}}}},
        "capability_gate": {"free_prefixes": {"iam": "placeholder"}},
    }
    p = tmp_path / "narrow.yaml"
    p.write_text(yaml.safe_dump(narrowed))
    d = tmp_path / "x.diff"
    d.write_text(_diff("a.py", 'boto3.client("s3")'))

    assert cli.main(["--diff-file", str(d)]) == 0
    assert cli.main(["--diff-file", str(d), "--ssot", str(p)]) == 1
    assert str(p) in capsys.readouterr().out


def test_an_injected_resource_map_is_read(tmp_path, capsys):
    p = tmp_path / "rm.yaml"
    p.write_text("resource_types:\n  AWS::Lambda::Function: lambda\n")
    d = tmp_path / "x.diff"
    d.write_text(_diff("i.yaml", "    Type: AWS::S3::Bucket"))
    assert cli.main(["--diff-file", str(d), "--resource-map", str(p)]) == 1
    assert cli.main(["--diff-file", str(d)]) == 0
    capsys.readouterr()


def test_the_console_entry_and_the_package_data_are_declared():
    """A module nothing can invoke is not a shared tool, and a grader whose
    map is not packaged installs as a gate that cannot load its own SSoT."""
    try:  # Python 3.11+
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    root = Path(__file__).resolve().parents[1]
    cfg = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert cfg["project"]["scripts"]["nousergon-cost-gate"] == (
        "nousergon_lib.cost_gate.cli:main"
    )
    pkg_data = cfg["tool"]["setuptools"]["package-data"]
    assert "data/*.yaml" in pkg_data["nousergon_lib.cost_gate"]
    assert sys.version_info >= (3, 9)


# -- the resource map, and the CloudFormation reader --------------------------


def test_the_map_declares_both_bounds():
    rules = RMAP["schedule_rules"]
    # Pinned, not merely present: both numbers are the whole grading rule for
    # the schedule class, and a silent widening of either is a silent widening
    # of the gate.
    assert rules["max_retry_attempts"] == 3
    assert rules["min_gha_cron_interval_minutes"] == 15


def test_every_entry_is_a_cfn_type_mapped_to_a_lowercase_prefix():
    import re as _re

    for type_name, prefix in RMAP["resource_types"].items():
        assert _re.fullmatch(r"AWS::[A-Za-z0-9]+::[A-Za-z0-9]+", type_name), type_name
        assert _re.fullmatch(r"[a-z0-9-]+", prefix), (type_name, prefix)


def test_a_malformed_map_raises_rather_than_loading_empty(tmp_path):
    """A cost gate that degrades to `nothing to grade` when its map fails to
    parse is worse than no gate: it prints a green zero for a class it did not
    look at."""
    bad = tmp_path / "m.yaml"
    for text in (
        "resource_types: {}\n",
        "resource_types:\n  NotACfnType: s3\n",
        "resource_types:\n  AWS::S3::Bucket: S3\n",
        "resource_types:\n  AWS::S3::Bucket: s3\nschedule_rules:\n  max_retry_attempts: -1\n",
        "resource_types:\n  AWS::S3::Bucket: s3\nschedule_rules: 4\n",
        "- a\n- b\n",
    ):
        bad.write_text(text)
        with pytest.raises(crm.ResourceMapError):
            crm.load(bad)

    with pytest.raises(crm.ResourceMapError):
        crm.load(tmp_path / "absent.yaml")


def test_a_map_omitting_a_bound_takes_this_packages_default(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("resource_types:\n  AWS::S3::Bucket: s3\n")
    rules = crm.load(p)["schedule_rules"]
    assert rules["max_retry_attempts"] == crm.DEFAULT_MAX_RETRY_ATTEMPTS
    assert rules["min_gha_cron_interval_minutes"] == (
        crm.DEFAULT_MIN_GHA_CRON_INTERVAL_MINUTES
    )


def test_intrinsics_survive_the_parse():
    """``yaml.safe_load`` refuses ``!GetAtt`` outright, and dropping the tag
    would make ``!Ref Foo`` indistinguishable from the string "Foo" — which is
    exactly the resolution this module performs."""
    resources = crm.parse_cfn_resources(
        "Resources:\n"
        "  S:\n"
        "    Type: AWS::Scheduler::Schedule\n"
        "    Properties:\n"
        "      Target:\n"
        "        Arn: !GetAtt D.Arn\n"
        "        Tags: !Sub ['a', 'b']\n"
        "        Meta: !Ref {a: b}\n"
    )
    arn = resources[0].properties["Target"]["Arn"]
    assert arn == crm.Intrinsic("GetAtt", "D.Arn")
    assert arn != crm.Intrinsic("Ref", "D.Arn")
    assert arn != "D.Arn"
    assert resources[0].properties["Target"]["Tags"].value == ["a", "b"]
    assert resources[0].properties["Target"]["Meta"].value == {"a": "b"}


def test_line_spans_let_the_gate_ask_which_resource_the_diff_touched():
    resources = crm.parse_cfn_resources(
        "Resources:\n"
        "  A:\n"
        "    Type: AWS::S3::Bucket\n"
        "  B:\n"
        "    Type: AWS::SNS::Topic\n"
    )
    a, b = resources
    assert a.touches({3}) and not a.touches({5})
    assert b.touches({5}) and not b.touches({3})


def test_a_document_with_no_resources_mapping_is_not_a_template():
    assert crm.parse_cfn_resources("on:\n  push: {}\n") == []
    assert crm.parse_cfn_resources("- a\n- b\n") == []
    # Malformed, but nothing about it claims to be a template.
    assert crm.parse_cfn_resources("a: [\n") == []
    # An entry with no `Type:` declares nothing to grade.
    assert crm.parse_cfn_resources("Resources:\n  A:\n    Properties: {}\n") == []


def test_the_json_long_form_resolves_like_the_yaml_short_form():
    """A gate that read ``!GetAtt`` and not ``{"Fn::GetAtt": [...]}`` would be
    blind to every JSON stack in the fleet."""
    types = {"AWS::Lambda::Function": "lambda"}
    by_lid = {"D": "AWS::Lambda::Function"}
    assert crm.resolve_target_service({"Fn::GetAtt": ["D", "Arn"]}, by_lid, types) == "lambda"
    assert crm.resolve_target_service(crm.Intrinsic("GetAtt", "D.Arn"), by_lid, types) == "lambda"
    assert crm.resolve_target_service({"Ref": "D"}, by_lid, types) == "lambda"


def test_a_literal_arn_names_its_own_service():
    assert crm.resolve_target_service(
        "arn:aws:states:us-east-1:1:stateMachine:x", {}, {}
    ) == "states"
    assert crm.resolve_target_service(
        crm.Intrinsic("Sub", ["arn:aws:sns:${AWS::Region}:1:t", {}]), {}, {}
    ) == "sns"


def test_an_unresolvable_target_is_none_and_never_a_guess():
    """``None`` is reported by the caller as NOT GRADED. A guess would be the
    one thing worse than an honest gap."""
    assert crm.resolve_target_service(crm.Intrinsic("Ref", "Missing"), {}, {}) is None
    assert crm.resolve_target_service(None, {}, {}) is None
    assert crm.resolve_target_service("not-an-arn", {}, {}) is None
    assert crm.resolve_target_service(crm.Intrinsic("GetAtt", 7), {}, {}) is None
    assert crm.resolve_target_service(crm.Intrinsic("Ref", 7), {}, {}) is None
    assert crm.resolve_target_service(crm.Intrinsic("Sub", 7), {}, {}) is None
    assert crm.resolve_target_service(crm.Intrinsic("Unknown", "x"), {}, {}) is None
    assert crm.resolve_target_service({"Ref": "D"}, {"D": "AWS::Unmapped::Thing"}, {}) is None


# -- the action class is context-aware ---------------------------------------
#
# Refs alpha-engine-config-I11228 (the false positives), alpha-engine-config-I11285
# (this fix). A token shaped `word:Word` is graded as an IAM action ONLY when
# it is the value of an `Action`/`NotAction` key in a JSON/YAML document — not
# a `Condition` key, not a `Principal`, not a bare component id that happens to
# contain a colon.


def test_a_component_id_with_a_colon_is_not_an_action(tmp_path):
    """`nousergon-data`'s false positive: a registry entry id shaped
    `pipeline:unit`, not a `service:Action` grant. No `Action`/`NotAction` key
    is anywhere in the document, so nothing structural is found."""
    issues, counts = _graded(
        tmp_path, "registry.d/units/x.yaml",
        "id: ne-weekly-freshness-pipeline:daily-close\n"
        "kind: unit\n",
    )
    assert issues == []
    assert counts["actions_context_fallback"] == 0


def test_a_condition_key_is_not_an_action(tmp_path):
    """`nous-ergon-ops`'s false positive, reproduced as a synthetic trust
    policy: `aws:SourceAccount` is a Condition operator's argument, not a
    grant of a service called `aws`. `sts:AssumeRole` is the real Action and
    is declared free, so this is a clean pass end to end."""
    issues, counts = _graded(tmp_path, "infrastructure/iam/trust-policy.json", (
        '{\n'
        '  "Version": "2012-10-17",\n'
        '  "Statement": [\n'
        '    {\n'
        '      "Effect": "Allow",\n'
        '      "Principal": {"Service": "example.amazonaws.com"},\n'
        '      "Action": "sts:AssumeRole",\n'
        '      "Condition": {\n'
        '        "StringEquals": {"aws:SourceAccount": "000000000000"}\n'
        '      }\n'
        '    }\n'
        '  ]\n'
        '}\n'
    ))
    assert issues == []
    assert counts["actions_context_fallback"] == 0


def test_a_creation_actions_registry_key_is_not_an_action(tmp_path):
    """The private twin's worked-around shape: a registry key NAMED
    `creation_actions` whose string values read like grants
    (`rds:CreateDBInstance`) but are not under an `Action`/`NotAction` key."""
    issues, counts = _graded(tmp_path, "private-docs/EXPENSE_BUDGETS.yaml", (
        "capability_gate:\n"
        "  creation_actions:\n"
        "    - rds:CreateDBInstance\n"
    ))
    assert issues == []
    assert counts["actions_context_fallback"] == 0


# -- the LINE-LEVEL fallback is also context-aware ---------------------------
#
# Refs alpha-engine-config-I11289. Every case above grades a JSON/YAML file
# that parses structurally. A `.py`/`.sh` source file never does — it always
# goes through the line-level fallback, which had no context awareness at
# all until this fix. `flow-doctor`'s false positive was exactly this shape,
# in a `.py` file the structural pass never sees.


def test_a_boto3_filter_dict_name_value_is_not_an_action():
    """`flow-doctor`'s false positive, reproduced: a boto3 EC2 `describe_*`
    filter dict entry (`Filters=[{"Name": "tag:Name", "Values": [...]}]`) is
    not an IAM grant — `tag:Name` here is the VALUE of a `Name` key, not an
    Action string. Source is a `.py` file, so this exercises the LINE-LEVEL
    path directly (no structural parse exists for Python)."""
    issues, counts = gate.findings(
        _diff(
            "flow_doctor/remediation/executor.py",
            '    {"Name": "tag:Name", "Values": [target]},',
        ),
        DOC,
    )
    assert issues == []
    assert counts["actions"] == 1


def test_an_aws_condition_key_in_a_python_source_line_is_not_an_action():
    """`aws:SourceAccount`-style condition keys are IAM's global-condition-key
    namespace, never a service prefix, so the line-level fallback must clear
    them the same way the structural pass already does."""
    issues, _ = gate.findings(
        _diff(
            "scripts/build_trust_policy.py",
            '    condition = {"aws:SourceAccount": account_id}',
        ),
        DOC,
    )
    assert issues == []


def test_a_component_id_value_in_a_python_source_line_is_not_an_action():
    """A component-id string keyed by something other than Action
    (`"id": "pipeline:Unit"`) reads the same way `tag:Name` does — the value
    of an unrelated key, not a grant. Capitalised suffix so it actually
    matches the entitlement shape (`_IAM_ACTION` requires an uppercase/`*`
    character after the colon) and so exercises the disambiguation rule
    rather than merely missing the regex, as the all-lowercase
    `pipeline:daily-close` id shape does."""
    issues, counts = gate.findings(
        _diff(
            "scripts/registry.py",
            '    unit = {"id": "ne-weekly-freshness-pipeline:DailyClose"}',
        ),
        DOC,
    )
    assert issues == []
    assert counts["actions"] == 1


def test_a_quoted_action_with_no_preceding_key_in_python_is_still_a_finding():
    """A bare list element — the shape a real grant takes when building a
    policy document by hand in Python — has no key on the same line at all,
    so it is NOT suppressed. Uses `textract` (unbudgeted): the canonical
    positive fixture in this file — the PR that motivated this fix (I11289)
    names `ce:GetCostAndUsage` as the illustrative "must stay a finding"
    shape, but `ce` is itself budgeted (see
    `test_the_packaged_ssot_and_resource_map_load`), so that exact string
    would pass on approval, not on being excluded from grading — this test
    isolates the mechanism instead."""
    issues, counts = gate.findings(
        _diff("scripts/build_policy.py", '    actions = ["textract:DetectDocumentText"]'),
        DOC,
    )
    assert len(issues) == 1
    assert "textract" in issues[0]
    assert counts["actions"] == 1


def test_a_quoted_action_keyed_by_action_in_python_is_still_a_finding():
    """The value of an Action-shaped key stays graded even at the line level —
    the disambiguation rule narrows what is EXCLUDED, not what is included."""
    issues, _ = gate.findings(
        _diff(
            "scripts/build_policy.py",
            '    stmt = {"Effect": "Allow", "Action": "textract:DetectDocumentText"}',
        ),
        DOC,
    )
    assert len(issues) == 1
    assert "textract" in issues[0]


def test_ce_get_cost_and_usage_in_a_policy_dict_is_graded_as_an_action():
    """The issue's own illustrative shape (I11289): `ce:GetCostAndUsage`
    inside a policy-building Python dict must still be GRADED as an
    entitlement string — it passes here because `ce` is budgeted, not
    because the line-level fallback stopped looking at it."""
    issues, counts = gate.findings(
        _diff(
            "scripts/build_policy.py",
            '    stmt = {"Effect": "Allow", "Action": "ce:GetCostAndUsage"}',
        ),
        DOC,
    )
    assert issues == []
    assert counts["actions"] == 1


def test_an_action_string_value_on_an_unbudgeted_prefix_is_still_a_finding(tmp_path):
    issues, _ = _graded(tmp_path, "infra/policy.json", (
        '{\n'
        '  "Statement": [\n'
        '    {"Effect": "Allow", "Action": "sagemaker:CreateEndpoint", "Resource": "*"}\n'
        '  ]\n'
        '}\n'
    ))
    assert len(issues) == 1
    assert "sagemaker" in issues[0]


def test_an_action_list_value_on_an_unbudgeted_prefix_is_still_a_finding(tmp_path):
    issues, _ = _graded(tmp_path, "infra/policy.json", (
        '{\n'
        '  "Statement": [\n'
        '    {"Effect": "Allow", "Action": ["s3:GetObject", "sagemaker:CreateEndpoint"]}\n'
        '  ]\n'
        '}\n'
    ))
    assert len(issues) == 1
    assert "sagemaker" in issues[0]


def test_an_action_value_in_yaml_form_is_still_graded(tmp_path):
    issues, _ = _graded(tmp_path, "infra/policy.yaml", (
        "Statement:\n"
        "  - Effect: Allow\n"
        "    Action: sagemaker:CreateEndpoint\n"
    ))
    assert len(issues) == 1
    assert "sagemaker" in issues[0]


def test_a_not_action_value_on_an_unbudgeted_prefix_is_still_a_finding(tmp_path):
    issues, _ = _graded(tmp_path, "infra/policy.json", (
        '{\n'
        '  "Statement": [\n'
        '    {"Effect": "Deny", "NotAction": "sagemaker:CreateEndpoint"}\n'
        '  ]\n'
        '}\n'
    ))
    assert len(issues) == 1
    assert "sagemaker" in issues[0]


def test_a_condition_next_to_a_real_unbudgeted_action_flags_only_the_action(tmp_path):
    """A `Condition` block containing `aws:` keys sits in the SAME statement
    as a real unbudgeted `Action` — exactly the action is flagged, and
    nothing about the condition is."""
    issues, _ = _graded(tmp_path, "infra/policy.json", (
        '{\n'
        '  "Statement": [\n'
        '    {\n'
        '      "Effect": "Allow",\n'
        '      "Action": "sagemaker:CreateEndpoint",\n'
        '      "Condition": {\n'
        '        "StringEquals": {"aws:SourceAccount": "000000000000"}\n'
        '      }\n'
        '    }\n'
        '  ]\n'
        '}\n'
    ))
    assert len(issues) == 1
    assert "sagemaker" in issues[0]


def test_a_non_parsing_iac_file_falls_back_loudly(tmp_path):
    """Malformed JSON must never read as zero actions found — the gate falls
    back to the line-level scan and COUNTS that it did, so the fallback is
    visible in the output rather than silent."""
    issues, counts = _graded(tmp_path, "infra/broken.json", (
        '{\n  "Statement": [\n    {"Action": "textract:DetectDocumentText"\n'
        # deliberately truncated / malformed JSON
    ))
    assert len(issues) == 1
    assert "textract" in issues[0]
    assert counts["actions_context_fallback"] == 1


def test_a_head_file_that_cannot_be_read_falls_back_too():
    """No ``root`` and no ``read_file`` — the same shape a caller hits when
    grading a diff without a checkout to read from."""
    issues, counts = gate.findings(
        _diff("infra/x.json", '        "textract:DetectDocumentText",'), DOC
    )
    assert len(issues) == 1
    assert "textract" in issues[0]
    assert counts["actions_context_fallback"] == 1


def test_a_test_fixture_json_naming_an_unbudgeted_action_is_not_graded(tmp_path):
    """The action class, like the client class, must not red the test file
    that proves it works."""
    issues, _ = _graded(tmp_path, "tests/fixtures/policy.json", (
        '{"Statement": [{"Action": "textract:DetectDocumentText"}]}\n'
    ))
    assert issues == []


def test_a_captured_policy_backup_json_is_not_a_new_grant(tmp_path):
    issues, _ = _graded(tmp_path, "private-docs/retirement/roles_backup_260920.json", (
        '{"Statement": [{"Action": "textract:DetectDocumentText"}]}\n'
    ))
    assert issues == []


# -- an SDK schedule call site is SYNTAX, not a token ------------------------
#
# Refs alpha-engine-config-I11289. The gate reported one permanently
# unresolvable schedule over 150 commits of this repo's own history, and the
# file it could not grade was THE GRADER'S OWN SOURCE: `grade.py` names
# `create_schedule`/`put_rule`/`put_targets`/`ScheduleExpression=` because
# those are the patterns it searches FOR, and the scanner matched its own
# pattern list. Same class as the `Condition`-key, component-id and
# `tag:Name` false positives above — a token graded outside the context that
# gives it meaning — so it is fixed the same way, structurally, and NOT with
# a path exclusion: a hardcoded `grade.py` denylist would blind the gate to a
# real schedule added to that file and would not cover the next module that
# names these strings.


_GRADE_PY = "src/nousergon_lib/cost_gate/grade.py"
_GRADE_PY_SOURCE = Path(gate.__file__).read_text()


def test_the_graders_own_pattern_list_is_not_a_schedule_call_site():
    """The regression that closes I11289's unidentified schedule: `grade.py`'s
    REAL source, added in full, grades with zero NOT-GRADED schedules. Uses
    the live file rather than a copy of its pattern list, so it keeps holding
    as that list changes."""
    issues, counts = gate.findings(
        _whole_file_diff(_GRADE_PY, _GRADE_PY_SOURCE),
        DOC, resource_map=RMAP,
        read_file=lambda path: _GRADE_PY_SOURCE if path == _GRADE_PY else None,
    )
    assert counts["ungraded_schedules"] == 0
    assert counts["ungraded"] == []
    assert issues == []


def test_a_real_put_rule_call_is_still_reported_ungraded(tmp_path):
    """The fail-closed half: a genuine call site in a file that parses
    perfectly is still NOT GRADED — its cadence and retry bound are not
    readable from Python source. Narrowing what counts as a call site must
    not narrow what happens to a real one."""
    issues, counts = _graded(
        tmp_path, "scripts/make_rule.py",
        'import boto3\n'
        'client = boto3.client("events")\n'
        'client.put_rule(Name="x", ScheduleExpression="rate(1 hour)")\n',
    )
    assert counts["ungraded_schedules"] == 1
    assert "scripts/make_rule.py" in counts["ungraded"][0]
    assert issues == []  # `events` is budgeted; the client class is separate


def test_a_bare_schedule_expression_assignment_is_still_a_call_site(tmp_path):
    """The `ScheduleExpression\\s*=` branch of the pre-filter was written for an
    assignment, not only a keyword argument. It keeps its meaning."""
    _, counts = _graded(
        tmp_path, "scripts/build.py", 'ScheduleExpression = "cron(0 3 * * ? *)"\n'
    )
    assert counts["ungraded_schedules"] == 1


def test_a_schedule_token_only_in_a_docstring_is_not_a_call_site(tmp_path):
    """The general case the self-match is one instance of: a module that
    merely NAMES these tokens — a doc example, a test fixture, a second
    grader — creates no schedule."""
    issues, counts = _graded(
        tmp_path, "scripts/docs.py",
        '"""Create one with `put_rule(ScheduleExpression=...)`."""\n'
        'PATTERNS = ("create_schedule", "update_schedule", "put_targets")\n',
    )
    assert counts["ungraded_schedules"] == 0
    assert issues == []


def test_a_python_file_that_will_not_parse_is_ungraded_not_passed(tmp_path):
    """FAIL-CLOSED. Syntax this cannot read must not read as "no schedule
    here" — the same rule the Action class applies on a parse failure."""
    _, counts = _graded(
        tmp_path, "scripts/broken.py", 'def f(:\n    put_rule(ScheduleExpression="x")\n'
    )
    assert counts["ungraded_schedules"] == 1


def test_a_python_head_file_that_cannot_be_read_is_ungraded_not_passed():
    """No `--root` and no `read_file`, so the question cannot be answered at
    all. Fail-closed, exactly as before this fix."""
    _, counts = gate.findings(
        _diff("scripts/x.py", '    c.create_schedule(ScheduleExpression="x")'),
        DOC, resource_map=RMAP,
    )
    assert counts["ungraded_schedules"] == 1


def test_a_call_site_the_diff_did_not_touch_is_not_regraded(tmp_path):
    """The span check is the same one the schedule and Action classes use: a
    diff that adds an unrelated line to a file holding a call site elsewhere
    has added no schedule."""
    target = tmp_path / "scripts/m.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        'import boto3\n'
        'c = boto3.client("events")\n'
        'c.put_rule(Name="x", ScheduleExpression="rate(1 hour)")\n'
        '# unrelated\n'
    )
    diff = (
        "diff --git a/scripts/m.py b/scripts/m.py\n"
        "--- a/scripts/m.py\n+++ b/scripts/m.py\n"
        "@@ -3,0 +4 @@\n+# the real put_rule(ScheduleExpression=...) call is on line 3, untouched\n"
    )
    _, counts = gate.findings(diff, DOC, resource_map=RMAP, root=tmp_path)
    assert counts["ungraded_schedules"] == 0


def test_parse_sdk_schedule_calls_refuses_unparseable_python():
    with pytest.raises(gate.SdkScheduleParseError):
        gate.parse_sdk_schedule_calls("def f(:\n")


# -- the FALLBACK line names the file too ------------------------------------
#
# Refs alpha-engine-config-I11289. The fallback notice told a caller that some
# JSON/YAML file in the diff had been graded by the weaker line-level scan and
# named no file, so the one action it asks for — fix that file's syntax —
# could not be taken. Measured live against a `krepis` replay, which printed
# `FALLBACK 1 JSON/YAML file(s)` and nothing else.


def test_the_fallback_detail_names_the_file_in_its_counts():
    issues, counts = gate.findings(
        _diff("infra/broken.json", '  "Action": "textract:DetectDocumentText",'),
        DOC, resource_map=RMAP,
        read_file=lambda path: '{"Action": [oops\n',
    )
    assert counts["actions_context_fallback"] == 1
    assert "infra/broken.json" in counts["fallback"][0]
    assert len(issues) == 1  # fail-closed: the line scan still grades it


def test_an_unreadable_head_file_is_named_in_the_fallback_detail():
    _, counts = gate.findings(
        _diff("infra/gone.json", '  "Action": "sts:AssumeRole",'),
        DOC, resource_map=RMAP,
    )
    assert counts["actions_context_fallback"] == 1
    assert "infra/gone.json" in counts["fallback"][0]
    assert "could not be read" in counts["fallback"][0]


def test_the_cli_prints_the_fallback_file_not_just_a_count(tmp_path, capsys):
    broken = tmp_path / "infra" / "broken.json"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text('{"Action": [oops\n')
    d = tmp_path / "x.diff"
    d.write_text(_diff("infra/broken.json", '  "Action": "textract:DetectDocumentText",'))
    cli.main(["--diff-file", str(d), "--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "FALLBACK" in out
    # Named on its OWN line under the FALLBACK notice — the finding text also
    # contains the path, so asserting only "the path appears somewhere" would
    # pass on the pre-fix output that named no file at all.
    assert "    ? infra/broken.json" in out


# -- the real-world shapes of the component-id false positive -----------------
#
# Refs alpha-engine-config-I11285. The synthetic fixture above uses an
# all-lowercase id (`pipeline:daily-close`), which does not even match the
# entitlement regex — it proves the pass is clean without exercising the
# disambiguation. These use the LITERAL strings from `nousergon-data`'s
# generated registry rows, whose suffix IS capitalised and therefore does
# match: `ne-weekly-freshness-pipeline:DataPhase1`, both as a bare YAML
# scalar and quoted inside a prose field.


def test_the_real_registry_row_component_id_shape_is_not_an_action(tmp_path):
    issues, counts = _graded(
        tmp_path, "registry.d/data-collector-d09-signal-returns.yaml",
        "component_id: data-collector-d09-signal-returns\n"
        "owning_repo: nousergon-data\n"
        "origin: ne-weekly-freshness-pipeline:DataPhase1\n"
        "log_location_reason: Owning pipeline 'ne-weekly-freshness-pipeline:DataPhase1'\n"
        "  (plan 4.4).\n"
        "alert_channel: sns:alpha-engine-alerts\n",
    )
    assert issues == []
    assert counts["actions_context_fallback"] == 0
