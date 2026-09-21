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
