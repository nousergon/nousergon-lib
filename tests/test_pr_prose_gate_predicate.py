"""Tests for the `pr-prose-gate-guard` predicate (alpha-engine-config-I8683).

The guard exists because ``crucible-predictor-PR328`` merged on 2026-08-26
while its body said "Do not merge". Every case below is written against that
measurement or against the false positives the conjunction has to avoid.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github" / "scripts" / "pr_prose_gate_predicate.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("_prose_gate", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gate = _load()


# The verbatim opening of crucible-predictor-PR328's body.
PR328_BODY = (
    "> **DRAFT — validated on synthetic data; the closes-#1524 real-cohort "
    "run is DATA-GATED.** Everything here (estimator + kill-gate machinery "
    "+ synthetic separation) runs and is green now; what remains is running "
    "the *same* harness against our accrued champion/challenger cohorts "
    "once >=4 have realized on >=1 substrate. **Do not merge until that "
    "real-cohort run reports (PASS -> merge & close; FALSIFIED -> close "
    "won't-do per S0.6 + log the negative result).**"
)


class TestTheMeasuredIncident:
    def test_pr328_as_merged_is_caught(self):
        """Not a draft, zero labels, blocking prose — the live case."""
        passes, reason = gate.evaluate(PR328_BODY, [], is_draft=False)
        assert passes is False
        assert "gate:*" in reason
        assert "crucible-predictor-PR328" in reason

    def test_pr328_body_as_an_actual_draft_passes(self):
        """A draft cannot merge, so prose and machine state already agree."""
        passes, reason = gate.evaluate(PR328_BODY, [], is_draft=True)
        assert passes is True
        assert "Draft PR" in reason

    def test_pr328_body_with_a_gate_label_passes(self):
        """The whole ask is that the block be machine-readable. It is."""
        passes, reason = gate.evaluate(
            PR328_BODY, ["gate:data", "P1"], is_draft=False,
        )
        assert passes is True
        assert "gate:data" in reason


class TestTheConjunction:
    """Blocking prose ALONE is never a failure — that is the false-positive
    surface that would make the check unusable and get it removed."""

    def test_clean_body_passes(self):
        passes, _ = gate.evaluate(
            "Fixes the floor. Tests: 2008 passed.", [], is_draft=False,
        )
        assert passes is True

    def test_ordinary_prose_containing_draft_is_not_a_block(self):
        passes, _ = gate.evaluate(
            "I drafted the schema first, then wrote the migration. "
            "A draft of the RFC is linked below.",
            [], is_draft=False,
        )
        assert passes is True

    def test_a_retrospective_quoting_the_phrase_still_fails(self):
        """Deliberate, and stated: the check cannot tell a quotation from an
        instruction, so a PR whose body says 'do not merge' for ANY reason
        must carry a label. This is the conservative direction — the cost is
        one label, the cost of the other direction is PR328."""
        passes, _ = gate.evaluate(
            "Postmortem: the body said 'do not merge' and nothing enforced it.",
            [], is_draft=False,
        )
        assert passes is False


class TestPhraseShapes:
    @pytest.mark.parametrize("body", [
        "Do not merge until the data lands.",
        "DO NOT MERGE",
        "Don't merge this yet.",
        "Dont merge yet",
        "This is not for merge.",
        "Do not land this before Tuesday.",
        "Hold this until the ruling.",
        "Hold off — waiting on the backfill.",
        "**DRAFT — needs the real cohort**",
        "> DRAFT: pending review",
        "The estimator is DATA-GATED on >=4 cohorts.",
    ])
    def test_blocking_shapes_are_caught(self, body):
        passes, _ = gate.evaluate(body, [], is_draft=False)
        assert passes is False

    @pytest.mark.parametrize("body", [
        "Merge order: nousergon-lib-PR355 first.",
        "Ready to merge once CI is green.",
        "The holder object is reused across calls.",
        "Withholding judgement on the second estimator.",
    ])
    def test_near_misses_pass(self, body):
        passes, _ = gate.evaluate(body, [], is_draft=False)
        assert passes is True

    def test_markdown_furniture_does_not_hide_a_leading_draft(self):
        """`> **DRAFT — ...` is how these warnings are actually written; an
        anchored `^draft` only works if the furniture is stripped first."""
        assert gate.find_blocking_phrase("> **DRAFT — blocked**") is not None
        assert gate.find_blocking_phrase("  - *draft: blocked*") is not None


class TestEmptyAndDegenerateInput:
    def test_empty_body_passes(self):
        assert gate.evaluate("", [], is_draft=False)[0] is True

    def test_none_body_passes(self):
        assert gate.evaluate(None, [], is_draft=False)[0] is True

    def test_gate_label_check_is_prefix_anchored(self):
        """`investigate:gated` is not a gate:* label."""
        passes, _ = gate.evaluate(
            "do not merge", ["investigate:gated"], is_draft=False,
        )
        assert passes is False


class TestTheScriptEntryPoint:
    def _run(self, tmp_path, payload):
        target = tmp_path / "pr.json"
        target.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(_SCRIPT), str(target)],
            capture_output=True, text=True,
        )

    def test_exit_1_and_an_error_annotation_on_a_block(self, tmp_path):
        proc = self._run(tmp_path, {
            "body": PR328_BODY, "labels": [], "isDraft": False,
        })
        assert proc.returncode == 1
        assert "::error::" in proc.stdout

    def test_exit_0_and_a_notice_on_a_pass(self, tmp_path):
        proc = self._run(tmp_path, {
            "body": "clean", "labels": [], "isDraft": False,
        })
        assert proc.returncode == 0
        assert "::notice::" in proc.stdout

    def test_label_objects_from_the_event_payload_are_understood(self, tmp_path):
        """A `pull_request` payload carries labels as objects, `gh pr view`
        as objects too — both shapes must resolve to names."""
        proc = self._run(tmp_path, {
            "body": PR328_BODY,
            "labels": [{"name": "gate:data"}],
            "isDraft": False,
        })
        assert proc.returncode == 0

    def test_draft_key_spelling_from_the_event_payload(self, tmp_path):
        """The event payload spells it `draft`; `gh pr view` spells it
        `isDraft`. Both must be honoured or the guard fails every draft."""
        proc = self._run(tmp_path, {
            "body": PR328_BODY, "labels": [], "draft": True,
        })
        assert proc.returncode == 0

    def test_unreadable_input_fails_closed(self, tmp_path):
        proc = subprocess.run(
            [sys.executable, str(_SCRIPT), str(tmp_path / "absent.json")],
            capture_output=True, text=True,
        )
        assert proc.returncode == 1
        assert "Failing closed" in proc.stdout

    def test_malformed_json_fails_closed(self, tmp_path):
        target = tmp_path / "pr.json"
        target.write_text("{not json", encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(_SCRIPT), str(target)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 1
        assert "Failing closed" in proc.stdout

    def test_missing_argument_fails_closed(self):
        proc = subprocess.run(
            [sys.executable, str(_SCRIPT)], capture_output=True, text=True,
        )
        assert proc.returncode == 1
