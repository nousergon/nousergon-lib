"""Tests for the universe-scope CI lint (alpha-engine-config#5809 deliverable 3, #11142)."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from nousergon_lib.contracts.universe_scope_lint import main, scan_tree


def test_signals_schema_universe_has_a_machine_checkable_bound() -> None:
    # I5809 deliverable 3: a cardinality note or bound on `universe`. The field is
    # genuinely board-width (no fixed decision-set size), so this is a sanity
    # ceiling against a runaway/duplicated write, not a scope cap — see the
    # schema's own description on `properties.universe` for why it must never be
    # lowered to approximate a decision set.
    schema_text = resources.files("nousergon_lib.contracts").joinpath(
        "signals.schema.json"
    ).read_text(encoding="utf-8")
    schema = json.loads(schema_text)
    universe = schema["properties"]["universe"]
    assert "maxItems" in universe, "universe must carry a machine-checkable cardinality bound"
    assert universe["maxItems"] >= 900, "bound must not be tighter than today's board width"


def _write(root: Path, rel: str, content: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_flags_subscript_read_on_signals_shaped_name(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "model/research_features.py",
        "def f(signals_payload, ticker):\n"
        "    ticker_sig = next(\n"
        "        (s for s in signals_payload.get(\"universe\", []) if s.get(\"ticker\") == ticker),\n"
        "        None,\n"
        "    )\n"
        "    return ticker_sig\n",
    )
    hits = scan_tree(tmp_path)
    assert len(hits) == 1
    assert hits[0].path == Path("model/research_features.py")
    assert "universe" in hits[0].snippet


def test_flags_bracket_subscript_too(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "loaders/signal_loader.py",
        "def f(signals_data):\n"
        "    return signals_data[\"universe\"]\n",
    )
    hits = scan_tree(tmp_path)
    assert len(hits) == 1


def test_does_not_flag_unrelated_variable_named_universe_key(tmp_path: Path) -> None:
    # A dict keyed "universe" that isn't a signals payload at all (e.g. an ArcticDB
    # library map) must not false-fire off an unrelated base-variable name.
    _write(
        tmp_path,
        "collectors/arctic_probe.py",
        "def f(libs):\n"
        "    return libs[\"universe\"]\n",
    )
    hits = scan_tree(tmp_path)
    assert hits == []


def test_ambiguous_name_needs_context_hint(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "a/no_hint.py",
        "def f(data):\n"
        "    return data.get(\"universe\", [])\n",
    )
    _write(
        tmp_path,
        "b/with_hint.py",
        "def f(data):\n"
        "    # signals payload\n"
        "    return data.get(\"universe\", [])\n",
    )
    hits = scan_tree(tmp_path)
    assert len(hits) == 1
    assert hits[0].path == Path("b/with_hint.py")


def test_executor_sizing_exit_path_is_allowlisted(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "crucible-executor/executor/deciders.py",
        "def f(signals_raw):\n"
        "    return signals_raw.get(\"universe\", [])\n",
    )
    hits = scan_tree(tmp_path)
    assert hits == []


def test_envelope_producer_is_allowlisted(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "crucible-research/scoring/signals_envelope.py",
        "def f(envelope):\n"
        "    return len(envelope[\"universe\"])\n",
    )
    hits = scan_tree(tmp_path)
    assert hits == []


def test_inline_exemption_comment_on_hit_line_suppresses(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "scoring/boost_signals.py",
        "def f(signals_payload):\n"
        "    universe = signals_payload.get(\"universe\") or []  "
        "# universe-scope-lint: allow -- reason=reviewed board-width fanout, ADR-42\n"
        "    return universe\n",
    )
    hits = scan_tree(tmp_path)
    assert hits == []


def test_inline_exemption_on_preceding_line_suppresses(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "scoring/morning_brief.py",
        "def f(envelope):\n"
        "    # universe-scope-lint: allow -- reason=board-width digest, reviewed 2026-09-19\n"
        "    return envelope.get(\"universe\") or []\n",
    )
    hits = scan_tree(tmp_path)
    assert hits == []


def test_exemption_without_reason_does_not_suppress(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "scoring/boost_signals.py",
        "def f(signals_payload):\n"
        "    return signals_payload.get(\"universe\") or []  "
        "# universe-scope-lint: allow -- reason=\n",
    )
    hits = scan_tree(tmp_path)
    assert len(hits) == 1


def test_tests_excluded_by_default_but_includable(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/test_something.py",
        "def f(signals_payload):\n"
        "    return signals_payload[\"universe\"]\n",
    )
    assert scan_tree(tmp_path) == []
    assert scan_tree(tmp_path, include_tests=True) != []


def test_main_returns_nonzero_and_prints_hits(tmp_path: Path, capsys) -> None:
    _write(
        tmp_path,
        "inference/stages/load_universe.py",
        "def f(ctx):\n"
        "    return ctx.signals_data.get(\"universe\") or []\n",
    )
    rc = main([str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "load_universe.py" in out


def test_main_returns_zero_when_clean(tmp_path: Path, capsys) -> None:
    _write(tmp_path, "clean.py", "x = 1\n")
    rc = main([str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no unexempted" in out
