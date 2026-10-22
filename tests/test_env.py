"""nousergon_lib.env — the environment registry resolver (alpha-engine-config-I12491)."""

from __future__ import annotations

import copy
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from nousergon_lib import env

PACKAGED = env.PACKAGED_REGISTRY


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(env.ENV_VAR, raising=False)
    monkeypatch.delenv(env.REGISTRY_VAR, raising=False)


def _doc() -> dict:
    return yaml.safe_load(PACKAGED.read_text())


def _write(tmp_path: Path, doc: dict) -> Path:
    p = tmp_path / "environments.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


# ---------------------------------------------------------- packaged registry


def test_packaged_registry_is_valid_and_defaults_to_prod():
    reg = env.load_registry()
    assert reg.default == "prod"
    assert env.environment_names() == ["prod"]
    assert env.current_name() == "prod"


def test_packaged_registry_ships_in_the_wheel_manifest():
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert '"environments.yaml"' in text


def test_resolve_defaults_to_prod():
    e = env.resolve()
    assert e.name == "prod"
    assert re.fullmatch(r"[0-9]{12}", e.account)
    assert e.region == "us-east-1"
    assert e.bucket("research")
    assert e.bucket("crucible")


def test_derived_values_compose_from_declared_parts():
    e = env.resolve("prod")
    assert e.topic_arn("pages") == f"arn:aws:sns:{e.region}:{e.account}:{e.topic('pages')}"
    assert e.s3_uri("crucible") == f"s3://{e.bucket('crucible')}/crucible"
    assert e.universe_uri.startswith(f"s3://{e.bucket('research')}/")
    assert e.schedule_et("weekly") == ("SAT", "07:00")
    assert e.neon_branch is None
    assert e.llm_cap_usd == 5.0
    assert e.universe_scope == "full"
    assert "research" in e.roles("buckets")


# ------------------------------------------------------------ fails loudly


def test_unknown_environment_from_argument_raises():
    with pytest.raises(env.UnknownEnvironmentError, match=r"'staging' \(from argument\).*declared: prod"):
        env.resolve("staging")


def test_unknown_environment_from_ne_env_raises(monkeypatch):
    monkeypatch.setenv(env.ENV_VAR, "qa")
    with pytest.raises(env.UnknownEnvironmentError, match=r"\$NE_ENV"):
        env.resolve()


def test_ne_env_selects_the_environment(monkeypatch, tmp_path):
    doc = _doc()
    staging = copy.deepcopy(doc["environments"]["prod"])
    staging["account"] = "000000000000"
    doc["environments"]["staging"] = staging
    monkeypatch.setenv(env.REGISTRY_VAR, str(_write(tmp_path, doc)))
    monkeypatch.setenv(env.ENV_VAR, "staging")
    assert env.current_name() == "staging"
    e = env.resolve()
    assert (e.name, e.account) == ("staging", "000000000000")
    # an explicit argument wins over NE_ENV
    assert env.resolve("prod").name == "prod"


@pytest.mark.parametrize(
    "accessor", ["bucket", "stack", "topic", "topic_arn", "instance", "ssm_prefix", "s3_uri", "schedule_et"]
)
def test_unknown_role_raises(accessor):
    e = env.resolve()
    with pytest.raises(env.UnknownRoleError, match="declares no"):
        getattr(e, accessor)("no_such_role")


def test_unknown_role_section_raises():
    with pytest.raises(env.UnknownRoleError):
        env.resolve().roles("account")


# ------------------------------------------------------------------- schema


def _mutated(mutate) -> dict:
    doc = _doc()
    mutate(doc)
    return doc


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda d: d.update(surprise=1), "surprise"),
        (lambda d: d["environments"]["prod"].update(colour="blue"), "colour"),
        (lambda d: d["environments"]["prod"]["state_store"].update(extra="x"), "extra"),
        (lambda d: d["environments"]["prod"].pop("region"), "region"),
        (lambda d: d["environments"]["prod"]["state_store"].pop("neon_branch"), "neon_branch"),
        (lambda d: d["environments"]["prod"].update(account="12345"), "account"),
        (lambda d: d["environments"]["prod"].update(account=711398986525), "account"),
        (lambda d: d["environments"]["prod"].update(region="US-EAST-1"), "region"),
        (lambda d: d["environments"]["prod"]["buckets"].update(bad="Not_A_Bucket"), "buckets.bad"),
        (lambda d: d["environments"]["prod"]["buckets"].update({"Bad-Role": "ok-bucket"}), "Bad-Role"),
        (lambda d: d["environments"]["prod"]["instances"].update(box="vm-1"), "instances.box"),
        (lambda d: d["environments"]["prod"]["ssm_prefixes"].update(x="no-slash"), "ssm_prefixes.x"),
        (lambda d: d["environments"]["prod"]["stores"]["crucible"].update(bucket="nope"), "not a declared bucket"),
        (lambda d: d["environments"]["prod"]["universe"].update(bucket="nope"), "not a declared bucket"),
        (lambda d: d["environments"]["prod"]["universe"].update(scope="most"), "scope"),
        (lambda d: d["environments"]["prod"]["schedules_et"]["weekly"].update(time_et="7:00"), "time_et"),
        (lambda d: d["environments"]["prod"]["schedules_et"]["weekly"].update(days="SATURDAY"), "days"),
        (lambda d: d["environments"]["prod"].update(llm_cap_usd=-1), "llm_cap_usd"),
        (lambda d: d.update(default="staging"), "default"),
        (lambda d: d.update(schema_version=2), "schema_version"),
        (lambda d: d.update(environments={}), "environments"),
        (lambda d: d["environments"].update({"Prod": d["environments"]["prod"]}), "Prod"),
    ],
)
def test_schema_rejects(mutate, needle, tmp_path):
    path = _write(tmp_path, _mutated(mutate))
    with pytest.raises(env.RegistrySchemaError, match=re.escape(needle)):
        env.load_registry(path)


def test_non_mapping_registry_is_rejected(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("- just\n- a list\n")
    with pytest.raises(env.RegistrySchemaError, match="mapping"):
        env.load_registry(p)


def test_invalid_yaml_and_missing_file_raise(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("a: [unclosed\n")
    with pytest.raises(env.RegistrySchemaError, match="not valid YAML"):
        env.load_registry(p)
    with pytest.raises(env.EnvRegistryError, match="cannot read"):
        env.load_registry(tmp_path / "missing.yaml")


def test_registry_override_path_must_exist(monkeypatch, tmp_path):
    monkeypatch.setenv(env.REGISTRY_VAR, str(tmp_path / "missing.yaml"))
    with pytest.raises(env.EnvRegistryError):
        env.resolve()


# ------------------------------------------------------------------- parity


def test_parity_ignores_comments_and_formatting(tmp_path):
    assert env.diff_against_packaged(_write(tmp_path, _doc())) == []


def test_parity_reports_each_difference(tmp_path):
    doc = _doc()
    doc["environments"]["prod"]["buckets"]["research"] = "other-bucket"
    doc["environments"]["prod"]["topics"].pop("watchdog")
    diffs = env.diff_against_packaged(_write(tmp_path, doc))
    assert any("buckets.research" in d and "other-bucket" in d for d in diffs)
    assert any("topics.watchdog: only in the packaged copy" in d for d in diffs)


# ----------------------------------------------------------------------- CLI


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "nousergon_lib.env", *args], capture_output=True, text=True, check=False
    )


def test_cli_check_parity_show(tmp_path):
    good = _write(tmp_path, _doc())
    assert _cli("check", str(good)).returncode == 0
    assert _cli("parity", str(good)).returncode == 0
    shown = _cli("show")
    assert shown.returncode == 0 and '"name": "prod"' in shown.stdout

    drifted = _doc()
    drifted["environments"]["prod"]["region"] = "eu-west-1"
    bad = tmp_path / "drift.yaml"
    bad.write_text(yaml.safe_dump(drifted))
    r = _cli("parity", str(bad))
    assert r.returncode == 1 and "region" in r.stderr

    assert _cli("show", "nope").returncode == 2
