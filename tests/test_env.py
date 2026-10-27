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
    assert env.environment_names() == ["prod", "staging"]
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


# ------------------------------------------------------------------ staging
# Declared ahead of its account and stack (S-03 second slice); no consumer
# resolves it until NE_ENV=staging is set. These pin the isolation the plan
# promises (ENV-PROD-OWNS-DATA) without naming any literal here.


def test_staging_is_a_separate_account_in_the_same_region():
    prod, staging = env.resolve("prod"), env.resolve("staging")
    assert re.fullmatch(r"[0-9]{12}", staging.account)
    assert staging.account != prod.account
    assert staging.region == prod.region


def test_staging_writes_only_its_own_bucket_and_reads_prod_data():
    prod, staging = env.resolve("prod"), env.resolve("staging")
    prod_buckets = {prod.bucket(r) for r in prod.roles("buckets")}
    # the crucible store is where staging writes; it must never be a prod bucket
    assert staging.bucket("crucible") not in prod_buckets
    assert staging.s3_uri("crucible").startswith(f"s3://{staging.bucket('crucible')}/")
    # the data of record is prod's, read-only: same research bucket and ArcticDB root
    assert staging.bucket("research") == prod.bucket("research")
    assert staging.s3_uri("arcticdb") == prod.s3_uri("arcticdb")
    # staging's universe lives in its own bucket, and is a subset
    assert staging.universe_scope == "subset"
    assert staging.universe_uri.startswith(f"s3://{staging.bucket('crucible')}/")


def test_staging_pages_are_muted_and_land_in_the_staging_account():
    staging = env.resolve("staging")
    assert staging.topic("pages") == staging.topic("pages_muted")
    assert staging.topic("pages").endswith("-muted")
    assert f":{staging.account}:" in staging.topic_arn("pages")


def test_staging_data_collection_stack_and_alerts_topic_resolve_muted():
    """alpha-engine-config-I12646: the S-18 parameters of nousergon-data-collection derive."""
    staging = env.resolve("staging")
    assert staging.stack("data_collection") == "nousergon-data-collection"
    assert staging.topic("alerts").endswith("-muted")
    assert staging.topic("alerts") == staging.topic("pages_muted")
    assert staging.topic_arn("alerts").startswith(f"arn:aws:sns:{staging.region}:{staging.account}:")


def test_staging_has_no_fixed_boxes_and_a_lower_llm_cap():
    prod, staging = env.resolve("prod"), env.resolve("staging")
    assert staging.roles("instances") == []
    assert 0 < staging.llm_cap_usd < prod.llm_cap_usd


def test_staging_schedules_are_offset_from_prod():
    prod, staging = env.resolve("prod"), env.resolve("staging")
    for role in staging.roles("schedules_et"):
        assert role in prod.roles("schedules_et"), f"staging schedule {role!r} has no prod counterpart"
        assert staging.schedule_et(role) != prod.schedule_et(role), f"{role} runs at the same time as prod"


def test_staging_lacks_prod_only_roles_and_says_so():
    staging = env.resolve("staging")
    with pytest.raises(env.UnknownRoleError, match="declared: crucible"):
        staging.stack("orchestration")
    with pytest.raises(env.UnknownRoleError):
        staging.instance("trading")


# ------------------------------------------------------------ fails loudly


def test_unknown_environment_from_argument_raises():
    with pytest.raises(env.UnknownEnvironmentError, match=r"'qa' \(from argument\).*declared: prod, staging"):
        env.resolve("qa")


def test_unknown_environment_from_ne_env_raises(monkeypatch):
    monkeypatch.setenv(env.ENV_VAR, "qa")
    with pytest.raises(env.UnknownEnvironmentError, match=r"\$NE_ENV"):
        env.resolve()


def test_staging_never_routes_and_prod_routes_to_ib_paper_only():
    """alpha-engine-config-I12685: the registry, not only the trader's NE_ENV
    branch, forbids ib_paper in staging."""
    staging = env.resolve("staging")
    assert staging.routing_allowed == ()
    with pytest.raises(env.RoutingNotAllowedError, match="never routes"):
        staging.assert_routing_allowed("ib_paper")

    prod = env.resolve("prod")
    assert prod.routing_allowed == ("ib_paper",)
    prod.assert_routing_allowed("ib_paper")
    with pytest.raises(env.RoutingNotAllowedError, match="ib_live"):
        prod.assert_routing_allowed("ib_live")


def test_every_declared_environment_states_its_routing():
    for name in env.environment_names():
        assert isinstance(env.resolve(name).routing_allowed, tuple)


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
        (lambda d: d["environments"]["prod"].pop("trader"), "trader"),
        (lambda d: d["environments"]["prod"]["trader"].pop("routing_allowed"), "routing_allowed"),
        (lambda d: d["environments"]["prod"]["trader"].update(routing_allowed=["IB-Paper"]), "IB-Paper"),
        (lambda d: d["environments"]["prod"]["trader"].update(routing_allowed=["ib_paper", "ib_paper"]), "repeats"),
        (lambda d: d["environments"]["prod"]["trader"].update(routing_allowed="ib_paper"), "routing_allowed"),
        (lambda d: d["environments"]["prod"]["trader"].update(extra="x"), "extra"),
        (lambda d: d.update(default="qa"), "default"),
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


def test_parity_catches_a_missing_environment_in_either_direction(tmp_path):
    # governance file lost staging -> only in the packaged copy
    doc = _doc()
    doc["environments"].pop("staging")
    diffs = env.diff_against_packaged(_write(tmp_path, doc))
    assert diffs == ["environments.staging: only in the packaged copy"]
    # governance file gained an environment the library does not ship -> only in the file
    doc = _doc()
    doc["environments"]["qa"] = copy.deepcopy(doc["environments"]["staging"])
    path = _write(tmp_path, doc)
    assert env.diff_against_packaged(path) == [f"environments.qa: only in {path}"]


def test_parity_catches_a_staging_value_drift(tmp_path):
    doc = _doc()
    doc["environments"]["staging"]["llm_cap_usd"] = 50.0
    doc["environments"]["staging"]["topics"]["pages"] = "crucible-v2-pages"
    diffs = env.diff_against_packaged(_write(tmp_path, doc))
    assert any(d.startswith("environments.staging.llm_cap_usd:") for d in diffs)
    assert any(d.startswith("environments.staging.topics.pages:") for d in diffs)


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
