"""Tests for the spot-bootstrap shell guards (alpha-engine-config#6916).

Each case is the real production defect the check exists for, reduced to the
smallest script that reproduces it — plus the legitimate shape that must NOT be
flagged, since a guard that fires on correct code gets disabled.
"""

from __future__ import annotations

from nousergon_lib.shell_guards import (
    embedded_units,
    endless_execstart_violations,
    unprovided_binary_violations,
    unsupplied_variable_violations,
)

_ENDLESS_ONESHOT = """
cat > /tmp/w.service <<'UNIT'
[Unit]
Description=Watchdog
[Service]
Type=oneshot
ExecStart=/usr/local/bin/w.sh
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
UNIT
cat > /usr/local/bin/w.sh <<'WDSH'
#!/usr/bin/env bash
while true; do
  sleep 60
done
WDSH
systemctl enable --now w
"""

_ENDLESS_SIMPLE = _ENDLESS_ONESHOT.replace("Type=oneshot", "Type=simple")

_GENUINE_ONESHOT = """
cat > /tmp/pull.service <<'UNIT'
[Unit]
Description=Boot pull
[Service]
Type=oneshot
ExecStart=/usr/local/bin/pull.sh
[Install]
WantedBy=multi-user.target
UNIT
cat > /usr/local/bin/pull.sh <<'PSH'
#!/usr/bin/env bash
git -C /srv/app pull --ff-only
PSH
"""


# ── embedded_units ───────────────────────────────────────────────────────────

def test_units_are_found_with_their_type_and_execstart():
    units = embedded_units(_ENDLESS_ONESHOT)
    assert len(units) == 1
    assert units[0].service_type == "oneshot"
    assert units[0].execstart == "/usr/local/bin/w.sh"


def test_a_unit_without_an_explicit_type_defaults_to_simple():
    text = _ENDLESS_ONESHOT.replace("Type=oneshot\n", "")
    assert embedded_units(text)[0].service_type == "simple"


def test_a_script_with_no_units_yields_none():
    assert embedded_units("echo hello") == []


# ── endless_execstart_violations ─────────────────────────────────────────────

def test_an_endless_loop_under_oneshot_is_flagged():
    violations = endless_execstart_violations(_ENDLESS_ONESHOT)
    assert len(violations) == 1
    assert "Type=simple" in violations[0]


def test_the_same_loop_under_simple_is_clean():
    assert endless_execstart_violations(_ENDLESS_SIMPLE) == []


def test_a_genuinely_oneshot_job_is_not_flagged():
    """A boot-time pull is correctly oneshot — flagging it would train people
    to ignore this check."""
    assert endless_execstart_violations(_GENUINE_ONESHOT) == []


def test_an_execstart_written_elsewhere_is_not_guessed_at():
    text = _ENDLESS_ONESHOT.split("cat > /usr/local/bin/w.sh")[0]
    assert endless_execstart_violations(text) == []


# ── unprovided_binary_violations ─────────────────────────────────────────────

def test_a_fatal_guard_with_no_install_is_flagged():
    block = 'command -v python3.12 >/dev/null || { echo "missing" >&2; exit 1; }'
    violations = unprovided_binary_violations(block)
    assert len(violations) == 1
    assert "python3.12" in violations[0]


def test_an_install_before_the_guard_clears_it():
    block = (
        "dnf install -y -q python3.12 python3.12-pip git gcc\n"
        'command -v python3.12 >/dev/null || { echo "missing" >&2; exit 1; }'
    )
    assert unprovided_binary_violations(block) == []


def test_an_install_AFTER_the_guard_does_not_clear_it():
    """Order matters — the guard runs first and exits."""
    block = (
        'command -v python3.12 >/dev/null || { echo "missing" >&2; exit 1; }\n'
        "dnf install -y -q python3.12\n"
    )
    assert len(unprovided_binary_violations(block)) == 1


def test_a_non_fatal_probe_is_not_a_guard():
    """`command -v X && A || B` chooses a path; it does not assert."""
    block = "command -v python3.12 >/dev/null && PY=python3.12 || PY=python3"
    assert unprovided_binary_violations(block) == []


# ── unsupplied_variable_violations ───────────────────────────────────────────

def test_a_variable_nobody_supplies_is_flagged():
    body = 'git clone --branch "${BRANCH:-main}" "${REPO_URL}" /srv/app'
    violations = unsupplied_variable_violations(body, {"BRANCH"})
    assert len(violations) == 1
    assert "REPO_URL" in violations[0]


def test_an_exported_variable_is_clean():
    body = 'git clone "${REPO_URL}" /srv/app'
    assert unsupplied_variable_violations(body, {"REPO_URL"}) == []


def test_a_defaulted_read_is_clean():
    body = 'git clone --branch "${BRANCH:-main}" /srv/app'
    assert unsupplied_variable_violations(body, set()) == []


def test_a_variable_assigned_in_the_body_is_clean():
    body = 'REPO_URL=https://example.invalid/x.git\ngit clone "${REPO_URL}" /srv/app'
    assert unsupplied_variable_violations(body, set()) == []


def test_a_variable_exported_inline_in_the_body_is_clean():
    body = 'export HOME=/home/ec2-user FOO=bar\necho "$FOO"'
    assert unsupplied_variable_violations(body, set()) == []


def test_runtime_ambient_variables_are_not_demanded_of_the_launcher():
    body = 'echo "$HOME $AWS_REGION $PATH"'
    assert unsupplied_variable_violations(body, set()) == []


def test_bare_dollar_reads_are_covered_too():
    body = "git clone $REPO_URL /srv/app"
    assert len(unsupplied_variable_violations(body, set())) == 1


def test_violations_are_sorted_for_stable_output():
    body = 'echo "${ZULU}${ALPHA}${MIKE}"'
    violations = unsupplied_variable_violations(body, set())
    names = [v.split("${")[1].split("}")[0] for v in violations]
    assert names == sorted(names)
