"""Tests for the install-start-dependency parser (alpha-engine-config-I9099).

Each case is either a real defect this parser was written for or a real shape in
the fleet's installers that must NOT be flagged — a guard that fires on correct
code gets disabled, which is worse than not having it.

The two seeded-regression directions are pinned here because the causal chain
has two links and breaking EITHER is a valid fix, so a test that only covers one
would call a half-fixed repo clean:

    installer says `enable --now <timer>` / `start <unit>`
        -> the unit's [Unit] dependencies pull in <service>
            -> <service> is a scheduled workload and runs off-schedule
"""

from __future__ import annotations

from nousergon_lib.systemd_install_guard import (
    directive,
    load_units,
    may_start,
    scheduled_workloads,
    sections,
    start_closure,
    started_units,
    triggered_service,
    violations,
)

_TIMER_CLEAN = """\
[Unit]
Description=Arm the widget
[Timer]
OnCalendar=*-*-* 04:00:00
[Install]
WantedBy=timers.target
"""

_TIMER_WITH_REQUIRES = """\
[Unit]
Description=Arm the widget
Requires=widget.service
[Timer]
OnCalendar=*-*-* 04:00:00
[Install]
WantedBy=timers.target
"""

_WORKLOAD = """\
[Unit]
Description=The widget job
[Service]
Type=oneshot
ExecStart=/usr/local/bin/widget.sh
"""

_DAEMON = """\
[Unit]
Description=A long-running thing
[Service]
Type=simple
ExecStart=/usr/local/bin/daemon.sh
"""


def _repo(tmp_path, units, script):
    """A minimal repo: `systemd/` unit files plus one installer script."""
    systemd = tmp_path / "systemd"
    systemd.mkdir(exist_ok=True)
    for name, text in units.items():
        path = systemd / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    installer = tmp_path / "install-widget.sh"
    installer.write_text(script, encoding="utf-8")
    return load_units(systemd), installer


# --------------------------------------------------------------------------
# the two seeded-regression directions
# --------------------------------------------------------------------------


def test_a_requires_edge_on_the_timer_is_the_violation(tmp_path):
    """Link 2: arming a clean-looking timer pulls in its own workload."""
    units, installer = _repo(
        tmp_path,
        {"widget.timer": _TIMER_WITH_REQUIRES, "widget.service": _WORKLOAD},
        "sudo systemctl enable --now widget.timer\n",
    )
    assert violations(installer, units) == {"widget.timer": {"widget.service"}}


def test_dropping_the_requires_edge_clears_it(tmp_path):
    """The same installer against the fixed timer is clean."""
    units, installer = _repo(
        tmp_path,
        {"widget.timer": _TIMER_CLEAN, "widget.service": _WORKLOAD},
        "sudo systemctl enable --now widget.timer\n",
    )
    assert violations(installer, units) == {}


def test_a_direct_start_of_the_workload_is_the_violation(tmp_path):
    """Link 1: clean timers do not save an installer that starts the service."""
    units, installer = _repo(
        tmp_path,
        {"widget.timer": _TIMER_CLEAN, "widget.service": _WORKLOAD},
        "sudo systemctl enable --now widget.timer\nsudo systemctl start widget.service\n",
    )
    assert violations(installer, units) == {"widget.service": {"widget.service"}}


def test_the_escape_hatch_lives_in_the_unit(tmp_path):
    """`X-InstallMayStart=yes` in the service's own [Unit] clears the start."""
    allowed = _WORKLOAD.replace(
        "Description=The widget job",
        "Description=The widget job\n# deliberate: the box needs one run to seed its cache\nX-InstallMayStart=yes",
    )
    units, installer = _repo(
        tmp_path,
        {"widget.timer": _TIMER_CLEAN, "widget.service": allowed},
        "sudo systemctl start widget.service\n",
    )
    assert may_start("widget.service", units) is True
    assert violations(installer, units) == {}


# --------------------------------------------------------------------------
# the divergence the lift closes: shell noise and printed guidance
# --------------------------------------------------------------------------


def test_a_redirection_is_not_a_unit_name(tmp_path):
    """`crucible-dashboard-PR792` reported `>/dev/null.service` and `2>.service`.

    `_START_RE` stops at the `&` of `2>&1`, so a naive split hands `>/dev/null`
    and `2>` to the normaliser. They can never match a real unit and so change
    no verdict — but garbage in a failure message is how a guard loses the
    argument the day it fires for real.
    """
    units, installer = _repo(
        tmp_path,
        {"widget.timer": _TIMER_CLEAN, "widget.service": _WORKLOAD},
        "sudo systemctl enable --now widget.timer >/dev/null 2>&1 || true\n",
    )
    assert started_units(installer) == {"widget.timer"}


def test_a_full_line_comment_is_not_an_execution(tmp_path):
    """The real line from `nousergon-data/infrastructure/install-metron-intraday.sh`.

    A parser that reads it as an execution path reports a unit named `without`.
    """
    units, installer = _repo(
        tmp_path,
        {"widget.timer": _TIMER_CLEAN, "widget.service": _WORKLOAD},
        "# This script used to copy units and `systemctl enable --now` without\n"
        "# ever checking the result.\n"
        "sudo systemctl enable --now widget.timer\n",
    )
    assert started_units(installer) == {"widget.timer"}


def test_a_trailing_comment_does_not_hide_the_execution(tmp_path):
    """Only FULL-line comments are stripped.

    Truncating at any `#` would turn this file's false-positive fix into a false
    negative — the failure mode that is invisible rather than noisy.
    """
    units, installer = _repo(
        tmp_path,
        {"widget.timer": _TIMER_CLEAN, "widget.service": _WORKLOAD},
        "sudo systemctl start widget.service  # seed the cache\n",
    )
    assert started_units(installer) == {"widget.service"}


def test_an_echoed_systemctl_is_printed_guidance(tmp_path):
    """Installers print "Run now: ..." at the end. That is not a start."""
    units, installer = _repo(
        tmp_path,
        {"widget.timer": _TIMER_CLEAN, "widget.service": _WORKLOAD},
        'echo "Run now: sudo systemctl start widget.service"\n'
        'printf "or: systemctl restart widget.service\\n"\n'
        'say "systemctl start widget.service"\n'
        "sudo systemctl enable --now widget.timer\n",
    )
    assert started_units(installer) == {"widget.timer"}


def test_a_variable_unit_name_is_skipped_and_the_gap_is_real(tmp_path):
    """`nous-ergon-ops`'s `install-box-config.sh` arms timers through `$_timer`.

    Skipping is deliberate — the alternative is inventing a unit name — and it
    is a named scope gap, not an oversight. Pinned so a future change that
    starts emitting `$_timer.service` fails here rather than in a report.
    """
    units, installer = _repo(
        tmp_path,
        {"widget.timer": _TIMER_WITH_REQUIRES, "widget.service": _WORKLOAD},
        'for _timer in widget; do sudo systemctl enable --now "${_timer}.timer"; done\n',
    )
    assert started_units(installer) == set()
    assert violations(installer, units) == {}


def test_enable_without_now_arms_but_does_not_start(tmp_path):
    """`enable` alone is the documented fix, so it must not be flagged."""
    units, installer = _repo(
        tmp_path,
        {"widget.timer": _TIMER_WITH_REQUIRES, "widget.service": _WORKLOAD},
        "sudo systemctl enable widget.timer\nsudo systemctl daemon-reload\n",
    )
    assert started_units(installer) == set()


def test_reenable_rewrites_symlinks_without_activating(tmp_path):
    units, installer = _repo(
        tmp_path,
        {"widget.timer": _TIMER_WITH_REQUIRES, "widget.service": _WORKLOAD},
        "sudo systemctl reenable widget.timer\n",
    )
    assert started_units(installer) == set()


def test_restart_and_try_restart_count(tmp_path):
    units, installer = _repo(
        tmp_path,
        {"widget.timer": _TIMER_CLEAN, "widget.service": _WORKLOAD},
        "sudo systemctl restart widget.service\nsudo systemctl try-restart other.service\n",
    )
    assert started_units(installer) == {"widget.service", "other.service"}


def test_a_bare_name_is_read_as_a_service(tmp_path):
    units, installer = _repo(
        tmp_path,
        {"widget.timer": _TIMER_CLEAN, "widget.service": _WORKLOAD},
        "sudo systemctl start widget\n",
    )
    assert started_units(installer) == {"widget.service"}


def test_a_template_instance_keeps_its_instance_name(tmp_path):
    """`foo@bar.service` must survive intact — `@` is not shell noise."""
    units, installer = _repo(
        tmp_path,
        {"widget.timer": _TIMER_CLEAN, "widget.service": _WORKLOAD},
        "sudo systemctl start tunnel@prod.service\n",
    )
    assert started_units(installer) == {"tunnel@prod.service"}


def test_a_template_timer_resolves_to_its_own_instance(tmp_path):
    """A `foo@bar.timer` defaults to `foo@bar.service`, not to `foo@.service`."""
    units, _ = _repo(
        tmp_path,
        {"sync@prod.timer": _TIMER_CLEAN, "sync@prod.service": _WORKLOAD},
        "true\n",
    )
    assert triggered_service("sync@prod.timer", units["sync@prod.timer"]) == ("sync@prod.service")
    assert scheduled_workloads(units) == {"sync@prod.service"}


# --------------------------------------------------------------------------
# the unit graph
# --------------------------------------------------------------------------


def test_scheduled_workloads_is_derived_not_listed(tmp_path):
    """Only `Type=oneshot` services a timer triggers. A daemon is not one."""
    units, _ = _repo(
        tmp_path,
        {
            "widget.timer": _TIMER_CLEAN,
            "widget.service": _WORKLOAD,
            "daemon.service": _DAEMON,
            "orphan.service": _WORKLOAD,
        },
        "true\n",
    )
    assert scheduled_workloads(units) == {"widget.service"}


def test_an_explicit_unit_directive_beats_the_basename_default(tmp_path):
    units, _ = _repo(
        tmp_path,
        {
            "arm.timer": _TIMER_CLEAN.replace(
                "OnCalendar=*-*-* 04:00:00",
                "OnCalendar=*-*-* 04:00:00\nUnit=widget.service",
            ),
            "widget.service": _WORKLOAD,
        },
        "true\n",
    )
    assert triggered_service("arm.timer", units["arm.timer"]) == "widget.service"
    assert scheduled_workloads(units) == {"widget.service"}


def test_the_closure_is_transitive_over_every_dep_key(tmp_path):
    units, _ = _repo(
        tmp_path,
        {
            "a.service": _DAEMON.replace(
                "Description=A long-running thing",
                "Description=A\nRequires=b.service",
            ),
            "b.service": _DAEMON.replace(
                "Description=A long-running thing",
                "Description=B\nWants=c.service\nBindsTo=d.service",
            ),
            "c.service": _DAEMON.replace(
                "Description=A long-running thing",
                "Description=C\nRequisite=e.service",
            ),
            "d.service": _DAEMON,
            "e.service": _DAEMON,
        },
        "true\n",
    )
    assert start_closure("a.service", units) == {
        "a.service",
        "b.service",
        "c.service",
        "d.service",
        "e.service",
    }


def test_the_closure_terminates_on_a_dependency_cycle(tmp_path):
    units, _ = _repo(
        tmp_path,
        {
            "a.service": _DAEMON.replace("Description=A long-running thing", "Description=A\nRequires=b.service"),
            "b.service": _DAEMON.replace("Description=A long-running thing", "Description=B\nRequires=a.service"),
        },
        "true\n",
    )
    assert start_closure("a.service", units) == {"a.service", "b.service"}


def test_only_unit_section_edges_count(tmp_path):
    """A timer's `Unit=` lives in [Timer] and is what the timer TRIGGERS.

    Reading it as a start edge would flag every correctly-written timer, which
    is the guard crying wolf on the exact shape it is asking people to adopt.
    """
    units, _ = _repo(
        tmp_path,
        {
            "arm.timer": _TIMER_CLEAN.replace(
                "OnCalendar=*-*-* 04:00:00",
                "OnCalendar=*-*-* 04:00:00\nUnit=widget.service",
            ),
            "widget.service": _WORKLOAD,
        },
        "true\n",
    )
    assert start_closure("arm.timer", units) == {"arm.timer"}


def test_drop_ins_are_merged_into_the_base_unit(tmp_path):
    """`<name>.service.d/*.conf` edges are as real as the base file's."""
    systemd = tmp_path / "systemd"
    (systemd / "widget.service.d").mkdir(parents=True)
    (systemd / "widget.service").write_text(_WORKLOAD, encoding="utf-8")
    (systemd / "widget.timer").write_text(_TIMER_CLEAN, encoding="utf-8")
    (systemd / "widget.service.d" / "10-after-news.conf").write_text(
        "[Unit]\nWants=news.service\nAfter=news.service\n", encoding="utf-8"
    )
    units = load_units(systemd)
    assert start_closure("widget.service", units) == {
        "widget.service",
        "news.service",
    }
    # `After=` is ordering, not a start dependency, and must not be an edge.
    assert "news.service" in directive(units["widget.service"]["Unit"], "Wants")


def test_a_drop_in_for_a_foreign_base_unit_still_yields_its_edges(tmp_path):
    """A repo may ship only a drop-in for a unit another repo installs."""
    systemd = tmp_path / "systemd"
    (systemd / "foreign.service.d").mkdir(parents=True)
    (systemd / "foreign.service.d" / "10-x.conf").write_text("[Unit]\nRequires=widget.service\n", encoding="utf-8")
    (systemd / "widget.service").write_text(_WORKLOAD, encoding="utf-8")
    (systemd / "widget.timer").write_text(_TIMER_CLEAN, encoding="utf-8")
    units = load_units(systemd)
    assert violations(_script(tmp_path, "sudo systemctl start foreign.service\n"), units) == {
        "foreign.service": {"widget.service"}
    }


def _script(tmp_path, text):
    path = tmp_path / "install-x.sh"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_unit_the_repo_does_not_own_is_named_but_not_expanded(tmp_path):
    """Closure includes a named foreign unit; it cannot see beyond it.

    That blindness is the reason the live box-side scan exists, and is named in
    the module docstring rather than left to be discovered.
    """
    units, _ = _repo(
        tmp_path,
        {
            "a.service": _DAEMON.replace(
                "Description=A long-running thing",
                "Description=A\nRequires=elsewhere.service",
            )
        },
        "true\n",
    )
    assert start_closure("a.service", units) == {"a.service", "elsewhere.service"}


def test_comments_in_unit_files_are_ignored(tmp_path):
    units, _ = _repo(
        tmp_path,
        {
            "widget.timer": "[Unit]\n#Requires=widget.service\n;Wants=widget.service\n[Timer]\nOnCalendar=daily\n",
            "widget.service": _WORKLOAD,
        },
        "sudo systemctl enable --now widget.timer\n",
    )
    assert start_closure("widget.timer", units) == {"widget.timer"}


def test_sections_and_directive_accumulate_repeats():
    parsed = sections("[Unit]\nRequires=a.service b.service\nRequires=c.service\n[Service]\nType=oneshot\n")
    assert directive(parsed["Unit"], "Requires") == [
        "a.service",
        "b.service",
        "c.service",
    ]
    assert directive(parsed["Service"], "Type") == ["oneshot"]


def test_load_units_on_a_missing_directory_is_empty(tmp_path):
    assert load_units(tmp_path / "nope") == {}
