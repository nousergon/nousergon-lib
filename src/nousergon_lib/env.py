"""The environment registry resolver: environment is one setting.

``nousergon_lib.env`` is the ONLY reader of the fleet environment registry
(``nous-ergon-ops/governance/environments.yaml``; clause ``ENV-REGISTRY`` of
the delivery policy, plan S-03, ``alpha-engine-config-I12491``). Code asks it
for an account, region, bucket, stack, topic, instance or SSM prefix by ROLE
and never names one literally (``ENV-NO-LITERALS``)::

    from nousergon_lib import env

    e = env.resolve()                 # NE_ENV, default "prod"
    e.bucket("research")              # -> the research bucket's name
    e.topic_arn("pages")              # -> arn:aws:sns:<region>:<account>:<topic>
    e.s3_uri("crucible")              # -> s3://<crucible bucket>/<prefix>

Which environment: ``resolve(name)`` if given, else ``$NE_ENV``, else the
registry's declared ``default`` (``prod``).

Which registry: ``$NE_ENV_REGISTRY`` (a local path) if set, else the copy
packaged with this library (``environments.yaml`` beside this module). The
packaged copy is a verbatim copy of the governance file; the private repo's CI
validates its file with :func:`validate_registry` and fails when it differs
from the copy in the pinned library (:func:`diff_against_packaged`). So a
caller whose pin is behind resolves against an older registry rather than
against nothing.

Everything fails loudly: an unknown environment, an unknown role, an unknown
key anywhere in the document, a missing key, or a malformed value raises.
Nothing is guessed and nothing falls back to a literal.

CLI (what the private repo's CI runs)::

    python -m nousergon_lib.env check PATH   # schema-validate a registry file
    python -m nousergon_lib.env parity PATH  # PATH vs the packaged copy
    python -m nousergon_lib.env show [NAME]  # print the resolved environment
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

__all__ = [
    "ENV_VAR",
    "REGISTRY_VAR",
    "PACKAGED_REGISTRY",
    "SCHEMA_VERSION",
    "EnvRegistryError",
    "RegistrySchemaError",
    "UnknownEnvironmentError",
    "UnknownRoleError",
    "Environment",
    "load_registry",
    "validate_registry",
    "environment_names",
    "current_name",
    "resolve",
    "diff_against_packaged",
]

#: Selects the environment. Unset means the registry's ``default``.
ENV_VAR = "NE_ENV"
#: Overrides which registry file is read (tests, and the private repo's CI).
REGISTRY_VAR = "NE_ENV_REGISTRY"
#: The registry copy shipped with this library.
PACKAGED_REGISTRY = Path(__file__).parent / "environments.yaml"
#: The only schema version this resolver reads.
SCHEMA_VERSION = 1


class EnvRegistryError(Exception):
    """Base class for every environment-registry failure."""


class RegistrySchemaError(EnvRegistryError, ValueError):
    """The registry document does not match the schema."""


class UnknownEnvironmentError(EnvRegistryError, KeyError):
    """The requested environment is not declared in the registry."""

    def __str__(self) -> str:  # KeyError's default repr-quotes the message
        return str(self.args[0]) if self.args else ""


class UnknownRoleError(EnvRegistryError, KeyError):
    """The requested role (bucket, stack, topic, ...) is not declared."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


# --------------------------------------------------------------------- schema

_ROLE = r"^[a-z][a-z0-9_]*$"
_ENV_NAME = re.compile(r"^[a-z][a-z0-9-]*$")
_ROLE_RE = re.compile(_ROLE)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _Store(_Strict):
    bucket: str = Field(pattern=_ROLE)
    prefix: str = Field(pattern=r"^[A-Za-z0-9_.\-/]*[A-Za-z0-9_.\-]$|^$")


class _Universe(_Strict):
    scope: str = Field(pattern=r"^(full|subset)$")
    bucket: str = Field(pattern=_ROLE)
    key: str = Field(pattern=r"^[A-Za-z0-9_.\-/]+$")


class _Schedule(_Strict):
    days: str = Field(pattern=r"^(DAILY|(MON|TUE|WED|THU|FRI|SAT|SUN)(-(MON|TUE|WED|THU|FRI|SAT|SUN))?)$")
    time_et: str = Field(pattern=r"^([01][0-9]|2[0-3]):[0-5][0-9]$")


class _StateStore(_Strict):
    #: ``None`` declares the branch absent (not yet provisioned), which is a
    #: statement, not an omission: the key itself is still required.
    neon_branch: str | None = Field(pattern=r"^[a-z0-9][a-z0-9_\-]*$")


_BUCKET_NAME = r"^[a-z0-9][a-z0-9.\-]{1,61}[a-z0-9]$"


class _Environment(_Strict):
    account: str = Field(pattern=r"^[0-9]{12}$")
    region: str = Field(pattern=r"^[a-z]{2}(-gov)?-[a-z]+-[0-9]$")
    buckets: dict[str, str] = Field(min_length=1)
    stores: dict[str, _Store]
    stacks: dict[str, str]
    state_store: _StateStore
    topics: dict[str, str]
    instances: dict[str, str]
    ssm_prefixes: dict[str, str]
    llm_cap_usd: float = Field(ge=0)
    universe: _Universe
    schedules_et: dict[str, _Schedule]

    @model_validator(mode="after")
    def _check(self) -> _Environment:
        checks: list[tuple[str, dict[str, Any], str | None]] = [
            ("buckets", self.buckets, _BUCKET_NAME),
            ("stores", self.stores, None),
            ("stacks", self.stacks, r"^[A-Za-z][A-Za-z0-9\-]{0,127}$"),
            ("topics", self.topics, r"^[A-Za-z0-9_\-]{1,256}$"),
            ("instances", self.instances, r"^i-[0-9a-f]{8,17}$"),
            ("ssm_prefixes", self.ssm_prefixes, r"^/[A-Za-z0-9_.\-/]*/$"),
            ("schedules_et", self.schedules_et, None),
        ]
        for section, mapping, value_re in checks:
            for role, value in mapping.items():
                if not _ROLE_RE.match(role):
                    raise ValueError(f"{section}: role {role!r} must match {_ROLE}")
                if value_re is not None and not re.match(value_re, value):
                    raise ValueError(f"{section}.{role}: {value!r} must match {value_re}")
        for role, store in self.stores.items():
            if store.bucket not in self.buckets:
                raise ValueError(f"stores.{role}.bucket: {store.bucket!r} is not a declared bucket role")
        if self.universe.bucket not in self.buckets:
            raise ValueError(f"universe.bucket: {self.universe.bucket!r} is not a declared bucket role")
        return self


class _Registry(_Strict):
    schema_version: int
    default: str
    environments: dict[str, _Environment] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> _Registry:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {self.schema_version} is not supported (this resolver reads {SCHEMA_VERSION})"
            )
        for name in self.environments:
            if not _ENV_NAME.match(name):
                raise ValueError(f"environment name {name!r} must match {_ENV_NAME.pattern}")
        if self.default not in self.environments:
            raise ValueError(f"default {self.default!r} is not a declared environment")
        return self


def validate_registry(doc: Any) -> _Registry:
    """Validate a parsed registry document; raise :class:`RegistrySchemaError`."""
    if not isinstance(doc, Mapping):
        raise RegistrySchemaError(f"registry must be a mapping, got {type(doc).__name__}")
    try:
        return _Registry.model_validate(dict(doc))
    except ValidationError as exc:
        raise RegistrySchemaError(f"environment registry does not match the schema:\n{exc}") from exc


# ------------------------------------------------------------------- loading


def _registry_path(path: str | os.PathLike[str] | None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get(REGISTRY_VAR, "").strip()
    return Path(override) if override else PACKAGED_REGISTRY


def _read(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EnvRegistryError(f"cannot read environment registry {path}: {exc}") from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RegistrySchemaError(f"environment registry {path} is not valid YAML: {exc}") from exc


def load_registry(path: str | os.PathLike[str] | None = None) -> _Registry:
    """Read and validate the registry (``path``, ``$NE_ENV_REGISTRY``, or packaged)."""
    return validate_registry(_read(_registry_path(path)))


def environment_names(path: str | os.PathLike[str] | None = None) -> list[str]:
    """The declared environment names, in registry order."""
    return list(load_registry(path).environments)


def current_name(path: str | os.PathLike[str] | None = None) -> str:
    """``$NE_ENV`` if set, else the registry's ``default``."""
    name = os.environ.get(ENV_VAR, "").strip()
    return name or load_registry(path).default


# ------------------------------------------------------------------ resolving


def _lookup(section: str, mapping: Mapping[str, Any], role: str, env: str) -> Any:
    try:
        return mapping[role]
    except KeyError:
        known = ", ".join(sorted(mapping)) or "(none)"
        raise UnknownRoleError(f"environment {env!r} declares no {section} role {role!r}; declared: {known}") from None


@dataclass(frozen=True)
class Environment:
    """One resolved environment. Every accessor raises on an unknown role."""

    name: str
    account: str
    region: str
    llm_cap_usd: float
    neon_branch: str | None
    universe_scope: str
    _spec: _Environment

    def bucket(self, role: str) -> str:
        return _lookup("bucket", self._spec.buckets, role, self.name)

    def stack(self, role: str) -> str:
        return _lookup("stack", self._spec.stacks, role, self.name)

    def topic(self, role: str) -> str:
        return _lookup("topic", self._spec.topics, role, self.name)

    def topic_arn(self, role: str) -> str:
        return f"arn:aws:sns:{self.region}:{self.account}:{self.topic(role)}"

    def instance(self, role: str) -> str:
        return _lookup("instance", self._spec.instances, role, self.name)

    def ssm_prefix(self, role: str) -> str:
        return _lookup("ssm prefix", self._spec.ssm_prefixes, role, self.name)

    def s3_uri(self, store: str) -> str:
        """``s3://<bucket>/<prefix>`` of a declared store."""
        spec = _lookup("store", self._spec.stores, store, self.name)
        base = f"s3://{self.bucket(spec.bucket)}"
        return f"{base}/{spec.prefix}" if spec.prefix else base

    @property
    def universe_uri(self) -> str:
        u = self._spec.universe
        return f"s3://{self.bucket(u.bucket)}/{u.key}"

    def schedule_et(self, role: str) -> tuple[str, str]:
        """``(days, "HH:MM")`` in America/New_York for a declared schedule."""
        s = _lookup("schedule", self._spec.schedules_et, role, self.name)
        return s.days, s.time_et

    def roles(self, section: str) -> list[str]:
        """Declared role names of one section (``buckets``, ``stacks``, ...)."""
        mapping = getattr(self._spec, section, None)
        if not isinstance(mapping, dict):
            raise UnknownRoleError(f"{section!r} is not a role section of the registry")
        return sorted(mapping)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, **self._spec.model_dump()}


def resolve(name: str | None = None, *, registry: str | os.PathLike[str] | None = None) -> Environment:
    """Resolve one environment (``name``, else ``$NE_ENV``, else the default)."""
    reg = load_registry(registry)
    chosen = name or os.environ.get(ENV_VAR, "").strip() or reg.default
    spec = reg.environments.get(chosen)
    if spec is None:
        known = ", ".join(reg.environments)
        source = "argument" if name else (f"${ENV_VAR}" if os.environ.get(ENV_VAR, "").strip() else "default")
        raise UnknownEnvironmentError(
            f"environment {chosen!r} (from {source}) is not declared in the registry; declared: {known}"
        )
    return Environment(
        name=chosen,
        account=spec.account,
        region=spec.region,
        llm_cap_usd=spec.llm_cap_usd,
        neon_branch=spec.state_store.neon_branch,
        universe_scope=spec.universe.scope,
        _spec=spec,
    )


# -------------------------------------------------------------------- parity


def diff_against_packaged(path: str | os.PathLike[str]) -> list[str]:
    """Differences between ``path`` and the packaged copy, as readable lines.

    Compared after parsing, so comments and formatting never count; both
    sides must also pass the schema. Empty list means in sync.
    """
    theirs = load_registry(path).model_dump()
    ours = load_registry(PACKAGED_REGISTRY).model_dump()
    out: list[str] = []

    def walk(a: Any, b: Any, where: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                here = f"{where}.{k}" if where else str(k)
                if k not in b:
                    out.append(f"{here}: only in {path}")
                elif k not in a:
                    out.append(f"{here}: only in the packaged copy")
                else:
                    walk(a[k], b[k], here)
        elif a != b:
            out.append(f"{where}: {path} has {a!r}, packaged copy has {b!r}")

    walk(theirs, ours, "")
    return out


# ----------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m nousergon_lib.env", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="schema-validate a registry file")
    c.add_argument("path")
    q = sub.add_parser("parity", help="compare a registry file with the packaged copy")
    q.add_argument("path")
    s = sub.add_parser("show", help="print one resolved environment as JSON")
    s.add_argument("name", nargs="?")
    args = p.parse_args(argv)
    try:
        if args.cmd == "check":
            reg = load_registry(args.path)
            print(f"OK: {args.path} declares {', '.join(reg.environments)} (default {reg.default})")
            return 0
        if args.cmd == "parity":
            diffs = diff_against_packaged(args.path)
            if diffs:
                print(f"DRIFT: {args.path} differs from the registry packaged in nousergon-lib:", file=sys.stderr)
                for d in diffs:
                    print(f"  {d}", file=sys.stderr)
                print(
                    f"Fix: copy {args.path} verbatim to nousergon-lib "
                    "src/nousergon_lib/environments.yaml in a PR, then move the pin.",
                    file=sys.stderr,
                )
                return 1
            print(f"OK: {args.path} matches the packaged copy")
            return 0
        print(json.dumps(resolve(args.name).as_dict(), indent=2, sort_keys=True))
        return 0
    except EnvRegistryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
