"""Resource-type -> billing-prefix map, and the CloudFormation reader the
pre-merge cost gate grades schedules with.

Lifted from ``alpha-engine-config/scripts/cost_resource_map.py`` on its second
adoption (``policy-shared-code``: the second consumer is the trigger, not the
third). The private repo keeps the single source of truth; this package ships
the DERIVED, prefix-only half of it as package data so any repo in the fleet —
public or private — runs the same grader from the library it already pins,
with no cross-repo fetch and no credential.

**Fail loud.** Every function here raises on malformed input. A cost gate that
degrades to "nothing to grade" when its map fails to parse is worse than no
gate: it reports a green zero for a class it did not look at.

Refs alpha-engine-config-I11228.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

#: Where the packaged, derived documents live. Regenerated in the private
#: repo and published here by PR; never hand-edited (see the header each file
#: carries).
DATA_DIR = Path(__file__).resolve().parent / "data"
RESOURCE_MAP_PATH = DATA_DIR / "resource_map.yaml"

#: Bounds used when a map omits them. Both are deliberately the SAME numbers
#: the shipped document declares, so a test asserting the document's values
#: cannot pass by accident against a silently-empty one.
DEFAULT_MAX_RETRY_ATTEMPTS = 3
DEFAULT_MIN_GHA_CRON_INTERVAL_MINUTES = 15


class ResourceMapError(ValueError):
    """The map is unusable. Raised, never returned as an empty map."""


def load(path: Path | str | None = None) -> dict:
    """Parse and validate the map. Raises :class:`ResourceMapError`.

    ``path`` defaults to the document packaged with this library.
    """
    p = Path(path) if path is not None else RESOURCE_MAP_PATH
    if not p.is_file():
        raise ResourceMapError(f"{p} does not exist")
    doc = yaml.safe_load(p.read_text())
    if not isinstance(doc, dict):
        raise ResourceMapError(f"{p} is not a mapping")

    types = doc.get("resource_types")
    if not isinstance(types, dict) or not types:
        raise ResourceMapError(f"{p}: `resource_types` must be a non-empty mapping")
    for key, prefix in types.items():
        if not isinstance(key, str) or key.count("::") != 2 or not key.startswith("AWS::"):
            raise ResourceMapError(
                f"{p}: resource_types key {key!r} is not an `AWS::Service::Type` name"
            )
        if not isinstance(prefix, str) or not prefix or prefix != prefix.lower():
            raise ResourceMapError(
                f"{p}: resource_types[{key!r}] must be a lowercase billing prefix, "
                f"got {prefix!r}"
            )

    rules = doc.get("schedule_rules") or {}
    if not isinstance(rules, dict):
        raise ResourceMapError(f"{p}: `schedule_rules` must be a mapping")
    for field, default in (
        ("max_retry_attempts", DEFAULT_MAX_RETRY_ATTEMPTS),
        ("min_gha_cron_interval_minutes", DEFAULT_MIN_GHA_CRON_INTERVAL_MINUTES),
    ):
        value = rules.get(field, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ResourceMapError(
                f"{p}: schedule_rules.{field} must be a non-negative integer, got {value!r}"
            )
        rules[field] = value
    doc["schedule_rules"] = rules
    return doc


# -- CloudFormation, read structurally ---------------------------------------
#
# A schedule cannot be graded line by line. `MaximumRetryAttempts` sits four
# levels under the resource it bounds, and the target service is usually a
# `!GetAtt`/`!Ref` at a logical id declared hundreds of lines away. Grading the
# ADDED LINES alone would demand a retry policy be re-added every time someone
# edits a cron expression -- noise, and noise is how a check becomes warn-only
# forever.


class Intrinsic:
    """A CloudFormation short-form tag: ``!GetAtt Dispatcher.Arn``.

    ``yaml.safe_load`` refuses these outright, and the permissive alternative —
    dropping the tag and keeping the scalar — would make ``!Ref Foo`` and the
    string ``"Foo"`` indistinguishable, which is exactly the resolution this
    module performs.
    """

    __slots__ = ("tag", "value")

    def __init__(self, tag: str, value: Any) -> None:
        self.tag = tag
        self.value = value

    def __repr__(self) -> str:  # pragma: no cover — diagnostics only
        return f"Intrinsic({self.tag!r}, {self.value!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Intrinsic)
            and self.tag == other.tag
            and self.value == other.value
        )

    def __hash__(self) -> int:  # pragma: no cover — defined for dict/set use
        return hash((self.tag, repr(self.value)))


#: A top-level ``Resources:`` mapping — the one thing every CloudFormation
#: template has and nothing else in a fleet repo's YAML does.
_LOOKS_LIKE_TEMPLATE = re.compile(r"^\s*[\"']?Resources[\"']?\s*:", re.MULTILINE)


class CfnLoader(yaml.SafeLoader):
    """SafeLoader that keeps CloudFormation intrinsics instead of dying on them."""


def _construct_intrinsic(loader: yaml.Loader, tag_suffix: str, node: yaml.Node) -> Intrinsic:
    if isinstance(node, yaml.ScalarNode):
        value: Any = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    elif isinstance(node, yaml.MappingNode):
        value = loader.construct_mapping(node, deep=True)
    else:  # pragma: no cover — PyYAML emits no fourth node kind
        # Not a swallow: an unknown node kind under an intrinsic tag means the
        # template is a shape this reader has never seen, and guessing at it is
        # how a gate reports a green zero for something it did not understand.
        raise ResourceMapError(
            f"a `!{tag_suffix}` intrinsic carries an unsupported node kind "
            f"{type(node).__name__}, so the template cannot be graded"
        )
    return Intrinsic(tag_suffix, value)


CfnLoader.add_multi_constructor("!", _construct_intrinsic)


class CfnResource:
    """One ``Resources:`` entry, with the line span it occupies in the file.

    The span is what lets the gate ask "did this diff touch THIS resource?"
    rather than "does this file contain a schedule anywhere?".
    """

    __slots__ = ("logical_id", "type", "properties", "first_line", "last_line")

    def __init__(
        self,
        logical_id: str,
        type_: str,
        properties: dict,
        first_line: int,
        last_line: int,
    ) -> None:
        self.logical_id = logical_id
        self.type = type_
        self.properties = properties
        self.first_line = first_line
        self.last_line = last_line

    def touches(self, lines: set) -> bool:
        return any(self.first_line <= n <= self.last_line for n in lines)


def parse_cfn_resources(text: str) -> list:
    """Every ``Resources:`` entry with a ``Type:``, with 1-based line spans.

    Returns ``[]`` for a document that is not a CloudFormation template — a
    workflow file, a docs page, a JSON blob with no ``Resources`` key. That is
    a legitimate answer, not a swallow: a document with no ``Resources``
    mapping declares no cloud resource, so "no schedules here" is the true
    answer rather than an unexamined one.

    Raises :class:`ResourceMapError` on a document that LOOKS like a template
    and does not parse. A template the gate cannot read must never read as a
    template with no schedules — that is the silent pass this gate exists to
    remove.
    """
    try:
        doc = yaml.load(text, Loader=CfnLoader)  # noqa: S506 — CfnLoader is a SafeLoader
    except yaml.YAMLError as exc:
        if _LOOKS_LIKE_TEMPLATE.search(text):
            raise ResourceMapError(
                f"a CloudFormation template failed to parse and so cannot be "
                f"graded: {exc}"
            ) from exc
        return []
    if not isinstance(doc, dict) or not isinstance(doc.get("Resources"), dict):
        return []

    # `compose` is parsed a second time rather than shared with `load`: PyYAML
    # exposes no supported way to keep marks off a constructed object, and the
    # templates involved are tens of kilobytes. Correctness over one parse.
    root = yaml.compose(text, Loader=CfnLoader)
    spans: dict = {}
    if isinstance(root, yaml.MappingNode):
        for key_node, value_node in root.value:
            if getattr(key_node, "value", None) != "Resources":
                continue
            if not isinstance(value_node, yaml.MappingNode):
                continue
            for lid_node, body_node in value_node.value:
                spans[str(lid_node.value)] = (
                    lid_node.start_mark.line + 1,
                    body_node.end_mark.line + 1,
                )

    out: list = []
    for lid, body in doc["Resources"].items():
        if not isinstance(body, dict) or not isinstance(body.get("Type"), str):
            continue
        first, last = spans.get(str(lid), (0, 0))
        props = body.get("Properties")
        out.append(
            CfnResource(
                str(lid),
                body["Type"],
                props if isinstance(props, dict) else {},
                first,
                last,
            )
        )
    return out


def resolve_target_service(
    arn: Any,
    by_logical_id: dict,
    resource_types: dict,
) -> str | None:
    """The billing prefix a schedule target bills to, or ``None`` if unresolvable.

    ``None`` is reported by the caller as NOT GRADED and never as a pass. The
    three shapes that occur in fleet templates:

      * a literal ``arn:aws:states:...`` (or one wrapped in ``!Sub``)
      * ``!GetAtt Dispatcher.Arn``
      * ``!Ref EodStateMachine``
    """
    if isinstance(arn, dict) and len(arn) == 1:
        # A JSON template, or a YAML one written in long form. `{"Fn::GetAtt":
        # ["Dispatcher", "Arn"]}` and `!GetAtt Dispatcher.Arn` are the same
        # thing, and a gate that read one and not the other would be blind to
        # every JSON stack in the fleet.
        ((key, value),) = arn.items()
        if isinstance(key, str):
            arn = Intrinsic(key.removeprefix("Fn::"), value)

    if isinstance(arn, Intrinsic):
        if arn.tag == "Ref" and isinstance(arn.value, str):
            return _prefix_for_logical_id(arn.value, by_logical_id, resource_types)
        if arn.tag == "GetAtt":
            raw = arn.value
            if isinstance(raw, list):
                raw = ".".join(str(part) for part in raw)
            if isinstance(raw, str):
                return _prefix_for_logical_id(
                    raw.split(".", 1)[0], by_logical_id, resource_types
                )
            return None
        if arn.tag == "Sub":
            raw = arn.value[0] if isinstance(arn.value, list) and arn.value else arn.value
            return _prefix_from_arn_literal(raw) if isinstance(raw, str) else None
        return None
    if isinstance(arn, str):
        return _prefix_from_arn_literal(arn)
    return None


def _prefix_from_arn_literal(text: str) -> str | None:
    parts = text.split(":")
    if len(parts) >= 3 and parts[0] == "arn" and parts[2]:
        return parts[2].lower()
    return None


def _prefix_for_logical_id(
    logical_id: str,
    by_logical_id: dict,
    resource_types: dict,
) -> str | None:
    type_ = by_logical_id.get(logical_id)
    if type_ is None:
        return None
    # An unmapped TYPE here is not the same question as an unmapped type in the
    # resource class: the caller reports it as unresolved so the resource class
    # raises it once, with the right message, instead of twice.
    return resource_types.get(type_)
